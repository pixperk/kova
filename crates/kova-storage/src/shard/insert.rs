//! `Shard::insert` and `Shard::insert_many`.
//!
//! Both follow the same 3-phase discipline (validate / commit / apply),
//! with WAL `sync` as the single commit point. `insert_many` adds
//! pre-grow + group-commit + put-many on top.

use std::collections::HashSet;

use kova_core::{Metadata, MetadataStore, Vector, VectorId, VectorStore};
use kova_index::{Index, KovaIndexError};

use crate::{Record, Wal};

use super::{Shard, ShardError};

impl<D, V, M, W> Shard<D, V, M, W>
where
    D: kova_core::Distance,
    V: VectorStore,
    M: MetadataStore,
    W: Wal,
{
    /// Insert `(id, vector, metadata)` under the log-then-mutate discipline.
    ///
    /// Three phases, with the WAL `sync` as the **commit point** :
    ///
    /// 1. **Pre-commit validation.** Cheap upfront checks (duplicate id,
    ///    dim mismatch). On failure, returns Err *before* touching the
    ///    WAL ; no state change anywhere.
    /// 2. **Commit.** `wal.append` + `wal.sync`. After `sync` returns Ok,
    ///    the operation is durable.
    /// 3. **Apply.** Mutate the in-memory index (which writes through to
    ///    the [`VectorStore`]) and the [`MetadataStore`].
    ///
    /// # Failure semantics
    ///
    /// - Returning `Ok(())` means the operation is committed **and** applied.
    /// - Returning `Err(...)` means the operation was rejected in phase 1
    ///   and the shard state is unchanged.
    /// - A failure in **phase 3** (after WAL commit) is treated as a
    ///   broken invariant : the WAL says the op happened, the in-memory
    ///   state disagrees. The only safe move is to abort the process so
    ///   replay on reopen reconciles. This impl **panics** in that case.
    ///   The crash test exercises this path ; pre-commit validation is
    ///   aggressive precisely to keep phase-3 failures genuinely rare
    ///   (disk full, EIO, etc., not "user passed a bad input").
    ///
    /// # Errors
    /// - [`ShardError::Index`] with `KovaIndexError::DuplicateId` if `id`
    ///   already exists.
    /// - [`ShardError::Index`] with `KovaIndexError::DimensionMismatch` if
    ///   `vector.dim()` doesn't match the shard's pinned dim.
    /// - [`ShardError::Backend`] from `wal.append` / `wal.sync`.
    ///
    /// # Panics
    /// Panics with a clear message if the in-memory apply (phase 3) fails
    /// after a successful WAL commit. See "Failure semantics" above.
    pub fn insert(
        &mut self,
        id: VectorId,
        vector: Vector,
        metadata: Metadata,
    ) -> Result<(), ShardError> {
        // ----------------------------------------------------------------
        // Phase 1 : pre-commit validation.
        //
        // Every failure-mode we can detect statically is rejected here,
        // before the WAL is touched. The WAL stays clean of records that
        // could not have applied successfully, which keeps replay simple
        // (every replayed Insert can be assumed valid by construction).
        // ----------------------------------------------------------------

        // Duplicate id : O(1) `HashMap` lookup on the in-memory node map.
        if self.index.top_layer_of(id).is_some() {
            return Err(KovaIndexError::DuplicateId { id }.into());
        }

        // Dim mismatch : check against the index's pinned dim first (set
        // by the first insert), falling back to the underlying store's
        // pinned dim (e.g. MmapVectorStore reads it from the file header
        // at open time, so it's known even before the first insert).
        if let Some(expected) = self.index.dim().or_else(|| self.index.store_dim())
            && vector.dim() != expected
        {
            return Err(KovaIndexError::DimensionMismatch {
                expected,
                got: vector.dim(),
            }
            .into());
        }

        // ----------------------------------------------------------------
        // Phase 2 : commit.
        //
        // Clones are unavoidable : Record needs to own the data for
        // serialisation, and phase 3 needs it for the apply.
        // ----------------------------------------------------------------
        let record = Record::Insert {
            id,
            vector: vector.clone(),
            metadata: metadata.clone(),
        };

        self.wal.append(&record).map_err(ShardError::backend)?;
        self.wal.sync().map_err(ShardError::backend)?;

        // ----------------------------------------------------------------
        // Phase 3 : apply (post-commit).
        //
        // Any failure here means the WAL committed an op that in-memory
        // state failed to apply. There is no clean recovery in-process :
        // the caller's view ("did it commit?") and the WAL's view
        // ("yes") have already diverged from the apply layer's view ("no").
        //
        // Panic. The process dies, the next reopen replays the WAL, and
        // the durable record gets re-applied to a fresh in-memory state.
        // ----------------------------------------------------------------
        if let Err(e) = self.index.insert(id, vector) {
            panic!(
                "Shard::insert phase-3 apply failure on index.insert: {e:?} \
                 (WAL has committed the record ; aborting so replay can reconcile)"
            );
        }
        if let Err(e) = self.metadata.put(id, metadata) {
            panic!(
                "Shard::insert phase-3 apply failure on metadata.put: {e:?} \
                 (WAL has committed the record ; aborting so replay can reconcile)"
            );
        }

        Ok(())
    }

    /// Insert many `(id, vector, metadata)` triples as a single batch.
    ///
    /// Same 3-phase discipline as [`Self::insert`], but the entire batch
    /// commits or rejects together :
    ///
    /// 1. **Pre-commit validation.** Every item is checked (duplicate
    ///    against the existing index, duplicate within the batch, dim
    ///    mismatch). The first failure rejects the *whole* batch ; no
    ///    WAL append, no state change. The vector store is pre-grown to
    ///    fit the whole batch here too, so disk-full surfaces upfront.
    /// 2. **Commit.** All `Insert` records are appended to the WAL, then
    ///    **one** `wal.sync` covers the whole batch. This is the
    ///    headline win : group-committing N records amortises the fsync
    ///    cost across the batch.
    /// 3. **Apply.** Index inserts run one-at-a-time (HNSW construction
    ///    is inherently sequential), but the metadata store's
    ///    `put_many` collapses N full-file rewrites into one. Post-commit
    ///    failures **panic** per the same Postgres-style rule as
    ///    [`Self::insert`].
    ///
    /// # Performance shape
    ///
    /// Per-op fsync cost dominates singleton `insert`. For N inserts :
    /// - Singleton `insert` x N  : ~N × (wal.sync + metadata flush)
    /// - `insert_many`           : 1 × wal.sync + 1 × metadata flush + N × mmap writes
    ///
    /// At realistic batch sizes (100-10k) the speedup is multiple orders
    /// of magnitude on disk-backed shards. In-memory composition sees
    /// only the WAL append amortisation (already cheap, no real fsync).
    ///
    /// # Errors
    /// See [`Self::insert`] for the per-failure-mode error set. The
    /// batch is rejected on the first failure.
    ///
    /// # Panics
    /// On phase-3 apply failure (rare ; pre-commit validation aims to
    /// eliminate the easy cases). See [`Self::insert`] for the rationale.
    pub fn insert_many<I>(&mut self, items: I) -> Result<(), ShardError>
    where
        I: IntoIterator<Item = (VectorId, Vector, Metadata)>,
    {
        let items: Vec<(VectorId, Vector, Metadata)> = items.into_iter().collect();
        if items.is_empty() {
            return Ok(());
        }

        // -------- Phase 1 : validate the whole batch --------
        //
        // `expected_dim` starts from whatever's pinned (index first,
        // store-fallback second) and may be set by the first item in
        // the batch when both are None.
        let mut expected_dim = self.index.dim().or_else(|| self.index.store_dim());
        let mut seen_in_batch: HashSet<VectorId> = HashSet::with_capacity(items.len());

        for (id, vector, _) in &items {
            if self.index.top_layer_of(*id).is_some() {
                return Err(KovaIndexError::DuplicateId { id: *id }.into());
            }
            if !seen_in_batch.insert(*id) {
                // Duplicate within the batch itself.
                return Err(KovaIndexError::DuplicateId { id: *id }.into());
            }
            match expected_dim {
                Some(d) if vector.dim() != d => {
                    return Err(KovaIndexError::DimensionMismatch {
                        expected: d,
                        got: vector.dim(),
                    }
                    .into());
                }
                None => {
                    // First insert into a totally fresh shard sets the dim
                    // for the rest of the batch.
                    expected_dim = Some(vector.dim());
                }
                _ => {}
            }
        }

        // Pre-grow the vector store to fit the whole batch. Failures
        // here (e.g. ENOSPC) reject the batch before WAL commit, which
        // is exactly what we want.
        self.index.reserve_store(items.len())?;

        // -------- Phase 2 : group-commit --------
        for (id, vector, metadata) in &items {
            let record = Record::Insert {
                id: *id,
                vector: vector.clone(),
                metadata: metadata.clone(),
            };
            self.wal.append(&record).map_err(ShardError::backend)?;
        }
        self.wal.sync().map_err(ShardError::backend)?;

        // -------- Phase 3 : apply, panic on failure --------
        // HNSW insertion is sequential by nature ; we just run it for
        // each. The wins for batching live in WAL + metadata, not here.
        for (id, vector, _) in &items {
            if let Err(e) = self.index.insert(*id, vector.clone()) {
                panic!(
                    "Shard::insert_many phase-3 apply failure on index.insert ({id}): {e:?} \
                     (WAL has committed the batch ; aborting so replay can reconcile)"
                );
            }
        }

        // ONE metadata flush for the whole batch.
        let metadata_items = items.into_iter().map(|(id, _, m)| (id, m));
        if let Err(e) = self.metadata.put_many(metadata_items) {
            panic!(
                "Shard::insert_many phase-3 apply failure on metadata.put_many: {e:?} \
                 (WAL has committed the batch ; aborting so replay can reconcile)"
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use kova_core::{
        InMemoryMetadataStore, InMemoryVectorStore, L2, Metadata, Value, Vector, VectorId,
    };
    use kova_index::{HnswParams, KovaIndexError};
    use tempfile::tempdir;

    use crate::InMemoryWal;

    use super::super::{Shard, ShardError};

    fn v(data: Vec<f32>) -> Vector {
        Vector::try_new(data).unwrap()
    }
    fn id(n: u64) -> VectorId {
        VectorId::new(n)
    }
    fn tag_meta(tag: &str) -> Metadata {
        let mut m = Metadata::new();
        m.insert("tag".into(), Value::String(tag.into()));
        m
    }

    // ---------- singleton insert ----------

    /// Smoke test : compose in-memory primitives, insert a couple of
    /// vectors, search, verify hits + metadata. Exercises every public
    /// method on the generic impl in one flow.
    #[test]
    fn insert_then_search_with_in_memory_primitives() {
        let mut shard = Shard::from_parts(
            L2,
            HnswParams::default(),
            InMemoryVectorStore::new(),
            InMemoryMetadataStore::new(),
            InMemoryWal::new(),
        )
        .unwrap();

        assert!(shard.is_empty());
        assert_eq!(shard.len(), 0);

        shard
            .insert(id(1), v(vec![1.0, 0.0]), tag_meta("alpha"))
            .unwrap();
        shard
            .insert(id(2), v(vec![0.0, 1.0]), tag_meta("beta"))
            .unwrap();

        assert_eq!(shard.len(), 2);
        assert!(shard.contains(id(1)));
        assert!(!shard.contains(id(99)));

        let hits = shard.search(&v(vec![1.0, 0.05]), 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, id(1));
        assert_eq!(
            hits[0].metadata.get("tag"),
            Some(&Value::String("alpha".into()))
        );
    }

    /// Inserting the same id twice errors with `DuplicateId` and does not
    /// append a second WAL record (the duplicate check fires before any
    /// state change).
    #[test]
    fn duplicate_insert_is_rejected() {
        let mut shard = Shard::from_parts(
            L2,
            HnswParams::default(),
            InMemoryVectorStore::new(),
            InMemoryMetadataStore::new(),
            InMemoryWal::new(),
        )
        .unwrap();

        shard.insert(id(7), v(vec![1.0]), Metadata::new()).unwrap();
        let err = shard
            .insert(id(7), v(vec![2.0]), Metadata::new())
            .unwrap_err();
        match err {
            ShardError::Index(KovaIndexError::DuplicateId { id: got }) => {
                assert_eq!(got, id(7));
            }
            other => panic!("expected DuplicateId, got {other:?}"),
        }
        assert_eq!(shard.len(), 1);
    }

    /// Dim mismatch on `insert` is caught in the pre-commit phase, so no
    /// WAL record is written. Verified by reopening : if the bad insert
    /// had appended, replay would see two records and fail on apply.
    #[test]
    fn dim_mismatch_on_insert_does_not_poison_wal() {
        let dir = tempdir().unwrap();
        let mut shard = Shard::open(dir.path(), 3, L2, HnswParams::default()).unwrap();

        // First good insert : establishes the index's dim.
        shard
            .insert(id(1), v(vec![1.0, 2.0, 3.0]), Metadata::new())
            .unwrap();

        // Wrong-dim insert : should fail before WAL append.
        let err = shard
            .insert(id(2), v(vec![1.0, 2.0]), Metadata::new())
            .unwrap_err();
        match err {
            ShardError::Index(KovaIndexError::DimensionMismatch { expected, got }) => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }

        drop(shard);
        let shard = Shard::open(dir.path(), 3, L2, HnswParams::default()).unwrap();
        assert_eq!(shard.len(), 1);
        assert!(shard.contains(id(1)));
        assert!(!shard.contains(id(2)));
    }

    /// Dim mismatch caught even on the FIRST insert (before the index has
    /// pinned its own dim), because the underlying `MmapVectorStore` has
    /// its dim from the file header. Same no-poison guarantee.
    #[test]
    fn dim_mismatch_on_first_insert_is_caught_via_store_dim() {
        let dir = tempdir().unwrap();
        let mut shard = Shard::open(dir.path(), 3, L2, HnswParams::default()).unwrap();

        let err = shard
            .insert(id(1), v(vec![1.0, 2.0]), Metadata::new())
            .unwrap_err();
        assert!(matches!(
            err,
            ShardError::Index(KovaIndexError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));

        drop(shard);
        let shard = Shard::open(dir.path(), 3, L2, HnswParams::default()).unwrap();
        assert_eq!(shard.len(), 0);
    }

    // ---------- batched insert ----------

    /// Happy path : [`Shard::insert_many`] a small batch, every id is
    /// searchable with its metadata, `len` reflects the batch size.
    #[test]
    fn insert_many_happy_path_in_memory() {
        let mut shard = Shard::from_parts(
            L2,
            HnswParams::default(),
            InMemoryVectorStore::new(),
            InMemoryMetadataStore::new(),
            InMemoryWal::new(),
        )
        .unwrap();

        let batch = (0u16..10)
            .map(|n| {
                (
                    id(u64::from(n)),
                    v(vec![f32::from(n), f32::from(n + 1)]),
                    tag_meta(&format!("t{n}")),
                )
            })
            .collect::<Vec<_>>();

        shard.insert_many(batch).unwrap();

        assert_eq!(shard.len(), 10);
        for n in 0..10 {
            assert!(shard.contains(id(n)), "id {n} missing after batch");
        }

        let hits = shard.search(&v(vec![5.0, 6.0]), 1).unwrap();
        assert_eq!(hits[0].id, id(5));
        assert_eq!(
            hits[0].metadata.get("tag"),
            Some(&Value::String("t5".into()))
        );
    }

    /// Empty batch is a clean no-op : returns Ok, no state change, no
    /// WAL record.
    #[test]
    fn insert_many_empty_batch_is_noop() {
        let mut shard = Shard::from_parts(
            L2,
            HnswParams::default(),
            InMemoryVectorStore::new(),
            InMemoryMetadataStore::new(),
            InMemoryWal::new(),
        )
        .unwrap();
        let empty: Vec<(VectorId, Vector, Metadata)> = Vec::new();
        shard.insert_many(empty).unwrap();
        assert!(shard.is_empty());
    }

    /// Duplicate id within the batch itself rejects the whole batch
    /// pre-commit ; nothing lands in the shard.
    #[test]
    fn insert_many_rejects_duplicate_within_batch() {
        let mut shard = Shard::from_parts(
            L2,
            HnswParams::default(),
            InMemoryVectorStore::new(),
            InMemoryMetadataStore::new(),
            InMemoryWal::new(),
        )
        .unwrap();

        let batch = vec![
            (id(1), v(vec![1.0, 0.0]), Metadata::new()),
            (id(2), v(vec![0.0, 1.0]), Metadata::new()),
            (id(1), v(vec![9.9, 9.9]), Metadata::new()),
        ];

        let err = shard.insert_many(batch).unwrap_err();
        assert!(matches!(
            err,
            ShardError::Index(KovaIndexError::DuplicateId { id }) if id == VectorId::new(1)
        ));
        assert!(shard.is_empty(), "no item should have landed");
    }

    /// Duplicate against an id already in the index also rejects the
    /// whole batch pre-commit.
    #[test]
    fn insert_many_rejects_duplicate_against_existing() {
        let mut shard = Shard::from_parts(
            L2,
            HnswParams::default(),
            InMemoryVectorStore::new(),
            InMemoryMetadataStore::new(),
            InMemoryWal::new(),
        )
        .unwrap();

        shard
            .insert(id(42), v(vec![1.0, 2.0]), Metadata::new())
            .unwrap();

        let batch = vec![
            (id(1), v(vec![1.0, 0.0]), Metadata::new()),
            (id(42), v(vec![9.0, 9.0]), Metadata::new()),
        ];

        let err = shard.insert_many(batch).unwrap_err();
        assert!(matches!(
            err,
            ShardError::Index(KovaIndexError::DuplicateId { id }) if id == VectorId::new(42)
        ));
        assert_eq!(shard.len(), 1);
        assert!(shard.contains(id(42)));
        assert!(!shard.contains(id(1)));
    }

    /// Dim mismatch in any item rejects the whole batch pre-commit.
    #[test]
    fn insert_many_rejects_dim_mismatch() {
        let mut shard = Shard::from_parts(
            L2,
            HnswParams::default(),
            InMemoryVectorStore::new(),
            InMemoryMetadataStore::new(),
            InMemoryWal::new(),
        )
        .unwrap();

        let batch = vec![
            (id(1), v(vec![1.0, 0.0]), Metadata::new()),
            (id(2), v(vec![0.0, 1.0, 0.0]), Metadata::new()),
        ];

        let err = shard.insert_many(batch).unwrap_err();
        assert!(matches!(
            err,
            ShardError::Index(KovaIndexError::DimensionMismatch {
                expected: 2,
                got: 3
            })
        ));
        assert!(shard.is_empty());
    }

    /// File-backed end-to-end : `insert_many` a batch, drop, reopen,
    /// verify every id survives. Exercises the WAL group-commit replay path.
    #[test]
    fn insert_many_persists_across_reopen() {
        let dir = tempdir().unwrap();
        let batch: Vec<_> = (0u16..50)
            .map(|n| {
                (
                    id(u64::from(n)),
                    v(vec![f32::from(n), f32::from(100 - n)]),
                    tag_meta(&format!("k{n}")),
                )
            })
            .collect();

        {
            let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
            shard.insert_many(batch).unwrap();
            assert_eq!(shard.len(), 50);
        }

        let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        assert_eq!(shard.len(), 50);
        for n in 0..50 {
            assert!(shard.contains(id(n)), "id {n} missing after reopen");
        }
        let hits = shard.search(&v(vec![5.0, 95.0]), 1).unwrap();
        assert_eq!(hits[0].id, id(5));
        assert_eq!(
            hits[0].metadata.get("tag"),
            Some(&Value::String("k5".into()))
        );
    }
}

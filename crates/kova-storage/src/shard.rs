//! `Shard` : the composition layer.
//!
//! Composes one of each persistence primitive ([`Wal`], [`VectorStore`],
//! [`MetadataStore`]) and an in-memory [`HnswIndex`] on top of the vector
//! store. Generic over all four so the same `Shard` type drives both the
//! production combo (`FileWal + MmapVectorStore + FileMetadataStore`) and
//! test fixtures (`InMemoryWal + InMemoryVectorStore + InMemoryMetadataStore`),
//! and so future backends (S3, distributed log, columnar metadata) plug in
//! without touching this file.
//!
//! The load-bearing rule is **log-then-mutate** : every `insert` appends a
//! [`Record`](crate::Record) to the WAL and `sync`s it before any in-memory
//! or on-disk state changes. On crash, the WAL is the witness ; recovery
//! replays it on reopen, idempotently re-applying every record to the
//! stores and the index.
//!
//! # Layout on disk (when composed with file-backed primitives)
//!
//! ```text
//! data_dir/
//!   wal/             <- FileWal segments
//!   vectors.mmap     <- MmapVectorStore file
//!   metadata.bin     <- FileMetadataStore file
//! ```

use std::collections::HashSet;
use std::error::Error as StdError;
use std::path::Path;

use kova_core::{Distance, Metadata, MetadataStore, Vector, VectorId, VectorStore};
use kova_index::{HnswIndex, HnswParams, Index, KovaIndexError};
use thiserror::Error;

use crate::{FileMetadataStore, FileWal, Lsn, MmapVectorStore, Record, Wal};

/// Default RNG seed for [`Shard::from_parts`] / [`Shard::open`], mirroring
/// `HnswIndex::new`'s default. Tests that need a different seed call
/// [`Shard::from_parts_seeded`] / [`Shard::open_seeded`].
const DEFAULT_SEED: u64 = 0xDEAD_BEEF_DEAD_BEEF;

/// Errors produced by [`Shard`] operations.
///
/// Two-level shape :
///
/// - [`Self::Index`] carries `KovaIndexError` directly (the index lives
///   inside `kova-index`, which has its own concrete error type).
/// - [`Self::Backend`] boxes any of `V::Error`, `M::Error`, or `W::Error`.
///   Generic composition cannot enumerate the three at the type level, but
///   `Box<dyn Error + Send + Sync>` preserves the full error chain so
///   callers can `downcast` if they need the original type.
///
/// Marked `#[non_exhaustive]` so future variants (`DimMismatch`,
/// `OperationRejected`, etc.) can land without a breaking change.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ShardError {
    /// Error from the index layer.
    #[error(transparent)]
    Index(#[from] KovaIndexError),

    /// Error from one of the composed backend primitives (WAL, vector
    /// store, metadata store). Boxed because the three error types differ
    /// per impl and aren't knowable at the `Shard` definition site.
    #[error("backend error: {0}")]
    Backend(#[source] Box<dyn StdError + Send + Sync + 'static>),
}

impl ShardError {
    /// Wrap any backend error into [`Self::Backend`]. Call-site shorthand :
    /// `.map_err(ShardError::backend)?` instead of `.map_err(|e| ShardError::Backend(Box::new(e)))?`.
    #[must_use]
    pub fn backend<E>(err: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::Backend(Box::new(err))
    }
}

/// A single search result returned by [`Shard::search`].
///
/// Carries the matched id, its distance to the query under the shard's
/// configured metric (smaller = closer, per the [`Distance`] convention),
/// and the metadata attached to that id at insert time. Metadata is
/// returned alongside the hit because the whole point of having a
/// metadata store is that callers can read attributes on the results path
/// without a second round-trip.
#[derive(Debug, Clone)]
pub struct SearchHit {
    /// The matched vector's id.
    pub id: VectorId,
    /// Distance under the shard's configured metric. Smaller is closer.
    pub distance: f32,
    /// Attribute bag attached to `id`. Empty if the insert carried no metadata.
    pub metadata: Metadata,
}

/// The composition layer : ties a [`Wal`], a [`VectorStore`], a
/// [`MetadataStore`], and an [`HnswIndex`] together under a single
/// log-then-mutate discipline.
///
/// Generic over all four primitives so the same struct drives production
/// (file/mmap backends) and tests (in-memory backends), and so future
/// backends (S3, distributed log) compose without code changes here.
///
/// See the module-level docs for the on-disk layout (when composed with
/// file-backed primitives) and the recovery model.
pub struct Shard<D, V, M, W>
where
    D: Distance,
    V: VectorStore,
    M: MetadataStore,
    W: Wal,
{
    index: HnswIndex<D, V>,
    metadata: M,
    wal: W,
}

// -----------------------------------------------------------------------------
// Generic impl : works for any compatible quartet of primitives.
// -----------------------------------------------------------------------------

impl<D, V, M, W> Shard<D, V, M, W>
where
    D: Distance,
    V: VectorStore,
    M: MetadataStore,
    W: Wal,
{
    /// Compose a shard from already-constructed primitives, using the
    /// default RNG seed for the index. Runs WAL replay before returning,
    /// so the index catches up with whatever state the stores were left in.
    ///
    /// # Errors
    /// Returns [`ShardError`] if WAL iteration or any replayed mutation fails.
    pub fn from_parts(
        metric: D,
        params: HnswParams,
        vectors: V,
        metadata: M,
        wal: W,
    ) -> Result<Self, ShardError> {
        Self::from_parts_seeded(metric, params, DEFAULT_SEED, vectors, metadata, wal)
    }

    /// Like [`Self::from_parts`] but with an explicit RNG seed. Used by
    /// tests that need reproducible graph construction.
    ///
    /// # Errors
    /// See [`Self::from_parts`].
    pub fn from_parts_seeded(
        metric: D,
        params: HnswParams,
        seed: u64,
        vectors: V,
        metadata: M,
        wal: W,
    ) -> Result<Self, ShardError> {
        let index = HnswIndex::seeded_with_store(metric, params, seed, vectors);
        let mut shard = Self {
            index,
            metadata,
            wal,
        };
        shard.replay()?;
        Ok(shard)
    }

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
    /// Returning a misleading `Err` after WAL commit is the alternative
    ///  and the worse one : the caller can't safely retry, since the
    /// duplicate-check would pass while the WAL already holds an
    /// `Insert{id}` record. Two records for the same id would then crash
    /// replay on next reopen. See the design note in the README.
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

        // Duplicate id : O(1) HashMap lookup on the in-memory node map.
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

    /// k-nearest search. Returns hits in increasing distance order, each
    /// with its attached metadata read from the metadata store.
    ///
    /// Missing metadata (e.g. an id present in the index but absent from
    /// the metadata store, which shouldn't happen under normal operation
    /// but can after partial recovery) is filled with an empty `Metadata`
    /// rather than failing the whole query.
    ///
    /// # Errors
    /// Returns [`ShardError::Index`] if the index search fails (e.g.
    /// dimension mismatch).
    pub fn search(&self, query: &Vector, k: usize) -> Result<Vec<SearchHit>, ShardError> {
        let hits = self.index.search(query, k)?;
        let results = hits
            .into_iter()
            .map(|(id, distance)| {
                let metadata = self.metadata.get(id).unwrap_or_default();
                SearchHit {
                    id,
                    distance,
                    metadata,
                }
            })
            .collect();
        Ok(results)
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

    /// Delete `id` from the shard, under the same 3-phase discipline as
    /// [`Self::insert`].
    ///
    /// Logical, not structural : the graph node and the vector bytes
    /// stay in place so `search_layer` can keep traversing through this
    /// id ; subsequent [`Self::search`] calls just filter it out of the
    /// returned hits, and [`Self::contains`] returns `false`.
    /// [`MetadataStore::delete`] is called so the attribute bag is gone
    /// immediately. Vacuum (a future milestone) is what actually frees
    /// the storage bytes and clears the tombstones.
    ///
    /// # Id reuse
    /// `id`s **cannot** be re-inserted after delete until vacuum runs ;
    /// the graph node is still in place, so [`Self::insert`]'s duplicate
    /// check fires. This is a deliberate v1 limitation, not a bug.
    ///
    /// # Errors
    /// - [`ShardError::Index`] with `KovaIndexError::NotFound` if `id`
    ///   was never inserted.
    /// - [`ShardError::Index`] with `KovaIndexError::AlreadyDeleted` if
    ///   `id` is already tombstoned.
    /// - [`ShardError::Backend`] from `wal.append` / `wal.sync`.
    ///
    /// # Panics
    /// Panics with a clear message on phase-3 apply failure ; see
    /// [`Self::insert`] for the rationale.
    pub fn delete(&mut self, id: VectorId) -> Result<(), ShardError> {
        // Phase 1 : pre-commit validation.
        if self.index.top_layer_of(id).is_none() {
            return Err(KovaIndexError::NotFound { id }.into());
        }
        if self.index.is_tombstoned(id) {
            return Err(KovaIndexError::AlreadyDeleted { id }.into());
        }

        // Phase 2 : commit.
        let record = Record::Delete { id };
        self.wal.append(&record).map_err(ShardError::backend)?;
        self.wal.sync().map_err(ShardError::backend)?;

        // Phase 3 : apply. Post-commit failures panic.
        if let Err(e) = self.index.tombstone(id) {
            panic!(
                "Shard::delete phase-3 apply failure on index.tombstone: {e:?} \
                 (WAL has committed the record ; aborting so replay can reconcile)"
            );
        }
        if let Err(e) = self.metadata.delete(id) {
            panic!(
                "Shard::delete phase-3 apply failure on metadata.delete: {e:?} \
                 (WAL has committed the record ; aborting so replay can reconcile)"
            );
        }

        Ok(())
    }

    /// Physically remove all tombstoned ids : rewire the HNSW graph
    /// so no live node points at a deleted one, free their slots in the
    /// vector store, reset the tombstone set. Returns the number of
    /// ids physically removed.
    ///
    /// **No on-disk commit.** Vacuum touches in-memory graph state and
    /// the mmap (clearing slot `present` bytes for reuse), but the WAL
    /// is unchanged and no manifest is written. The vacuum work is
    /// recoverable but *wasted* on crash : if the process dies before
    /// the next [`Shard::checkpoint`], reopen replays the full WAL
    /// (including the `Delete` records), tombstones come back, and you
    /// have to vacuum again. Operators usually call `checkpoint()` not
    /// long after `vacuum()` to lock the work in.
    ///
    /// Metadata is **not** touched here : tombstoned ids were already
    /// pruned from the metadata store at [`Self::delete`] time.
    ///
    /// # Errors
    /// [`ShardError::Index`] surfaces any HNSW-level error during the
    /// rewiring or the underlying `vectors.remove` call.
    pub fn vacuum(&mut self) -> Result<usize, ShardError> {
        Ok(self.index.vacuum_tombstones()?)
    }

    /// Whether the shard currently holds a live entry for `id`.
    ///
    /// Returns `false` for ids that were inserted then deleted : the
    /// graph node is still in memory (until vacuum) but the id is
    /// logically gone, and that's what callers care about.
    ///
    /// O(1) ; does not touch the vector store or metadata store.
    #[must_use]
    pub fn contains(&self, id: VectorId) -> bool {
        self.index.top_layer_of(id).is_some() && !self.index.is_tombstoned(id)
    }

    /// Number of live (non-tombstoned) vectors in the shard.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len() - self.index.tombstone_count()
    }

    /// Whether the shard has any live entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Replay every WAL record (LSN order) into the index + stores.
    ///
    /// Called once during construction. The stores may already hold
    /// persisted state from a previous run ; the index is always fresh
    /// (graph structure is in-memory only). Every backend `put` is
    /// idempotent (overwrite-in-place), so re-applying a record that was
    /// already partially applied before a crash is safe.
    ///
    /// We materialise the records into a `Vec` up front because the
    /// `iter_from` borrow conflicts with the mutable borrows we need on
    /// `self.index` / `self.metadata` to apply them.
    fn replay(&mut self) -> Result<(), ShardError> {
        let records: Vec<(Lsn, Record)> = self
            .wal
            .iter_from(Lsn::ZERO)
            .collect::<Result<_, _>>()
            .map_err(ShardError::backend)?;

        for (_lsn, record) in records {
            match record {
                Record::Insert {
                    id,
                    vector,
                    metadata,
                } => {
                    self.index.insert(id, vector)?;
                    self.metadata
                        .put(id, metadata)
                        .map_err(ShardError::backend)?;
                }
                Record::Delete { id } => {
                    // Tombstone in the index + drop from metadata. The
                    // graph node and vector bytes stay (vacuum reclaims).
                    //
                    // Ordering matters : `Shard::insert` rejects duplicate
                    // ids by graph-node presence, so the WAL never holds
                    // Delete{id} without a prior Insert{id} (modulo the
                    // crash test which inserts then dies — that has a
                    // matching Insert in the same WAL).
                    self.index.tombstone(id)?;
                    self.metadata.delete(id).map_err(ShardError::backend)?;
                }
            }
        }
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// Concrete impl : the production file-backed combo. Wires up `FileWal +
// MmapVectorStore + FileMetadataStore` from a single data directory.
// -----------------------------------------------------------------------------

impl<D: Distance> Shard<D, MmapVectorStore, FileMetadataStore, FileWal> {
    /// Open (or create) a file-backed shard rooted at `dir`. Uses the
    /// default RNG seed.
    ///
    /// On first open, creates the directory and empty backing files. On
    /// subsequent opens, the stores recover their persisted state and the
    /// WAL is replayed to rebuild the in-memory index.
    ///
    /// `dim` pins the vector dimension. If the underlying `vectors.mmap`
    /// already exists with a different dim, [`MmapVectorStore::open`]
    /// surfaces the mismatch as [`crate::KovaStorageError::CorruptRecord`].
    ///
    /// # Errors
    /// Any [`crate::KovaStorageError`] from the backing primitives bubbles
    /// up as [`ShardError::Backend`].
    pub fn open(
        dir: impl AsRef<Path>,
        dim: usize,
        metric: D,
        params: HnswParams,
    ) -> Result<Self, ShardError> {
        Self::open_seeded(dir, dim, metric, params, DEFAULT_SEED)
    }

    /// Like [`Self::open`] but with an explicit RNG seed.
    ///
    /// # Errors
    /// See [`Self::open`].
    pub fn open_seeded(
        dir: impl AsRef<Path>,
        dim: usize,
        metric: D,
        params: HnswParams,
        seed: u64,
    ) -> Result<Self, ShardError> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).map_err(ShardError::backend)?;

        let wal = FileWal::open(dir.join("wal")).map_err(ShardError::backend)?;
        let vectors =
            MmapVectorStore::open(dir.join("vectors.mmap"), dim).map_err(ShardError::backend)?;
        let metadata =
            FileMetadataStore::open(dir.join("metadata.bin")).map_err(ShardError::backend)?;

        Self::from_parts_seeded(metric, params, seed, vectors, metadata, wal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kova_core::{InMemoryMetadataStore, InMemoryVectorStore, L2, Value};
    use kova_index::HnswParams;
    use tempfile::tempdir;

    use crate::InMemoryWal;

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
        assert_eq!(hits[0].id, id(1)); // nearest to (1.0, 0.05)
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

    // ---------- file-backed tests : exercise the concrete `Shard::open` path ----------

    /// First `open` on an empty directory creates the three backing files
    /// and the shard reports zero size.
    #[test]
    fn open_creates_empty_shard_and_files() {
        let dir = tempdir().unwrap();
        let shard = Shard::open(dir.path(), 3, L2, HnswParams::default()).unwrap();

        assert!(shard.is_empty());
        assert_eq!(shard.len(), 0);

        assert!(
            dir.path().join("wal").exists(),
            "wal/ directory should exist"
        );
        assert!(
            dir.path().join("vectors.mmap").exists(),
            "vectors.mmap should exist"
        );
        assert!(
            dir.path().join("metadata.bin").exists(),
            "metadata.bin should exist"
        );
    }

    /// Same insert+search flow as the in-memory smoke test, but through
    /// the file-backed combo. Confirms `Shard::open` wires up the right
    /// primitives end-to-end.
    #[test]
    fn insert_then_search_file_backed() {
        let dir = tempdir().unwrap();
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();

        shard
            .insert(id(1), v(vec![1.0, 0.0]), tag_meta("alpha"))
            .unwrap();
        shard
            .insert(id(2), v(vec![0.0, 1.0]), tag_meta("beta"))
            .unwrap();

        let hits = shard.search(&v(vec![1.0, 0.05]), 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, id(1));
        assert_eq!(
            hits[0].metadata.get("tag"),
            Some(&Value::String("alpha".into()))
        );
    }

    /// The whole point of the persistence layer : drop a shard, open it
    /// again on the same directory, and the inserts are still there with
    /// their metadata intact.
    ///
    /// Mechanically this exercises every recovery path :
    /// - `MmapVectorStore::open` walks slots to rebuild `id_to_slot`
    /// - `FileMetadataStore::open` deserializes the bincode blob
    /// - `FileWal::open` enumerates segments
    /// - `Shard::replay` walks the WAL and re-applies every record into
    ///   the fresh in-memory HNSW index
    #[test]
    fn reopen_recovers_inserted_vectors() {
        let dir = tempdir().unwrap();

        {
            let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
            shard
                .insert(id(1), v(vec![1.0, 0.0]), tag_meta("alpha"))
                .unwrap();
            shard
                .insert(id(2), v(vec![0.0, 1.0]), tag_meta("beta"))
                .unwrap();
            shard
                .insert(id(3), v(vec![0.5, 0.5]), tag_meta("gamma"))
                .unwrap();
        } // shard dropped : releases mmap + files

        let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        assert_eq!(shard.len(), 3);
        assert!(shard.contains(id(1)));
        assert!(shard.contains(id(2)));
        assert!(shard.contains(id(3)));

        let hits = shard.search(&v(vec![1.0, 0.0]), 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, id(1));
        assert_eq!(
            hits[0].metadata.get("tag"),
            Some(&Value::String("alpha".into()))
        );
    }

    /// Dim mismatch on `insert` is caught in the pre-commit phase, so no
    /// WAL record is written. Verified by counting WAL records via
    /// `iter_from(Lsn::ZERO)` after the failed insert.
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

        // Reopen : WAL replay should see exactly ONE record. If the bad
        // insert had appended, replay would see two and fail on apply
        // (the second `Insert{2}` with wrong dim).
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

        // Index hasn't seen any inserts ; dim() is None. But the store
        // (MmapVectorStore) has dim = 3 from the file header. The
        // pre-commit check uses store_dim() as fallback.
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

        // No WAL record poison : reopen, search, all good.
        drop(shard);
        let shard = Shard::open(dir.path(), 3, L2, HnswParams::default()).unwrap();
        assert_eq!(shard.len(), 0);
    }

    /// Reopening with a `dim` that doesn't match the existing
    /// `vectors.mmap` header surfaces as a `ShardError::Backend` whose
    /// source mentions the dim mismatch. Caller can downcast to inspect
    /// the original `KovaStorageError` if needed ; here we just check the
    /// message contains "dim".
    #[test]
    fn reopen_with_wrong_dim_errors() {
        let dir = tempdir().unwrap();

        {
            let mut shard = Shard::open(dir.path(), 3, L2, HnswParams::default()).unwrap();
            shard
                .insert(id(1), v(vec![1.0, 2.0, 3.0]), Metadata::new())
                .unwrap();
        }

        // Can't use `.unwrap_err()` : the Ok variant (`Shard`) isn't Debug
        // (HnswIndex isn't Debug, deliberately), so we destructure manually.
        let Err(err) = Shard::open(dir.path(), 4, L2, HnswParams::default()) else {
            panic!("expected error on dim mismatch");
        };
        match err {
            ShardError::Backend(ref source) => {
                let msg = format!("{source}");
                assert!(msg.contains("dim"), "expected dim error, got: {msg}");
            }
            other => panic!("expected Backend error, got {other:?}"),
        }
    }

    /// Replay is idempotent : opening, dropping, and opening again N
    /// times leaves the shard in the same state every time. This is the
    /// invariant the crash recovery test (next milestone) will lean on.
    #[test]
    fn replay_is_idempotent_across_multiple_reopens() {
        let dir = tempdir().unwrap();

        {
            let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
            shard
                .insert(id(1), v(vec![1.0, 0.0]), tag_meta("a"))
                .unwrap();
            shard
                .insert(id(2), v(vec![0.0, 1.0]), tag_meta("b"))
                .unwrap();
        }

        for round in 0..3 {
            let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
            assert_eq!(shard.len(), 2, "round {round}: wrong len");
            assert!(shard.contains(id(1)), "round {round}: missing id 1");
            assert!(shard.contains(id(2)), "round {round}: missing id 2");
            let hits = shard.search(&v(vec![1.0, 0.0]), 1).unwrap();
            assert_eq!(hits[0].id, id(1), "round {round}: wrong nearest");
        }
    }

    // ---------- delete behaviour ----------

    /// `delete` flips `contains`, drops `len`, and filters the id from
    /// future search hits. Exercises the happy path end-to-end on
    /// in-memory primitives.
    #[test]
    fn delete_then_contains_and_search_reflect_it() {
        let mut shard = Shard::from_parts(
            L2,
            HnswParams::default(),
            InMemoryVectorStore::new(),
            InMemoryMetadataStore::new(),
            InMemoryWal::new(),
        )
        .unwrap();

        shard
            .insert(id(1), v(vec![1.0, 0.0]), tag_meta("alpha"))
            .unwrap();
        shard
            .insert(id(2), v(vec![0.0, 1.0]), tag_meta("beta"))
            .unwrap();
        assert_eq!(shard.len(), 2);

        shard.delete(id(1)).unwrap();
        assert_eq!(shard.len(), 1);
        assert!(!shard.contains(id(1)));
        assert!(shard.contains(id(2)));

        // The nearest neighbour of (1.0, 0.0) is id 1, but it's tombstoned.
        let hits = shard.search(&v(vec![1.0, 0.0]), 2).unwrap();
        let ids: Vec<_> = hits.iter().map(|h| h.id).collect();
        assert!(!ids.contains(&id(1)), "tombstoned id should not appear");
        assert!(ids.contains(&id(2)));
    }

    /// Deleting a nonexistent id errors with `NotFound` before any WAL
    /// append. Verified by counting log records after the failed call.
    #[test]
    fn delete_unknown_id_errors_no_poison() {
        let mut shard = Shard::from_parts(
            L2,
            HnswParams::default(),
            InMemoryVectorStore::new(),
            InMemoryMetadataStore::new(),
            InMemoryWal::new(),
        )
        .unwrap();
        shard.insert(id(1), v(vec![1.0]), Metadata::new()).unwrap();

        let err = shard.delete(id(99)).unwrap_err();
        assert!(matches!(
            err,
            ShardError::Index(KovaIndexError::NotFound { id }) if id == VectorId::new(99)
        ));

        // Sanity : the live id is still present.
        assert!(shard.contains(id(1)));
        assert_eq!(shard.len(), 1);
    }

    /// Deleting an already-deleted id errors with `AlreadyDeleted` ;
    /// state and WAL are unchanged.
    #[test]
    fn delete_already_deleted_errors() {
        let mut shard = Shard::from_parts(
            L2,
            HnswParams::default(),
            InMemoryVectorStore::new(),
            InMemoryMetadataStore::new(),
            InMemoryWal::new(),
        )
        .unwrap();
        shard.insert(id(1), v(vec![1.0]), Metadata::new()).unwrap();
        shard.delete(id(1)).unwrap();

        let err = shard.delete(id(1)).unwrap_err();
        assert!(matches!(
            err,
            ShardError::Index(KovaIndexError::AlreadyDeleted { id }) if id == VectorId::new(1)
        ));
        assert_eq!(shard.len(), 0);
    }

    /// V1 limitation : ids can't be reused after delete (the graph node
    /// is still in place, so insert's duplicate check fires). Vacuum
    /// is the future milestone that lifts this restriction.
    #[test]
    fn reinsert_after_delete_errors_until_vacuum() {
        let mut shard = Shard::from_parts(
            L2,
            HnswParams::default(),
            InMemoryVectorStore::new(),
            InMemoryMetadataStore::new(),
            InMemoryWal::new(),
        )
        .unwrap();
        shard.insert(id(1), v(vec![1.0]), Metadata::new()).unwrap();
        shard.delete(id(1)).unwrap();

        let err = shard
            .insert(id(1), v(vec![9.0]), Metadata::new())
            .unwrap_err();
        assert!(matches!(
            err,
            ShardError::Index(KovaIndexError::DuplicateId { id }) if id == VectorId::new(1)
        ));
    }

    /// Deletes survive reopen : the WAL `Delete` record is replayed, the
    /// re-built in-memory index has the tombstone, and the metadata
    /// entry is gone from `metadata.bin`.
    #[test]
    fn delete_persists_across_reopen() {
        let dir = tempdir().unwrap();

        {
            let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
            shard
                .insert(id(1), v(vec![1.0, 0.0]), tag_meta("a"))
                .unwrap();
            shard
                .insert(id(2), v(vec![0.0, 1.0]), tag_meta("b"))
                .unwrap();
            shard.delete(id(1)).unwrap();
        }

        let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        assert_eq!(shard.len(), 1);
        assert!(!shard.contains(id(1)));
        assert!(shard.contains(id(2)));

        let hits = shard.search(&v(vec![1.0, 0.0]), 2).unwrap();
        let ids: Vec<_> = hits.iter().map(|h| h.id).collect();
        assert!(!ids.contains(&id(1)));
        assert!(ids.contains(&id(2)));
    }

    // ---------- batched insert behaviour ----------

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
            (id(1), v(vec![9.9, 9.9]), Metadata::new()), // duplicate of id 1
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
            (id(42), v(vec![9.0, 9.0]), Metadata::new()), // collides with existing
        ];

        let err = shard.insert_many(batch).unwrap_err();
        assert!(matches!(
            err,
            ShardError::Index(KovaIndexError::DuplicateId { id }) if id == VectorId::new(42)
        ));
        // Original id 42 still present, id 1 was never written.
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
            (id(2), v(vec![0.0, 1.0, 0.0]), Metadata::new()), // 3-dim vs 2-dim
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
        // Spot-check metadata round-tripped.
        let hits = shard.search(&v(vec![5.0, 95.0]), 1).unwrap();
        assert_eq!(hits[0].id, id(5));
        assert_eq!(
            hits[0].metadata.get("tag"),
            Some(&Value::String("k5".into()))
        );
    }

    // ---------- Shard::vacuum ----------

    /// `vacuum` on a shard with no tombstones is a clean no-op : returns
    /// 0 and leaves state untouched.
    #[test]
    fn vacuum_on_no_tombstones_is_noop() {
        let mut shard = Shard::from_parts(
            L2,
            HnswParams::default(),
            InMemoryVectorStore::new(),
            InMemoryMetadataStore::new(),
            InMemoryWal::new(),
        )
        .unwrap();
        shard
            .insert(id(1), v(vec![1.0, 0.0]), Metadata::new())
            .unwrap();
        shard
            .insert(id(2), v(vec![0.0, 1.0]), Metadata::new())
            .unwrap();

        let freed = shard.vacuum().unwrap();
        assert_eq!(freed, 0);
        assert_eq!(shard.len(), 2);
        assert!(shard.contains(id(1)));
        assert!(shard.contains(id(2)));
    }

    /// After delete + vacuum, the tombstoned id is fully gone from the
    /// index (no longer in nodes, not just tombstoned).
    #[test]
    fn vacuum_physically_removes_tombstoned_ids() {
        let mut shard = Shard::from_parts(
            L2,
            HnswParams::default(),
            InMemoryVectorStore::new(),
            InMemoryMetadataStore::new(),
            InMemoryWal::new(),
        )
        .unwrap();
        shard
            .insert(id(1), v(vec![1.0, 0.0]), Metadata::new())
            .unwrap();
        shard
            .insert(id(2), v(vec![0.0, 1.0]), Metadata::new())
            .unwrap();
        shard.delete(id(1)).unwrap();

        let freed = shard.vacuum().unwrap();
        assert_eq!(freed, 1);
        assert_eq!(shard.len(), 1);
        assert!(!shard.contains(id(1)));
        assert!(shard.contains(id(2)));
    }

    /// The headline lift of the v1 limitation : after vacuum, a deleted
    /// id can be re-inserted with a fresh vector. Pre-vacuum, the same
    /// insert would fail with `DuplicateId` because the graph node was
    /// still around (just tombstoned).
    #[test]
    fn vacuum_enables_id_reuse() {
        let mut shard = Shard::from_parts(
            L2,
            HnswParams::default(),
            InMemoryVectorStore::new(),
            InMemoryMetadataStore::new(),
            InMemoryWal::new(),
        )
        .unwrap();
        shard
            .insert(id(7), v(vec![1.0, 0.0]), tag_meta("first"))
            .unwrap();
        shard.delete(id(7)).unwrap();

        // Pre-vacuum : re-insert errors.
        let err = shard.insert(id(7), v(vec![2.0, 0.0]), tag_meta("second"));
        assert!(matches!(
            err,
            Err(ShardError::Index(KovaIndexError::DuplicateId { .. }))
        ));

        // Vacuum frees the id ; re-insert now succeeds.
        shard.vacuum().unwrap();
        shard
            .insert(id(7), v(vec![2.0, 0.0]), tag_meta("second"))
            .unwrap();
        assert_eq!(shard.len(), 1);
        assert!(shard.contains(id(7)));

        let hits = shard.search(&v(vec![2.0, 0.0]), 1).unwrap();
        assert_eq!(hits[0].id, id(7));
        assert_eq!(
            hits[0].metadata.get("tag"),
            Some(&Value::String("second".into()))
        );
    }

    /// File-backed end-to-end : insert, delete, vacuum, drop, reopen.
    /// Since vacuum doesn't write a snapshot or truncate the WAL, the
    /// reopen replays the full WAL (including the `Delete` records),
    /// tombstones come back, and the vacuum work is lost. This is the
    /// documented "vacuum without checkpoint is wasted on crash" trade.
    ///
    /// Asserts the post-reopen state is still *correct* (tombstoned id
    /// is invisible), and that a second vacuum still works.
    #[test]
    fn vacuum_work_is_wasted_on_reopen_but_state_stays_correct() {
        let dir = tempdir().unwrap();

        {
            let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
            shard
                .insert(id(1), v(vec![1.0, 0.0]), tag_meta("a"))
                .unwrap();
            shard
                .insert(id(2), v(vec![0.0, 1.0]), tag_meta("b"))
                .unwrap();
            shard.delete(id(1)).unwrap();
            let freed = shard.vacuum().unwrap();
            assert_eq!(freed, 1);
            assert_eq!(shard.len(), 1);
        }

        // Reopen : WAL replay re-inserts id 1, then re-tombstones it.
        // From the caller's view the shard still has only id 2 visible.
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        assert!(!shard.contains(id(1)));
        assert!(shard.contains(id(2)));
        assert_eq!(shard.len(), 1);

        // Vacuum-again works : frees id 1's reborn-then-killed graph node.
        let freed = shard.vacuum().unwrap();
        assert_eq!(freed, 1);
        assert_eq!(shard.len(), 1);
    }
}

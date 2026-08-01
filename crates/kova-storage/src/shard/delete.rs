//! `Shard::delete` and `Shard::vacuum`.
//!
//! `delete` follows the 3-phase discipline : pre-commit validation,
//! WAL commit, panic on apply failure. Logical-only ; the graph node
//! and vector bytes stay until vacuum.
//!
//! `vacuum` is the cleanup pass : rewires the HNSW graph to physically
//! remove tombstones and frees their vector store slots. No on-disk
//! commit ; the WAL still holds the `Delete` records, so the vacuum
//! work is wasted on crash (recovered, just redone).

use kova_core::{Metadata, MetadataStore, VectorId, VectorStore};
use kova_index::KovaIndexError;

use crate::{Record, Wal};

use super::{Shard, ShardError};

impl<D, V, M, W> Shard<D, V, M, W>
where
    D: kova_core::Distance,
    V: VectorStore,
    M: MetadataStore,
    W: Wal,
{
    /// Delete `id` from the shard, under the same 3-phase discipline as
    /// [`Self::insert`].
    ///
    /// Logical, not structural : the graph node and the vector bytes
    /// stay in place so `search_layer` can keep traversing through this
    /// id ; subsequent [`Self::search`] calls just filter it out of the
    /// returned hits, and [`Self::contains`] returns `false`.
    /// [`MetadataStore::delete`] is called so the attribute bag is gone
    /// immediately. Vacuum is what actually frees the storage bytes and
    /// clears the tombstones.
    ///
    /// # Id reuse
    /// `id`s **cannot** be re-inserted after delete until [`Self::vacuum`]
    /// runs ; the graph node is still in place, so [`Self::insert`]'s
    /// duplicate check fires.
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

        // Snapshot the old metadata bag BEFORE the WAL commit so it
        // can be embedded in the record. Replay needs it for catalog
        // bucket cleanup ; the metadata store on disk doesn't carry
        // the old bag because it persists eagerly.
        let old_meta = self.metadata.get(id);

        // Phase 2 : commit.
        let record = Record::Delete {
            id,
            old_metadata: old_meta.clone().unwrap_or_default(),
        };
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
        if let Some(ref meta) = old_meta {
            self.catalog.on_delete(id, meta);
        }

        Ok(())
    }

    /// Batched delete : tombstone every id in `ids` under one WAL
    /// group-commit. Three-phase discipline mirrors [`Self::insert_many`].
    ///
    /// Returns the number of ids actually tombstoned (always equal to
    /// `ids.len()` on success ; pre-commit validation rejects the whole
    /// batch if any id is missing or already tombstoned).
    ///
    /// # Wins over a `delete()` loop
    /// Only one `wal.sync()` for the batch instead of N. Append cost is
    /// linear either way, but `sync` is the disk barrier.
    ///
    /// # Errors
    /// - [`ShardError::Index`] with `KovaIndexError::NotFound` on the
    ///   first id that was never inserted.
    /// - [`ShardError::Index`] with `KovaIndexError::AlreadyDeleted` on
    ///   the first id that's already tombstoned.
    /// - [`ShardError::Backend`] from `wal.append` / `wal.sync`.
    ///
    /// # Panics
    /// Panics on phase-3 apply failure ; see [`Self::insert`] for the
    /// rationale.
    pub fn delete_many<I>(&mut self, ids: I) -> Result<usize, ShardError>
    where
        I: IntoIterator<Item = VectorId>,
    {
        let ids: Vec<VectorId> = ids.into_iter().collect();
        if ids.is_empty() {
            return Ok(0);
        }

        // Phase 1 : pre-commit validation. Reject the whole batch on
        // the first bad id rather than processing partials.
        for &id in &ids {
            if self.index.top_layer_of(id).is_none() {
                return Err(KovaIndexError::NotFound { id }.into());
            }
            if self.index.is_tombstoned(id) {
                return Err(KovaIndexError::AlreadyDeleted { id }.into());
            }
        }

        // Snapshot every old metadata bag before WAL commit so each
        // bag rides along in the record for the catalog's benefit at
        // replay time.
        let items: Vec<(VectorId, Metadata)> = ids
            .iter()
            .map(|id| (*id, self.metadata.get(*id).unwrap_or_default()))
            .collect();

        // Phase 2 : group-commit. One DeleteMany frame for the whole
        // batch instead of N Delete frames ; replay applies each id
        // independently so the on-disk effect is identical.
        let record = Record::DeleteMany {
            items: items.clone(),
        };
        self.wal.append(&record).map_err(ShardError::backend)?;
        self.wal.sync().map_err(ShardError::backend)?;

        // Phase 3 : apply, panic on failure.
        //
        // Snapshot each old metadata bag just before its delete so the
        // catalog can remove the row from every bucket the value lived
        // in. Missing bag means there was no metadata to index.
        for &id in &ids {
            let old_meta = self.metadata.get(id);
            if let Err(e) = self.index.tombstone(id) {
                panic!(
                    "Shard::delete_many phase-3 apply failure on index.tombstone ({id}): {e:?} \
                     (WAL has committed the batch ; aborting so replay can reconcile)"
                );
            }
            if let Err(e) = self.metadata.delete(id) {
                panic!(
                    "Shard::delete_many phase-3 apply failure on metadata.delete ({id}): {e:?} \
                     (WAL has committed the batch ; aborting so replay can reconcile)"
                );
            }
            if let Some(ref meta) = old_meta {
                self.catalog.on_delete(id, meta);
            }
        }

        Ok(ids.len())
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
        // Phase 1 : nothing to validate. Vacuuming an empty tombstone
        // set is a no-op, not an error.
        //
        // Phase 2 : commit. Vacuum changes no rows, but it *does*
        // rewire the graph in a way that depends on when it ran, so it
        // has to occupy a definite position in the log — see
        // [`Record::Vacuum`].
        self.wal
            .append(&Record::Vacuum)
            .map_err(ShardError::backend)?;
        self.wal.sync().map_err(ShardError::backend)?;

        // Phase 3 : apply. Post-commit failure panics, as everywhere
        // else : the log says the vacuum happened.
        match self.index.vacuum_tombstones() {
            Ok(removed) => Ok(removed),
            Err(e) => panic!(
                "Shard::vacuum phase-3 apply failure on index.vacuum_tombstones: {e:?} \
                 (WAL has committed the record ; aborting so replay can reconcile)"
            ),
        }
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

    // ---------- delete ----------

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

        let hits = shard.search(&v(vec![1.0, 0.0]), 2).unwrap();
        let ids: Vec<_> = hits.iter().map(|h| h.id).collect();
        assert!(!ids.contains(&id(1)), "tombstoned id should not appear");
        assert!(ids.contains(&id(2)));
    }

    /// Deleting a nonexistent id errors with `NotFound` before any WAL
    /// append.
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

        assert!(shard.contains(id(1)));
        assert_eq!(shard.len(), 1);
    }

    /// Deleting an already-deleted id errors with `AlreadyDeleted`.
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

    /// V1 limitation : ids can't be reused after delete until vacuum runs.
    /// (Lifted by [`Shard::vacuum`] ; see vacuum tests below.)
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

    /// Deletes survive reopen : the WAL `Delete` record is replayed,
    /// the re-built in-memory index has the tombstone, and the metadata
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

    // ---------- vacuum ----------

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
    /// id can be re-inserted with a fresh vector.
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

        let err = shard.insert(id(7), v(vec![2.0, 0.0]), tag_meta("second"));
        assert!(matches!(
            err,
            Err(ShardError::Index(KovaIndexError::DuplicateId { .. }))
        ));

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

    /// File-backed end-to-end : vacuum doesn't write a snapshot or
    /// truncate the WAL, so reopen replays the full WAL (tombstones
    /// come back), the vacuum work is lost. State is still correct
    /// and a second vacuum still works.
    #[test]
    fn vacuum_survives_reopen_because_it_is_logged() {
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

        let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        assert!(!shard.contains(id(1)));
        assert!(shard.contains(id(2)));
        assert_eq!(shard.len(), 1);

        // This test used to be called `vacuum_work_is_wasted_on_reopen`
        // and asserted the tombstone came back : vacuum made no on-disk
        // commit, so replay re-applied the `Delete` and the node was
        // tombstoned again, forcing a second vacuum.
        //
        // Vacuum is now a logged record (`Record::Vacuum`), because
        // replicas have to vacuum at the same log position or their
        // graphs diverge. Durability across reopen is the pleasant
        // side effect: replay applies the vacuum too, so the work is
        // no longer thrown away.
        assert_eq!(
            shard.index.tombstone_count(),
            0,
            "replay should have applied Record::Vacuum, leaving nothing to redo"
        );
    }

    // ---------- delete_many ----------

    fn fresh_in_memory() -> Shard<L2, InMemoryVectorStore, InMemoryMetadataStore, InMemoryWal> {
        Shard::from_parts(
            L2,
            HnswParams::default(),
            InMemoryVectorStore::new(),
            InMemoryMetadataStore::new(),
            InMemoryWal::new(),
        )
        .unwrap()
    }

    #[test]
    fn delete_many_tombstones_each_id_and_returns_count() {
        let mut shard = fresh_in_memory();
        for i in 1..=4 {
            #[allow(clippy::cast_precision_loss)]
            let x = i as f32;
            shard.insert(id(i), v(vec![x, 0.0]), tag_meta("a")).unwrap();
        }
        let n = shard.delete_many(vec![id(1), id(3)]).unwrap();
        assert_eq!(n, 2);
        assert!(!shard.contains(id(1)));
        assert!(shard.contains(id(2)));
        assert!(!shard.contains(id(3)));
        assert!(shard.contains(id(4)));
        assert_eq!(shard.len(), 2);
    }

    #[test]
    fn delete_many_empty_batch_is_noop() {
        let mut shard = fresh_in_memory();
        shard
            .insert(id(1), v(vec![1.0, 0.0]), tag_meta("a"))
            .unwrap();
        let n = shard.delete_many(std::iter::empty::<VectorId>()).unwrap();
        assert_eq!(n, 0);
        assert_eq!(shard.len(), 1);
    }

    #[test]
    fn delete_many_unknown_id_rejects_whole_batch() {
        let mut shard = fresh_in_memory();
        shard
            .insert(id(1), v(vec![1.0, 0.0]), tag_meta("a"))
            .unwrap();
        shard
            .insert(id(2), v(vec![0.0, 1.0]), tag_meta("b"))
            .unwrap();
        let err = shard
            .delete_many(vec![id(1), id(99), id(2)])
            .expect_err("phase-1 should reject");
        assert!(matches!(
            err,
            ShardError::Index(KovaIndexError::NotFound { id }) if id == VectorId::new(99)
        ));
        // Both live ids stay live : phase-1 ran before any WAL append.
        assert!(shard.contains(id(1)));
        assert!(shard.contains(id(2)));
    }

    #[test]
    fn delete_many_already_tombstoned_rejects_whole_batch() {
        let mut shard = fresh_in_memory();
        shard
            .insert(id(1), v(vec![1.0, 0.0]), tag_meta("a"))
            .unwrap();
        shard
            .insert(id(2), v(vec![0.0, 1.0]), tag_meta("b"))
            .unwrap();
        shard.delete(id(1)).unwrap();
        let err = shard
            .delete_many(vec![id(2), id(1)])
            .expect_err("phase-1 should reject");
        assert!(matches!(
            err,
            ShardError::Index(KovaIndexError::AlreadyDeleted { id }) if id == VectorId::new(1)
        ));
        // id 2 stays live.
        assert!(shard.contains(id(2)));
    }

    #[test]
    fn delete_many_persists_across_reopen() {
        let dir = tempdir().expect("tempdir");
        {
            let mut shard = Shard::open(dir.path(), 4, L2, HnswParams::default()).unwrap();
            for i in 1..=4usize {
                let mut vec = vec![0.0_f32; 4];
                vec[(i - 1) % 4] = 1.0;
                shard
                    .insert(VectorId::new(i as u64), v(vec), tag_meta("a"))
                    .unwrap();
            }
            let n = shard
                .delete_many(vec![VectorId::new(1), VectorId::new(3)])
                .unwrap();
            assert_eq!(n, 2);
        }
        let shard = Shard::open(dir.path(), 4, L2, HnswParams::default()).unwrap();
        assert!(!shard.contains(VectorId::new(1)));
        assert!(shard.contains(VectorId::new(2)));
        assert!(!shard.contains(VectorId::new(3)));
        assert!(shard.contains(VectorId::new(4)));
    }
}

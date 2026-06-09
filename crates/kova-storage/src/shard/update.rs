//! `Shard::update_metadata` : batched metadata-bag replacement under
//! the same 3-phase discipline as [`Shard::insert_many`] and
//! [`Shard::delete_many`]. Vector data is immutable ; only the
//! attribute bag attached to each id mutates.

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
    /// Batched metadata replacement : for every `(id, metadata)` pair
    /// in `updates`, replace the bag attached to `id` with `metadata`.
    /// One WAL group-commit for the whole batch.
    ///
    /// Returns the number of ids updated (always equal to
    /// `updates.len()` on success ; pre-commit validation rejects the
    /// whole batch on the first missing or tombstoned id).
    ///
    /// Vector and graph state are untouched. UPDATE is metadata-only ;
    /// the embedding is immutable for graph-integrity reasons.
    ///
    /// # Errors
    /// - [`ShardError::Index`] with `KovaIndexError::NotFound` on the
    ///   first id that was never inserted.
    /// - [`ShardError::Index`] with `KovaIndexError::AlreadyDeleted` on
    ///   the first id that's tombstoned (updating a deleted row is
    ///   meaningless ; reject loudly).
    /// - [`ShardError::Backend`] from `wal.append` / `wal.sync` or the
    ///   underlying `metadata.put_many` apply.
    ///
    /// # Panics
    /// Panics on phase-3 apply failure ; see [`Self::insert`] for the
    /// rationale.
    pub fn update_metadata<I>(&mut self, updates: I) -> Result<usize, ShardError>
    where
        I: IntoIterator<Item = (VectorId, Metadata)>,
    {
        let updates: Vec<(VectorId, Metadata)> = updates.into_iter().collect();
        if updates.is_empty() {
            return Ok(0);
        }

        // Phase 1 : pre-commit validation. Both id-must-exist and
        // not-tombstoned are caught here, so phase-3 only deals with
        // backend-layer failures.
        for (id, _) in &updates {
            if self.index.top_layer_of(*id).is_none() {
                return Err(KovaIndexError::NotFound { id: *id }.into());
            }
            if self.index.is_tombstoned(*id) {
                return Err(KovaIndexError::AlreadyDeleted { id: *id }.into());
            }
        }

        // Snapshot the old bag for every update before the WAL commit,
        // so each record carries (old, new) for the catalog to consume
        // at replay time.
        let olds: Vec<Metadata> = updates
            .iter()
            .map(|(id, _)| self.metadata.get(*id).unwrap_or_default())
            .collect();

        // Phase 2 : group-commit. One `UpdateMetadata` frame per id ;
        // a future Record::UpdateMetadataMany would compact further,
        // mirroring the Delete -> DeleteMany compaction already done.
        for ((id, metadata), old) in updates.iter().zip(olds.iter()) {
            let record = Record::UpdateMetadata {
                id: *id,
                old_metadata: old.clone(),
                metadata: metadata.clone(),
            };
            self.wal.append(&record).map_err(ShardError::backend)?;
        }
        self.wal.sync().map_err(ShardError::backend)?;

        // Phase 3 : apply. One batched `put_many` for the metadata
        // store (the optimisation `insert_many` uses too) ; failure
        // here panics because the WAL has already committed.
        //
        // Catalog hooks reuse the `olds` snapshot already taken
        // pre-commit ; runs BEFORE put_many consumes `updates`.
        for ((id, new), old) in updates.iter().zip(olds.iter()) {
            self.catalog.on_update(*id, old, new);
        }

        let count = updates.len();
        if let Err(e) = self.metadata.put_many(updates) {
            panic!(
                "Shard::update_metadata phase-3 apply failure on metadata.put_many: {e:?} \
                 (WAL has committed the batch ; aborting so replay can reconcile)"
            );
        }

        Ok(count)
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
    fn update_metadata_replaces_bag_and_returns_count() {
        let mut shard = fresh_in_memory();
        shard
            .insert(id(1), v(vec![1.0, 0.0]), tag_meta("old"))
            .unwrap();
        shard
            .insert(id(2), v(vec![0.0, 1.0]), tag_meta("old"))
            .unwrap();

        let n = shard
            .update_metadata(vec![(id(1), tag_meta("new")), (id(2), tag_meta("fresh"))])
            .unwrap();
        assert_eq!(n, 2);
        assert_eq!(
            shard.get_metadata(id(1)).unwrap().get("tag"),
            Some(&Value::String("new".into()))
        );
        assert_eq!(
            shard.get_metadata(id(2)).unwrap().get("tag"),
            Some(&Value::String("fresh".into()))
        );
    }

    #[test]
    fn update_metadata_empty_batch_is_noop() {
        let mut shard = fresh_in_memory();
        shard
            .insert(id(1), v(vec![1.0, 0.0]), tag_meta("a"))
            .unwrap();
        let n = shard
            .update_metadata(std::iter::empty::<(VectorId, Metadata)>())
            .unwrap();
        assert_eq!(n, 0);
        assert_eq!(
            shard.get_metadata(id(1)).unwrap().get("tag"),
            Some(&Value::String("a".into()))
        );
    }

    #[test]
    fn update_metadata_unknown_id_rejects_whole_batch() {
        let mut shard = fresh_in_memory();
        shard
            .insert(id(1), v(vec![1.0, 0.0]), tag_meta("a"))
            .unwrap();
        let err = shard
            .update_metadata(vec![(id(1), tag_meta("b")), (id(99), tag_meta("c"))])
            .expect_err("phase-1 should reject");
        assert!(matches!(
            err,
            ShardError::Index(KovaIndexError::NotFound { id }) if id == VectorId::new(99)
        ));
        // id 1's bag is untouched : phase-1 ran before any WAL append.
        assert_eq!(
            shard.get_metadata(id(1)).unwrap().get("tag"),
            Some(&Value::String("a".into()))
        );
    }

    #[test]
    fn update_metadata_tombstoned_id_rejects_whole_batch() {
        let mut shard = fresh_in_memory();
        shard
            .insert(id(1), v(vec![1.0, 0.0]), tag_meta("a"))
            .unwrap();
        shard
            .insert(id(2), v(vec![0.0, 1.0]), tag_meta("b"))
            .unwrap();
        shard.delete(id(1)).unwrap();
        let err = shard
            .update_metadata(vec![(id(2), tag_meta("c")), (id(1), tag_meta("d"))])
            .expect_err("phase-1 should reject");
        assert!(matches!(
            err,
            ShardError::Index(KovaIndexError::AlreadyDeleted { id }) if id == VectorId::new(1)
        ));
        // id 2's bag stays at 'b'.
        assert_eq!(
            shard.get_metadata(id(2)).unwrap().get("tag"),
            Some(&Value::String("b".into()))
        );
    }

    #[test]
    fn update_metadata_persists_across_reopen() {
        let dir = tempdir().expect("tempdir");
        {
            let mut shard = Shard::open(dir.path(), 4, L2, HnswParams::default()).unwrap();
            for i in 1..=3usize {
                let mut vec = vec![0.0_f32; 4];
                vec[(i - 1) % 4] = 1.0;
                shard
                    .insert(VectorId::new(i as u64), v(vec), tag_meta("old"))
                    .unwrap();
            }
            let n = shard
                .update_metadata(vec![
                    (VectorId::new(1), tag_meta("new")),
                    (VectorId::new(3), tag_meta("fresh")),
                ])
                .unwrap();
            assert_eq!(n, 2);
        }
        let shard = Shard::open(dir.path(), 4, L2, HnswParams::default()).unwrap();
        assert_eq!(
            shard.get_metadata(VectorId::new(1)).unwrap().get("tag"),
            Some(&Value::String("new".into()))
        );
        assert_eq!(
            shard.get_metadata(VectorId::new(2)).unwrap().get("tag"),
            Some(&Value::String("old".into())),
            "id 2 wasn't updated, should still be 'old'"
        );
        assert_eq!(
            shard.get_metadata(VectorId::new(3)).unwrap().get("tag"),
            Some(&Value::String("fresh".into()))
        );
    }

    #[test]
    fn update_metadata_full_replacement_drops_old_fields() {
        // Old bag has tag="a" and category="docs".
        // New bag has only tag="b".
        // After update : tag="b", category is gone.
        let mut shard = fresh_in_memory();
        let mut old = Metadata::new();
        old.insert("tag".into(), Value::String("a".into()));
        old.insert("category".into(), Value::String("docs".into()));
        shard.insert(id(1), v(vec![1.0, 0.0]), old).unwrap();

        shard.update_metadata(vec![(id(1), tag_meta("b"))]).unwrap();
        let bag = shard.get_metadata(id(1)).unwrap();
        assert_eq!(bag.get("tag"), Some(&Value::String("b".into())));
        assert_eq!(bag.get("category"), None, "old field should be gone");
    }
}

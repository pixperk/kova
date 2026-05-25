//! `Shard::search` : k-nearest with metadata attached to each hit.

use kova_core::{MetadataStore, Vector, VectorStore};
use kova_index::Index;

use crate::Wal;

use super::{SearchHit, Shard, ShardError};

impl<D, V, M, W> Shard<D, V, M, W>
where
    D: kova_core::Distance,
    V: VectorStore,
    M: MetadataStore,
    W: Wal,
{
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
}

#[cfg(test)]
mod tests {
    use kova_core::{
        InMemoryMetadataStore, InMemoryVectorStore, L2, Metadata, Value, Vector, VectorId,
    };
    use kova_index::{HnswParams, KovaIndexError};

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

    /// Search on an empty shard returns an empty hit list (not an error).
    #[test]
    fn search_empty_shard_returns_empty() {
        let shard = fresh_in_memory();
        let hits = shard.search(&v(vec![1.0, 0.0]), 5).unwrap();
        assert!(hits.is_empty());
    }

    /// `k = 0` returns an empty hit list regardless of contents.
    #[test]
    fn search_with_k_zero_returns_empty() {
        let mut shard = fresh_in_memory();
        shard.insert(id(1), v(vec![1.0, 0.0]), Metadata::new()).unwrap();
        shard.insert(id(2), v(vec![0.0, 1.0]), Metadata::new()).unwrap();

        let hits = shard.search(&v(vec![1.0, 0.0]), 0).unwrap();
        assert!(hits.is_empty());
    }

    /// Hits come back sorted by distance ascending (the nearest is first).
    /// `Distance` follows the "smaller = closer" convention.
    #[test]
    fn search_returns_hits_in_ascending_distance_order() {
        let mut shard = fresh_in_memory();
        shard.insert(id(1), v(vec![10.0, 0.0]), Metadata::new()).unwrap();
        shard.insert(id(2), v(vec![1.0, 0.0]), Metadata::new()).unwrap();
        shard.insert(id(3), v(vec![5.0, 0.0]), Metadata::new()).unwrap();

        let hits = shard.search(&v(vec![0.0, 0.0]), 3).unwrap();
        assert_eq!(hits.len(), 3);
        // Distances : id 2 = 1, id 3 = 5, id 1 = 10
        assert_eq!(hits[0].id, id(2));
        assert_eq!(hits[1].id, id(3));
        assert_eq!(hits[2].id, id(1));
        for window in hits.windows(2) {
            assert!(
                window[0].distance <= window[1].distance,
                "hits not sorted: {:?} > {:?}",
                window[0].distance,
                window[1].distance,
            );
        }
    }

    /// Each `SearchHit` carries the metadata attached to its id at
    /// insert time. The whole point of having metadata in the hit.
    #[test]
    fn search_returns_metadata_attached_to_each_hit() {
        let mut shard = fresh_in_memory();
        shard.insert(id(1), v(vec![1.0, 0.0]), tag_meta("alpha")).unwrap();
        shard.insert(id(2), v(vec![0.0, 1.0]), tag_meta("beta")).unwrap();

        let hits = shard.search(&v(vec![1.0, 0.05]), 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(
            hits[0].metadata.get("tag"),
            Some(&Value::String("alpha".into()))
        );
        assert_eq!(
            hits[1].metadata.get("tag"),
            Some(&Value::String("beta".into()))
        );
    }

    /// `k` larger than the shard's size returns all available hits, not
    /// an error and not zero. (`Vec` is naturally bounded by what's
    /// available.)
    #[test]
    fn search_with_k_larger_than_len_returns_all_available() {
        let mut shard = fresh_in_memory();
        shard.insert(id(1), v(vec![1.0, 0.0]), Metadata::new()).unwrap();
        shard.insert(id(2), v(vec![0.0, 1.0]), Metadata::new()).unwrap();

        let hits = shard.search(&v(vec![1.0, 0.0]), 100).unwrap();
        assert_eq!(hits.len(), 2);
    }

    /// Query vector with a different dim than the index's pinned dim
    /// surfaces as `ShardError::Index(DimensionMismatch)`.
    #[test]
    fn search_with_dim_mismatch_errors() {
        let mut shard = fresh_in_memory();
        shard.insert(id(1), v(vec![1.0, 0.0]), Metadata::new()).unwrap();

        let err = shard.search(&v(vec![1.0, 2.0, 3.0]), 5).unwrap_err();
        assert!(matches!(
            err,
            ShardError::Index(KovaIndexError::DimensionMismatch { expected: 2, got: 3 })
        ));
    }

    /// Tombstoned ids never appear in search results, even when they
    /// would otherwise be the nearest neighbour. This is the existing
    /// HNSW search-layer post-filter, but pinned at the Shard surface.
    #[test]
    fn search_filters_tombstoned_ids() {
        let mut shard = fresh_in_memory();
        shard.insert(id(1), v(vec![1.0, 0.0]), tag_meta("a")).unwrap();
        shard.insert(id(2), v(vec![0.0, 1.0]), tag_meta("b")).unwrap();
        shard.insert(id(3), v(vec![1.0, 1.0]), tag_meta("c")).unwrap();

        shard.delete(id(1)).unwrap();

        let hits = shard.search(&v(vec![1.0, 0.0]), 3).unwrap();
        let returned_ids: Vec<_> = hits.iter().map(|h| h.id).collect();
        assert!(!returned_ids.contains(&id(1)), "tombstoned id 1 surfaced");
        assert!(returned_ids.contains(&id(2)) || returned_ids.contains(&id(3)));
    }
}

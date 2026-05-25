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

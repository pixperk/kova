//! HNSW index implementation.
//!
//! `HnswIndex<D, V>` is generic over a [`Distance`] metric `D` and a
//! [`VectorStore`] backend `V`. Graph structure (nodes + neighbour lists)
//! is owned by the index; vector bytes live in `V`. Different storage
//! strategies (in-memory, mmap, distributed) plug in by implementing
//! [`VectorStore`].

use std::collections::{HashMap, HashSet};

use kova_core::{Distance, InMemoryVectorStore, Vector, VectorId, VectorStore};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::{Index, KovaIndexError};

mod insert;
mod layer;
mod node;
mod params;
mod search;
mod select;

pub(crate) use node::Node;
pub use params::HnswParams;

/// Default RNG seed for `new` / `with_params`. Users wanting non-deterministic
/// behaviour or a custom seed call [`HnswIndex::seeded`].
const DEFAULT_SEED: u64 = 0xDEAD_BEEF_DEAD_BEEF;

/// Hierarchical Navigable Small World index.
///
/// Generic over a [`Distance`] metric and a [`VectorStore`] backend.
/// Defaults `V = InMemoryVectorStore` so callers who don't care about
/// storage just write `HnswIndex::new(L2)`.
pub struct HnswIndex<D: Distance, V: VectorStore = InMemoryVectorStore> {
    metric: D,
    params: HnswParams,
    /// Graph structure only : `nodes[id]` holds neighbour lists per layer.
    /// Vector bytes are in `vectors`.
    nodes: HashMap<VectorId, Node>,
    vectors: V,
    /// Logically-deleted ids. Their graph nodes and vectors stay in place
    /// so `search_layer` can still traverse through them (preserves graph
    /// connectivity), but `search` filters them out of the returned hits.
    /// Vacuum (future milestone) actually frees the storage.
    tombstones: HashSet<VectorId>,
    entry_point: Option<VectorId>,
    dim: Option<usize>,
    rng: StdRng,
}

// --- constructors that default V to InMemoryVectorStore ---

impl<D: Distance> HnswIndex<D, InMemoryVectorStore> {
    /// Build an empty index with the default in-memory store and seed.
    #[must_use]
    pub fn new(metric: D) -> Self {
        Self::seeded(metric, HnswParams::default(), DEFAULT_SEED)
    }

    /// Build an empty index with custom params and the default in-memory store.
    #[must_use]
    pub fn with_params(metric: D, params: HnswParams) -> Self {
        Self::seeded(metric, params, DEFAULT_SEED)
    }

    /// Build an empty index with an explicit RNG seed (for reproducible tests).
    #[must_use]
    pub fn seeded(metric: D, params: HnswParams, seed: u64) -> Self {
        Self::seeded_with_store(metric, params, seed, InMemoryVectorStore::new())
    }
}

// --- constructor that lets the caller supply any VectorStore ---

impl<D: Distance, V: VectorStore> HnswIndex<D, V> {
    /// Build an empty index with a caller-supplied [`VectorStore`] and seed.
    #[must_use]
    pub fn seeded_with_store(metric: D, params: HnswParams, seed: u64, vectors: V) -> Self {
        Self {
            metric,
            params,
            nodes: HashMap::new(),
            vectors,
            tombstones: HashSet::new(),
            entry_point: None,
            dim: None,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Name of the configured distance metric.
    #[must_use]
    pub fn metric_name(&self) -> &'static str {
        self.metric.name()
    }

    /// Read-only view of the tuning parameters.
    #[must_use]
    pub fn params(&self) -> &HnswParams {
        &self.params
    }

    /// Pinned vector dimension, set on first insert.
    #[must_use]
    pub fn dim(&self) -> Option<usize> {
        self.dim
    }

    /// Current entry point, `None` while the index is empty.
    #[must_use]
    pub fn entry_point(&self) -> Option<VectorId> {
        self.entry_point
    }

    /// Fetch a vector by id (delegates to the underlying [`VectorStore`]).
    #[must_use]
    pub fn get(&self, id: VectorId) -> Option<Vector> {
        self.vectors.get(id)
    }

    /// Pinned dim of the underlying [`VectorStore`], if any.
    ///
    /// Distinct from [`Self::dim`] : [`Self::dim`] is the index's own
    /// pinned dim (set on the first insert). `store_dim` asks the
    /// underlying store directly. Useful when a caller wants to validate
    /// an input vector against the store's pinned dim *before* the index
    /// has seen any inserts.
    #[must_use]
    pub fn store_dim(&self) -> Option<usize> {
        self.vectors.dim()
    }

    /// Highest layer the node with `id` occupies, or `None` if absent.
    #[must_use]
    pub fn top_layer_of(&self, id: VectorId) -> Option<usize> {
        self.nodes.get(&id).map(Node::top_layer)
    }

    /// Mark `id` as logically deleted.
    ///
    /// The node and its vector stay in place so search can still traverse
    /// through the graph ; subsequent searches simply filter `id` out of
    /// the returned hits. Vacuum (future milestone) is what actually
    /// reclaims storage.
    ///
    /// # Errors
    /// - [`KovaIndexError::NotFound`] if `id` was never inserted.
    /// - [`KovaIndexError::AlreadyDeleted`] if `id` is already tombstoned.
    pub fn tombstone(&mut self, id: VectorId) -> Result<(), KovaIndexError> {
        if !self.nodes.contains_key(&id) {
            return Err(KovaIndexError::NotFound { id });
        }
        if !self.tombstones.insert(id) {
            return Err(KovaIndexError::AlreadyDeleted { id });
        }
        Ok(())
    }

    /// Whether `id` is currently tombstoned.
    #[must_use]
    pub fn is_tombstoned(&self, id: VectorId) -> bool {
        self.tombstones.contains(&id)
    }

    /// Number of tombstoned ids.
    ///
    /// `self.len() - self.tombstone_count()` gives the count of live ids
    /// — what a user-facing layer typically wants to report.
    #[must_use]
    pub fn tombstone_count(&self) -> usize {
        self.tombstones.len()
    }
}

impl<D: Distance, V: VectorStore> Index<D> for HnswIndex<D, V> {
    type Error = KovaIndexError;

    fn insert(&mut self, id: VectorId, vector: Vector) -> Result<(), Self::Error> {
        self.insert_impl(id, vector)
    }

    fn search(&self, query: &Vector, k: usize) -> Result<Vec<(VectorId, f32)>, Self::Error> {
        self.search_impl(query, k)
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kova_core::L2;

    #[test]
    fn new_creates_empty_index() {
        let idx = HnswIndex::new(L2);
        assert_eq!(idx.len(), 0);
        assert!(idx.is_empty());
        assert!(idx.entry_point().is_none());
        assert!(idx.dim().is_none());
    }

    #[test]
    fn defaults_match_hnsw_params_default() {
        let idx = HnswIndex::new(L2);
        assert_eq!(idx.params().m, 16);
        assert_eq!(idx.metric_name(), "l2");
    }

    #[test]
    fn with_params_overrides_defaults() {
        let idx = HnswIndex::with_params(L2, HnswParams::new(32));
        assert_eq!(idx.params().m, 32);
        assert_eq!(idx.params().m_max0, 64);
    }

    #[test]
    fn get_returns_none_for_missing_id() {
        let idx = HnswIndex::new(L2);
        assert!(idx.get(VectorId::new(1)).is_none());
        assert!(idx.top_layer_of(VectorId::new(1)).is_none());
    }

    // ---------- tombstone behaviour ----------

    fn vec(data: Vec<f32>) -> Vector {
        Vector::try_new(data).unwrap()
    }

    fn vid(n: u64) -> VectorId {
        VectorId::new(n)
    }

    #[test]
    fn tombstone_unknown_id_errors_not_found() {
        let mut idx = HnswIndex::new(L2);
        let err = idx.tombstone(vid(7)).unwrap_err();
        assert!(matches!(err, KovaIndexError::NotFound { id } if id == vid(7)));
        assert_eq!(idx.tombstone_count(), 0);
    }

    #[test]
    fn tombstone_existing_id_succeeds() {
        let mut idx = HnswIndex::new(L2);
        idx.insert(vid(1), vec(vec![1.0, 2.0])).unwrap();
        assert!(!idx.is_tombstoned(vid(1)));

        idx.tombstone(vid(1)).unwrap();
        assert!(idx.is_tombstoned(vid(1)));
        assert_eq!(idx.tombstone_count(), 1);
    }

    #[test]
    fn tombstone_already_deleted_errors() {
        let mut idx = HnswIndex::new(L2);
        idx.insert(vid(1), vec(vec![1.0, 2.0])).unwrap();
        idx.tombstone(vid(1)).unwrap();

        let err = idx.tombstone(vid(1)).unwrap_err();
        assert!(matches!(err, KovaIndexError::AlreadyDeleted { id } if id == vid(1)));
        assert_eq!(idx.tombstone_count(), 1);
    }

    #[test]
    fn search_filters_tombstoned_ids() {
        let mut idx = HnswIndex::new(L2);
        idx.insert(vid(1), vec(vec![1.0, 0.0])).unwrap();
        idx.insert(vid(2), vec(vec![0.0, 1.0])).unwrap();
        idx.insert(vid(3), vec(vec![1.0, 1.0])).unwrap();

        // Sanity : all three are searchable.
        let hits = idx.search(&vec(vec![1.0, 0.0]), 3).unwrap();
        let ids: Vec<_> = hits.iter().map(|(i, _)| *i).collect();
        assert!(ids.contains(&vid(1)));
        assert!(ids.contains(&vid(2)));
        assert!(ids.contains(&vid(3)));

        // Tombstone id 1 ; it should disappear from results even though
        // it's the nearest neighbour of the query.
        idx.tombstone(vid(1)).unwrap();
        let hits = idx.search(&vec(vec![1.0, 0.0]), 3).unwrap();
        let ids: Vec<_> = hits.iter().map(|(i, _)| *i).collect();
        assert!(
            !ids.contains(&vid(1)),
            "tombstoned id 1 should not be returned"
        );
        assert!(ids.contains(&vid(2)));
        assert!(ids.contains(&vid(3)));
    }
}

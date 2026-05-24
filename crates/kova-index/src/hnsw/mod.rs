//! HNSW index implementation.
//!
//! `HnswIndex<D, V>` is generic over a [`Distance`] metric `D` and a
//! [`VectorStore`] backend `V`. Graph structure (nodes + neighbour lists)
//! is owned by the index; vector bytes live in `V`. Different storage
//! strategies (in-memory, mmap, distributed) plug in by implementing
//! [`VectorStore`].

use std::collections::HashMap;

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
}

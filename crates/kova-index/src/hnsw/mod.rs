//! HNSW index implementation.

use std::collections::HashMap;

use kova_core::{Distance, Vector, VectorId};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::{Index, KovaIndexError};

mod insert;
mod layer;
mod node;
mod params;
mod search;
mod select;

use node::Node;
pub use params::HnswParams;

/// Default RNG seed for `new` / `with_params`. Users wanting non-deterministic
/// behaviour or a custom seed call [`HnswIndex::seeded`].
const DEFAULT_SEED: u64 = 0xDEAD_BEEF_DEAD_BEEF;

/// Hierarchical Navigable Small World index.
pub struct HnswIndex<D: Distance> {
    metric: D,
    params: HnswParams,
    nodes: HashMap<VectorId, Node>,
    entry_point: Option<VectorId>,
    dim: Option<usize>,
    rng: StdRng,
}

impl<D: Distance> HnswIndex<D> {
    /// Build an empty index using [`HnswParams::default`] and the default seed.
    #[must_use]
    pub fn new(metric: D) -> Self {
        Self::seeded(metric, HnswParams::default(), DEFAULT_SEED)
    }

    /// Build an empty index with caller-supplied parameters and the default seed.
    #[must_use]
    pub fn with_params(metric: D, params: HnswParams) -> Self {
        Self::seeded(metric, params, DEFAULT_SEED)
    }

    /// Build an empty index with an explicit RNG seed (for reproducible tests).
    #[must_use]
    pub fn seeded(metric: D, params: HnswParams, seed: u64) -> Self {
        Self {
            metric,
            params,
            nodes: HashMap::new(),
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

    /// Fetch a vector by id.
    #[must_use]
    pub fn get(&self, id: VectorId) -> Option<&Vector> {
        self.nodes.get(&id).map(|n| &n.vector)
    }

    /// Highest layer the node with `id` occupies, or `None` if absent.
    #[must_use]
    pub fn top_layer_of(&self, id: VectorId) -> Option<usize> {
        self.nodes.get(&id).map(Node::top_layer)
    }
}

impl<D: Distance> Index<D> for HnswIndex<D> {
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

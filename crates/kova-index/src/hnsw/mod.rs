//! HNSW index implementation.

use std::collections::HashMap;

use kova_core::{Distance, Vector, VectorId};

use crate::{Index, KovaIndexError};

mod node;
mod params;

use node::Node;
pub use params::HnswParams;

/// Hierarchical Navigable Small World index.
///
/// Constructors and read-only accessors are wired now; the [`Index`] trait
/// implementation has `insert` and `search` stubbed with `todo!` until
/// Algorithms 1 / 2 / 5 land.
pub struct HnswIndex<D: Distance> {
    metric: D,
    params: HnswParams,
    nodes: HashMap<VectorId, Node>,
    entry_point: Option<VectorId>,
    dim: Option<usize>,
}

impl<D: Distance> HnswIndex<D> {
    /// Build a new empty index using [`HnswParams::default`].
    #[must_use]
    pub fn new(metric: D) -> Self {
        Self::with_params(metric, HnswParams::default())
    }

    /// Build a new empty index with caller-supplied parameters.
    #[must_use]
    pub fn with_params(metric: D, params: HnswParams) -> Self {
        Self {
            metric,
            params,
            nodes: HashMap::new(),
            entry_point: None,
            dim: None,
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

    fn insert(&mut self, _id: VectorId, _vector: Vector) -> Result<(), Self::Error> {
        todo!("HNSW Algorithm 1 (insert)")
    }

    fn search(&self, _query: &Vector, _k: usize) -> Result<Vec<(VectorId, f32)>, Self::Error> {
        todo!("HNSW Algorithms 2 + 5 (search)")
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

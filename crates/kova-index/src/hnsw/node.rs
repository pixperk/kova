//! HNSW graph node : vector data + per-layer neighbour lists.

use kova_core::{Vector, VectorId};

/// A node in the HNSW graph.
///
/// Each node owns its [`Vector`] and a per-layer list of neighbour IDs.
/// `neighbors[L]` is the adjacency list at layer `L`; the total number of
/// layers a node occupies is `neighbors.len()`, i.e. `top_layer() + 1`.
#[derive(Debug)]
pub(crate) struct Node {
    /// The vector this node holds.
    pub(crate) vector: Vector,
    /// `neighbors[L]` is the list of neighbour IDs at layer L.
    /// `neighbors.len() == top_layer + 1`.
    pub(crate) neighbors: Vec<Vec<VectorId>>, //neighbor ids per layer, indexed by layer
}

impl Node {
    /// Creates a node that lives at every layer from `0` through `top_layer`,
    /// each with an empty neighbour list.
    #[allow(dead_code)] // called by HnswIndex::insert once Algorithm 1 lands
    pub(crate) fn new(vector: Vector, top_layer: usize) -> Self {
        Self {
            vector,
            neighbors: vec![Vec::new(); top_layer + 1],
        }
    }

    /// Returns the highest layer this node occupies.
    pub(crate) fn top_layer(&self) -> usize {
        self.neighbors.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_vec() -> Vector {
        Vector::try_new(vec![1.0, 2.0]).expect("test vector")
    }

    #[test]
    fn new_allocates_one_list_per_layer() {
        let node = Node::new(dummy_vec(), 3);
        assert_eq!(node.neighbors.len(), 4); // layers 0..=3
        assert!(node.neighbors.iter().all(Vec::is_empty));
    }

    #[test]
    fn top_layer_returns_highest_index() {
        assert_eq!(Node::new(dummy_vec(), 3).top_layer(), 3);
    }

    #[test]
    fn new_at_layer_zero_has_single_list() {
        let node = Node::new(dummy_vec(), 0);
        assert_eq!(node.neighbors.len(), 1);
        assert_eq!(node.top_layer(), 0);
    }
}

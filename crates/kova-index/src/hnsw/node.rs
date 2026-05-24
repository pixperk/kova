//! HNSW graph node : per-layer neighbour lists.
//!
//! Nodes hold only graph structure (which neighbours at which layer). The
//! actual vector bytes live in a [`kova_core::VectorStore`] composed into
//! the index. This separation lets the storage strategy (in-memory, mmap,
//! distributed) vary without touching HNSW.

use kova_core::VectorId;

/// A node in the HNSW graph.
///
/// `neighbors[L]` is the adjacency list at layer `L`. The total number of
/// layers a node occupies is `neighbors.len()`, i.e. `top_layer() + 1`.
#[derive(Debug)]
pub(crate) struct Node {
    /// `neighbors[L]` is the list of neighbour IDs at layer L.
    /// `neighbors.len() == top_layer + 1`.
    pub(crate) neighbors: Vec<Vec<VectorId>>,
}

impl Node {
    /// Creates a node that lives at every layer from `0` through `top_layer`,
    /// each with an empty neighbour list.
    pub(crate) fn new(top_layer: usize) -> Self {
        Self {
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

    #[test]
    fn new_allocates_one_list_per_layer() {
        let node = Node::new(3);
        assert_eq!(node.neighbors.len(), 4); // layers 0..=3
        assert!(node.neighbors.iter().all(Vec::is_empty));
    }

    #[test]
    fn top_layer_returns_highest_index() {
        assert_eq!(Node::new(3).top_layer(), 3);
    }

    #[test]
    fn new_at_layer_zero_has_single_list() {
        let node = Node::new(0);
        assert_eq!(node.neighbors.len(), 1);
        assert_eq!(node.top_layer(), 0);
    }
}

//! `search_layer` : bounded best-first search at one layer of the graph.
//!
//! This is the workhorse called by both insert (Algorithm 1) and the
//! user-facing search (Algorithms 2 + 5). The user-facing wrapper lands once
//! insert is in place.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use kova_core::{Distance, Vector, VectorId};

use super::HnswIndex;
use crate::scored::ScoredId;

impl<D: Distance> HnswIndex<D> {
    /// Bounded best-first walk at `layer`, seeded by `entry_points`.
    ///
    /// Returns up to `ef` `(id, distance)` pairs sorted ascending by distance.
    /// Allocates fresh state per call so concurrent reads do not interfere.
    pub(crate) fn search_layer(
        &self,
        query: &Vector,
        entry_points: &[VectorId],
        ef: usize,
        layer: usize,
    ) -> Vec<(VectorId, f32)> {
        if ef == 0 || entry_points.is_empty() {
            return Vec::new();
        }

        let mut visited: HashSet<VectorId> = HashSet::with_capacity(ef * 4);
        let mut candidates: BinaryHeap<Reverse<ScoredId>> = BinaryHeap::with_capacity(ef);
        let mut results: BinaryHeap<ScoredId> = BinaryHeap::with_capacity(ef);

        // Seed both heaps with the entry points.
        for &ep in entry_points {
            if !visited.insert(ep) {
                continue;
            }
            let Some(node) = self.nodes.get(&ep) else {
                continue;
            };
            let distance = self.metric.distance(query, &node.vector);
            let scored = ScoredId { id: ep, distance };
            candidates.push(Reverse(scored));
            results.push(scored);
            if results.len() > ef {
                results.pop();
            }
        }

        while let Some(Reverse(c)) = candidates.pop() {
            // No remaining candidate can improve results : the closest
            // unexplored is already worse than our current worst result.
            if let Some(worst) = results.peek()
                && c.distance > worst.distance
            {
                break;
            }

            let Some(c_node) = self.nodes.get(&c.id) else {
                continue;
            };
            // Defensively skip if c does not live at this layer.
            let Some(neighbours) = c_node.neighbors.get(layer) else {
                continue;
            };

            for &n_id in neighbours {
                if !visited.insert(n_id) {
                    continue;
                }
                let Some(n_node) = self.nodes.get(&n_id) else {
                    continue;
                };
                let n_dist = self.metric.distance(query, &n_node.vector);
                let worst_dist = results.peek().map_or(f32::INFINITY, |w| w.distance);

                if results.len() < ef || n_dist < worst_dist {
                    let scored = ScoredId {
                        id: n_id,
                        distance: n_dist,
                    };
                    candidates.push(Reverse(scored));
                    results.push(scored);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        results
            .into_sorted_vec()
            .into_iter()
            .map(|s| (s.id, s.distance))
            .collect()
    }
}

#[cfg(test)]
impl<D: Distance> HnswIndex<D> {
    /// Test-only: stash a node directly with empty neighbour lists.
    pub(crate) fn test_insert_node(&mut self, id: VectorId, vector: Vector, top_layer: usize) {
        let dim = vector.dim();
        self.nodes.insert(id, super::Node::new(vector, top_layer));
        if self.dim.is_none() {
            self.dim = Some(dim);
        }
        if self.entry_point.is_none() {
            self.entry_point = Some(id);
        }
    }

    /// Test-only: add a directed edge from `from` to `to` at `layer`.
    pub(crate) fn test_add_edge(&mut self, from: VectorId, to: VectorId, layer: usize) {
        if let Some(node) = self.nodes.get_mut(&from) {
            node.neighbors[layer].push(to);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kova_core::L2;

    fn v(data: Vec<f32>) -> Vector {
        Vector::try_new(data).expect("test vector")
    }

    fn id(n: u64) -> VectorId {
        VectorId::new(n)
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn empty_index_returns_empty() {
        let idx: HnswIndex<L2> = HnswIndex::new(L2);
        let q = v(vec![0.0]);
        let out = idx.search_layer(&q, &[id(1)], 4, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn ef_zero_returns_empty() {
        let mut idx = HnswIndex::new(L2);
        idx.test_insert_node(id(1), v(vec![0.0]), 0);
        let q = v(vec![0.0]);
        assert!(idx.search_layer(&q, &[id(1)], 0, 0).is_empty());
    }

    #[test]
    fn single_node_search() {
        let mut idx = HnswIndex::new(L2);
        idx.test_insert_node(id(1), v(vec![5.0]), 0);
        let q = v(vec![0.0]);
        let out = idx.search_layer(&q, &[id(1)], 4, 0);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, id(1));
        assert!(approx(out[0].1, 5.0));
    }

    #[test]
    fn finds_closer_neighbour_via_edge() {
        // A at 10, B at 1, edge between them at layer 0. Query is at 0.
        let mut idx = HnswIndex::new(L2);
        idx.test_insert_node(id(1), v(vec![10.0]), 0); // A
        idx.test_insert_node(id(2), v(vec![1.0]), 0); // B
        idx.test_add_edge(id(1), id(2), 0);
        idx.test_add_edge(id(2), id(1), 0);

        let q = v(vec![0.0]);
        // Enter at A; B should win at ef=1 because B is closer to Q.
        let out = idx.search_layer(&q, &[id(1)], 1, 0);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, id(2));
    }

    #[test]
    fn ef_bounds_result_size() {
        // 5 nodes on a fully-connected layer-0 graph.
        let mut idx = HnswIndex::new(L2);
        let points = [(1, 0.1), (2, 0.2), (3, 5.0), (4, 10.0), (5, 100.0)];
        for &(n, x) in &points {
            idx.test_insert_node(id(n), v(vec![x]), 0);
        }
        for &(a, _) in &points {
            for &(b, _) in &points {
                if a != b {
                    idx.test_add_edge(id(a), id(b), 0);
                }
            }
        }

        let q = v(vec![0.0]);
        let out = idx.search_layer(&q, &[id(3)], 2, 0); // enter at the middle node
        assert_eq!(out.len(), 2);
        // The two closest are id 1 (0.1) and id 2 (0.2).
        assert_eq!(out[0].0, id(1));
        assert_eq!(out[1].0, id(2));
    }

    #[test]
    fn multiple_entry_points() {
        // No edges. Each entry point is its own island.
        let mut idx = HnswIndex::new(L2);
        idx.test_insert_node(id(1), v(vec![1.0]), 0);
        idx.test_insert_node(id(2), v(vec![2.0]), 0);
        idx.test_insert_node(id(3), v(vec![3.0]), 0);

        let q = v(vec![0.0]);
        let out = idx.search_layer(&q, &[id(1), id(2), id(3)], 3, 0);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].0, id(1));
        assert_eq!(out[1].0, id(2));
        assert_eq!(out[2].0, id(3));
    }

    #[test]
    fn results_sorted_ascending() {
        // Walk from id 1 across edges to discover 2 and 3.
        let mut idx = HnswIndex::new(L2);
        idx.test_insert_node(id(1), v(vec![5.0]), 0);
        idx.test_insert_node(id(2), v(vec![1.0]), 0);
        idx.test_insert_node(id(3), v(vec![3.0]), 0);
        idx.test_add_edge(id(1), id(2), 0);
        idx.test_add_edge(id(1), id(3), 0);

        let q = v(vec![0.0]);
        let out = idx.search_layer(&q, &[id(1)], 3, 0);
        assert_eq!(out.len(), 3);
        // Ascending: 2 (d=1), 3 (d=3), 1 (d=5).
        assert_eq!(out[0].0, id(2));
        assert_eq!(out[1].0, id(3));
        assert_eq!(out[2].0, id(1));
        assert!(out[0].1 < out[1].1);
        assert!(out[1].1 < out[2].1);
    }
}

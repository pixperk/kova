//! HNSW Algorithm 1 : insertion into the graph.
//!
//! Orchestrates the three helpers built on previous days:
//! - [`super::layer::random_level`] picks the new node's top layer
//! - [`HnswIndex::search_layer`] finds candidate neighbours per layer
//! - [`HnswIndex::select_neighbors_heuristic`] picks `M` from those candidates

use kova_core::{Distance, Vector, VectorId};

use super::HnswIndex;
use super::layer::random_level;
use super::node::Node;
use crate::KovaIndexError;

impl<D: Distance> HnswIndex<D> {
    /// Insert `(id, vector)` into the index. Implements HNSW Algorithm 1.
    pub(crate) fn insert_impl(
        &mut self,
        id: VectorId,
        vector: Vector,
    ) -> Result<(), KovaIndexError> {
        // ---- Validate dimension and uniqueness ----
        if let Some(d) = self.dim
            && vector.dim() != d
        {
            return Err(KovaIndexError::DimensionMismatch {
                expected: d,
                got: vector.dim(),
            });
        }
        if self.nodes.contains_key(&id) {
            return Err(KovaIndexError::DuplicateId { id });
        }

        let new_level = random_level(self.params.m_l, &mut self.rng);

        // ---- First node: becomes the entry point and we're done ----
        if self.entry_point.is_none() {
            let dim = vector.dim();
            self.nodes.insert(id, Node::new(vector, new_level));
            self.dim = Some(dim);
            self.entry_point = Some(id);
            return Ok(());
        }

        let entry_id = self.entry_point.expect("non-empty checked above");
        let top_level = self.nodes[&entry_id].top_layer();

        // ---- Phase A : greedy descent through layers above new_level ----
        let mut current_ep = entry_id;
        if new_level < top_level {
            for layer in ((new_level + 1)..=top_level).rev() {
                let nearest = self.search_layer(&vector, &[current_ep], 1, layer);
                if let Some(&(best_id, _)) = nearest.first() {
                    current_ep = best_id;
                }
            }
        }

        // ---- Phase B : per-layer candidate + neighbour selection ----
        let max_insert_layer = new_level.min(top_level);
        let mut entry_points = vec![current_ep];
        let mut neighbours_per_layer: Vec<Vec<(VectorId, f32)>> =
            vec![Vec::new(); max_insert_layer + 1];

        for layer in (0..=max_insert_layer).rev() {
            let candidates =
                self.search_layer(&vector, &entry_points, self.params.ef_construction, layer);
            let m = self.params.m_for_layer(layer);
            let chosen = self.select_neighbors_heuristic(&candidates, m);
            entry_points = chosen.iter().map(|(nid, _)| *nid).collect();
            neighbours_per_layer[layer] = chosen;
        }

        // ---- Phase C : register the node, wire edges, prune overflow ----
        self.nodes.insert(id, Node::new(vector, new_level));

        // Take the neighbour lists out so we can index by layer without
        // tripping the borrow checker once we start mutating self.nodes.
        let layers_with_neighbours: Vec<(usize, Vec<(VectorId, f32)>)> =
            neighbours_per_layer.into_iter().enumerate().collect();

        for (layer, chosen) in &layers_with_neighbours {
            for &(n_id, _) in chosen {
                if let Some(new_node) = self.nodes.get_mut(&id) {
                    new_node.neighbors[*layer].push(n_id);
                }
                if let Some(neighbour_node) = self.nodes.get_mut(&n_id) {
                    neighbour_node.neighbors[*layer].push(id);
                }
            }

            // Re-select neighbours of any node that overflowed its degree cap.
            for &(n_id, _) in chosen {
                self.prune_neighbours(n_id, *layer);
            }
        }

        // Promote entry point if the new node now lives at a higher top layer.
        if new_level > top_level {
            self.entry_point = Some(id);
        }

        Ok(())
    }

    /// If `node_id`'s neighbour list at `layer` exceeds the degree cap,
    /// re-select neighbours via the diversity heuristic.
    fn prune_neighbours(&mut self, node_id: VectorId, layer: usize) {
        let cap = self.params.m_for_layer(layer);

        // Snapshot the candidate list while holding the immutable borrow,
        // then drop it before mutating self.
        let candidates = {
            let Some(node) = self.nodes.get(&node_id) else {
                return;
            };
            if node.neighbors[layer].len() <= cap {
                return;
            }
            let node_vec = &node.vector;
            let mut scored: Vec<(VectorId, f32)> = node.neighbors[layer]
                .iter()
                .filter_map(|&nn_id| {
                    self.nodes
                        .get(&nn_id)
                        .map(|nn| (nn_id, self.metric.distance(node_vec, &nn.vector)))
                })
                .collect();
            scored.sort_by(|a, b| a.1.total_cmp(&b.1));
            scored
        };

        let kept = self.select_neighbors_heuristic(&candidates, cap);

        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.neighbors[layer] = kept.into_iter().map(|(nid, _)| nid).collect();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Index;
    use kova_core::L2;

    fn v(data: Vec<f32>) -> Vector {
        Vector::try_new(data).expect("test vector")
    }

    fn id(n: u64) -> VectorId {
        VectorId::new(n)
    }

    #[test]
    fn first_insert_sets_entry_point_and_dim() {
        let mut idx = HnswIndex::new(L2);
        idx.insert(id(1), v(vec![1.0, 2.0])).unwrap();

        assert_eq!(idx.len(), 1);
        assert_eq!(idx.entry_point(), Some(id(1)));
        assert_eq!(idx.dim(), Some(2));
        assert!(idx.get(id(1)).is_some());
    }

    #[test]
    fn duplicate_id_returns_error() {
        let mut idx = HnswIndex::new(L2);
        idx.insert(id(1), v(vec![1.0])).unwrap();
        let err = idx.insert(id(1), v(vec![2.0])).unwrap_err();
        assert!(matches!(err, KovaIndexError::DuplicateId { .. }));
    }

    #[test]
    fn dim_mismatch_returns_error() {
        let mut idx = HnswIndex::new(L2);
        idx.insert(id(1), v(vec![1.0, 2.0])).unwrap();
        let err = idx.insert(id(2), v(vec![1.0])).unwrap_err();
        assert!(matches!(
            err,
            KovaIndexError::DimensionMismatch {
                expected: 2,
                got: 1,
            }
        ));
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn insert_many_grows_len() {
        let mut idx = HnswIndex::seeded(L2, super::super::HnswParams::default(), 7);
        for i in 0..50 {
            let f = i as f32;
            idx.insert(id(i), v(vec![f, f + 1.0])).unwrap();
        }
        assert_eq!(idx.len(), 50);
        let ep = idx.entry_point().unwrap();
        assert!(idx.get(ep).is_some());
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn seeded_inserts_are_deterministic() {
        // Same seed + same input order = structurally identical graph.
        let mut a = HnswIndex::seeded(L2, super::super::HnswParams::default(), 42);
        let mut b = HnswIndex::seeded(L2, super::super::HnswParams::default(), 42);

        for i in 0..30 {
            let f = i as f32;
            a.insert(id(i), v(vec![f, f * 2.0])).unwrap();
            b.insert(id(i), v(vec![f, f * 2.0])).unwrap();
        }

        assert_eq!(a.entry_point(), b.entry_point());
        for i in 0..30 {
            assert_eq!(a.top_layer_of(id(i)), b.top_layer_of(id(i)));
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn neighbour_lists_respect_layer_cap() {
        // After many inserts, no node's layer-0 neighbour list may exceed m_max0
        // (the pruning step must enforce this).
        let mut idx = HnswIndex::seeded(L2, super::super::HnswParams::default(), 11);
        let cap = idx.params().m_max0;
        for i in 0..40 {
            let f = i as f32;
            idx.insert(id(i), v(vec![f, f + 0.5])).unwrap();
        }
        // Direct invariant check via the private nodes map (same module tree).
        for node in idx.nodes.values() {
            assert!(node.neighbors[0].len() <= cap);
        }
    }
}

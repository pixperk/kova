//! HNSW search : the workhorse `search_layer` (Algorithm 2) and the
//! user-facing `search_impl` (Algorithm 5).
//!
//! Path A : neighbours' vectors are fetched via [`kova_core::VectorStore`]
//! at each distance computation. Returns owned [`Vector`] per call; cloning
//! cost is the tradeoff for the storage abstraction.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use kova_core::{Distance, Vector, VectorId, VectorStore};

use super::HnswIndex;
use crate::KovaIndexError;
use crate::scored::ScoredId;

impl<D: Distance, V: VectorStore> HnswIndex<D, V> {
    /// Bounded best-first walk at `layer`, seeded by `entry_points`.
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
            let Some(ep_vec) = self.vectors.get(ep) else {
                continue;
            };
            let distance = self.metric.distance(query, &ep_vec);
            let scored = ScoredId { id: ep, distance };
            candidates.push(Reverse(scored));
            results.push(scored);
            if results.len() > ef {
                results.pop();
            }
        }

        while let Some(Reverse(c)) = candidates.pop() {
            if let Some(worst) = results.peek()
                && c.distance > worst.distance
            {
                break;
            }

            let Some(c_node) = self.nodes.get(&c.id) else {
                continue;
            };
            let Some(neighbours) = c_node.neighbors.get(layer) else {
                continue;
            };

            for &n_id in neighbours {
                if !visited.insert(n_id) {
                    continue;
                }
                let Some(n_vec) = self.vectors.get(n_id) else {
                    continue;
                };
                let n_dist = self.metric.distance(query, &n_vec);
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

    /// User-facing search : HNSW Algorithm 5.
    pub(crate) fn search_impl(
        &self,
        query: &Vector,
        k: usize,
    ) -> Result<Vec<(VectorId, f32)>, KovaIndexError> {
        if k == 0 {
            return Ok(Vec::new());
        }

        let Some(entry_id) = self.entry_point else {
            return Ok(Vec::new());
        };

        if let Some(d) = self.dim
            && query.dim() != d
        {
            return Err(KovaIndexError::DimensionMismatch {
                expected: d,
                got: query.dim(),
            });
        }

        let top_level = self.nodes[&entry_id].top_layer();
        let ef = self.params.ef_search.max(k);

        let mut current_ep = entry_id;
        for layer in (1..=top_level).rev() {
            let nearest = self.search_layer(query, &[current_ep], 1, layer);
            if let Some(&(best_id, _)) = nearest.first() {
                current_ep = best_id;
            }
        }

        let mut results = self.search_layer(query, &[current_ep], ef, 0);
        results.truncate(k);
        Ok(results)
    }
}

// Test-only helpers : only available on the default in-memory store so we
// don't have to worry about fallible `put` in test code.
#[cfg(test)]
impl<D: Distance> HnswIndex<D, kova_core::InMemoryVectorStore> {
    /// Test-only: stash a node directly with empty neighbour lists.
    pub(crate) fn test_insert_node(&mut self, id: VectorId, vector: Vector, top_layer: usize) {
        let dim = vector.dim();
        self.vectors.put(id, vector).expect("infallible store");
        self.nodes.insert(id, super::Node::new(top_layer));
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
    use kova_core::{InMemoryVectorStore, L2};

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
        let idx: HnswIndex<L2, InMemoryVectorStore> = HnswIndex::new(L2);
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
        let mut idx = HnswIndex::new(L2);
        idx.test_insert_node(id(1), v(vec![10.0]), 0);
        idx.test_insert_node(id(2), v(vec![1.0]), 0);
        idx.test_add_edge(id(1), id(2), 0);
        idx.test_add_edge(id(2), id(1), 0);

        let q = v(vec![0.0]);
        let out = idx.search_layer(&q, &[id(1)], 1, 0);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, id(2));
    }

    #[test]
    fn ef_bounds_result_size() {
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
        let out = idx.search_layer(&q, &[id(3)], 2, 0);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, id(1));
        assert_eq!(out[1].0, id(2));
    }

    #[test]
    fn multiple_entry_points() {
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
        let mut idx = HnswIndex::new(L2);
        idx.test_insert_node(id(1), v(vec![5.0]), 0);
        idx.test_insert_node(id(2), v(vec![1.0]), 0);
        idx.test_insert_node(id(3), v(vec![3.0]), 0);
        idx.test_add_edge(id(1), id(2), 0);
        idx.test_add_edge(id(1), id(3), 0);

        let q = v(vec![0.0]);
        let out = idx.search_layer(&q, &[id(1)], 3, 0);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].0, id(2));
        assert_eq!(out[1].0, id(3));
        assert_eq!(out[2].0, id(1));
        assert!(out[0].1 < out[1].1);
        assert!(out[1].1 < out[2].1);
    }

    // ---------- user-facing search_impl ----------

    #[test]
    fn search_impl_empty_index_returns_empty() {
        let idx: HnswIndex<L2, InMemoryVectorStore> = HnswIndex::new(L2);
        let out = idx.search_impl(&v(vec![0.0]), 5).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn search_impl_k_zero_returns_empty() {
        let mut idx = HnswIndex::new(L2);
        crate::Index::insert(&mut idx, id(1), v(vec![1.0])).unwrap();
        let out = idx.search_impl(&v(vec![0.0]), 0).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn search_impl_dim_mismatch_errors() {
        let mut idx = HnswIndex::new(L2);
        crate::Index::insert(&mut idx, id(1), v(vec![1.0, 2.0])).unwrap();
        let err = idx.search_impl(&v(vec![1.0]), 1).unwrap_err();
        assert!(matches!(
            err,
            crate::KovaIndexError::DimensionMismatch {
                expected: 2,
                got: 1,
            }
        ));
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn search_impl_returns_sorted_ascending() {
        let mut idx = HnswIndex::seeded(L2, super::super::HnswParams::default(), 5);
        for i in 0..20 {
            let f = i as f32;
            crate::Index::insert(&mut idx, id(i), v(vec![f])).unwrap();
        }
        let out = idx.search_impl(&v(vec![0.0]), 5).unwrap();
        assert!(out.len() <= 5);
        for w in out.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn measure_recall(n: u64, dim: usize, k: usize, queries: usize, data_seed: u64) -> f32 {
        use crate::{FlatIndex, Index};
        use rand::{RngExt, SeedableRng, rngs::StdRng};
        use std::collections::HashSet;

        let mut rng = StdRng::seed_from_u64(data_seed);
        let mut hnsw = HnswIndex::seeded(L2, super::super::HnswParams::default(), 13);
        let mut flat: FlatIndex<L2> = FlatIndex::new(L2);

        for i in 0..n {
            let data: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            let vec = Vector::try_new(data).unwrap();
            hnsw.insert(id(i), vec.clone()).unwrap();
            flat.insert(id(i), vec).unwrap();
        }

        let mut total = 0.0_f32;
        for _ in 0..queries {
            let qdata: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            let q = Vector::try_new(qdata).unwrap();

            let h_ids: HashSet<VectorId> = hnsw
                .search(&q, k)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let f_ids: HashSet<VectorId> = flat
                .search(&q, k)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();

            total += h_ids.intersection(&f_ids).count() as f32 / k as f32;
        }
        total / queries as f32
    }

    #[test]
    fn recall_at_10_vs_flat_on_300_vectors() {
        let r = measure_recall(300, 8, 10, 30, 99);
        assert!(r > 0.9, "recall@10 at n=300 was {r:.3}, expected > 0.9");
    }

    #[test]
    fn recall_at_10_vs_flat_on_10k_dim32() {
        let r = measure_recall(10_000, 32, 10, 20, 99);
        assert!(
            r > 0.9,
            "recall@10 at n=10k dim=32 was {r:.3}, expected > 0.9"
        );
    }

    #[test]
    #[ignore = "slow: ~75s; run with --ignored"]
    fn recall_at_10_vs_flat_on_50k_dim32() {
        let r = measure_recall(50_000, 32, 10, 20, 99);
        assert!(
            r > 0.9,
            "recall@10 at n=50k dim=32 was {r:.3}, expected > 0.9"
        );
    }
}

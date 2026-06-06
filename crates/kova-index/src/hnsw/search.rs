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

        // Tombstoned ids stay in the graph (so traversal works) but are
        // filtered out of returned hits. Filtering after `search_layer`
        // means the result count may be smaller than `k` when many
        // candidates in the neighbourhood are deleted ; vacuum eventually
        // restores result quality.
        if !self.tombstones.is_empty() {
            results.retain(|(id, _)| !self.tombstones.contains(id));
        }

        results.truncate(k);
        Ok(results)
    }
}

impl<D: Distance, V: VectorStore> HnswIndex<D, V> {
    /// Bounded best-first walk at `layer` with a predicate filter.
    ///
    /// Differs from [`Self::search_layer`] in two ways :
    ///
    /// - **What goes into the results heap** : only nodes for which
    ///   `filter(id)` returns `true`. Out-of-filter nodes are still
    ///   *visited* (their neighbours can still get expanded), but they
    ///   never become candidates for the final top-k. This is the
    ///   "soft filter" of plan C : filtering is woven into the walk,
    ///   not bolted on as a post-filter.
    ///
    /// - **Termination** : we only short-circuit on
    ///   `c.distance > worst_result.distance` once we've accumulated
    ///   at least `ef` results. While the results heap is short of
    ///   `ef`, every popped candidate is worth expanding , the next
    ///   neighbour might be the first filter-passing node we see.
    ///
    /// Worst-case visits every node in `layer` when `filter` is empty
    /// on the candidate region ; this is the cost of the soft-filter
    /// strategy and is bounded by graph size.
    pub(crate) fn search_layer_filtered<F>(
        &self,
        query: &Vector,
        entry_points: &[VectorId],
        ef: usize,
        layer: usize,
        filter: &F,
    ) -> Vec<(VectorId, f32)>
    where
        F: Fn(VectorId) -> bool,
    {
        if ef == 0 || entry_points.is_empty() {
            return Vec::new();
        }

        let mut visited: HashSet<VectorId> = HashSet::with_capacity(ef * 4);
        let mut candidates: BinaryHeap<Reverse<ScoredId>> = BinaryHeap::with_capacity(ef);
        let mut results: BinaryHeap<ScoredId> = BinaryHeap::with_capacity(ef);

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
            if filter(ep) {
                results.push(scored);
                if results.len() > ef {
                    results.pop();
                }
            }
        }

        while let Some(Reverse(c)) = candidates.pop() {
            // Early termination requires both : results heap is full
            // (so further candidates can only displace, not seed it)
            // AND the popped candidate is worse than the worst result.
            if results.len() >= ef
                && let Some(worst) = results.peek()
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
                let scored = ScoredId {
                    id: n_id,
                    distance: n_dist,
                };
                let worst_dist = results.peek().map_or(f32::INFINITY, |w| w.distance);
                // Worth exploring through if the heap isn't full yet
                // (no upper bound on usefulness) or this node is
                // closer than the worst result (could route to a
                // closer filter-match).
                if results.len() < ef || n_dist < worst_dist {
                    candidates.push(Reverse(scored));
                    if filter(n_id) {
                        results.push(scored);
                        if results.len() > ef {
                            results.pop();
                        }
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

    /// Filtered kNN search : returns the top-`k` nodes that satisfy
    /// `filter`, ordered by distance to `query`. Plan C's primitive :
    /// the filter threads into the graph walk instead of getting
    /// applied as a post-filter.
    ///
    /// `filter` is called once per visited node. For predicates that
    /// require a metadata lookup, the caller composes that lookup
    /// inside the closure.
    pub(crate) fn search_filtered_impl<F>(
        &self,
        query: &Vector,
        k: usize,
        filter: &F,
    ) -> Result<Vec<(VectorId, f32)>, KovaIndexError>
    where
        F: Fn(VectorId) -> bool,
    {
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

        // Descent : unfiltered. The upper layers are sparse , we just
        // want any entry into the dense layer-0 region. Filtering only
        // there.
        let mut current_ep = entry_id;
        for layer in (1..=top_level).rev() {
            let nearest = self.search_layer(query, &[current_ep], 1, layer);
            if let Some(&(best_id, _)) = nearest.first() {
                current_ep = best_id;
            }
        }

        let mut results = self.search_layer_filtered(query, &[current_ep], ef, 0, filter);

        if !self.tombstones.is_empty() {
            results.retain(|(id, _)| !self.tombstones.contains(id));
        }

        results.truncate(k);
        Ok(results)
    }

    /// User-facing radius search.
    ///
    /// Strategy : descend the upper layers with kNN-1 to land near
    /// `query`, then run a kNN-style `search_layer` at layer 0 with
    /// doubling `ef`. We stop expanding once the result set contains
    /// *any* node outside the radius , that proves the radius ball is
    /// fully enclosed within the returned set (HNSW's locality property)
    /// , or once `ef` reaches the index size. Then filter by radius.
    ///
    /// Why doubling instead of a true radius walk : a naive "expand
    /// while in-radius" walk can't escape a local minimum where the
    /// entry point is outside the ball but a neighbour two hops away
    /// is inside. Bumping `ef` reuses the well-tested `search_layer`
    /// and inherits HNSW's recall guarantees.
    pub(crate) fn search_radius_impl(
        &self,
        query: &Vector,
        radius: f32,
    ) -> Result<Vec<(VectorId, f32)>, KovaIndexError> {
        if !radius.is_finite() || radius < 0.0 {
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

        let mut current_ep = entry_id;
        for layer in (1..=top_level).rev() {
            let nearest = self.search_layer(query, &[current_ep], 1, layer);
            if let Some(&(best_id, _)) = nearest.first() {
                current_ep = best_id;
            }
        }

        let total = self.nodes.len();
        let mut ef = self.params.ef_search.max(16);
        let mut layer0_hits;
        loop {
            layer0_hits = self.search_layer(query, &[current_ep], ef, 0);
            let saw_outside = layer0_hits.iter().any(|(_, d)| *d > radius);
            if saw_outside || ef >= total {
                break;
            }
            ef = ef.saturating_mul(2).min(total);
        }

        let mut results: Vec<(VectorId, f32)> = layer0_hits
            .into_iter()
            .filter(|(_, d)| *d <= radius)
            .collect();

        if !self.tombstones.is_empty() {
            results.retain(|(id, _)| !self.tombstones.contains(id));
        }

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

    // ---------- search_radius_impl ----------

    #[test]
    fn search_radius_empty_index_returns_empty() {
        let idx: HnswIndex<L2, InMemoryVectorStore> = HnswIndex::new(L2);
        let out = idx.search_radius_impl(&v(vec![0.0]), 10.0).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn search_radius_negative_returns_empty() {
        let mut idx = HnswIndex::new(L2);
        crate::Index::insert(&mut idx, id(1), v(vec![0.0])).unwrap();
        let out = idx.search_radius_impl(&v(vec![0.0]), -1.0).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn search_radius_zero_includes_exact_match() {
        let mut idx = HnswIndex::new(L2);
        crate::Index::insert(&mut idx, id(1), v(vec![0.0])).unwrap();
        crate::Index::insert(&mut idx, id(2), v(vec![5.0])).unwrap();
        let out = idx.search_radius_impl(&v(vec![0.0]), 0.0).unwrap();
        let ids: Vec<_> = out.iter().map(|(i, _)| *i).collect();
        assert_eq!(ids, vec![id(1)]);
    }

    #[test]
    fn search_radius_dim_mismatch_errors() {
        let mut idx = HnswIndex::new(L2);
        crate::Index::insert(&mut idx, id(1), v(vec![1.0, 2.0])).unwrap();
        let err = idx.search_radius_impl(&v(vec![1.0]), 10.0).unwrap_err();
        assert!(matches!(
            err,
            crate::KovaIndexError::DimensionMismatch {
                expected: 2,
                got: 1,
            }
        ));
    }

    #[test]
    fn search_radius_filters_tombstones() {
        let mut idx = HnswIndex::new(L2);
        for i in 0..5 {
            #[allow(clippy::cast_precision_loss)]
            let f = i as f32;
            crate::Index::insert(&mut idx, id(i), v(vec![f])).unwrap();
        }
        idx.tombstone(id(0)).unwrap();
        let out = idx.search_radius_impl(&v(vec![0.0]), 2.5).unwrap();
        let ids: Vec<_> = out.iter().map(|(i, _)| *i).collect();
        assert!(!ids.contains(&id(0)));
        assert!(ids.contains(&id(1)));
        assert!(ids.contains(&id(2)));
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn search_radius_results_sorted_ascending() {
        let mut idx = HnswIndex::seeded(L2, super::super::HnswParams::default(), 5);
        for i in 0..20 {
            let f = i as f32;
            crate::Index::insert(&mut idx, id(i), v(vec![f])).unwrap();
        }
        let out = idx.search_radius_impl(&v(vec![0.0]), 5.0).unwrap();
        for w in out.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
        for (_, d) in &out {
            assert!(*d <= 5.0);
        }
    }

    /// Radius parity harness : compare `HnswIndex::search_radius`
    /// against a flat-scan ground truth. Mirrors `measure_recall` for
    /// the kNN side. Skips queries whose ground-truth ball is empty
    /// (they contribute nothing to the recall numerator/denominator).
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn measure_radius_recall(
        n: u64,
        dim: usize,
        radius: f32,
        queries: usize,
        data_seed: u64,
    ) -> f32 {
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

        let mut sum = 0.0_f32;
        let mut counted = 0_usize;
        for _ in 0..queries {
            let qdata: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            let q = Vector::try_new(qdata).unwrap();

            let h_ids: HashSet<VectorId> = hnsw
                .search_radius(&q, radius)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let f_ids: HashSet<VectorId> = flat
                .search(&q, n as usize)
                .unwrap()
                .into_iter()
                .filter(|(_, d)| *d <= radius)
                .map(|(id, _)| id)
                .collect();

            if f_ids.is_empty() {
                continue;
            }
            sum += h_ids.intersection(&f_ids).count() as f32 / f_ids.len() as f32;
            counted += 1;
        }
        if counted == 0 {
            1.0
        } else {
            sum / counted as f32
        }
    }

    #[test]
    fn radius_recall_on_500_dim4() {
        let r = measure_radius_recall(500, 4, 0.3, 30, 42);
        assert!(
            r >= 0.9,
            "radius recall (n=500, dim=4, r=0.3) was {r:.3}, expected >= 0.9"
        );
    }

    #[test]
    fn radius_recall_on_2k_dim16() {
        let r = measure_radius_recall(2_000, 16, 0.6, 20, 43);
        assert!(
            r >= 0.9,
            "radius recall (n=2k, dim=16, r=0.6) was {r:.3}, expected >= 0.9"
        );
    }

    // ---------- search_filtered_impl ----------

    #[test]
    fn search_filtered_empty_index_returns_empty() {
        let idx: HnswIndex<L2, InMemoryVectorStore> = HnswIndex::new(L2);
        let always = |_id: VectorId| true;
        let out = idx.search_filtered_impl(&v(vec![0.0]), 5, &always).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn search_filtered_k_zero_returns_empty() {
        let mut idx = HnswIndex::new(L2);
        crate::Index::insert(&mut idx, id(1), v(vec![0.0])).unwrap();
        let always = |_id: VectorId| true;
        let out = idx.search_filtered_impl(&v(vec![0.0]), 0, &always).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn search_filtered_dim_mismatch_errors() {
        let mut idx = HnswIndex::new(L2);
        crate::Index::insert(&mut idx, id(1), v(vec![1.0, 2.0])).unwrap();
        let always = |_id: VectorId| true;
        let err = idx
            .search_filtered_impl(&v(vec![1.0]), 1, &always)
            .unwrap_err();
        assert!(matches!(
            err,
            crate::KovaIndexError::DimensionMismatch {
                expected: 2,
                got: 1,
            }
        ));
    }

    #[test]
    fn search_filtered_drops_filter_rejects_from_results() {
        let mut idx = HnswIndex::new(L2);
        for i in 0..10 {
            #[allow(clippy::cast_precision_loss)]
            let f = i as f32;
            crate::Index::insert(&mut idx, id(i), v(vec![f])).unwrap();
        }
        // Only even ids pass the filter.
        let even_only = |i: VectorId| i.get().is_multiple_of(2);
        let out = idx
            .search_filtered_impl(&v(vec![0.0]), 5, &even_only)
            .unwrap();
        for (i, _) in &out {
            assert_eq!(i.get() % 2, 0, "filter let odd id {i:?} through");
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn search_filtered_finds_filter_match_through_out_of_filter_neighbours() {
        // Verifies the soft-filter property : even when the entry
        // point is filtered out, the walk routes through it and
        // surfaces a filter-passing neighbour.
        let mut idx = HnswIndex::seeded(L2, super::super::HnswParams::default(), 11);
        for i in 0..20 {
            let f = i as f32;
            crate::Index::insert(&mut idx, id(i), v(vec![f])).unwrap();
        }
        // Filter keeps only id 10.
        let only_ten = |i: VectorId| i.get() == 10;
        let out = idx
            .search_filtered_impl(&v(vec![10.0]), 1, &only_ten)
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, id(10));
    }

    #[test]
    fn search_filtered_tombstoned_ids_excluded() {
        let mut idx = HnswIndex::new(L2);
        for i in 0..5 {
            #[allow(clippy::cast_precision_loss)]
            let f = i as f32;
            crate::Index::insert(&mut idx, id(i), v(vec![f])).unwrap();
        }
        idx.tombstone(id(1)).unwrap();
        let always = |_id: VectorId| true;
        let out = idx.search_filtered_impl(&v(vec![0.0]), 5, &always).unwrap();
        let ids: Vec<_> = out.iter().map(|(i, _)| *i).collect();
        assert!(!ids.contains(&id(1)));
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

    /// Recall harness for filtered search : compare
    /// `HnswIndex::search_filtered` against a flat-scan-then-filter
    /// ground truth. Filter keeps `keep_fraction` of the dataset by
    /// id, so we can dial selectivity directly without metadata.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn measure_filtered_recall(
        n: u64,
        dim: usize,
        k: usize,
        queries: usize,
        keep_fraction: f32,
        data_seed: u64,
    ) -> f32 {
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

        // Keep ids in `[0, n * keep_fraction)`. Simple, deterministic,
        // and independent of vector content , proves the filter
        // mechanism rather than some metadata coincidence.
        let cutoff = (f64::from(n as u32) * f64::from(keep_fraction)) as u64;
        let in_filter = |i: VectorId| i.get() < cutoff;

        let mut total = 0.0_f32;
        for _ in 0..queries {
            let qdata: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            let q = Vector::try_new(qdata).unwrap();

            let h_ids: HashSet<VectorId> = hnsw
                .search_filtered(&q, k, &in_filter)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();

            // Ground truth : every vector that passes the filter,
            // ranked by distance, take top-k.
            let mut f_all: Vec<_> = flat
                .search(&q, n as usize)
                .unwrap()
                .into_iter()
                .filter(|(i, _)| in_filter(*i))
                .collect();
            f_all.truncate(k);
            let f_ids: HashSet<VectorId> = f_all.into_iter().map(|(i, _)| i).collect();

            if f_ids.is_empty() {
                continue;
            }
            total += h_ids.intersection(&f_ids).count() as f32 / f_ids.len() as f32;
        }
        total / queries as f32
    }

    #[test]
    fn filtered_recall_at_10_on_500_dim8_half_filter() {
        // Mid-selectivity (50% kept) is plan C's hot spot.
        let r = measure_filtered_recall(500, 8, 10, 30, 0.5, 77);
        assert!(
            r > 0.9,
            "filtered recall@10 (n=500, dim=8, 50% kept) was {r:.3}, expected > 0.9"
        );
    }

    #[test]
    fn filtered_recall_at_10_on_2k_dim16_tight_filter() {
        // Tighter filter (20% kept) is the harder case : the filter
        // disqualifies most neighbours, so the walk has to traverse
        // farther to assemble the top-10.
        let r = measure_filtered_recall(2_000, 16, 10, 20, 0.2, 78);
        assert!(
            r > 0.9,
            "filtered recall@10 (n=2k, dim=16, 20% kept) was {r:.3}, expected > 0.9"
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

    /// Parametrised sweep over (recall variant) x (size, dim, selectivity).
    /// Prints a compact table so regressions are visible at a glance.
    /// Asserts every cell clears 0.9 ; if any one drops, the test fails
    /// with the specific cell named.
    #[test]
    #[ignore = "slow ; run with --ignored"]
    fn recall_sweep_baseline() {
        let mut report: Vec<(String, f32, bool)> = Vec::new();

        // ---- kNN recall : (n, dim) sweep at k=10 ----
        for &(n, dim) in &[
            (500u64, 4usize),
            (500, 16),
            (2_000, 8),
            (2_000, 32),
            (10_000, 16),
        ] {
            let r = measure_recall(n, dim, 10, 20, 99);
            let pass = r >= 0.9;
            report.push((format!("kNN @10  n={n:>5} dim={dim:>2}"), r, pass));
        }

        // ---- Filtered recall : (n, dim, keep_fraction) ----
        for &(n, dim, keep) in &[
            (500u64, 8usize, 0.5f32),
            (2_000, 16, 0.2),
            (2_000, 16, 0.5),
            (2_000, 16, 0.8),
            (5_000, 16, 0.3),
        ] {
            let r = measure_filtered_recall(n, dim, 10, 20, keep, 77);
            let pass = r >= 0.9;
            report.push((
                format!("flt @10  n={n:>5} dim={dim:>2} keep={keep:.2}"),
                r,
                pass,
            ));
        }

        // ---- Radius recall : (n, dim, r) ----
        for &(n, dim, radius) in &[
            (500u64, 4usize, 0.3f32),
            (2_000, 16, 0.6),
            (5_000, 16, 0.5),
            (5_000, 32, 1.0),
        ] {
            let recall = measure_radius_recall(n, dim, radius, 20, 43);
            let pass = recall >= 0.9;
            report.push((
                format!("rad      n={n:>5} dim={dim:>2} r={radius:.2}"),
                recall,
                pass,
            ));
        }

        // Print the report regardless of pass/fail so a maintainer can
        // see the full landscape, not just the first regression.
        eprintln!("\n=== HNSW recall sweep ===");
        for (label, r, pass) in &report {
            let marker = if *pass { "  " } else { "!!" };
            eprintln!("  {marker}  {label}  recall = {r:.3}");
        }

        // Now fail the test if any cell missed.
        let failures: Vec<_> = report.iter().filter(|(_, _, pass)| !*pass).collect();
        assert!(
            failures.is_empty(),
            "{} of {} recall cells dropped below 0.9 ; see report above",
            failures.len(),
            report.len(),
        );
    }
}

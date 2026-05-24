//! `select_neighbors_heuristic` (HNSW Algorithm 4, simple variant).
//!
//! Picks up to `m` neighbours from a sorted candidate list, preferring
//! *directional diversity* over raw proximity. A candidate `c` is accepted
//! only if no already-selected neighbour `s` is closer to `c` than the query
//! is : i.e. `c` opens a new direction not yet covered.
//!
//! This is what gives the HNSW graph its small-world property. Replacing it
//! with naive "top-m by distance to query" silently caps recall around 70%.

use kova_core::{Distance, VectorId, VectorStore};

use super::HnswIndex;

impl<D: Distance, V: VectorStore> HnswIndex<D, V> {
    /// Pick up to `m` neighbours from `candidates`, preferring diversity.
    ///
    /// `candidates` must be sorted ascending by distance to the query
    /// (which is what [`Self::search_layer`] returns). Order is preserved
    /// in the output : earlier acceptances win.
    pub(crate) fn select_neighbors_heuristic(
        &self,
        candidates: &[(VectorId, f32)],
        m: usize,
    ) -> Vec<(VectorId, f32)> {
        if m == 0 || candidates.is_empty() {
            return Vec::new();
        }

        let mut result: Vec<(VectorId, f32)> = Vec::with_capacity(m);

        for &(c_id, c_to_query) in candidates {
            if result.len() >= m {
                break;
            }

            let Some(c_vec) = self.vectors.get(c_id) else {
                continue;
            };

            // c is "dominated" if any already-selected s is closer to c than
            // the query is : i.e. s covers c's direction.
            let dominated = result.iter().any(|&(s_id, _)| {
                let Some(s_vec) = self.vectors.get(s_id) else {
                    return false;
                };
                self.metric.distance(&c_vec, &s_vec) < c_to_query
            });

            if !dominated {
                result.push((c_id, c_to_query));
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kova_core::{L2, Vector};

    fn v(data: Vec<f32>) -> Vector {
        Vector::try_new(data).expect("test vector")
    }

    fn id(n: u64) -> VectorId {
        VectorId::new(n)
    }

    #[test]
    fn empty_candidates_returns_empty() {
        let idx: HnswIndex<L2> = HnswIndex::new(L2);
        let out = idx.select_neighbors_heuristic(&[], 5);
        assert!(out.is_empty());
    }

    #[test]
    fn m_zero_returns_empty() {
        let mut idx = HnswIndex::new(L2);
        idx.test_insert_node(id(1), v(vec![1.0]), 0);
        let out = idx.select_neighbors_heuristic(&[(id(1), 1.0)], 0);
        assert!(out.is_empty());
    }

    #[test]
    fn single_candidate_accepted() {
        let mut idx = HnswIndex::new(L2);
        idx.test_insert_node(id(1), v(vec![1.0]), 0);
        let out = idx.select_neighbors_heuristic(&[(id(1), 1.0)], 3);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, id(1));
    }

    #[test]
    fn cluster_trap_picks_diverse_directions() {
        // Q = [0]. A, B, C cluster around x = 1; D is in the opposite
        // direction.  Naive top-3 would return {A, B, C} (all near each other).
        // The heuristic should return {A, D} : two directions, not three
        // duplicates.
        let mut idx = HnswIndex::new(L2);
        idx.test_insert_node(id(1), v(vec![1.0]), 0); // A
        idx.test_insert_node(id(2), v(vec![1.05]), 0); // B (close to A)
        idx.test_insert_node(id(3), v(vec![1.1]), 0); // C (close to A)
        idx.test_insert_node(id(4), v(vec![-1.2]), 0); // D (opposite)

        // Candidates sorted ascending by distance to Q = [0].
        let candidates = vec![
            (id(1), 1.0),  // A
            (id(2), 1.05), // B
            (id(3), 1.1),  // C
            (id(4), 1.2),  // D
        ];

        let out = idx.select_neighbors_heuristic(&candidates, 3);

        // A is closest, accepted. B and C are dominated by A
        // (d(B,A) = 0.05 < 1.05; d(C,A) = 0.1 < 1.1).
        // D is far from A (d(D,A) = 2.2 > 1.2), accepted.
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, id(1));
        assert_eq!(out[1].0, id(4));
    }

    #[test]
    fn all_diverse_directions_fill_m() {
        // Four orthogonal candidates in 4D. None dominates another, so all
        // are accepted up to M.
        let mut idx = HnswIndex::new(L2);
        idx.test_insert_node(id(1), v(vec![1.0, 0.0, 0.0, 0.0]), 0);
        idx.test_insert_node(id(2), v(vec![0.0, 1.1, 0.0, 0.0]), 0);
        idx.test_insert_node(id(3), v(vec![0.0, 0.0, 1.2, 0.0]), 0);
        idx.test_insert_node(id(4), v(vec![0.0, 0.0, 0.0, 1.3]), 0);

        let candidates = vec![(id(1), 1.0), (id(2), 1.1), (id(3), 1.2), (id(4), 1.3)];

        let out = idx.select_neighbors_heuristic(&candidates, 3);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].0, id(1));
        assert_eq!(out[1].0, id(2));
        assert_eq!(out[2].0, id(3));
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn respects_m_cap_even_with_more_candidates() {
        let mut idx = HnswIndex::new(L2);
        for n in 1..=5 {
            idx.test_insert_node(id(n), v(vec![n as f32, 0.0, 0.0, 0.0]), 0);
        }
        let candidates: Vec<(VectorId, f32)> = (1..=5).map(|n| (id(n), n as f32)).collect();

        let out = idx.select_neighbors_heuristic(&candidates, 2);
        assert!(out.len() <= 2);
    }
}

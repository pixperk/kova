//! A simple brute-force vector index that stores all vectors in memory and
//! computes distances on demand during search operations.
use std::collections::{BinaryHeap, HashMap};

use kova_core::{Distance, Vector, VectorId};

use crate::scored::ScoredId;
use crate::{Index, KovaIndexError};

/// A simple brute-force vector index that stores all vectors in memory and
/// computes distances on demand during search operations.
pub struct FlatIndex<D: Distance> {
    /// The distance metric this index uses for search operations.
    metric: D,
    /// Maps vector IDs to their corresponding vectors.
    nodes: HashMap<VectorId, Vector>,
    /// The dimension of vectors in this index, if any have been inserted yet.
    dim: Option<usize>,
}

impl<D: Distance> FlatIndex<D> {
    /// Creates a new empty `FlatIndex` with the specified distance metric.
    pub fn new(metric: D) -> Self {
        Self {
            metric,
            nodes: HashMap::new(),
            dim: None,
        }
    }
}

impl<D: Distance> Index<D> for FlatIndex<D> {
    type Error = KovaIndexError;

    fn insert(&mut self, id: VectorId, vector: Vector) -> Result<(), Self::Error> {
        // Check for dimension consistency.
        if let Some(expected_dim) = self.dim {
            if vector.dim() != expected_dim {
                return Err(KovaIndexError::DimensionMismatch {
                    expected: expected_dim,
                    got: vector.dim(),
                });
            }
        } else {
            // First inserted vector sets the dimension for the index.
            self.dim = Some(vector.dim());
        }

        // Check for duplicate ID.
        if self.nodes.contains_key(&id) {
            return Err(KovaIndexError::DuplicateId { id });
        }

        self.nodes.insert(id, vector);
        Ok(())
    }

    fn search(&self, query: &Vector, k: usize) -> Result<Vec<(VectorId, f32)>, Self::Error> {
        // Check for dimension consistency.
        if let Some(expected_dim) = self.dim {
            if query.dim() != expected_dim {
                return Err(KovaIndexError::DimensionMismatch {
                    expected: expected_dim,
                    got: query.dim(),
                });
            }
        } else {
            // If the index is empty, we can return an empty result without error.
            return Ok(Vec::new());
        }

        if k == 0 {
            return Ok(Vec::new());
        }

        // Bounded max-heap of size k. The top of the heap is the current worst
        // (largest distance) candidate, so we can evict it when something closer
        // arrives. O(n log k) instead of sort+truncate's O(n log n).
        let mut heap: BinaryHeap<ScoredId> = BinaryHeap::with_capacity(k);

        for (&id, vec) in &self.nodes {
            let distance = self.metric.distance(query, vec);

            if heap.len() < k {
                heap.push(ScoredId { id, distance });
            } else if let Some(worst) = heap.peek()
                && distance < worst.distance
            {
                heap.pop();
                heap.push(ScoredId { id, distance });
            }
        }

        // `into_sorted_vec` drains the heap in ascending order per our `Ord` impl:
        // closest first, as the `search` contract requires.
        let results = heap
            .into_sorted_vec()
            .into_iter()
            .map(|s| (s.id, s.distance))
            .collect();

        Ok(results)
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kova_core::{Cosine, L2};

    fn v(data: Vec<f32>) -> Vector {
        Vector::try_new(data).expect("test vector should be valid")
    }

    fn id(n: u64) -> VectorId {
        VectorId::new(n)
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    // ---------- shape ----------

    #[test]
    fn new_index_is_empty() {
        let idx: FlatIndex<L2> = FlatIndex::new(L2);
        assert_eq!(idx.len(), 0);
        assert!(idx.is_empty());
    }

    #[test]
    fn insert_pins_dim_and_increments_len() {
        let mut idx = FlatIndex::new(L2);
        idx.insert(id(1), v(vec![1.0, 2.0, 3.0])).unwrap();
        assert_eq!(idx.len(), 1);
        assert!(!idx.is_empty());
    }

    #[test]
    fn insert_same_dim_succeeds() {
        let mut idx = FlatIndex::new(L2);
        idx.insert(id(1), v(vec![1.0, 2.0])).unwrap();
        idx.insert(id(2), v(vec![3.0, 4.0])).unwrap();
        assert_eq!(idx.len(), 2);
    }

    // ---------- insert validation ----------

    #[test]
    fn insert_dim_mismatch_errors() {
        let mut idx = FlatIndex::new(L2);
        idx.insert(id(1), v(vec![1.0, 2.0])).unwrap();
        let err = idx.insert(id(2), v(vec![1.0])).unwrap_err();
        assert!(matches!(
            err,
            KovaIndexError::DimensionMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn insert_duplicate_id_errors() {
        let mut idx = FlatIndex::new(L2);
        idx.insert(id(1), v(vec![1.0, 2.0])).unwrap();
        let err = idx.insert(id(1), v(vec![3.0, 4.0])).unwrap_err();
        assert!(matches!(err, KovaIndexError::DuplicateId { .. }));
    }

    // ---------- search edge cases ----------

    #[test]
    fn search_empty_index_returns_empty() {
        let idx: FlatIndex<L2> = FlatIndex::new(L2);
        let results = idx.search(&v(vec![1.0, 2.0]), 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_dim_mismatch_errors() {
        let mut idx = FlatIndex::new(L2);
        idx.insert(id(1), v(vec![1.0, 2.0])).unwrap();
        let err = idx.search(&v(vec![1.0]), 1).unwrap_err();
        assert!(matches!(
            err,
            KovaIndexError::DimensionMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn search_k_zero_returns_empty() {
        let mut idx = FlatIndex::new(L2);
        idx.insert(id(1), v(vec![1.0])).unwrap();
        let results = idx.search(&v(vec![0.0]), 0).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_k_greater_than_len_returns_all() {
        let mut idx = FlatIndex::new(L2);
        idx.insert(id(1), v(vec![1.0])).unwrap();
        idx.insert(id(2), v(vec![2.0])).unwrap();
        let results = idx.search(&v(vec![0.0]), 100).unwrap();
        assert_eq!(results.len(), 2);
    }

    // ---------- search correctness ----------

    #[test]
    fn search_returns_sorted_ascending() {
        let mut idx = FlatIndex::new(L2);
        // Insert in non-sorted order so the result ordering is not an artifact
        // of insertion order.
        idx.insert(id(1), v(vec![10.0, 0.0])).unwrap(); // far
        idx.insert(id(2), v(vec![1.0, 0.0])).unwrap(); // closest
        idx.insert(id(3), v(vec![5.0, 0.0])).unwrap(); // middle

        let results = idx.search(&v(vec![0.0, 0.0]), 3).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, id(2));
        assert_eq!(results[1].0, id(3));
        assert_eq!(results[2].0, id(1));
        assert!(results[0].1 < results[1].1);
        assert!(results[1].1 < results[2].1);
    }

    #[test]
    fn search_l2_three_four_five_triangle() {
        let mut idx = FlatIndex::new(L2);
        idx.insert(id(1), v(vec![0.0, 0.0])).unwrap();
        idx.insert(id(2), v(vec![3.0, 4.0])).unwrap();

        let results = idx.search(&v(vec![0.0, 0.0]), 2).unwrap();
        assert_eq!(results[0].0, id(1));
        assert!(approx(results[0].1, 0.0));
        assert_eq!(results[1].0, id(2));
        assert!(approx(results[1].1, 5.0));
    }

    #[test]
    fn search_cosine_orthogonal() {
        let mut idx = FlatIndex::new(Cosine);
        idx.insert(id(1), v(vec![1.0, 0.0])).unwrap(); // same direction as query
        idx.insert(id(2), v(vec![0.0, 1.0])).unwrap(); // orthogonal to query

        let results = idx.search(&v(vec![1.0, 0.0]), 2).unwrap();
        assert_eq!(results[0].0, id(1));
        assert!(approx(results[0].1, 0.0));
        assert_eq!(results[1].0, id(2));
        assert!(approx(results[1].1, 1.0));
    }

    #[test]
    fn search_top_k_does_not_include_outliers() {
        // 5 vectors, ask for top 2. Make sure only the two closest come back.
        let mut idx = FlatIndex::new(L2);
        idx.insert(id(1), v(vec![0.1])).unwrap(); // closest
        idx.insert(id(2), v(vec![0.2])).unwrap(); // second
        idx.insert(id(3), v(vec![5.0])).unwrap();
        idx.insert(id(4), v(vec![10.0])).unwrap();
        idx.insert(id(5), v(vec![100.0])).unwrap();

        let results = idx.search(&v(vec![0.0]), 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, id(1));
        assert_eq!(results[1].0, id(2));
    }
}

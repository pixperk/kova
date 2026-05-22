//! The [`Index`] trait — the common interface every vector index implements.
//!
//! Each implementation (brute force, HNSW) pins down a distance metric at the
//! type level via the `D: Distance` parameter.

use kova_core::{Distance, Vector, VectorId};

/// A trait representing a vector index that supports insertion and search operations.
pub trait Index<D: Distance> {
    /// The error type returned by the index's operations.
    type Error;
    /// Inserts a vector with the given ID into the index.
    fn insert(&mut self, id: VectorId, vector: Vector) -> Result<(), Self::Error>;

    /// Searches for the k nearest neighbors to the query vector.
    fn search(&self, query: &Vector, k: usize) -> Result<Vec<(VectorId, f32)>, Self::Error>;
    /// Returns the number of vectors in the index.
    fn len(&self) -> usize;

    /// Returns true if the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

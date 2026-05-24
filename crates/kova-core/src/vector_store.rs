//! The [`VectorStore`] trait : id-to-vector storage abstraction.
//!
//! HNSW stores graph structure only; the actual vector bytes live in a
//! `VectorStore`. Concrete impls plug in different backends : an
//! `InMemoryVectorStore` (`HashMap`, for tests and small workloads), an
//! mmap-backed store (Phase 3 day 8), eventually distributed stores.

use std::collections::HashMap;

use crate::{Vector, VectorId};

/// Abstraction over id-to-vector storage.
///
/// Implementations decide how vectors are persisted (in memory, mmap, S3, ...).
/// HNSW composes one and looks up vectors via [`Self::get`] during search.
///
/// [`Self::get`] returns an owned [`Vector`] (clones from underlying storage).
/// This trades a per-call allocation for a simpler trait shape; a future
/// borrowed-view variant can be added if benches show the clone dominates.
pub trait VectorStore {
    /// Error type returned by mutating operations.
    type Error: std::fmt::Debug;

    /// Store `vector` under `id`. Overwrites any existing entry for `id`.
    fn put(&mut self, id: VectorId, vector: Vector) -> Result<(), Self::Error>;

    /// Fetch the vector for `id`, if present. Returns an owned clone.
    fn get(&self, id: VectorId) -> Option<Vector>;

    /// Number of vectors stored.
    fn len(&self) -> usize;

    /// Whether the store is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether `id` exists in the store.
    fn contains(&self, id: VectorId) -> bool {
        self.get(id).is_some()
    }
}

/// Trivial in-memory [`VectorStore`] backed by a `HashMap`.
///
/// Used as the default storage for [`crate::Vector`] consumers (e.g., HNSW
/// tests) and as a baseline against which other implementations are compared.
#[derive(Debug, Default, Clone)]
pub struct InMemoryVectorStore {
    nodes: HashMap<VectorId, Vector>,
}

impl InMemoryVectorStore {
    /// Construct an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl VectorStore for InMemoryVectorStore {
    type Error = std::convert::Infallible;

    fn put(&mut self, id: VectorId, vector: Vector) -> Result<(), Self::Error> {
        self.nodes.insert(id, vector);
        Ok(())
    }

    fn get(&self, id: VectorId) -> Option<Vector> {
        self.nodes.get(&id).cloned()
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(data: Vec<f32>) -> Vector {
        Vector::try_new(data).unwrap()
    }

    fn id(n: u64) -> VectorId {
        VectorId::new(n)
    }

    #[test]
    fn new_is_empty() {
        let store = InMemoryVectorStore::new();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn put_then_get_roundtrip() {
        let mut store = InMemoryVectorStore::new();
        let original = v(vec![1.0, 2.0, 3.0]);
        store.put(id(42), original.clone()).unwrap();
        assert_eq!(store.get(id(42)), Some(original));
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn get_missing_returns_none() {
        let store = InMemoryVectorStore::new();
        assert!(store.get(id(99)).is_none());
        assert!(!store.contains(id(99)));
    }

    #[test]
    fn put_overwrites_existing() {
        let mut store = InMemoryVectorStore::new();
        store.put(id(1), v(vec![1.0])).unwrap();
        store.put(id(1), v(vec![2.0])).unwrap();
        assert_eq!(store.get(id(1)), Some(v(vec![2.0])));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn contains_matches_get() {
        let mut store = InMemoryVectorStore::new();
        store.put(id(5), v(vec![5.0])).unwrap();
        assert!(store.contains(id(5)));
        assert!(!store.contains(id(6)));
    }
}

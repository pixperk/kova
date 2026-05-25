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
    ///
    /// Bounded as `Error + Send + Sync + 'static` so callers (notably
    /// [`crate::MetadataStore`] consumers and `kova-storage::Shard`) can
    /// box the error into a single `Box<dyn Error + Send + Sync>` for the
    /// generic composition layer. All current impls satisfy this trivially
    /// (`Infallible` for in-memory ; concrete `Error` types for file/mmap).
    type Error: std::error::Error + Send + Sync + 'static;

    /// Store `vector` under `id`. Overwrites any existing entry for `id`.
    fn put(&mut self, id: VectorId, vector: Vector) -> Result<(), Self::Error>;

    /// Fetch the vector for `id`, if present. Returns an owned clone.
    fn get(&self, id: VectorId) -> Option<Vector>;

    /// Remove the entry for `id`. No-op if `id` isn't present.
    ///
    /// Idempotent : removing a missing id is **not** an error. Callers
    /// that want to distinguish "was actually removed" from "wasn't
    /// there" check [`Self::contains`] first.
    ///
    /// Implementations that store data on disk are free to retain the
    /// freed capacity for reuse (e.g. an mmap store reusing slots via a
    /// free list, rather than truncating the file).
    ///
    /// # Errors
    /// Returns `Self::Error` only on underlying I/O failure (mmap flush,
    /// etc.). The "id-not-present" case is not an error.
    fn remove(&mut self, id: VectorId) -> Result<(), Self::Error>;

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

    /// Pinned vector dimension, if the store has one.
    ///
    /// Returns `Some(d)` for stores that fix the dimension at construction
    /// (e.g. mmap stores reading dim from their file header). Returns
    /// `None` for stores that infer the dimension from the first `put`.
    ///
    /// Used by upstream composition layers (e.g. `kova-storage::Shard`)
    /// to validate inserts against the pinned dim *before* committing to
    /// the WAL, instead of discovering the mismatch during apply.
    ///
    /// Default returns `None` ; only stores that genuinely pin a dim
    /// should override.
    fn dim(&self) -> Option<usize> {
        None
    }

    /// Pre-grow capacity for `additional` upcoming `put`s.
    ///
    /// Hint, not a guarantee : implementations are free to ignore it
    /// (default does), but file-backed stores override to grow once
    /// instead of paying the per-`put` grow cost. Caller's responsibility
    /// to ensure the requested capacity actually fits ; this is an
    /// optimisation, not an allocation reservation.
    ///
    /// # Errors
    /// Returns `Self::Error` if the underlying grow operation fails (e.g.
    /// `ENOSPC` for a file-backed store).
    fn reserve(&mut self, _additional: usize) -> Result<(), Self::Error> {
        Ok(())
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

    fn remove(&mut self, id: VectorId) -> Result<(), Self::Error> {
        self.nodes.remove(&id);
        Ok(())
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

    #[test]
    fn remove_present_drops_entry() {
        let mut store = InMemoryVectorStore::new();
        store.put(id(1), v(vec![1.0])).unwrap();
        store.put(id(2), v(vec![2.0])).unwrap();
        assert_eq!(store.len(), 2);

        store.remove(id(1)).unwrap();
        assert!(!store.contains(id(1)));
        assert!(store.contains(id(2)));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn remove_missing_is_noop() {
        let mut store = InMemoryVectorStore::new();
        store.put(id(1), v(vec![1.0])).unwrap();
        // Removing an id that was never inserted is not an error.
        store.remove(id(99)).unwrap();
        assert_eq!(store.len(), 1);
        assert!(store.contains(id(1)));
    }
}

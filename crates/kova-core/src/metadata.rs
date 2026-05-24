//! The [`MetadataStore`] trait : id-to-metadata storage abstraction.
//!
//! A vector record has two halves : the raw vector (lives in a
//! [`crate::VectorStore`]) and an open-shaped bag of attributes (lives here).
//! Filtered search (`WHERE doc_type = 'invoice' AND created_at > X`) reads
//! from this store after the index returns ANN candidates.
//!
//! The trait is intentionally minimal. Concrete impls plug in different
//! backends : an [`InMemoryMetadataStore`] (`HashMap`, for tests and small
//! workloads), a file-backed store in `kova-storage`, eventually a
//! columnar/indexed store once filter selectivity matters.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::VectorId;

/// A single attribute value attached to a vector.
///
/// Mirrors the JSON-ish shape callers expect without pulling in `serde_json`.
/// `F64` is kept distinct from `I64` so round-trips don't silently widen ints
/// to floats; `Bool` is distinct from `I64` for the same reason.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// UTF-8 string.
    String(String),
    /// Signed 64-bit integer.
    I64(i64),
    /// 64-bit float. NaN is permitted here (unlike [`crate::Vector`]) because
    /// metadata is opaque to the index ; downstream KQL predicates decide
    /// whether NaN matches.
    F64(f64),
    /// Boolean.
    Bool(bool),
    /// Homogeneous-or-heterogeneous list. Used for tag arrays and similar.
    Array(Vec<Value>),
}

/// An attribute bag attached to a single [`VectorId`].
///
/// Keys are arbitrary strings; values are [`Value`]s. The shape is open : two
/// vectors in the same store may carry entirely different keys.
pub type Metadata = HashMap<String, Value>;

/// Abstraction over id-to-metadata storage.
///
/// Implementations decide how attributes are persisted (in memory, on disk,
/// columnar, ...). The shard composes one and consults it after the index
/// returns ANN candidates to apply WHERE-style filters.
pub trait MetadataStore {
    /// Error type returned by mutating operations.
    ///
    /// Bounded as `Error + Send + Sync + 'static` so callers (notably
    /// `kova-storage::Shard`) can box the error into a single
    /// `Box<dyn Error + Send + Sync>` for the generic composition layer.
    /// All current impls satisfy this trivially (`Infallible` for in-memory ;
    /// concrete `Error` types for file-backed).
    type Error: std::error::Error + Send + Sync + 'static;

    /// Store `meta` under `id`. Overwrites any existing entry for `id`.
    fn put(&mut self, id: VectorId, meta: Metadata) -> Result<(), Self::Error>;

    /// Fetch the metadata for `id`, if present. Returns an owned clone.
    fn get(&self, id: VectorId) -> Option<Metadata>;

    /// Remove the entry for `id`. No-op if absent.
    fn delete(&mut self, id: VectorId) -> Result<(), Self::Error>;

    /// Number of entries stored.
    fn len(&self) -> usize;

    /// Whether the store is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether `id` exists in the store.
    fn contains(&self, id: VectorId) -> bool {
        self.get(id).is_some()
    }

    /// Insert many `(id, metadata)` pairs as a single logical batch.
    ///
    /// Default implementation just calls [`Self::put`] for each item.
    /// Backends that benefit from batching (e.g. a file-backed store
    /// that rewrites the whole file on every `put`) should override to
    /// amortise the per-item cost across the batch.
    ///
    /// # Object safety
    /// This method takes a generic `IntoIterator`, which makes it
    /// callable only when `Self: Sized`. Use [`Self::put`] in a loop if
    /// you need to call through `&dyn MetadataStore`.
    ///
    /// # Errors
    /// Returns `Self::Error` on the first failure ; partial application
    /// of preceding items is implementation-defined.
    fn put_many<I>(&mut self, items: I) -> Result<(), Self::Error>
    where
        I: IntoIterator<Item = (VectorId, Metadata)>,
        Self: Sized,
    {
        for (id, meta) in items {
            self.put(id, meta)?;
        }
        Ok(())
    }
}

/// Trivial in-memory [`MetadataStore`] backed by a `HashMap`.
///
/// Used as the default metadata backend for shards in tests and as a
/// baseline against which persistent implementations are compared.
#[derive(Debug, Default, Clone)]
pub struct InMemoryMetadataStore {
    entries: HashMap<VectorId, Metadata>,
}

impl InMemoryMetadataStore {
    /// Construct an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl MetadataStore for InMemoryMetadataStore {
    type Error = std::convert::Infallible;

    fn put(&mut self, id: VectorId, meta: Metadata) -> Result<(), Self::Error> {
        self.entries.insert(id, meta);
        Ok(())
    }

    fn get(&self, id: VectorId) -> Option<Metadata> {
        self.entries.get(&id).cloned()
    }

    fn delete(&mut self, id: VectorId) -> Result<(), Self::Error> {
        self.entries.remove(&id);
        Ok(())
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u64) -> VectorId {
        VectorId::new(n)
    }

    fn sample_meta() -> Metadata {
        let mut m = Metadata::new();
        m.insert("user_id".into(), Value::String("u_9921".into()));
        m.insert("count".into(), Value::I64(42));
        m.insert("score".into(), Value::F64(0.87));
        m.insert("active".into(), Value::Bool(true));
        m.insert(
            "tags".into(),
            Value::Array(vec![
                Value::String("q2".into()),
                Value::String("urgent".into()),
            ]),
        );
        m
    }

    #[test]
    fn new_is_empty() {
        let store = InMemoryMetadataStore::new();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn put_then_get_roundtrip() {
        let mut store = InMemoryMetadataStore::new();
        let meta = sample_meta();
        store.put(id(1), meta.clone()).unwrap();
        assert_eq!(store.get(id(1)), Some(meta));
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn get_missing_returns_none() {
        let store = InMemoryMetadataStore::new();
        assert!(store.get(id(99)).is_none());
        assert!(!store.contains(id(99)));
    }

    #[test]
    fn put_overwrites_existing() {
        let mut store = InMemoryMetadataStore::new();
        let mut a = Metadata::new();
        a.insert("k".into(), Value::I64(1));
        let mut b = Metadata::new();
        b.insert("k".into(), Value::I64(2));
        store.put(id(1), a).unwrap();
        store.put(id(1), b.clone()).unwrap();
        assert_eq!(store.get(id(1)), Some(b));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn delete_removes_entry() {
        let mut store = InMemoryMetadataStore::new();
        store.put(id(7), sample_meta()).unwrap();
        assert!(store.contains(id(7)));
        store.delete(id(7)).unwrap();
        assert!(!store.contains(id(7)));
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn delete_missing_is_noop() {
        let mut store = InMemoryMetadataStore::new();
        store.delete(id(42)).unwrap();
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn contains_matches_get() {
        let mut store = InMemoryMetadataStore::new();
        store.put(id(5), sample_meta()).unwrap();
        assert!(store.contains(id(5)));
        assert!(!store.contains(id(6)));
    }

    #[test]
    fn value_roundtrips_through_bincode() {
        let meta = sample_meta();
        let bytes = bincode::serialize(&meta).unwrap();
        let back: Metadata = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back, meta);
    }
}

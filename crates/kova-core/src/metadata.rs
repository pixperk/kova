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
    /// Nested key-value bag. Same shape as [`Metadata`] itself ;
    /// enables hierarchical attributes (`location.city`, etc.) and
    /// subscripted assignment (`SET attrs['key'] = ...`).
    ///
    /// Ordering against other `Value` variants is undefined : the
    /// predicate evaluator surfaces comparisons against `Map` as
    /// errors rather than silently coercing.
    Map(HashMap<String, Value>),
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
    ///
    /// Prefer [`Self::with_metadata`] when the caller only needs to
    /// *read* the bag : `get` clones the whole `HashMap`, which
    /// dominates on hot paths that touch many rows.
    fn get(&self, id: VectorId) -> Option<Metadata>;

    /// Borrow the metadata for `id` and run `f` against it, returning
    /// `f`'s result. `None` if `id` isn't in the store.
    ///
    /// This is the read-only counterpart to [`Self::get`]. `get` has to
    /// hand back an owned [`Metadata`], which means cloning the entire
    /// attribute bag (a `HashMap<String, Value>`, itself holding owned
    /// `String`s and possibly nested `Array` / `Map` values). Callers
    /// that just want to evaluate a predicate, read one field, or test
    /// presence pay that allocation for nothing.
    ///
    /// The cost is not academic : the query planner's `c_metadata_get`
    /// coefficient was calibrated against `get` at ~310 ns, and the
    /// filtered-ANN plan pays it *per visited graph node*. Routing
    /// those paths through this method removes the clone from the hot
    /// loop entirely.
    ///
    /// The default implementation delegates to [`Self::get`] (so
    /// existing impls keep compiling unchanged) and therefore still
    /// clones. Concrete stores that hold bags in memory should override
    /// to hand out a borrow.
    ///
    /// # Object safety
    /// The generic closure and return type make this callable only
    /// when `Self: Sized`, matching [`Self::scan_ids`] and
    /// [`Self::walk_field`].
    fn with_metadata<F, R>(&self, id: VectorId, f: F) -> Option<R>
    where
        F: FnOnce(&Metadata) -> R,
        Self: Sized,
    {
        self.get(id).map(|m| f(&m))
    }

    /// Remove the entry for `id`. No-op if absent.
    fn delete(&mut self, id: VectorId) -> Result<(), Self::Error>;

    /// Number of entries stored.
    fn len(&self) -> usize;

    /// Push any in-memory state to durable storage.
    ///
    /// **Mutations are not required to be durable when they return.**
    /// Durability for metadata comes from the WAL : `Shard` logs and
    /// fsyncs every mutation *before* touching this store, so the log
    /// is the recovery source and this file is a checkpoint artifact.
    /// `Shard::checkpoint` is what calls this.
    ///
    /// That split matters because a file-backed store rewrites the whole
    /// file per flush : two fsyncs and O(rows) bytes. Doing that per
    /// mutation measured at ~7.9 ms per `put` regardless of store size
    /// (a hard ~125 writes/sec ceiling), and pushed 852 MB through the
    /// disk to store 436 KB. Flushing at checkpoint instead makes the
    /// cost proportional to checkpoints rather than to writes.
    ///
    /// In-memory implementations have nothing to do, hence the default.
    ///
    /// # Errors
    /// Returns `Self::Error` if the underlying write fails.
    fn flush(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

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

    /// Walk all `(id, metadata)` pairs in the store, returning the
    /// ids whose metadata satisfies `pred`. The predicate borrows
    /// each metadata bag (no clones), making this much cheaper than
    /// calling [`Self::get`] in a loop.
    ///
    /// Order is implementation-defined ; callers that need a specific
    /// order must sort the result.
    ///
    /// # Object safety
    /// The generic closure parameter makes this method callable only
    /// when `Self: Sized`. Trait-object callers must walk through
    /// [`Self::get`] explicitly.
    fn scan_ids<F>(&self, pred: F) -> Vec<VectorId>
    where
        F: FnMut(&Metadata) -> bool,
        Self: Sized;

    /// Walk every row that has a value at top-level `field`, calling
    /// `callback(id, &value)` for each. The intended use is secondary-
    /// index backfill : a single pass over the store produces the
    /// `(id, value)` stream the index needs, no per-row re-fetches.
    ///
    /// Default impl composes [`Self::scan_ids`] with [`Self::get`] so
    /// existing impls work without overriding ; concrete stores should
    /// override to walk their internal map directly and skip the
    /// double-traversal.
    ///
    /// # Object safety
    /// Same constraint as [`Self::scan_ids`] : the generic closure
    /// makes this `Self: Sized` only.
    fn walk_field<F>(&self, field: &str, mut callback: F)
    where
        F: FnMut(VectorId, &Value),
        Self: Sized,
    {
        let ids = self.scan_ids(|m| m.contains_key(field));
        for id in ids {
            if let Some(bag) = self.get(id)
                && let Some(v) = bag.get(field)
            {
                callback(id, v);
            }
        }
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

    /// Borrowing override : hands the closure a reference straight out
    /// of the map, skipping the clone the default impl would pay.
    fn with_metadata<F, R>(&self, id: VectorId, f: F) -> Option<R>
    where
        F: FnOnce(&Metadata) -> R,
    {
        self.entries.get(&id).map(f)
    }

    fn delete(&mut self, id: VectorId) -> Result<(), Self::Error> {
        self.entries.remove(&id);
        Ok(())
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn scan_ids<F>(&self, mut pred: F) -> Vec<VectorId>
    where
        F: FnMut(&Metadata) -> bool,
    {
        self.entries
            .iter()
            .filter(|(_, m)| pred(m))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Single-pass override of the default trait impl : iterate the
    /// in-memory `HashMap` once, calling `callback` for every row
    /// that has the field. Skips the predicate-then-`get` double
    /// traversal.
    fn walk_field<F>(&self, field: &str, mut callback: F)
    where
        F: FnMut(VectorId, &Value),
    {
        for (id, meta) in &self.entries {
            if let Some(v) = meta.get(field) {
                callback(*id, v);
            }
        }
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

    #[test]
    fn scan_ids_returns_matching_entries() {
        let mut store = InMemoryMetadataStore::new();
        for i in 1..=5_u64 {
            let mut m = Metadata::new();
            m.insert(
                "category".into(),
                Value::String(if i % 2 == 0 {
                    "even".into()
                } else {
                    "odd".into()
                }),
            );
            store.put(id(i), m).unwrap();
        }
        let mut matches =
            store.scan_ids(|m| matches!(m.get("category"), Some(Value::String(s)) if s == "even"));
        matches.sort_by_key(|v| v.get());
        assert_eq!(matches, vec![id(2), id(4)]);
    }

    #[test]
    fn scan_ids_on_empty_store_returns_empty() {
        let store = InMemoryMetadataStore::new();
        let matches = store.scan_ids(|_| true);
        assert!(matches.is_empty());
    }

    #[test]
    fn scan_ids_with_always_false_predicate_returns_empty() {
        let mut store = InMemoryMetadataStore::new();
        store.put(id(1), sample_meta()).unwrap();
        store.put(id(2), sample_meta()).unwrap();
        let matches = store.scan_ids(|_| false);
        assert!(matches.is_empty());
    }

    #[test]
    fn scan_ids_does_not_clone_metadata() {
        // The predicate gets `&Metadata`, so it can inspect the bag
        // without forcing an allocation. The store's `entries` field
        // still owns the data after the scan.
        let mut store = InMemoryMetadataStore::new();
        store.put(id(1), sample_meta()).unwrap();
        let _ = store.scan_ids(|_| true);
        assert!(store.contains(id(1)));
    }

    #[test]
    fn with_metadata_runs_closure_against_the_stored_bag() {
        let mut store = InMemoryMetadataStore::new();
        store.put(id(1), sample_meta()).unwrap();

        let count = store.with_metadata(id(1), |m| m.get("count").cloned());
        assert_eq!(count, Some(Some(Value::I64(42))));
    }

    #[test]
    fn with_metadata_missing_id_returns_none_without_calling_closure() {
        let store = InMemoryMetadataStore::new();
        let mut called = false;
        let out = store.with_metadata(id(99), |_| {
            called = true;
            1_u8
        });
        assert_eq!(out, None);
        assert!(!called, "closure must not run for a missing id");
    }

    #[test]
    fn with_metadata_can_return_any_type() {
        // The closure's return type is generic, so callers can project
        // out a bool (predicate eval), an owned clone, or anything else.
        let mut store = InMemoryMetadataStore::new();
        store.put(id(1), sample_meta()).unwrap();

        let is_active = store.with_metadata(id(1), |m| {
            matches!(m.get("active"), Some(Value::Bool(true)))
        });
        assert_eq!(is_active, Some(true));

        let key_count = store.with_metadata(id(1), HashMap::len);
        assert_eq!(key_count, Some(5));
    }

    #[test]
    fn with_metadata_leaves_the_store_intact() {
        // Borrowing, not moving : the bag is still owned by the store
        // after the closure returns.
        let mut store = InMemoryMetadataStore::new();
        store.put(id(1), sample_meta()).unwrap();
        let _ = store.with_metadata(id(1), |_| ());
        assert_eq!(store.get(id(1)), Some(sample_meta()));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn with_metadata_agrees_with_get() {
        // The override must be observationally identical to the default
        // impl (`get(id).map(|m| f(&m))`) for both present and absent ids.
        let mut store = InMemoryMetadataStore::new();
        store.put(id(1), sample_meta()).unwrap();

        for probe in [id(1), id(2)] {
            let via_get = store.get(probe).map(|m| m.len());
            let via_borrow = store.with_metadata(probe, HashMap::len);
            assert_eq!(via_get, via_borrow, "mismatch for id {}", probe.get());
        }
    }

    #[test]
    fn walk_field_visits_only_rows_with_the_field() {
        let mut store = InMemoryMetadataStore::new();
        store.put(id(1), sample_meta()).unwrap(); // has "count"
        store.put(id(2), sample_meta()).unwrap(); // has "count"
        let mut m_no_count = Metadata::new();
        m_no_count.insert("category".into(), Value::String("docs".into()));
        store.put(id(3), m_no_count).unwrap(); // no "count"

        let mut visited: Vec<(VectorId, Value)> = Vec::new();
        store.walk_field("count", |i, v| visited.push((i, v.clone())));
        visited.sort_by_key(|(i, _)| i.get());

        // ids 1 and 2 ; id 3 lacks the field.
        assert_eq!(visited.len(), 2);
        assert_eq!(visited[0].0, id(1));
        assert_eq!(visited[1].0, id(2));
        // Both have the same I64(42) from sample_meta.
        assert_eq!(visited[0].1, Value::I64(42));
        assert_eq!(visited[1].1, Value::I64(42));
    }

    #[test]
    fn walk_field_skips_rows_whose_value_is_at_other_keys() {
        // Defensive : `field` is the top-level key. If a row has
        // {"other": ...} but no "category", the walk skips it.
        let mut store = InMemoryMetadataStore::new();
        let mut m = Metadata::new();
        m.insert("other".into(), Value::String("docs".into()));
        store.put(id(1), m).unwrap();

        let mut visited = 0;
        store.walk_field("category", |_, _| visited += 1);
        assert_eq!(visited, 0);
    }

    #[test]
    fn walk_field_passes_owned_value_via_reference() {
        // The callback gets `&Value` ; we can clone or borrow.
        let mut store = InMemoryMetadataStore::new();
        store.put(id(1), sample_meta()).unwrap();
        let mut seen: Option<Value> = None;
        store.walk_field("active", |_, v| seen = Some(v.clone()));
        assert_eq!(seen, Some(Value::Bool(true)));
        assert!(store.contains(id(1)));
    }
}

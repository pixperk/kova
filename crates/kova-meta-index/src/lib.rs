//! Secondary indexes for Kova metadata predicates.
//!
//! Three index types sit behind one trait :
//!
//! - [`HashIndex`] : equality lookups (`field = X`, `field IN (...)`).
//! - [`BTreeIndex`] : range lookups (`field < x`, `BETWEEN lo AND hi`).
//! - [`InvertedIndex`] : array containment (`tags @> 'rust'`).
//!
//! All three return [`roaring::RoaringTreemap`] of matching
//! [`VectorId`]s. The executor's `MetadataScan` operator composes
//! the bitmaps for `AND` (intersection), `OR` (union), and `NOT`
//! (difference from the live-id set).
//!
//! ## What this crate is NOT
//!
//! - **Not a query language.** No KQL types leak in. The trait talks
//!   about [`Value`] (from `kova-core`) and [`IndexAtom`] (a slim
//!   local enum), never `PredAtom`.
//! - **Not a planner.** Cost estimation lives in `kova-query` ; this
//!   crate exposes [`MetaIndex::cardinality`] so the estimator can
//!   compute fractions cheaply, but does not pick plans.
//! - **Not a vector index.** Anything involving distance lives in
//!   `kova-index`.
//! - **Not persistent in M2.1.** Indexes live in memory and rebuild
//!   from the metadata store on `Shard::open`. Sidecar persistence
//!   lands as M2.2.
//!
//! See `docs/.notes/metaidx.md` (gitignored) for the full design
//! ref.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod btree;
pub mod catalog;
pub mod error;
pub mod hash;
pub mod inverted;

use kova_core::{Value, VectorId};
use roaring::RoaringTreemap;

pub use btree::BTreeIndex;
pub use catalog::IndexCatalog;
pub use error::KovaMetaIndexError;
pub use hash::HashIndex;
pub use inverted::InvertedIndex;

/// Comparison operator for [`IndexAtom::Cmp`].
///
/// Equality and inequality have their own atom variants
/// ([`IndexAtom::Eq`], reachable via [`IndexAtom::Cmp`] with
/// [`CmpOp::Ne`]) so that hash-only indexes can answer them without
/// having to think about ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    /// `field < value`.
    Lt,
    /// `field <= value`.
    Le,
    /// `field > value`.
    Gt,
    /// `field >= value`.
    Ge,
    /// `field != value`.
    Ne,
}

/// One predicate atom, as seen by an index.
///
/// This is a slim view of `kova_query::logical::PredAtom`. The
/// distinction matters because indexes are reusable infrastructure
/// that must not depend on KQL ; the executor at the boundary
/// translates `PredAtom` to `IndexAtom` before calling
/// [`MetaIndex::query`].
#[derive(Debug, Clone)]
pub enum IndexAtom {
    /// `field = value`.
    Eq(Value),
    /// `field <op> value` for ops other than `=`.
    Cmp(CmpOp, Value),
    /// `field IN (v1, v2, ...)`.
    In(Vec<Value>),
    /// `field BETWEEN lo AND hi` (inclusive both sides).
    Between(Value, Value),
    /// `field IS NOT NULL`. The binder normalises `IS NULL` to
    /// `NOT IsNotNull` so downstream code only handles one shape.
    IsNotNull,
    /// `field @> value` : array containment.
    ArrayContains(Value),
}

/// Common surface every index type implements.
///
/// All methods take `&self` or `&mut self` ; there is no lifetime
/// parameter. The indexes hold their state by value, and the
/// returned [`RoaringTreemap`] is always a fresh clone of the matching
/// rows (or a freshly-computed union/intersection over buckets).
pub trait MetaIndex: Send + Sync {
    /// Bulk-build from a snapshot of current state. Used by
    /// `Shard::open` (re-derive from metadata on reopen) and by
    /// `CREATE INDEX` (build from current shard state).
    fn build<I>(&mut self, rows: I)
    where
        I: IntoIterator<Item = (VectorId, Value)>,
        Self: Sized;

    /// Single-row insert. Called from `Shard::insert` /
    /// `Shard::insert_many` phase 3, after the WAL commit.
    fn insert(&mut self, id: VectorId, value: &Value);

    /// Single-row removal. Called from `Shard::delete` /
    /// `delete_many` phase 3. Note that the metadata bag is dropped
    /// at delete time, so the index loses the row immediately ;
    /// HNSW's tombstone-then-vacuum dance doesn't apply here.
    fn delete(&mut self, id: VectorId, value: &Value);

    /// Single-row replace. Some impls can avoid two bucket lookups
    /// when the old and new values are equal-for-indexing-purposes
    /// (e.g. an `UPDATE` that touches a different field entirely).
    fn update(&mut self, id: VectorId, old: &Value, new: &Value);

    /// Query against one predicate atom. Returns matching ids.
    /// Atoms the index does not support return an empty bitmap and
    /// a `None` from [`Self::cardinality`] ; the caller falls back
    /// to a metadata scan in that case.
    fn query(&self, atom: &IndexAtom) -> RoaringTreemap;

    /// Estimated count of rows matching this atom. Used by the
    /// selectivity estimator. `None` means "I can answer the query
    /// but I can't count it cheaply without scanning, so go ask the
    /// data."
    fn cardinality(&self, atom: &IndexAtom) -> Option<u64>;

    /// How many rows have a non-null value in the indexed field.
    fn len(&self) -> u64;

    /// Convenience over [`Self::len`].
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// What kinds of atoms this index can answer.
    fn supports(&self, atom: &IndexAtom) -> bool;
}

/// Normalised hashable / orderable key derived from a [`Value`].
///
/// Three concerns motivate this type :
///
/// 1. [`Value`] cannot be a `HashMap` key because [`Value::F64`]
///    contains `f64`, which is not [`Eq`] (NaN).
/// 2. [`Value`] cannot be a `BTreeMap` key for the same reason
///    ([`Ord`] requires [`Eq`]).
/// 3. The non-scalar variants ([`Value::Array`], [`Value::Map`])
///    do not have a single canonical key shape for indexing, so
///    they are excluded.
///
/// `NormalizedKey` solves all three by representing floats as
/// [`u64`] (via [`f64::to_bits`]) and excluding non-scalar variants
/// (returning `None` from [`NormalizedKey::from_value`]).
///
/// **Float ordering caveat.** Lexicographic ordering of
/// `f64::to_bits` matches numeric `<` for positive finite floats
/// only. Mixed signs and NaN diverge. The v1 [`BTreeIndex`]
/// therefore rejects ranged queries on float fields ; see
/// [`BTreeIndex::supports`] for the gate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NormalizedKey {
    /// UTF-8 string key.
    String(String),
    /// Signed 64-bit integer.
    I64(i64),
    /// IEEE 754 double's bit pattern.
    F64Bits(u64),
    /// Boolean.
    Bool(bool),
}

impl NormalizedKey {
    /// Convert a [`Value`] to a [`NormalizedKey`]. Returns `None`
    /// for [`Value::Array`] and [`Value::Map`] : those variants are
    /// not indexable as keys (they may be _values_ that get
    /// decomposed into element-keys by [`InvertedIndex`], but they
    /// are not single keys themselves).
    #[must_use]
    pub fn from_value(v: &Value) -> Option<Self> {
        match v {
            Value::String(s) => Some(NormalizedKey::String(s.clone())),
            Value::I64(n) => Some(NormalizedKey::I64(*n)),
            Value::F64(f) => Some(NormalizedKey::F64Bits(f.to_bits())),
            Value::Bool(b) => Some(NormalizedKey::Bool(*b)),
            Value::Array(_) | Value::Map(_) => None,
        }
    }

    /// True if the key was derived from a float. Used by
    /// [`BTreeIndex::supports`] to gate ranged float queries.
    #[must_use]
    pub fn is_float(&self) -> bool {
        matches!(self, NormalizedKey::F64Bits(_))
    }
}

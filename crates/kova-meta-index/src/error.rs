//! Errors surfaced from the meta-index crate.
//!
//! Most index operations are infallible by construction (the
//! underlying `HashMap` / `BTreeMap` / `RoaringTreemap` operations
//! cannot fail except via OOM, which we treat as panic territory).
//! This error type exists for the few cases that can legitimately
//! fail : building from a mistyped row set, querying with an atom
//! shape the index does not support, etc.

use thiserror::Error;

/// Failure modes for [`crate::MetaIndex`] operations.
#[derive(Debug, Error)]
pub enum KovaMetaIndexError {
    /// An attempt to index a [`kova_core::Value`] variant the index
    /// cannot key on (e.g. inserting a `Value::Map` into a
    /// `HashIndex`).
    #[error("value type is not indexable by this index : {kind}")]
    NonIndexableValue {
        /// Static label for the offending variant (e.g. "Map").
        kind: &'static str,
    },

    /// An attempt to query with an atom shape the index does not
    /// support (e.g. `Cmp { op: Lt }` on a `HashIndex`).
    ///
    /// In normal operation the planner calls
    /// [`crate::MetaIndex::supports`] first and never reaches this
    /// error path. It exists so misuse fails loudly rather than
    /// silently returning the wrong answer.
    #[error("atom shape not supported by this index : {atom_kind}")]
    UnsupportedAtom {
        /// Static label for the offending atom variant.
        atom_kind: &'static str,
    },
}

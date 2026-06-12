//! Errors surfaced from the meta-index crate.
//!
//! Most index operations are infallible by construction (the
//! underlying `HashMap` / `BTreeMap` / `RoaringTreemap` operations
//! cannot fail except via OOM, which we treat as panic territory).
//! This error type exists for the few cases that can legitimately
//! fail : building from a mistyped row set, querying with an atom
//! shape the index does not support, etc.

use std::io;

use thiserror::Error;

/// Failure modes for [`crate::MetaIndex`] operations.
#[derive(Debug, Error)]
pub enum KovaMetaIndexError {
    /// I/O failure while reading or writing a catalog file.
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    /// Catalog file's magic header is missing or wrong.
    #[error("catalog magic mismatch (file may not be a kova catalog)")]
    BadMagic,

    /// Catalog file's format version is not supported by this build.
    #[error("unsupported catalog format version : {got} (expected {expected})")]
    UnsupportedVersion {
        /// Version this build expects.
        expected: u32,
        /// Version found on disk.
        got: u32,
    },

    /// Catalog file is shorter than the fixed header.
    #[error("catalog file truncated : {bytes} bytes, need at least {min}")]
    Truncated {
        /// Bytes present.
        bytes: usize,
        /// Minimum bytes required.
        min: usize,
    },

    /// Bincode decode error on the catalog payload.
    #[error("decode error: {0}")]
    Decode(#[from] bincode::Error),

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

    /// Attempted to create a named index whose name is already in
    /// the catalog's name registry. DDL only ; the programmatic
    /// `add_*_index(field)` API is anonymous and never hits this.
    #[error("index named '{name}' already exists")]
    IndexNameInUse {
        /// Name the DDL asked to create.
        name: String,
    },

    /// Attempted to drop a named index by a name that was never
    /// registered.
    #[error("no index named '{name}'")]
    UnknownIndexName {
        /// Name the DDL asked to drop.
        name: String,
    },
}

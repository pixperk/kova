//! Error type returned by `kova-core` operations.

use thiserror::Error;

/// Errors produced by `kova-core` types and functions.
///
/// Marked `#[non_exhaustive]` so that adding new variants in future versions
/// is not a breaking change for downstream crates.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum KovaError {
    /// Two vectors had different dimensions when an operation required them
    /// to match (for example, computing a distance).
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The dimension the operation expected (usually the first operand's).
        expected: usize,
        /// The dimension actually supplied.
        got: usize,
    },

    /// Attempted to construct a [`crate::vector::Vector`] from an empty slice.
    #[error("vector must have at least one dimension")]
    EmptyVector,

    /// A vector component was non-finite (NaN or infinity), which is not allowed.
    #[error("vector component at index {index} is non-finite: {value}")]
    NonFinite {
        /// The index of the component that was non-finite.
        index: usize,
        /// The non-finite value that was found.
        value: f32,
    },
}

//! Kova Index error types.

use kova_core::VectorId;
use thiserror::Error;

/// Errors produced by `kova-index` types and functions.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum KovaIndexError {
    /// Inserted/queried vector has different dim than the index's pinned dim.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The dimension the index expects (usually the first inserted vector's).
        expected: usize,
        /// The dimension actually supplied.
        got: usize,
    },

    /// Tried to insert a `VectorId` that's already in the index.
    #[error("duplicate vector ID: {id}")]
    DuplicateId {
        /// The `VectorId` that was attempted to be inserted.
        id: VectorId,
    },

    /// Tried to delete / tombstone a `VectorId` that is not in the index.
    #[error("vector ID not found: {id}")]
    NotFound {
        /// The `VectorId` that was looked up.
        id: VectorId,
    },

    /// Tried to delete a `VectorId` that is already tombstoned.
    #[error("vector ID already deleted: {id}")]
    AlreadyDeleted {
        /// The `VectorId` that was already deleted.
        id: VectorId,
    },

    /// Underlying validation from `kova-core`.
    #[error(transparent)]
    Core(#[from] kova_core::KovaError),

    /// Error from the underlying [`kova_core::VectorStore`] impl.
    /// Carries a stringified `Debug` so we don't have to be generic over
    /// the store's concrete error type here.
    #[error("vector store error: {0}")]
    Storage(String),
}

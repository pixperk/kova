//! Error type for the `kova-query` crate.
//!
//! One concrete enum, because `kova-query` has a single implementation
//! end-to-end (no pluggable parser or planner backends). Errors from
//! the storage layer that bubble up through the executor are boxed
//! inside the [`KovaQueryError::Backend`] variant ; the chain is
//! preserved via `source()` for callers that want to downcast.

use thiserror::Error;

/// Top-level error type for KQL operations.
#[derive(Debug, Error)]
pub enum KovaQueryError {
    /// Parser failed to recognise the input as a valid KQL statement.
    /// Message carries line/column from Pest.
    #[error("parse error: {0}")]
    Parse(String),

    /// Binder rejected a syntactically-valid statement. Examples :
    /// unknown field, type mismatch, embedding update attempted.
    #[error("bind error: {0}")]
    Bind(String),

    /// Planner could not produce a valid physical plan for the given
    /// logical statement.
    #[error("plan error: {0}")]
    Plan(String),

    /// Executor failed at runtime for a reason that originates in the
    /// executor itself (parameter binding mismatch, unbound named
    /// parameter, etc). Storage-layer failures go through
    /// [`KovaQueryError::Backend`].
    #[error("execution error: {0}")]
    Execution(String),

    /// Storage layer error from a `Shard` operation invoked by the
    /// executor.
    #[error("storage backend error")]
    Backend(#[source] Box<dyn std::error::Error + Send + Sync>),
}

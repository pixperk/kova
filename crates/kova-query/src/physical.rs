//! Physical plan : the executor's IR.
//!
//! Operator tree the executor walks against a `Shard`. Where the
//! [`crate::logical::LogicalStatement`] captures *what to compute*,
//! the physical plan captures *how* : which operator runs, in what
//! order, with which parameter slots resolved at execute time.
//!
//! Operators land incrementally. CHECKPOINT is first ; INSERT /
//! DELETE / VACUUM / SELECT follow.

use kova_core::VectorId;

use crate::ast::ParamRef;

/// Physical operator. v1 grows this enum as each statement gets its
/// executor support. Explicit variants (no catchall) so the executor's
/// dispatch is exhaustive and the compiler complains when an arm goes
/// missing.
#[derive(Debug, Clone)]
pub enum PhysicalPlan {
    /// Stop-the-world checkpoint : vacuum + WAL fsync + snapshot
    /// write + manifest commit + WAL truncate. Dispatches directly
    /// to `Shard::checkpoint`. Returns the committed LSN.
    Checkpoint,
    /// Tombstone reclaim + HNSW graph repair on `table`. Dispatches
    /// to `Shard::vacuum`. Returns the count of nodes physically
    /// removed.
    Vacuum {
        /// Target table name (validated against the engine's shard).
        table: String,
    },
    /// Single-row INSERT. Three parameter slots ; the executor
    /// resolves each one against the caller's [`crate::executor::ParamBindings`]
    /// and dispatches to `Shard::insert`.
    InsertOne {
        /// Target table.
        table: String,
        /// Parameter slot for the row's `id`.
        id: ParamRef,
        /// Parameter slot for the row's `embedding` vector.
        embedding: ParamRef,
        /// Parameter slot for the row's `metadata` bag.
        metadata: ParamRef,
    },
    /// Batch INSERT. One parameter slot bound to an array of
    /// `(id, embedding, metadata)` tuples ; dispatches to
    /// `Shard::insert_many` for a single WAL group-commit fsync.
    InsertMany {
        /// Target table.
        table: String,
        /// Parameter slot for the batch array.
        batch: ParamRef,
    },
    /// DELETE by literal id : the fast path. Produced when the
    /// binder's single-id hint was set (predicate of the form
    /// `WHERE id = <integer-literal>`). Dispatches to `Shard::delete`.
    DeleteById {
        /// Target table.
        table: String,
        /// Pre-resolved id (no parameter lookup needed at execute time).
        id: VectorId,
    },
}

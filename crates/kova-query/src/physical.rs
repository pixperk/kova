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

use crate::ast::{DistanceOp, ParamRef};
use crate::logical::{PredicateExpr, ProjectionSpec};

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
    /// kNN search : the read path's entry point. Calls
    /// `Shard::search` for `k` candidates, then applies `post_filter`
    /// (if any) to drop rows that fail the predicate. Returns a
    /// stream of [`kova_storage::SearchHit`] up the operator tree.
    ///
    /// The planner already inflated `k` to `user_limit * overfetch`
    /// so the post-filter has room to drop some candidates without
    /// starving the final LIMIT.
    KnnSearch {
        /// Target table.
        table: String,
        /// Parameter slot for the query vector.
        query: ParamRef,
        /// Distance metric requested (planner records ; executor uses
        /// the shard's native metric in v1).
        metric: DistanceOp,
        /// kNN result count (already overfetched).
        k: usize,
        /// Optional predicate applied after the kNN returns. Drops
        /// rows whose metadata fails the predicate.
        post_filter: Option<PredicateExpr>,
    },
    /// Truncate the input row stream to at most `limit` rows.
    Limit {
        /// Sub-plan producing the input rows.
        input: Box<PhysicalPlan>,
        /// Maximum rows to emit.
        limit: u64,
    },
    /// Shape the output rows according to a projection spec. Always
    /// the outermost read operator ; converts internal hits into
    /// user-facing [`crate::executor::Row`] values.
    Projection {
        /// Sub-plan producing the input rows.
        input: Box<PhysicalPlan>,
        /// Projection list with wildcards already expanded.
        spec: ProjectionSpec,
    },
}

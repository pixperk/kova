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
use crate::logical::{LogicalAssignment, PredicateExpr, ProjectionSpec};

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
    /// UPDATE by literal id. Produced when the binder spotted
    /// `WHERE id = <integer-literal>` ; the executor fetches the
    /// current metadata, applies each assignment to a copy, and
    /// writes the new bag via `Shard::update_metadata`.
    UpdateById {
        /// Target table.
        table: String,
        /// Pre-resolved id (no parameter lookup needed at execute time).
        id: VectorId,
        /// One or more assignments, in source order.
        assignments: Vec<LogicalAssignment>,
    },
    /// UPDATE by an id sourced from a parameter slot. Mirror of
    /// `UpdateById` for `WHERE id = $param`.
    UpdateByParamId {
        /// Target table.
        table: String,
        /// Parameter slot carrying the id.
        id_param: ParamRef,
        /// One or more assignments, in source order.
        assignments: Vec<LogicalAssignment>,
    },
    /// UPDATE by predicate : scan metadata for ids whose bag passes
    /// `predicate`, then apply `assignments` to each matched bag in
    /// one WAL group-commit. Symmetric with [`Self::DeleteByPredicate`].
    UpdateByPredicate {
        /// Target table.
        table: String,
        /// Predicate evaluated against each row's metadata.
        predicate: PredicateExpr,
        /// Assignments applied to every matched row.
        assignments: Vec<LogicalAssignment>,
    },
    /// UPDATE by radius : every live id within `radius` of `query`
    /// gets its metadata bag rewritten with `assignments`. Same
    /// `post_filter` semantics as [`Self::DeleteByRadius`].
    UpdateByRadius {
        /// Target table.
        table: String,
        /// Parameter slot for the query vector.
        query: ParamRef,
        /// Distance metric requested.
        metric: DistanceOp,
        /// Distance bound.
        radius: f32,
        /// Strict (`<`) or inclusive (`<=`) boundary.
        inclusive: bool,
        /// Optional non-distance predicate residue from the WHERE.
        post_filter: Option<PredicateExpr>,
        /// Assignments applied to every matched row.
        assignments: Vec<LogicalAssignment>,
    },
    /// DELETE by an id sourced from a parameter slot. Mirror of
    /// `DeleteById` for `WHERE id = $param` — the executor resolves
    /// the param at run time, then dispatches to `Shard::delete`.
    DeleteByParamId {
        /// Target table.
        table: String,
        /// Parameter slot carrying the id.
        id_param: ParamRef,
    },
    /// DELETE by radius : every live id within `radius` of `query`
    /// is tombstoned. The executor uses `Shard::search_radius` to
    /// produce the id set, applies any AND-residue against each
    /// hit's metadata, then dispatches to `Shard::delete_many` for
    /// one WAL group-commit.
    DeleteByRadius {
        /// Target table.
        table: String,
        /// Parameter slot for the query vector.
        query: ParamRef,
        /// Distance metric requested.
        metric: DistanceOp,
        /// Distance bound.
        radius: f32,
        /// Whether the comparison was strict (`<`) or inclusive (`<=`).
        inclusive: bool,
        /// Optional non-distance predicate residue from the WHERE.
        post_filter: Option<PredicateExpr>,
    },
    /// DELETE by predicate : scan metadata for matching ids, then
    /// batch-tombstone the lot in a single WAL group-commit. Produced
    /// when the binder couldn't extract a single-id hint (predicate
    /// is param-bound, compound, or doesn't match the trivial form).
    DeleteByPredicate {
        /// Target table.
        table: String,
        /// Predicate evaluated against each row's metadata.
        predicate: PredicateExpr,
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
    /// Filtered kNN search : plan C's entry point. Threads `filter`
    /// into the HNSW walk so out-of-filter nodes never enter the
    /// results heap. `k` is the user's LIMIT directly , no overfetch
    /// needed, because filtering happens during the walk rather than
    /// after.
    ///
    /// Wins over plan A when selectivity is mid-range : low enough
    /// that plan A's overfetch would starve, high enough that plan B's
    /// metadata scan would be wasteful.
    FilteredKnnSearch {
        /// Target table.
        table: String,
        /// Parameter slot for the query vector.
        query: ParamRef,
        /// Distance metric requested.
        metric: DistanceOp,
        /// Top-k cap (user's LIMIT, not overfetched).
        k: usize,
        /// Predicate evaluated against each visited node's metadata.
        filter: PredicateExpr,
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
    /// Scan the metadata store for live ids whose bag passes
    /// `predicate`. Plan B's entry point : the "pre-filter" that
    /// produces a candidate id set the downstream `ExactDistance`
    /// can score.
    ///
    /// v1 uses `Shard::scan_metadata`, an O(N) walk. v2 dispatches
    /// to a secondary index when one exists for the predicate.
    MetadataScan {
        /// Target table.
        table: String,
        /// Predicate evaluated against each row's metadata.
        predicate: PredicateExpr,
    },
    /// Compute exact distance from `query` to each input id under
    /// the shard's metric, sort ascending, take top `k`. Plan B's
    /// "rank by distance" step ; replaces plan A's kNN-with-overfetch.
    ///
    /// Wins over plan A when the input candidate set is small enough
    /// that computing exact distances costs less than running the kNN.
    /// Selectivity-driven planner picks between them in v2.
    ExactDistance {
        /// Sub-plan producing the candidate ids (typically `MetadataScan`).
        input: Box<PhysicalPlan>,
        /// Parameter slot for the query vector.
        query: ParamRef,
        /// Distance metric. Recorded ; v1 uses the shard's native metric.
        metric: DistanceOp,
        /// Top-k cap after exact-distance scoring.
        k: usize,
    },
    /// Radius search : `WHERE embedding <-> $q < $r`. Returns every
    /// live row whose distance to `query` is below `radius`, ascending
    /// by distance. No top-k cap : result size is whatever falls inside
    /// the ball.
    ///
    /// `post_filter` carries any non-distance atoms peeled off the
    /// WHERE clause by the planner (e.g. `embedding <-> $q < $r AND
    /// category = 'a'` keeps `category = 'a'` here). Applied after the
    /// radius walk against each hit's metadata.
    RadiusSearch {
        /// Target table.
        table: String,
        /// Parameter slot for the query vector.
        query: ParamRef,
        /// Distance metric requested. Recorded ; executor uses the
        /// shard's native metric in v1.
        metric: DistanceOp,
        /// Distance bound. Baked in at bind time : the grammar requires
        /// a literal here (param-bound radii are rejected at the
        /// binder), so the planner doesn't need a [`ParamRef`] slot.
        radius: f32,
        /// Whether the comparison was strict (`<`) or inclusive (`<=`).
        /// The executor uses this to decide whether boundary hits keep
        /// the hit or drop it.
        inclusive: bool,
        /// Optional non-distance predicate residue from the WHERE.
        post_filter: Option<PredicateExpr>,
    },
    /// `SELECT COUNT(*) FROM <table> [WHERE pred]`. The only aggregate
    /// v1 supports. Bypasses the kNN-only check because there's no
    /// ordering to perform : just a single scalar result. Returns one
    /// row with one column.
    Count {
        /// Target table.
        table: String,
        /// Optional WHERE predicate. `None` means count all live rows.
        predicate: Option<PredicateExpr>,
        /// Output column name (alias or `"count"`).
        column_name: String,
    },
}

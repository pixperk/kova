//! [`LogicalStatement`] -> [`PhysicalPlan`].
//!
//! For write-side and management statements the planning is one-to-one
//! (one logical statement, one operator). For SELECT the planner
//! makes a real decision : plan A (overfetched kNN + post-filter) vs
//! plan B (predicate scan + exact distance). The choice is driven by
//! selectivity : if few rows match the predicate, plan B's bounded
//! work beats plan A's overfetch.

use kova_core::VectorId;

use crate::error::KovaQueryError;
use crate::executor::ParamBindings;
use crate::logical::{
    BoundProjection, LogicalDelete, LogicalInsert, LogicalInsertSource, LogicalQuery,
    LogicalStatement, LogicalVacuum, OrderingSpec, PredicateExpr, ProjectionSpec,
};
use crate::physical::PhysicalPlan;

/// kNN overfetch multiplier. The planner asks the kNN for `k_user *
/// OVERFETCH` candidates so the post-filter has room to drop some
/// without starving the final LIMIT. v2 tunes this from selectivity.
const KNN_OVERFETCH: usize = 4;

/// Selectivity threshold for plan A vs plan B. If matches/total is
/// below this fraction, plan B (scan + exact distance) wins ; above,
/// plan A (overfetched kNN + post-filter) wins. v1 hardcodes 0.5 ;
/// v2 derives it from measured cost coefficients.
const PLAN_B_SELECTIVITY_THRESHOLD: f64 = 0.5;

/// Estimate produced by [`SelectivityEstimator::estimate`].
#[derive(Debug, Clone, Copy)]
pub struct SelectivityEstimate {
    /// Rows whose metadata satisfies the predicate.
    pub matches: usize,
    /// Total live rows in the shard.
    pub total: usize,
}

impl SelectivityEstimate {
    /// Selectivity as a fraction in `[0.0, 1.0]`. Returns `1.0` for
    /// an empty shard (no rows to filter ; any plan is equivalent).
    #[must_use]
    pub fn fraction(self) -> f64 {
        if self.total == 0 {
            1.0
        } else {
            // usize -> f64 is precision-lossy for huge counts. For
            // shards under ~2^53 rows the cast is exact ; v2 may
            // switch to integer-math thresholds.
            #[allow(clippy::cast_precision_loss)]
            let m = self.matches as f64;
            #[allow(clippy::cast_precision_loss)]
            let t = self.total as f64;
            m / t
        }
    }
}

/// Estimates how many rows a predicate matches against a shard.
///
/// v1 impl runs the predicate against every row (cheap because
/// metadata is in-memory) and returns an exact count. v2 swaps in
/// index cardinality lookups for `O(log N)` per atom.
pub trait SelectivityEstimator {
    /// Estimate selectivity of `pred`. `params` is passed through so
    /// the estimator can resolve param-bound atoms (e.g.
    /// `WHERE category = $1`).
    fn estimate(&self, pred: &PredicateExpr, params: &ParamBindings) -> SelectivityEstimate;
}

/// Trivial estimator that always reports zero selectivity. Useful
/// for tests that don't care about the plan choice, and for the
/// bare `plan()` entry point (kept for backwards-compat with code
/// that doesn't have a shard handle).
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopEstimator;

impl SelectivityEstimator for NoopEstimator {
    fn estimate(&self, _: &PredicateExpr, _: &ParamBindings) -> SelectivityEstimate {
        // Zero selectivity reports "very few matches" : the planner
        // always picks plan B with this estimator, matching the
        // previous v1 stopgap behaviour.
        SelectivityEstimate {
            matches: 0,
            total: 1,
        }
    }
}

/// Pick the physical plan for a [`LogicalStatement`].
///
/// Convenience wrapper around [`plan_with_estimator`] that uses the
/// [`NoopEstimator`]. For SELECT queries with predicates, this always
/// picks plan B (matches the previous v1 stopgap behaviour). For
/// real cost-driven dispatch, use [`plan_with_estimator`] with a
/// `ShardEstimator` that has access to the shard.
///
/// # Errors
///
/// Returns [`KovaQueryError::Plan`] for any statement the planner
/// doesn't yet know how to handle.
//
// By-value : real arms move fields out of LogicalStatement payloads
// when they land (same shape as the binder dispatch).
#[allow(clippy::needless_pass_by_value)]
pub fn plan(stmt: LogicalStatement) -> Result<PhysicalPlan, KovaQueryError> {
    plan_with_estimator(stmt, &NoopEstimator, &ParamBindings::empty())
}

/// Pick the physical plan for a [`LogicalStatement`], driving SELECT
/// plan-choice with a [`SelectivityEstimator`].
///
/// # Errors
///
/// Returns [`KovaQueryError::Plan`] for any statement the planner
/// doesn't yet know how to handle.
//
// By-value on stmt : real arms move fields out of the payload.
#[allow(clippy::needless_pass_by_value)]
pub fn plan_with_estimator<E: SelectivityEstimator>(
    stmt: LogicalStatement,
    estimator: &E,
    params: &ParamBindings,
) -> Result<PhysicalPlan, KovaQueryError> {
    match stmt {
        LogicalStatement::Checkpoint => Ok(PhysicalPlan::Checkpoint),
        LogicalStatement::Vacuum(LogicalVacuum { table }) => Ok(PhysicalPlan::Vacuum { table }),
        LogicalStatement::Insert(LogicalInsert { table, rows }) => match rows {
            LogicalInsertSource::Single {
                id,
                embedding,
                metadata,
            } => Ok(PhysicalPlan::InsertOne {
                table,
                id,
                embedding,
                metadata,
            }),
            LogicalInsertSource::Batch { param } => Ok(PhysicalPlan::InsertMany {
                table,
                batch: param,
            }),
        },
        LogicalStatement::Delete(LogicalDelete {
            table,
            single_id_hint,
            predicate: _,
        }) => match single_id_hint {
            // Hint set : binder spotted `WHERE id = <integer-literal>`.
            // Skip straight to the fast path ; no predicate evaluation
            // needed.
            Some(id) => Ok(PhysicalPlan::DeleteById {
                table,
                id: VectorId::new(id),
            }),
            // Hint missing : predicate is param-bound, compound, or
            // doesn't match the simple-id shape. Full DELETE-by-predicate
            // is its own milestone (needs metadata scan + delete_many).
            None => Err(KovaQueryError::Plan(
                "DELETE WHERE <predicate> is not yet supported ; v1 supports \
                 DELETE WHERE id = <integer-literal> only"
                    .into(),
            )),
        },

        LogicalStatement::Query(q) => plan_query(q, estimator, params),

        // Filled in as each statement gains executor support. Explicit
        // arms (rather than `_`) so the compiler errors the moment a
        // new LogicalStatement variant is added without a planner arm.
        LogicalStatement::Update(_) => unimplemented("UPDATE"),
    }
}

/// Build the physical plan for a SELECT statement.
///
/// v1 (plan A) shape :
///
/// 1. Must be a kNN query : `ORDER BY embedding <op> $q` is the
///    first ordering key. Non-kNN selects (scan + filter without
///    distance ordering) are plan B / C territory and error here
///    until M1.5 lands.
/// 2. Must have a LIMIT (binder already enforces).
/// 3. Wildcard projection expands to `[id, metadata]` here, so
///    downstream operators never see `BoundProjection::Wildcard`.
/// 4. The predicate becomes the kNN's `post_filter`, applied to
///    candidates after the kNN returns ; the kNN itself runs
///    unfiltered with `k = LIMIT * OVERFETCH` so the post-filter
///    has room to drop without starving the final result.
fn plan_query<E: SelectivityEstimator>(
    q: LogicalQuery,
    estimator: &E,
    params: &ParamBindings,
) -> Result<PhysicalPlan, KovaQueryError> {
    let LogicalQuery {
        from_table,
        projection,
        predicate,
        ordering,
        limit,
    } = q;

    // Step 1 : check the shape. v1 needs a distance ordering as the
    // first (and only, for now) ordering key. Field ordering and
    // missing ordering are plan B territory.
    let knn = ordering
        .into_iter()
        .find_map(|o| match o {
            OrderingSpec::Distance { metric, param } => Some((metric, param)),
            OrderingSpec::Field { .. } => None,
        })
        .ok_or_else(|| {
            KovaQueryError::Plan(
                "v1 supports only kNN SELECTs : need `ORDER BY embedding <op> $q LIMIT k`. \
                 Scan-and-filter (without distance ordering) lands when plan B ships."
                    .into(),
            )
        })?;

    // Step 2 : LIMIT must be present. The binder enforces this for
    // kNN queries, but planning has its own check so the Plan-error
    // message is clear if someone bypasses the binder.
    let user_limit = limit.ok_or_else(|| {
        KovaQueryError::Plan("kNN SELECT requires LIMIT (binder should have caught this)".into())
    })?;

    // Step 3 : expand wildcards. v1 maps `*` to `[Id, Metadata]`.
    let projection = expand_wildcard(projection);

    let (metric, query_param) = knn;
    let user_k = usize::try_from(user_limit).unwrap_or(usize::MAX);

    // Step 4 : pick plan A or plan B based on selectivity.
    //
    //   selectivity < threshold  -> plan B (few matches, scan + exact
    //                                       distance ; bounded work)
    //   selectivity >= threshold -> plan A (most pass, kNN overfetch
    //                                       still saturates LIMIT)
    //   no predicate             -> plan A (nothing to filter)
    //
    // v1 uses `PLAN_B_SELECTIVITY_THRESHOLD = 0.5` as a flat cutoff.
    // v2 (M2.6) replaces this with a measured cost model that
    // accounts for k, recall target, and shard size.
    let plan = match predicate {
        Some(pred) => {
            let est = estimator.estimate(&pred, params);
            if est.fraction() < PLAN_B_SELECTIVITY_THRESHOLD {
                build_plan_b(from_table, pred, query_param, metric, user_k, user_limit)
            } else {
                build_plan_a(
                    from_table,
                    query_param,
                    metric,
                    user_k,
                    Some(pred),
                    user_limit,
                )
            }
        }
        None => build_plan_a(from_table, query_param, metric, user_k, None, user_limit),
    };

    Ok(PhysicalPlan::Projection {
        input: Box::new(plan),
        spec: projection,
    })
}

/// Plan A : `Limit(KnnSearch(overfetched, optional post_filter))`.
fn build_plan_a(
    table: String,
    query: crate::ast::ParamRef,
    metric: crate::ast::DistanceOp,
    user_k: usize,
    post_filter: Option<PredicateExpr>,
    user_limit: u64,
) -> PhysicalPlan {
    let overfetched_k = user_k.saturating_mul(KNN_OVERFETCH);
    let knn = PhysicalPlan::KnnSearch {
        table,
        query,
        metric,
        k: overfetched_k,
        post_filter,
    };
    PhysicalPlan::Limit {
        input: Box::new(knn),
        limit: user_limit,
    }
}

/// Plan B : `Limit(ExactDistance(MetadataScan(predicate)))`.
fn build_plan_b(
    table: String,
    predicate: PredicateExpr,
    query: crate::ast::ParamRef,
    metric: crate::ast::DistanceOp,
    user_k: usize,
    user_limit: u64,
) -> PhysicalPlan {
    let scan = PhysicalPlan::MetadataScan { table, predicate };
    let exact = PhysicalPlan::ExactDistance {
        input: Box::new(scan),
        query,
        metric,
        k: user_k,
    };
    PhysicalPlan::Limit {
        input: Box::new(exact),
        limit: user_limit,
    }
}

/// If the projection is just `[Wildcard]`, expand to the canonical
/// v1 column set : `[id, metadata]`. Otherwise leave it alone.
/// The binder already rejected "wildcard alongside other items," so
/// `[Wildcard]` is the only case we have to handle here.
fn expand_wildcard(spec: ProjectionSpec) -> ProjectionSpec {
    let is_solo_wildcard =
        spec.columns.len() == 1 && matches!(spec.columns.first(), Some(BoundProjection::Wildcard));
    if is_solo_wildcard {
        ProjectionSpec {
            columns: vec![
                BoundProjection::Id { alias: None },
                BoundProjection::Metadata { alias: None },
            ],
        }
    } else {
        spec
    }
}

fn unimplemented(name: &str) -> Result<PhysicalPlan, KovaQueryError> {
    Err(KovaQueryError::Plan(format!(
        "planner not yet implemented for {name}"
    )))
}

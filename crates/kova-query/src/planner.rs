//! [`LogicalStatement`] -> [`PhysicalPlan`].
//!
//! For write-side and management statements the planning is one-to-one
//! (one logical statement, one operator). For SELECT the planner
//! makes a real decision : plan A (overfetched kNN + post-filter) vs
//! plan B (predicate scan + exact distance). The choice is driven by
//! selectivity : if few rows match the predicate, plan B's bounded
//! work beats plan A's overfetch.

use kova_core::VectorId;

use crate::ast::{CmpOp, DistanceOp, ParamRef};
use crate::error::KovaQueryError;
use crate::executor::ParamBindings;
use crate::logical::{
    BoundProjection, LogicalDelete, LogicalInsert, LogicalInsertSource, LogicalQuery,
    LogicalStatement, LogicalVacuum, OrderingSpec, PredAtom, PredicateExpr, ProjectionSpec,
};
use crate::physical::PhysicalPlan;

/// kNN overfetch multiplier. The planner asks the kNN for `k_user *
/// OVERFETCH` candidates so the post-filter has room to drop some
/// without starving the final LIMIT. v2 tunes this from selectivity.
const KNN_OVERFETCH: usize = 4;

/// Lower selectivity boundary : below this fraction we pick plan B
/// (scan metadata + exact distance). The candidate set is small enough
/// that an O(matches * d) exact distance loop beats running the full
/// ANN walk.
const PLAN_B_UPPER: f64 = 0.05;

/// Upper selectivity boundary : at or above this fraction we pick
/// plan A (overfetched kNN + post-filter). The predicate is loose
/// enough that the post-filter rarely drops candidates, so the
/// cheaper overfetch wins over plan C's per-visit predicate eval.
///
/// In between (`PLAN_B_UPPER`, `PLAN_A_LOWER`) we pick plan C :
/// filter threaded into the graph walk.
const PLAN_A_LOWER: f64 = 0.5;

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
            predicate,
        }) => {
            // Hint set : binder spotted `WHERE id = <integer-literal>`.
            // Skip straight to the fast path ; no predicate evaluation
            // needed.
            if let Some(id) = single_id_hint {
                return Ok(PhysicalPlan::DeleteById {
                    table,
                    id: VectorId::new(id),
                });
            }
            // Hint missing : route to the predicate-driven path. The
            // executor scans metadata for matching ids and feeds them
            // to `Shard::delete_many` in one batch. Distance-threshold
            // predicates are rejected upfront because the metadata
            // evaluator can't score distances.
            if predicate_has_distance_threshold(&predicate) {
                return Err(KovaQueryError::Plan(
                    "DELETE WHERE <distance-threshold> isn't supported ; \
                     distance predicates need the radius operator, \
                     not the metadata scan"
                        .into(),
                ));
            }
            Ok(PhysicalPlan::DeleteByPredicate { table, predicate })
        }

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

    // Step 0 : COUNT(*) bypass. A solo `COUNT(*)` projection short-
    // circuits the rest of plan_query : there's no ordering, no kNN,
    // no projection rows to build , just a scalar count of matching
    // rows. Treat it independently from kNN SELECTs.
    if let Some(column_name) = solo_count_star_name(&projection) {
        // COUNT(*) ignores ordering and LIMIT (one row is all you get).
        return Ok(PhysicalPlan::Count {
            table: from_table,
            predicate,
            column_name,
        });
    }

    // Step 0.5 : radius bypass. A SELECT with no ORDER BY and a WHERE
    // that contains a `DistanceThreshold` atom with `<` or `<=`
    // becomes a RadiusSearch. Mixed AND-residue stays on as a
    // post_filter. OR-with-DistanceThreshold and `>`/`>=` radii are
    // rejected explicitly (defer to the union / inverse-ball
    // milestones).
    //
    // LIMIT is allowed and wraps the radius operator : `WHERE dist <
    // r LIMIT 10` returns up to 10 in-ball hits.
    //
    // Skipped when ordering is present : a query that says both
    // "within radius r" *and* "order by distance, take k" is really
    // a kNN with a distance post-filter, so we fall through into
    // the kNN-shape path and let `pred` become a post_filter on plan A.
    if ordering.is_empty()
        && let Some(p) = &predicate
    {
        reject_or_with_distance_threshold(p)?;
        if let Some(extracted) = extract_radius_atom(p) {
            let projection = expand_wildcard(projection);
            let radius_plan = PhysicalPlan::RadiusSearch {
                table: from_table,
                query: extracted.query_param,
                metric: extracted.metric,
                radius: extracted.radius,
                inclusive: extracted.inclusive,
                post_filter: extracted.residue,
            };
            let capped = match limit {
                Some(user_limit) => PhysicalPlan::Limit {
                    input: Box::new(radius_plan),
                    limit: user_limit,
                },
                None => radius_plan,
            };
            return Ok(PhysicalPlan::Projection {
                input: Box::new(capped),
                spec: projection,
            });
        }
    }

    // Step 0.75 : scan-and-limit bypass. See `build_scan_and_limit`.
    if ordering.is_empty()
        && let Some(user_limit) = limit
    {
        return build_scan_and_limit(from_table, predicate, projection, user_limit);
    }

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

    let plan = dispatch_knn_plan(
        KnnPlanInputs {
            from_table,
            predicate,
            query_param,
            metric,
            user_k,
            user_limit,
        },
        estimator,
        params,
    );

    Ok(PhysicalPlan::Projection {
        input: Box::new(plan),
        spec: projection,
    })
}

/// Inputs to the kNN plan dispatcher. Bundled so the arity stays
/// manageable when a cost model grows extra knobs.
struct KnnPlanInputs {
    from_table: String,
    predicate: Option<PredicateExpr>,
    query_param: ParamRef,
    metric: DistanceOp,
    user_k: usize,
    user_limit: u64,
}

/// Pick plan A / B / C for a kNN SELECT based on selectivity.
///
/// Bands :
///
/// - `< PLAN_B_UPPER` : plan B (tight predicate ; tiny candidate set,
///   exact distance).
/// - `[PLAN_B_UPPER, PLAN_A_LOWER)` : plan C (mid predicate ; filter
///   threads into the ANN walk).
/// - `>= PLAN_A_LOWER` : plan A (loose predicate ; overfetched kNN
///   plus cheap post-filter).
/// - No predicate : plan A (nothing to filter).
fn dispatch_knn_plan<E: SelectivityEstimator>(
    inputs: KnnPlanInputs,
    estimator: &E,
    params: &ParamBindings,
) -> PhysicalPlan {
    let KnnPlanInputs {
        from_table,
        predicate,
        query_param,
        metric,
        user_k,
        user_limit,
    } = inputs;
    match predicate {
        Some(pred) => {
            let fraction = estimator.estimate(&pred, params).fraction();
            if fraction < PLAN_B_UPPER {
                build_plan_b(from_table, pred, query_param, metric, user_k, user_limit)
            } else if fraction < PLAN_A_LOWER {
                build_plan_c(from_table, pred, query_param, metric, user_k, user_limit)
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
    }
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

/// Build the scan-and-limit plan for `SELECT ... WHERE pred LIMIT k`
/// with no ORDER BY. Order of returned rows is implementation-defined
/// and not stable across releases. Requires a predicate ; an
/// unbounded "give me N arbitrary rows" without WHERE is rejected
/// to avoid the foot-gun of returning random slices of large shards.
/// Predicate shapes that hide a `DistanceThreshold` (under NOT,
/// nested OR, etc.) are also rejected so we don't route them to
/// `MetadataScan` whose evaluator errors at runtime.
fn build_scan_and_limit(
    from_table: String,
    predicate: Option<PredicateExpr>,
    projection: ProjectionSpec,
    user_limit: u64,
) -> Result<PhysicalPlan, KovaQueryError> {
    let Some(pred) = predicate else {
        return Err(KovaQueryError::Plan(
            "LIMIT without ORDER BY requires a WHERE clause ; \
             unbounded slice-scans aren't supported"
                .into(),
        ));
    };
    if predicate_has_distance_threshold(&pred) {
        return Err(KovaQueryError::Plan(
            "distance-threshold predicate in a shape the radius \
             planner doesn't recognise (e.g. NOT, nested OR) ; only \
             `<distance> < r` / `<distance> <= r` at the top of the \
             WHERE is supported"
                .into(),
        ));
    }
    let projection = expand_wildcard(projection);
    let scan = PhysicalPlan::MetadataScan {
        table: from_table,
        predicate: pred,
    };
    let limited = PhysicalPlan::Limit {
        input: Box::new(scan),
        limit: user_limit,
    };
    Ok(PhysicalPlan::Projection {
        input: Box::new(limited),
        spec: projection,
    })
}

/// Plan C : `Limit(FilteredKnnSearch(filter))`. ANN walk with the
/// predicate threaded into the traversal. No overfetch (k is the
/// user's LIMIT) because filtering happens during the walk.
fn build_plan_c(
    table: String,
    filter: PredicateExpr,
    query: ParamRef,
    metric: DistanceOp,
    user_k: usize,
    user_limit: u64,
) -> PhysicalPlan {
    let knn = PhysicalPlan::FilteredKnnSearch {
        table,
        query,
        metric,
        k: user_k,
        filter,
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

/// If the projection is exactly `[CountStar { alias }]`, return the
/// output column name (`alias` or `"count"`). Otherwise return None,
/// signalling that this isn't a COUNT-only query.
///
/// `SELECT COUNT(*), id FROM ...` returns None : COUNT(*) mixed with
/// other columns would require GROUP BY semantics v1 doesn't ship.
fn solo_count_star_name(spec: &ProjectionSpec) -> Option<String> {
    if spec.columns.len() != 1 {
        return None;
    }
    match spec.columns.first()? {
        BoundProjection::CountStar { alias } => {
            Some(alias.clone().unwrap_or_else(|| "count".into()))
        }
        _ => None,
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

/// Payload returned by [`extract_radius_atom`].
struct ExtractedRadius {
    query_param: ParamRef,
    metric: DistanceOp,
    radius: f32,
    inclusive: bool,
    residue: Option<PredicateExpr>,
}

/// Pull a top-level `DistanceThreshold` atom out of `pred` if one
/// exists, returning its parts plus whatever predicate is left after
/// removing it. Recognised shapes :
///
/// - Bare atom : `DistanceThreshold` → consumed, residue is `None`
/// - `And([..., DistanceThreshold, ...])` → atom consumed, residue is
///   the rest (single child unwrapped, multiple children re-wrapped
///   in `And`)
///
/// Returns `None` for any other shape (no distance threshold, threshold
/// is `>`/`>=`, threshold buried under `Or` or `Not`, etc.). Callers
/// fall through to the kNN-shape check.
fn extract_radius_atom(pred: &PredicateExpr) -> Option<ExtractedRadius> {
    match pred {
        PredicateExpr::Atom(PredAtom::DistanceThreshold {
            metric,
            param,
            op,
            radius,
        }) => {
            let inclusive = cmp_to_inclusive(*op)?;
            Some(ExtractedRadius {
                query_param: param.clone(),
                metric: *metric,
                radius: *radius,
                inclusive,
                residue: None,
            })
        }
        PredicateExpr::And(children) => {
            let pos = children.iter().position(|c| {
                matches!(
                    c,
                    PredicateExpr::Atom(PredAtom::DistanceThreshold {
                        op: CmpOp::Lt | CmpOp::Le,
                        ..
                    })
                )
            })?;
            let PredicateExpr::Atom(PredAtom::DistanceThreshold {
                metric,
                param,
                op,
                radius,
            }) = &children[pos]
            else {
                return None;
            };
            let inclusive = cmp_to_inclusive(*op)?;
            let mut rest: Vec<PredicateExpr> = children
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != pos)
                .map(|(_, c)| c.clone())
                .collect();
            let residue = match rest.len() {
                0 => None,
                1 => Some(rest.pop().unwrap()),
                _ => Some(PredicateExpr::And(rest)),
            };
            Some(ExtractedRadius {
                query_param: param.clone(),
                metric: *metric,
                radius: *radius,
                inclusive,
                residue,
            })
        }
        _ => None,
    }
}

/// Map `CmpOp` to `inclusive: bool` for radius semantics. Only `<` and
/// `<=` make sense ; `>`/`>=` would be "outside the ball" (full-scan
/// territory) and `=` / `!=` are nonsense for a distance bound.
fn cmp_to_inclusive(op: CmpOp) -> Option<bool> {
    match op {
        CmpOp::Lt => Some(false),
        CmpOp::Le => Some(true),
        _ => None,
    }
}

/// True if `pred` contains a `DistanceThreshold` atom anywhere in
/// its subtree. Used by the scan-and-limit bypass to refuse predicate
/// shapes the radius extractor didn't recognise , otherwise we'd
/// route them to `MetadataScan` whose predicate evaluator errors out
/// on distance atoms at runtime.
fn predicate_has_distance_threshold(p: &PredicateExpr) -> bool {
    match p {
        PredicateExpr::Atom(PredAtom::DistanceThreshold { .. }) => true,
        PredicateExpr::And(cs) | PredicateExpr::Or(cs) => {
            cs.iter().any(predicate_has_distance_threshold)
        }
        PredicateExpr::Not(inner) => predicate_has_distance_threshold(inner),
        PredicateExpr::Atom(_) | PredicateExpr::True | PredicateExpr::False => false,
    }
}

/// Reject `OR`s that contain a `DistanceThreshold` atom anywhere in
/// their subtree.
///
/// We reject these unconditionally rather than attempting either of
/// the two paths a more capable planner might take :
///
/// - Per-branch selectivity (`union when every branch is selective,
///   fall through to plan A otherwise`) needs a decomposable
///   estimator. Today's `ShardEstimator` walks every row and can't
///   decompose an `Or` cheaply, and the "selective" threshold itself
///   would need measurement nobody has done.
/// - A `Union` operator (the implementation half of the union path)
///   would need multi-radius dedup, distance-merge, and tombstone
///   composition across branches. None of that exists.
/// - Silent fallback to plan A is a recall trap : "OR of two radii"
///   becomes "unfiltered kNN with overfetch and a complex post-filter"
///   which can starve the LIMIT.
///
/// Loud rejection is the honest answer until either an index-driven
/// estimator or a Union operator lands.
fn reject_or_with_distance_threshold(pred: &PredicateExpr) -> Result<(), KovaQueryError> {
    fn walk(p: &PredicateExpr) -> Result<(), KovaQueryError> {
        match p {
            PredicateExpr::Or(cs) => {
                if cs.iter().any(predicate_has_distance_threshold) {
                    return Err(KovaQueryError::Plan(
                        "OR containing a distance-threshold atom isn't supported ; \
                         needs per-branch selectivity plus a Union operator that \
                         merges radius balls, neither of which ships yet"
                            .into(),
                    ));
                }
                Ok(())
            }
            PredicateExpr::And(cs) => cs.iter().try_for_each(walk),
            PredicateExpr::Not(inner) => walk(inner),
            _ => Ok(()),
        }
    }
    walk(pred)
}

fn unimplemented(name: &str) -> Result<PhysicalPlan, KovaQueryError> {
    Err(KovaQueryError::Plan(format!(
        "planner not yet implemented for {name}"
    )))
}

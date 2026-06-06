//! [`LogicalStatement`] -> [`PhysicalPlan`].
//!
//! For write-side and management statements the planning is one-to-one
//! (one logical statement, one operator). The interesting work lands
//! when SELECT joins the dispatch and the cost-model picks between
//! scan / index / post-filter / soft-filtered-ANN.

use kova_core::VectorId;

use crate::error::KovaQueryError;
use crate::logical::{
    BoundProjection, LogicalDelete, LogicalInsert, LogicalInsertSource, LogicalQuery,
    LogicalStatement, LogicalVacuum, OrderingSpec, ProjectionSpec,
};
use crate::physical::PhysicalPlan;

/// kNN overfetch multiplier. The planner asks the kNN for `k_user *
/// OVERFETCH` candidates so the post-filter has room to drop some
/// without starving the final LIMIT. v2 tunes this from selectivity.
const KNN_OVERFETCH: usize = 4;

/// Pick the physical plan for a [`LogicalStatement`].
///
/// # Errors
///
/// Returns [`KovaQueryError::Plan`] for any statement the planner
/// doesn't yet know how to handle. As each statement's executor
/// support lands, its arm gets a real plan ; until then it errors
/// cleanly instead of panicking.
//
// By-value : real arms move fields out of LogicalStatement payloads
// when they land (same shape as the binder dispatch).
#[allow(clippy::needless_pass_by_value)]
pub fn plan(stmt: LogicalStatement) -> Result<PhysicalPlan, KovaQueryError> {
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

        LogicalStatement::Query(q) => plan_query(q),

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
fn plan_query(q: LogicalQuery) -> Result<PhysicalPlan, KovaQueryError> {
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

    // Step 4 : pick plan A or plan B.
    //
    // v1 stopgap rule : if there's a predicate, use plan B (scan + exact
    // distance). Otherwise plan A (pure kNN overfetch). This is wrong
    // when the predicate has high selectivity (plan A is faster), but
    // it's the simplest rule that produces correct answers without a
    // cost model. v2 (M2.6) replaces this with selectivity-based dispatch
    // : `estimate_selectivity(pred) < threshold` picks B, otherwise A.
    let plan = if let Some(pred) = predicate {
        // Plan B : MetadataScan -> ExactDistance -> Limit -> Projection
        let scan = PhysicalPlan::MetadataScan {
            table: from_table,
            predicate: pred,
        };
        let exact = PhysicalPlan::ExactDistance {
            input: Box::new(scan),
            query: query_param,
            metric,
            k: user_k,
        };
        PhysicalPlan::Limit {
            input: Box::new(exact),
            limit: user_limit,
        }
    } else {
        // Plan A : KnnSearch(overfetch) -> Limit -> Projection
        let overfetched_k = user_k.saturating_mul(KNN_OVERFETCH);
        let knn = PhysicalPlan::KnnSearch {
            table: from_table,
            query: query_param,
            metric,
            k: overfetched_k,
            post_filter: None,
        };
        PhysicalPlan::Limit {
            input: Box::new(knn),
            limit: user_limit,
        }
    };

    Ok(PhysicalPlan::Projection {
        input: Box::new(plan),
        spec: projection,
    })
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

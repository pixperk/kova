//! AST -> [`LogicalStatement`] : field resolution, type checks,
//! predicate normalisation, and the hard semantic rejects (embedding
//! update, v2-only statements, distance ordering direction, etc.).
//!
//! The binder is stateless in v1 because the schema is inferred ;
//! v2 grows a context carrying the strict-schema registry.

use crate::ast::{
    AstAssignment, AstDelete, AstExpr, AstInsert, AstInsertSource, AstLiteral, AstOrderBy,
    AstPredicate, AstProjection, AstQuery, AstStatement, AstUpdate, AstVacuum,
    OrderDir as AstOrderDir,
};
use crate::error::KovaQueryError;
use crate::logical::{
    BoundExpr, BoundLiteral, BoundProjection, LogicalAssignment, LogicalDelete, LogicalInsert,
    LogicalInsertSource, LogicalQuery, LogicalStatement, LogicalUpdate, LogicalVacuum, OrderDir,
    OrderingSpec, PredAtom, PredicateExpr, ProjectionSpec,
};

/// Canonical INSERT column shape. v1 accepts these three names, in
/// this order. v2 may relax (any permutation) when the executor's
/// column-to-slot mapping is ready.
const CANONICAL_COLUMNS: [&str; 3] = ["id", "embedding", "metadata"];

/// Bind an [`AstStatement`] into a [`LogicalStatement`].
///
/// # Errors
///
/// Returns [`KovaQueryError::Bind`] for any semantic violation :
/// unknown field, type mismatch, embedding update, v2-only DDL,
/// distance ordering with `DESC`, wildcard in a list, etc.
pub fn bind(ast: AstStatement) -> Result<LogicalStatement, KovaQueryError> {
    match ast {
        AstStatement::Checkpoint => Ok(LogicalStatement::Checkpoint),
        AstStatement::Vacuum(v) => bind_vacuum(v),
        AstStatement::Insert(i) => bind_insert(i),
        AstStatement::Update(u) => bind_update(u),
        AstStatement::Delete(d) => bind_delete(d),
        AstStatement::Select(q) => bind_select(q),

        // Filled in as each binder lands. Explicit arms (rather than
        // a `_` catchall) so the compiler complains the moment a new
        // AST variant is added without a binder.
        AstStatement::CreateIndex(_) => unimplemented(StatementKind::CreateIndex),
        AstStatement::DropIndex(_) => unimplemented(StatementKind::DropIndex),
    }
}

// =========================================================================
// Statement binders
// =========================================================================

/// Bind a [`AstVacuum`] : preserve the target table name into the
/// logical form. The binder does not gate the name against any
/// catalog ; that's the executor's job once it has a `Shard` (or
/// list of shards) to dispatch against. Keeping the name as opaque
/// data here means multi-shard / multi-table is a runtime config
/// change later, not a binder rewrite.
//
// Uniform binder signature across all statement kinds : every
// `bind_X` returns Result even when (today) it can't fail. Keeps the
// dispatcher's `?` consistent and forward-compatible with future
// checks (catalog lookup, permissions, etc.).
#[allow(clippy::unnecessary_wraps)]
fn bind_vacuum(v: AstVacuum) -> Result<LogicalStatement, KovaQueryError> {
    let AstVacuum { table } = v;
    Ok(LogicalStatement::Vacuum(LogicalVacuum { table }))
}

/// Bind an [`AstInsert`] : validate the column list against the
/// canonical v1 shape, then dispatch on row source.
fn bind_insert(i: AstInsert) -> Result<LogicalStatement, KovaQueryError> {
    let AstInsert {
        table,
        columns,
        source,
    } = i;

    validate_canonical_columns(&columns)?;

    let rows = match source {
        AstInsertSource::Rows(rows) => bind_insert_rows(rows)?,
        AstInsertSource::Param(p) => LogicalInsertSource::Batch { param: p },
    };

    Ok(LogicalStatement::Insert(LogicalInsert { table, rows }))
}

/// v1 only accepts a single explicit row ; multi-row inserts go
/// through the batch parameter form. Validate the row shape and
/// extract the three values into the named slots.
fn bind_insert_rows(rows: Vec<Vec<AstExpr>>) -> Result<LogicalInsertSource, KovaQueryError> {
    if rows.len() != 1 {
        return Err(KovaQueryError::Bind(format!(
            "INSERT VALUES (...) supports a single row in v1 (got {}). \
             Use VALUES $batch for multi-row inserts.",
            rows.len()
        )));
    }
    let row = rows.into_iter().next().expect("len checked");
    if row.len() != CANONICAL_COLUMNS.len() {
        return Err(KovaQueryError::Bind(format!(
            "INSERT row must have {} values (id, embedding, metadata) ; got {}",
            CANONICAL_COLUMNS.len(),
            row.len()
        )));
    }
    let mut iter = row.into_iter();
    let id = extract_param(iter.next().expect("0/3"))?;
    let embedding = extract_param(iter.next().expect("1/3"))?;
    let metadata = extract_param(iter.next().expect("2/3"))?;
    Ok(LogicalInsertSource::Single {
        id,
        embedding,
        metadata,
    })
}

/// Reject the canonical-shape violation cases for the INSERT column
/// list. v1 requires the three columns in the documented order ;
/// permutations are a v2 relaxation when the executor's slot
/// mapping supports them.
fn validate_canonical_columns(columns: &[String]) -> Result<(), KovaQueryError> {
    if columns.len() != CANONICAL_COLUMNS.len() {
        return Err(KovaQueryError::Bind(format!(
            "INSERT column list must be exactly ({}) ; got {} column(s)",
            CANONICAL_COLUMNS.join(", "),
            columns.len()
        )));
    }
    for (got, want) in columns.iter().zip(CANONICAL_COLUMNS.iter()) {
        if !got.eq_ignore_ascii_case(want) {
            return Err(KovaQueryError::Bind(format!(
                "INSERT column list must be ({}) in canonical order ; got ({})",
                CANONICAL_COLUMNS.join(", "),
                columns.join(", "),
            )));
        }
    }
    Ok(())
}

/// INSERT row values must be parameter-bound (the grammar already
/// enforces this for `row_value` ; this is the binder-side check
/// in case future grammar relaxations let literals through).
fn extract_param(e: AstExpr) -> Result<crate::ast::ParamRef, KovaQueryError> {
    match e {
        AstExpr::Param(p) => Ok(p),
        AstExpr::Literal(_) => Err(KovaQueryError::Bind(
            "INSERT values must be parameter-bound, not literal".into(),
        )),
    }
}

/// Bind an [`AstUpdate`] : reject embedding assignments (HNSW node
/// positions are immutable), translate the rest into logical form.
fn bind_update(u: AstUpdate) -> Result<LogicalStatement, KovaQueryError> {
    let AstUpdate {
        table,
        assignments,
        predicate,
    } = u;
    let assignments: Result<Vec<_>, _> = assignments.into_iter().map(bind_assignment).collect();
    let assignments = assignments?;
    let predicate = bind_predicate(predicate)?;
    Ok(LogicalStatement::Update(LogicalUpdate {
        table,
        predicate,
        assignments,
    }))
}

/// Translate one `SET <field> = <value>` (or `SET <field>['key'] =
/// <value>`) assignment, after rejecting embedding assignments.
fn bind_assignment(a: AstAssignment) -> Result<LogicalAssignment, KovaQueryError> {
    if a.field.eq_ignore_ascii_case("embedding") {
        return Err(KovaQueryError::Bind(
            "vectors are immutable ; delete and reinsert to change an embedding".into(),
        ));
    }
    let AstAssignment {
        field,
        subscript,
        value,
    } = a;
    Ok(LogicalAssignment {
        field,
        subscript,
        value: bind_expr(value),
    })
}

/// Bind an [`AstDelete`] : translate the predicate, then detect the
/// single-id hint so the planner gets the fast path for free.
fn bind_delete(d: AstDelete) -> Result<LogicalStatement, KovaQueryError> {
    let AstDelete { table, predicate } = d;
    let predicate = bind_predicate(predicate)?;
    let single_id_hint = detect_single_id_hint(&predicate);
    Ok(LogicalStatement::Delete(LogicalDelete {
        table,
        predicate,
        single_id_hint,
    }))
}

/// Pattern-match the top-level predicate for the trivial single-id
/// equality form (`WHERE id = <integer literal>`). Returns the id
/// for the planner to grab without re-walking the tree. Param-bound
/// ids (`WHERE id = $1`) don't qualify because the value isn't
/// known at bind time.
fn detect_single_id_hint(pred: &PredicateExpr) -> Option<u64> {
    let PredicateExpr::Atom(PredAtom::Eq { field, value }) = pred else {
        return None;
    };
    if !field.eq_ignore_ascii_case("id") {
        return None;
    }
    let BoundExpr::Literal(BoundLiteral::I64(n)) = value else {
        return None;
    };
    u64::try_from(*n).ok()
}

// =========================================================================
// Predicate / expression / literal binders
// =========================================================================

/// Translate an [`AstPredicate`] into the canonical [`PredicateExpr`].
/// One structural normalisation happens here : `IS NULL` becomes
/// `NOT IsNotNull(field)` so downstream code only handles the
/// positive form.
//
// Full normalisation (flatten, NOT push-down, constant fold,
// cost-ordering) is a separate pass landing in a later step.
#[allow(clippy::unnecessary_wraps)]
fn bind_predicate(p: AstPredicate) -> Result<PredicateExpr, KovaQueryError> {
    Ok(match p {
        AstPredicate::And(children) => {
            let bound: Result<Vec<_>, _> = children.into_iter().map(bind_predicate).collect();
            PredicateExpr::And(bound?)
        }
        AstPredicate::Or(children) => {
            let bound: Result<Vec<_>, _> = children.into_iter().map(bind_predicate).collect();
            PredicateExpr::Or(bound?)
        }
        AstPredicate::Not(inner) => PredicateExpr::Not(Box::new(bind_predicate(*inner)?)),
        AstPredicate::Eq(field, value) => PredicateExpr::Atom(PredAtom::Eq {
            field,
            value: bind_expr(value),
        }),
        AstPredicate::Cmp(field, op, value) => PredicateExpr::Atom(PredAtom::Cmp {
            field,
            op,
            value: bind_expr(value),
        }),
        AstPredicate::In(field, values) => PredicateExpr::Atom(PredAtom::In {
            field,
            values: values.into_iter().map(bind_literal).collect(),
        }),
        AstPredicate::Between(field, lo, hi) => PredicateExpr::Atom(PredAtom::Between {
            field,
            lo: bind_literal(lo),
            hi: bind_literal(hi),
        }),
        AstPredicate::IsNull(field, negated) => {
            // `IS NOT NULL` is the positive form, `IS NULL` wraps it in NOT.
            let atom = PredicateExpr::Atom(PredAtom::IsNotNull { field });
            if negated {
                atom
            } else {
                PredicateExpr::Not(Box::new(atom))
            }
        }
        AstPredicate::ArrayContains(field, value) => PredicateExpr::Atom(PredAtom::ArrayContains {
            field,
            value: bind_literal(value),
        }),
        AstPredicate::DistanceThreshold(dist, op, radius) => {
            PredicateExpr::Atom(PredAtom::DistanceThreshold {
                metric: dist.metric,
                param: dist.param,
                op,
                radius,
            })
        }
    })
}

/// Translate an [`AstExpr`] (value position) into a [`BoundExpr`].
/// One-to-one mapping today ; future strict-schema work makes this
/// the point that type-checks literals against field types.
fn bind_expr(e: AstExpr) -> BoundExpr {
    match e {
        AstExpr::Param(p) => BoundExpr::Param(p),
        AstExpr::Literal(l) => BoundExpr::Literal(bind_literal(l)),
    }
}

/// Translate an [`AstLiteral`] into a [`BoundLiteral`]. Same shape,
/// different layer so v2 can extend the bound side without churning
/// the AST.
fn bind_literal(l: AstLiteral) -> BoundLiteral {
    match l {
        AstLiteral::String(s) => BoundLiteral::String(s),
        AstLiteral::I64(n) => BoundLiteral::I64(n),
        AstLiteral::F64(f) => BoundLiteral::F64(f),
        AstLiteral::Bool(b) => BoundLiteral::Bool(b),
        AstLiteral::Null => BoundLiteral::Null,
    }
}

// =========================================================================
// SELECT
// =========================================================================

/// Bind an [`AstQuery`] : projection list checks, predicate binding,
/// ORDER BY direction validation, kNN-requires-LIMIT enforcement.
fn bind_select(q: AstQuery) -> Result<LogicalStatement, KovaQueryError> {
    let AstQuery {
        projection,
        from_table,
        predicate,
        order_by,
        limit,
    } = q;

    let projection = bind_projection_list(projection)?;
    let predicate = predicate.map(bind_predicate).transpose()?;
    let ordering = bind_ordering(order_by)?;

    // kNN queries (ordered by a distance expression) require LIMIT.
    // Without it, the user has implicitly asked for the entire shard
    // sorted by distance, which is rarely what they want and always
    // expensive. Force them to be explicit.
    let has_distance_ordering = ordering
        .iter()
        .any(|o| matches!(o, OrderingSpec::Distance { .. }));
    if has_distance_ordering && limit.is_none() {
        return Err(KovaQueryError::Bind(
            "kNN queries (ORDER BY embedding <op> $q) require a LIMIT clause".into(),
        ));
    }

    Ok(LogicalStatement::Query(LogicalQuery {
        from_table,
        projection,
        predicate,
        ordering,
        limit,
    }))
}

/// Validate projection-list-wide rules (wildcard appears alone),
/// then translate each item individually.
fn bind_projection_list(items: Vec<AstProjection>) -> Result<ProjectionSpec, KovaQueryError> {
    let has_wildcard = items.iter().any(|p| matches!(p, AstProjection::Wildcard));
    if has_wildcard && items.len() > 1 {
        return Err(KovaQueryError::Bind(
            "SELECT * cannot appear alongside other projection items".into(),
        ));
    }
    let columns: Result<Vec<_>, _> = items.into_iter().map(bind_projection_item).collect();
    Ok(ProjectionSpec { columns: columns? })
}

/// Translate one projection item. Distance expressions without an
/// alias are rejected here : there's no natural column name for a
/// distance computation, and downstream code (projection serializer,
/// gRPC response shape) needs every column to have a name.
fn bind_projection_item(p: AstProjection) -> Result<BoundProjection, KovaQueryError> {
    match p {
        AstProjection::Wildcard => Ok(BoundProjection::Wildcard),
        AstProjection::CountStar => Ok(BoundProjection::CountStar { alias: None }),
        AstProjection::Id => Ok(BoundProjection::Id { alias: None }),
        AstProjection::Metadata => Ok(BoundProjection::Metadata { alias: None }),
        AstProjection::Field(name) => Ok(BoundProjection::MetadataField { name, alias: None }),
        AstProjection::DistanceExpr(_) => Err(KovaQueryError::Bind(
            "distance expression in SELECT requires an alias (AS <name>)".into(),
        )),
        AstProjection::Aliased(inner, alias) => bind_aliased_projection(*inner, alias),
    }
}

/// Translate `<projection> AS <alias>` by pushing the alias into the
/// inner variant. Defensive rejects for shapes the grammar shouldn't
/// produce (wildcard-with-alias, nested alias).
fn bind_aliased_projection(
    inner: AstProjection,
    alias: String,
) -> Result<BoundProjection, KovaQueryError> {
    match inner {
        AstProjection::Wildcard => Err(KovaQueryError::Bind("SELECT * cannot be aliased".into())),
        AstProjection::Aliased(_, _) => Err(KovaQueryError::Bind(
            "nested aliases are not allowed (use only one AS)".into(),
        )),
        AstProjection::CountStar => Ok(BoundProjection::CountStar { alias: Some(alias) }),
        AstProjection::Id => Ok(BoundProjection::Id { alias: Some(alias) }),
        AstProjection::Metadata => Ok(BoundProjection::Metadata { alias: Some(alias) }),
        AstProjection::DistanceExpr(d) => Ok(BoundProjection::Distance {
            metric: d.metric,
            param: d.param,
            alias,
        }),
        AstProjection::Field(name) => Ok(BoundProjection::MetadataField {
            name,
            alias: Some(alias),
        }),
    }
}

/// Translate the ORDER BY list, item-by-item.
fn bind_ordering(items: Vec<AstOrderBy>) -> Result<Vec<OrderingSpec>, KovaQueryError> {
    items.into_iter().map(bind_ordering_item).collect()
}

/// Translate one ORDER BY item, enforcing the "distance is ASC-only"
/// rule. Sorting by distance descending means "farthest first" which
/// is almost never what the user actually wants ; the spec rejects
/// it so a typo doesn't silently become a wrong answer.
fn bind_ordering_item(item: AstOrderBy) -> Result<OrderingSpec, KovaQueryError> {
    match item {
        AstOrderBy::Distance(d, dir) => {
            if matches!(dir, AstOrderDir::Desc) {
                return Err(KovaQueryError::Bind(
                    "ORDER BY <distance> only supports ASC ; \
                     DESC would mean 'farthest first' which is rarely intentional"
                        .into(),
                ));
            }
            Ok(OrderingSpec::Distance {
                metric: d.metric,
                param: d.param,
            })
        }
        AstOrderBy::Field(name, dir) => Ok(OrderingSpec::Field {
            name,
            dir: bind_order_dir(dir),
        }),
    }
}

/// Map the AST direction enum to the logical one. Same shape today ;
/// two enums so future v2 changes (NULLS FIRST/LAST, etc.) on either
/// side don't churn the other.
fn bind_order_dir(d: AstOrderDir) -> OrderDir {
    match d {
        AstOrderDir::Asc => OrderDir::Asc,
        AstOrderDir::Desc => OrderDir::Desc,
    }
}

// =========================================================================
// Not-yet-implemented helpers
// =========================================================================

/// Shape of the not-yet-implemented stub so the error message stays
/// consistent across statement variants. Shrinks as each statement
/// binder lands ; the whole helper goes away when the last variant
/// is wired up.
#[derive(Debug, Clone, Copy)]
enum StatementKind {
    CreateIndex,
    DropIndex,
}

fn unimplemented(kind: StatementKind) -> Result<LogicalStatement, KovaQueryError> {
    Err(KovaQueryError::Bind(format!(
        "bind not yet implemented for {kind:?}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{CmpOp, DistanceOp, ParamRef};
    use crate::parser::parse_str;

    /// End-to-end sanity : parse a CHECKPOINT statement, hand the AST
    /// to the binder, expect [`LogicalStatement::Checkpoint`] back.
    #[test]
    fn binds_checkpoint() {
        let ast = parse_str("CHECKPOINT").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        assert_eq!(logical, LogicalStatement::Checkpoint);
    }

    /// Case-insensitivity at the parser level survives the binder
    /// (CHECKPOINT carries no payload, but the round-trip still
    /// catches "we accidentally rejected a case-folded keyword").
    #[test]
    fn binds_checkpoint_case_insensitive() {
        let ast = parse_str("checkpoint").expect("parse Ok");
        assert_eq!(bind(ast).expect("bind Ok"), LogicalStatement::Checkpoint);
    }

    /// Every statement type without a real binder yet must report a
    /// clean Bind error, not panic. CREATE INDEX is the chosen probe
    /// today (v2-only DDL).
    #[test]
    fn unimplemented_variants_return_bind_error() {
        let ast = parse_str("CREATE INDEX idx ON vectors USING HASH (category)").expect("parse Ok");
        let err = bind(ast).expect_err("expected Bind error");
        assert!(
            matches!(err, KovaQueryError::Bind(_)),
            "expected Bind, got {err:?}"
        );
    }

    // ----- VACUUM -----

    #[test]
    fn binds_vacuum_carries_table_name_through() {
        let ast = parse_str("VACUUM vectors").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Vacuum(LogicalVacuum { table }) = logical else {
            panic!("expected Vacuum");
        };
        assert_eq!(table, "vectors");
    }

    #[test]
    fn binds_vacuum_preserves_identifier_case() {
        let ast = parse_str("VACUUM MyShard").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Vacuum(LogicalVacuum { table }) = logical else {
            panic!("expected Vacuum");
        };
        assert_eq!(table, "MyShard");
    }

    #[test]
    fn binds_vacuum_accepts_arbitrary_table_name() {
        let ast = parse_str("VACUUM products").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Vacuum(LogicalVacuum { table }) = logical else {
            panic!("expected Vacuum");
        };
        assert_eq!(table, "products");
    }

    // ----- INSERT -----

    #[test]
    fn binds_insert_single_row_canonical() {
        let ast = parse_str("INSERT INTO vectors (id, embedding, metadata) VALUES ($1, $2, $3)")
            .expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Insert(LogicalInsert { table, rows }) = logical else {
            panic!("expected Insert");
        };
        assert_eq!(table, "vectors");
        let LogicalInsertSource::Single {
            id,
            embedding,
            metadata,
        } = rows
        else {
            panic!("expected Single");
        };
        assert!(matches!(id, ParamRef::Positional(1)));
        assert!(matches!(embedding, ParamRef::Positional(2)));
        assert!(matches!(metadata, ParamRef::Positional(3)));
    }

    #[test]
    fn binds_insert_single_row_with_named_params() {
        let ast =
            parse_str("INSERT INTO vectors (id, embedding, metadata) VALUES ($id, $vec, $meta)")
                .expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Insert(LogicalInsert { rows, .. }) = logical else {
            panic!("expected Insert");
        };
        let LogicalInsertSource::Single {
            id,
            embedding,
            metadata,
        } = rows
        else {
            panic!("expected Single");
        };
        assert!(matches!(id, ParamRef::Named(ref s) if s == "id"));
        assert!(matches!(embedding, ParamRef::Named(ref s) if s == "vec"));
        assert!(matches!(metadata, ParamRef::Named(ref s) if s == "meta"));
    }

    #[test]
    fn binds_insert_batch_form() {
        let ast =
            parse_str("INSERT INTO vectors (id, embedding, metadata) VALUES $1").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Insert(LogicalInsert { rows, .. }) = logical else {
            panic!("expected Insert");
        };
        assert!(matches!(
            rows,
            LogicalInsertSource::Batch {
                param: ParamRef::Positional(1)
            }
        ));
    }

    #[test]
    fn binds_insert_accepts_case_variation_in_columns() {
        // Column names are matched case-insensitively against the canonical shape.
        let ast = parse_str("INSERT INTO vectors (Id, EMBEDDING, Metadata) VALUES ($1, $2, $3)")
            .expect("parse Ok");
        assert!(matches!(bind(ast), Ok(LogicalStatement::Insert(_))));
    }

    #[test]
    fn rejects_insert_non_canonical_column_order() {
        // Permuted shape : (embedding, id, metadata) instead of canonical.
        let ast = parse_str("INSERT INTO vectors (embedding, id, metadata) VALUES ($1, $2, $3)")
            .expect("parse Ok");
        let err = bind(ast).expect_err("expected Bind error");
        let KovaQueryError::Bind(msg) = err else {
            panic!("expected Bind, got {err:?}");
        };
        assert!(
            msg.contains("canonical order"),
            "message should call out canonical order : {msg}"
        );
    }

    #[test]
    fn rejects_insert_wrong_column_count() {
        let ast =
            parse_str("INSERT INTO vectors (id, embedding) VALUES ($1, $2)").expect("parse Ok");
        let err = bind(ast).expect_err("expected Bind error");
        assert!(matches!(err, KovaQueryError::Bind(_)));
    }

    #[test]
    fn rejects_insert_unknown_column() {
        let ast = parse_str("INSERT INTO vectors (id, embedding, junk) VALUES ($1, $2, $3)")
            .expect("parse Ok");
        let err = bind(ast).expect_err("expected Bind error");
        assert!(matches!(err, KovaQueryError::Bind(_)));
    }

    // ----- UPDATE -----

    #[test]
    fn binds_update_whole_metadata_replace() {
        let ast = parse_str("UPDATE vectors SET metadata = $1 WHERE id = $2").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Update(LogicalUpdate {
            table,
            assignments,
            predicate,
        }) = logical
        else {
            panic!("expected Update");
        };
        assert_eq!(table, "vectors");
        assert_eq!(assignments.len(), 1);
        let LogicalAssignment {
            field,
            subscript,
            value,
        } = &assignments[0];
        assert_eq!(field, "metadata");
        assert_eq!(subscript.as_deref(), None);
        assert!(matches!(value, BoundExpr::Param(ParamRef::Positional(1))));
        // The WHERE id = $2 is an Eq atom with a param value.
        assert!(matches!(
            predicate,
            PredicateExpr::Atom(PredAtom::Eq { .. })
        ));
    }

    #[test]
    fn binds_update_subscript_patch() {
        let ast = parse_str("UPDATE vectors SET metadata['priority'] = 'high' WHERE id = $1")
            .expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Update(LogicalUpdate { assignments, .. }) = logical else {
            panic!("expected Update");
        };
        let LogicalAssignment {
            subscript, value, ..
        } = &assignments[0];
        assert_eq!(subscript.as_deref(), Some("priority"));
        assert!(matches!(
            value,
            BoundExpr::Literal(BoundLiteral::String(s)) if s == "high"
        ));
    }

    #[test]
    fn binds_update_multiple_assignments() {
        let ast =
            parse_str("UPDATE vectors SET metadata['a'] = 'x', metadata['b'] = 'y' WHERE id = $1")
                .expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Update(LogicalUpdate { assignments, .. }) = logical else {
            panic!("expected Update");
        };
        assert_eq!(assignments.len(), 2);
    }

    #[test]
    fn rejects_update_embedding_assignment() {
        let ast = parse_str("UPDATE vectors SET embedding = $1 WHERE id = $2").expect("parse Ok");
        let err = bind(ast).expect_err("expected Bind error");
        let KovaQueryError::Bind(msg) = err else {
            panic!("expected Bind, got {err:?}");
        };
        assert!(
            msg.contains("immutable"),
            "message should explain immutability : {msg}"
        );
    }

    #[test]
    fn rejects_update_embedding_assignment_case_insensitive() {
        let ast = parse_str("UPDATE vectors SET Embedding = $1 WHERE id = $2").expect("parse Ok");
        assert!(matches!(bind(ast), Err(KovaQueryError::Bind(_))));
    }

    // ----- DELETE -----

    #[test]
    fn binds_delete_simple() {
        let ast = parse_str("DELETE FROM vectors WHERE category = 'docs'").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Delete(LogicalDelete {
            table,
            predicate,
            single_id_hint,
        }) = logical
        else {
            panic!("expected Delete");
        };
        assert_eq!(table, "vectors");
        assert!(matches!(
            predicate,
            PredicateExpr::Atom(PredAtom::Eq { .. })
        ));
        assert_eq!(single_id_hint, None, "category isn't id ; no hint");
    }

    #[test]
    fn binds_delete_with_literal_id_sets_single_id_hint() {
        let ast = parse_str("DELETE FROM vectors WHERE id = 42").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Delete(LogicalDelete { single_id_hint, .. }) = logical else {
            panic!("expected Delete");
        };
        assert_eq!(single_id_hint, Some(42));
    }

    #[test]
    fn binds_delete_with_param_id_does_not_set_single_id_hint() {
        // The id value isn't known at bind time when it's a parameter,
        // so the planner doesn't get a fast path. The hint stays None.
        let ast = parse_str("DELETE FROM vectors WHERE id = $1").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Delete(LogicalDelete { single_id_hint, .. }) = logical else {
            panic!("expected Delete");
        };
        assert_eq!(single_id_hint, None);
    }

    #[test]
    fn binds_delete_with_compound_predicate_no_hint() {
        let ast =
            parse_str("DELETE FROM vectors WHERE id = 1 AND category = 'docs'").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Delete(LogicalDelete { single_id_hint, .. }) = logical else {
            panic!("expected Delete");
        };
        assert_eq!(single_id_hint, None, "compound predicate, no hint");
    }

    #[test]
    fn binds_delete_with_negative_id_rejects_hint() {
        // -5 is a valid i64 literal but not a valid VectorId (u64).
        // Predicate binds fine, but the hint stays None.
        let ast = parse_str("DELETE FROM vectors WHERE id = -5").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Delete(LogicalDelete { single_id_hint, .. }) = logical else {
            panic!("expected Delete");
        };
        assert_eq!(single_id_hint, None);
    }

    // ----- Predicate binder : structural translations -----

    #[test]
    fn binds_is_null_normalises_to_not_is_not_null() {
        // `category IS NULL` becomes NOT(IsNotNull(category)).
        let ast = parse_str("DELETE FROM vectors WHERE category IS NULL").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Delete(LogicalDelete { predicate, .. }) = logical else {
            panic!("expected Delete");
        };
        let PredicateExpr::Not(inner) = predicate else {
            panic!("expected NOT wrapper for IS NULL");
        };
        assert!(matches!(
            inner.as_ref(),
            PredicateExpr::Atom(PredAtom::IsNotNull { .. })
        ));
    }

    #[test]
    fn binds_is_not_null_stays_positive() {
        let ast = parse_str("DELETE FROM vectors WHERE category IS NOT NULL").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Delete(LogicalDelete { predicate, .. }) = logical else {
            panic!("expected Delete");
        };
        assert!(matches!(
            predicate,
            PredicateExpr::Atom(PredAtom::IsNotNull { .. })
        ));
    }

    #[test]
    fn binds_nested_and_or_preserves_structure() {
        // Pre-normalisation : the binder doesn't flatten yet, just
        // translates shape. Two-level AND/OR stays two-level.
        let ast =
            parse_str("DELETE FROM vectors WHERE (a = 1 OR b = 2) AND c = 3").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Delete(LogicalDelete { predicate, .. }) = logical else {
            panic!("expected Delete");
        };
        let PredicateExpr::And(children) = predicate else {
            panic!("expected top-level And");
        };
        assert_eq!(children.len(), 2);
        assert!(matches!(children[0], PredicateExpr::Or(_)));
        assert!(matches!(
            children[1],
            PredicateExpr::Atom(PredAtom::Eq { .. })
        ));
    }

    #[test]
    fn binds_distance_threshold() {
        let ast = parse_str("DELETE FROM vectors WHERE embedding <-> $1 < 0.5").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Delete(LogicalDelete { predicate, .. }) = logical else {
            panic!("expected Delete");
        };
        let PredicateExpr::Atom(PredAtom::DistanceThreshold {
            metric,
            param,
            op,
            radius,
        }) = predicate
        else {
            panic!("expected DistanceThreshold atom");
        };
        assert_eq!(metric, DistanceOp::L2);
        assert!(matches!(param, ParamRef::Positional(1)));
        assert_eq!(op, CmpOp::Lt);
        assert!((radius - 0.5).abs() < f32::EPSILON);
    }

    // ----- SELECT -----

    #[test]
    fn binds_select_star_carries_table() {
        let ast = parse_str("SELECT * FROM vectors").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Query(q) = logical else {
            panic!("expected Query");
        };
        assert_eq!(q.from_table, "vectors");
        assert_eq!(q.projection.columns.len(), 1);
        assert!(matches!(q.projection.columns[0], BoundProjection::Wildcard));
        assert!(q.predicate.is_none());
        assert!(q.ordering.is_empty());
        assert_eq!(q.limit, None);
    }

    #[test]
    fn binds_select_id_and_metadata_route_to_typed_variants() {
        let ast = parse_str("SELECT id, metadata FROM vectors").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Query(q) = logical else {
            panic!("expected Query");
        };
        assert!(matches!(
            q.projection.columns[0],
            BoundProjection::Id { alias: None }
        ));
        assert!(matches!(
            q.projection.columns[1],
            BoundProjection::Metadata { alias: None }
        ));
    }

    #[test]
    fn binds_select_regular_field_uses_metadata_field_variant() {
        let ast = parse_str("SELECT category, year FROM vectors").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Query(q) = logical else {
            panic!("expected Query");
        };
        let names: Vec<&str> = q
            .projection
            .columns
            .iter()
            .map(|p| match p {
                BoundProjection::MetadataField { name, .. } => name.as_str(),
                other => panic!("expected MetadataField, got {other:?}"),
            })
            .collect();
        assert_eq!(names, ["category", "year"]);
    }

    #[test]
    fn binds_select_count_star_with_alias() {
        let ast = parse_str("SELECT COUNT(*) AS n FROM vectors").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Query(q) = logical else {
            panic!("expected Query");
        };
        let BoundProjection::CountStar { alias } = &q.projection.columns[0] else {
            panic!("expected CountStar");
        };
        assert_eq!(alias.as_deref(), Some("n"));
    }

    #[test]
    fn binds_select_distance_projection_with_alias() {
        let ast = parse_str(
            "SELECT embedding <-> $1 AS distance FROM vectors ORDER BY embedding <-> $1 LIMIT 10",
        )
        .expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Query(q) = logical else {
            panic!("expected Query");
        };
        let BoundProjection::Distance {
            metric,
            param,
            alias,
        } = &q.projection.columns[0]
        else {
            panic!("expected Distance projection");
        };
        assert_eq!(*metric, DistanceOp::L2);
        assert!(matches!(param, ParamRef::Positional(1)));
        assert_eq!(alias, "distance");
    }

    #[test]
    fn binds_select_with_field_alias() {
        let ast = parse_str("SELECT id AS row_id FROM vectors").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Query(q) = logical else {
            panic!("expected Query");
        };
        let BoundProjection::Id { alias } = &q.projection.columns[0] else {
            panic!("expected Id");
        };
        assert_eq!(alias.as_deref(), Some("row_id"));
    }

    #[test]
    fn binds_select_with_where_clause() {
        let ast = parse_str("SELECT id FROM vectors WHERE id = $1").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Query(q) = logical else {
            panic!("expected Query");
        };
        assert!(matches!(
            q.predicate,
            Some(PredicateExpr::Atom(PredAtom::Eq { .. }))
        ));
    }

    #[test]
    fn binds_select_with_order_by_field_desc() {
        let ast = parse_str("SELECT id FROM vectors ORDER BY year DESC").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Query(q) = logical else {
            panic!("expected Query");
        };
        assert_eq!(q.ordering.len(), 1);
        let OrderingSpec::Field { name, dir } = &q.ordering[0] else {
            panic!("expected Field ordering");
        };
        assert_eq!(name, "year");
        assert_eq!(*dir, OrderDir::Desc);
    }

    #[test]
    fn binds_select_with_multiple_order_by_keys() {
        let ast =
            parse_str("SELECT id FROM vectors ORDER BY year DESC, score ASC").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Query(q) = logical else {
            panic!("expected Query");
        };
        assert_eq!(
            q.ordering.len(),
            2,
            "multi-key ordering survives binding (v1 ships this)"
        );
    }

    #[test]
    fn binds_select_with_distance_ordering_and_limit() {
        let ast = parse_str("SELECT id FROM vectors ORDER BY embedding <-> $1 LIMIT 10")
            .expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Query(q) = logical else {
            panic!("expected Query");
        };
        let OrderingSpec::Distance { metric, param } = &q.ordering[0] else {
            panic!("expected Distance ordering");
        };
        assert_eq!(*metric, DistanceOp::L2);
        assert!(matches!(param, ParamRef::Positional(1)));
        assert_eq!(q.limit, Some(10));
    }

    #[test]
    fn binds_select_full_hybrid_query() {
        let ast = parse_str(
            "SELECT id, embedding <-> $1 AS distance, metadata FROM vectors \
             WHERE category = 'docs' AND year >= 2024 \
             ORDER BY embedding <-> $1 LIMIT 10",
        )
        .expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Query(q) = logical else {
            panic!("expected Query");
        };
        assert_eq!(q.from_table, "vectors");
        assert_eq!(q.projection.columns.len(), 3);
        assert!(matches!(
            q.projection.columns[0],
            BoundProjection::Id { .. }
        ));
        assert!(matches!(
            q.projection.columns[1],
            BoundProjection::Distance { .. }
        ));
        assert!(matches!(
            q.projection.columns[2],
            BoundProjection::Metadata { .. }
        ));
        assert!(matches!(q.predicate, Some(PredicateExpr::And(_))));
        assert_eq!(q.ordering.len(), 1);
        assert_eq!(q.limit, Some(10));
    }

    // ----- SELECT : rejection paths -----

    #[test]
    fn rejects_select_wildcard_with_other_items() {
        // Parser allows the shape ; binder rejects.
        // Constructed by hand because `*, id` parses as wildcard+select_item which
        // the parser's select_list grammar accepts.
        let ast = AstStatement::Select(AstQuery {
            projection: vec![AstProjection::Wildcard, AstProjection::Id],
            from_table: "vectors".into(),
            predicate: None,
            order_by: vec![],
            limit: None,
        });
        let err = bind(ast).expect_err("expected Bind error");
        let KovaQueryError::Bind(msg) = err else {
            panic!("expected Bind, got {err:?}");
        };
        assert!(
            msg.contains("cannot appear alongside"),
            "message should call out wildcard collision : {msg}"
        );
    }

    #[test]
    fn rejects_select_distance_projection_without_alias() {
        // Construct by hand : the parser's select_item accepts the
        // distance form without alias, the binder rejects.
        let ast = AstStatement::Select(AstQuery {
            projection: vec![AstProjection::DistanceExpr(crate::ast::AstDistance {
                metric: DistanceOp::L2,
                param: ParamRef::Positional(1),
            })],
            from_table: "vectors".into(),
            predicate: None,
            order_by: vec![],
            limit: Some(10),
        });
        let err = bind(ast).expect_err("expected Bind error");
        let KovaQueryError::Bind(msg) = err else {
            panic!("expected Bind, got {err:?}");
        };
        assert!(
            msg.contains("requires an alias"),
            "message should ask for alias : {msg}"
        );
    }

    #[test]
    fn rejects_distance_ordering_with_desc() {
        let ast = AstStatement::Select(AstQuery {
            projection: vec![AstProjection::Id],
            from_table: "vectors".into(),
            predicate: None,
            order_by: vec![AstOrderBy::Distance(
                crate::ast::AstDistance {
                    metric: DistanceOp::L2,
                    param: ParamRef::Positional(1),
                },
                AstOrderDir::Desc,
            )],
            limit: Some(10),
        });
        let err = bind(ast).expect_err("expected Bind error");
        let KovaQueryError::Bind(msg) = err else {
            panic!("expected Bind, got {err:?}");
        };
        assert!(
            msg.contains("ASC"),
            "message should call out ASC-only rule : {msg}"
        );
    }

    #[test]
    fn rejects_knn_query_without_limit() {
        let ast = parse_str("SELECT id FROM vectors ORDER BY embedding <-> $1").expect("parse Ok");
        let err = bind(ast).expect_err("expected Bind error");
        let KovaQueryError::Bind(msg) = err else {
            panic!("expected Bind, got {err:?}");
        };
        assert!(
            msg.contains("LIMIT"),
            "message should call out missing LIMIT : {msg}"
        );
    }

    #[test]
    fn binds_select_without_distance_ordering_no_limit_ok() {
        // A query with field ordering doesn't need LIMIT — only kNN
        // queries do. `ORDER BY year DESC` (without LIMIT) binds fine.
        let ast = parse_str("SELECT id FROM vectors ORDER BY year DESC").expect("parse Ok");
        assert!(bind(ast).is_ok());
    }
}

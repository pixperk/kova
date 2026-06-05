//! AST -> [`LogicalStatement`] : field resolution, type checks,
//! predicate normalisation, and the hard semantic rejects (embedding
//! update, v2-only statements, distance ordering direction, etc.).
//!
//! The binder is stateless in v1 because the schema is inferred ;
//! v2 grows a context carrying the strict-schema registry.

use crate::ast::{
    AstAssignment, AstDelete, AstExpr, AstInsert, AstInsertSource, AstLiteral, AstPredicate,
    AstStatement, AstUpdate, AstVacuum,
};
use crate::error::KovaQueryError;
use crate::logical::{
    BoundExpr, BoundLiteral, LogicalAssignment, LogicalDelete, LogicalInsert, LogicalInsertSource,
    LogicalStatement, LogicalUpdate, LogicalVacuum, PredAtom, PredicateExpr,
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

        // Filled in as each binder lands. Explicit arms (rather than
        // a `_` catchall) so the compiler complains the moment a new
        // AST variant is added without a binder.
        AstStatement::Select(_) => unimplemented(StatementKind::Select),
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
// Not-yet-implemented helpers
// =========================================================================

/// Shape of the not-yet-implemented stub so the error message stays
/// consistent across statement variants. Shrinks as each statement
/// binder lands ; the whole helper goes away when the last variant
/// is wired up.
#[derive(Debug, Clone, Copy)]
enum StatementKind {
    Select,
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
    /// clean Bind error, not panic. SELECT is the chosen probe today.
    #[test]
    fn unimplemented_variants_return_bind_error() {
        let ast = parse_str("SELECT id FROM vectors").expect("parse Ok");
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
}

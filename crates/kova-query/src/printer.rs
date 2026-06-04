//! AST -> KQL pretty-printer.
//!
//! Output is canonical : keywords uppercase, identifiers preserved
//! as-written, single-line, minimal whitespace. The round-trip
//! property `parse(print(ast))` produces an AST that re-prints to
//! the same string. This is the M1.1 final correctness check :
//! lossy printers, ambiguous grammars, and missing AST fields all
//! show up as round-trip failures.
//!
//! Predicate parenthesization is precedence-aware : combinators are
//! parenthesized only when a parent operator binds tighter. So
//! `a = 1 AND b = 2` round-trips without growing parens each time.

use crate::ast::{
    AstAssignment, AstCreateIndex, AstDelete, AstDistance, AstDropIndex, AstExpr, AstInsert,
    AstInsertSource, AstLiteral, AstOrderBy, AstPredicate, AstProjection, AstQuery, AstStatement,
    AstUpdate, AstVacuum, CmpOp, DistanceOp, IndexMethod, OrderDir, ParamRef,
};

/// Pretty-print a top-level KQL statement to canonical form.
#[must_use]
pub fn print(stmt: &AstStatement) -> String {
    match stmt {
        AstStatement::Checkpoint => "CHECKPOINT".to_string(),
        AstStatement::Vacuum(v) => print_vacuum(v),
        AstStatement::Insert(i) => print_insert(i),
        AstStatement::Update(u) => print_update(u),
        AstStatement::Delete(d) => print_delete(d),
        AstStatement::Select(s) => print_select(s),
        AstStatement::CreateIndex(c) => print_create_index(c),
        AstStatement::DropIndex(d) => print_drop_index(d),
    }
}

fn print_vacuum(v: &AstVacuum) -> String {
    format!("VACUUM {}", v.table)
}

fn print_insert(i: &AstInsert) -> String {
    let cols = i.columns.join(", ");
    let vals = match &i.source {
        AstInsertSource::Rows(rows) => {
            let row_strs: Vec<String> = rows
                .iter()
                .map(|row| {
                    let vals: Vec<String> = row.iter().map(print_expr).collect();
                    format!("({})", vals.join(", "))
                })
                .collect();
            row_strs.join(", ")
        }
        AstInsertSource::Param(p) => print_param(p),
    };
    format!("INSERT INTO {} ({cols}) VALUES {vals}", i.table)
}

fn print_update(u: &AstUpdate) -> String {
    let assigns: Vec<String> = u.assignments.iter().map(print_assignment).collect();
    format!(
        "UPDATE {} SET {} WHERE {}",
        u.table,
        assigns.join(", "),
        print_predicate(&u.predicate),
    )
}

fn print_assignment(a: &AstAssignment) -> String {
    let lhs = match &a.subscript {
        Some(key) => format!("{}['{}']", a.field, key),
        None => a.field.clone(),
    };
    format!("{lhs} = {}", print_expr(&a.value))
}

fn print_delete(d: &AstDelete) -> String {
    format!(
        "DELETE FROM {} WHERE {}",
        d.table,
        print_predicate(&d.predicate),
    )
}

fn print_select(s: &AstQuery) -> String {
    let mut out = String::new();
    out.push_str("SELECT ");
    let projs: Vec<String> = s.projection.iter().map(print_projection).collect();
    out.push_str(&projs.join(", "));
    out.push_str(" FROM ");
    out.push_str(&s.from_table);
    if let Some(pred) = &s.predicate {
        out.push_str(" WHERE ");
        out.push_str(&print_predicate(pred));
    }
    if !s.order_by.is_empty() {
        out.push_str(" ORDER BY ");
        let items: Vec<String> = s.order_by.iter().map(print_order_by_item).collect();
        out.push_str(&items.join(", "));
    }
    if let Some(limit) = s.limit {
        use std::fmt::Write;
        write!(out, " LIMIT {limit}").expect("write to String never fails");
    }
    out
}

fn print_create_index(c: &AstCreateIndex) -> String {
    let name_part = match &c.name {
        Some(n) => format!(" {n}"),
        None => String::new(),
    };
    format!(
        "CREATE INDEX{name_part} ON {} USING {} ({})",
        c.table,
        print_index_method(c.method),
        c.field,
    )
}

fn print_drop_index(d: &AstDropIndex) -> String {
    format!("DROP INDEX {} ON {}", d.name, d.table)
}

fn print_projection(p: &AstProjection) -> String {
    match p {
        AstProjection::Wildcard => "*".to_string(),
        AstProjection::CountStar => "COUNT(*)".to_string(),
        AstProjection::Id => "id".to_string(),
        AstProjection::Metadata => "metadata".to_string(),
        AstProjection::DistanceExpr(d) => print_distance_expr(d),
        AstProjection::Field(s) => s.clone(),
        AstProjection::Aliased(inner, alias) => {
            format!("{} AS {alias}", print_projection(inner))
        }
    }
}

fn print_order_by_item(o: &AstOrderBy) -> String {
    match o {
        AstOrderBy::Distance(d, dir) => {
            format!("{} {}", print_distance_expr(d), print_dir(*dir))
        }
        AstOrderBy::Field(name, dir) => format!("{name} {}", print_dir(*dir)),
    }
}

fn print_dir(d: OrderDir) -> &'static str {
    match d {
        OrderDir::Asc => "ASC",
        OrderDir::Desc => "DESC",
    }
}

fn print_index_method(m: IndexMethod) -> &'static str {
    match m {
        IndexMethod::Hash => "HASH",
        IndexMethod::Btree => "BTREE",
        IndexMethod::Inverted => "INVERTED",
    }
}

fn print_expr(e: &AstExpr) -> String {
    match e {
        AstExpr::Param(p) => print_param(p),
        AstExpr::Literal(l) => print_literal(l),
    }
}

fn print_param(p: &ParamRef) -> String {
    match p {
        ParamRef::Positional(n) => format!("${n}"),
        ParamRef::Named(s) => format!("${s}"),
    }
}

fn print_literal(l: &AstLiteral) -> String {
    match l {
        AstLiteral::String(s) => format!("'{s}'"),
        AstLiteral::I64(n) => n.to_string(),
        AstLiteral::F64(f) => print_f64(*f),
        AstLiteral::Bool(true) => "TRUE".to_string(),
        AstLiteral::Bool(false) => "FALSE".to_string(),
        AstLiteral::Null => "NULL".to_string(),
    }
}

/// Emit an f64 in a form that round-trips back to [`AstLiteral::F64`]
/// (not [`AstLiteral::I64`]). Whole-number floats need an explicit
/// `.0` because the grammar splits literals on the decimal point.
fn print_f64(f: f64) -> String {
    if f.fract() == 0.0 && f.is_finite() {
        format!("{f}.0")
    } else {
        f.to_string()
    }
}

/// Same as [`print_f64`] but for the `f32`-sized distance threshold
/// radius.
fn print_f32(f: f32) -> String {
    if f.fract() == 0.0 && f.is_finite() {
        format!("{f}.0")
    } else {
        f.to_string()
    }
}

fn print_distance_expr(d: &AstDistance) -> String {
    format!(
        "embedding {} {}",
        print_distance_op(d.metric),
        print_param(&d.param),
    )
}

fn print_distance_op(op: DistanceOp) -> &'static str {
    match op {
        DistanceOp::L2 => "<->",
        DistanceOp::Cosine => "<=>",
        DistanceOp::InnerProduct => "<#>",
    }
}

fn print_cmp_op(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "=",
        CmpOp::Lt => "<",
        CmpOp::Le => "<=",
        CmpOp::Gt => ">",
        CmpOp::Ge => ">=",
        CmpOp::Ne => "!=",
    }
}

// ----- Predicate printing with precedence-aware parens -----

/// Binding precedence : higher = tighter. Used to decide when a
/// child predicate needs parens.
fn predicate_prec(p: &AstPredicate) -> u8 {
    match p {
        AstPredicate::Or(_) => 0,
        AstPredicate::And(_) => 1,
        AstPredicate::Not(_) => 2,
        // Every atom variant.
        AstPredicate::Eq(..)
        | AstPredicate::Cmp(..)
        | AstPredicate::In(..)
        | AstPredicate::Between(..)
        | AstPredicate::IsNull(..)
        | AstPredicate::ArrayContains(..)
        | AstPredicate::DistanceThreshold(..) => 3,
    }
}

fn print_predicate(p: &AstPredicate) -> String {
    // Top-level call : no surrounding context, so parent_prec = 0
    // (the lowest possible). Only the predicate-itself's children
    // get parens-checked.
    print_predicate_at(p, 0)
}

/// Print `p` with surrounding context at precedence `parent_prec`.
/// Wraps in parens iff `p`'s own precedence is lower than the
/// parent's (i.e. removing the parens would change associativity).
fn print_predicate_at(p: &AstPredicate, parent_prec: u8) -> String {
    let body = match p {
        AstPredicate::Or(children) => children
            .iter()
            .map(|c| print_predicate_at(c, 0))
            .collect::<Vec<_>>()
            .join(" OR "),
        AstPredicate::And(children) => children
            .iter()
            .map(|c| print_predicate_at(c, 1))
            .collect::<Vec<_>>()
            .join(" AND "),
        AstPredicate::Not(inner) => format!("NOT {}", print_predicate_at(inner, 2)),
        AstPredicate::Eq(field, val) => format!("{field} = {}", print_expr(val)),
        AstPredicate::Cmp(field, op, val) => {
            format!("{field} {} {}", print_cmp_op(*op), print_expr(val))
        }
        AstPredicate::In(field, values) => {
            let vals: Vec<String> = values.iter().map(print_literal).collect();
            format!("{field} IN ({})", vals.join(", "))
        }
        AstPredicate::Between(field, lo, hi) => {
            format!(
                "{field} BETWEEN {} AND {}",
                print_literal(lo),
                print_literal(hi),
            )
        }
        AstPredicate::IsNull(field, true) => format!("{field} IS NOT NULL"),
        AstPredicate::IsNull(field, false) => format!("{field} IS NULL"),
        AstPredicate::ArrayContains(field, val) => {
            format!("{field} @> {}", print_literal(val))
        }
        AstPredicate::DistanceThreshold(dist, op, radius) => {
            format!(
                "{} {} {}",
                print_distance_expr(dist),
                print_cmp_op(*op),
                print_f32(*radius),
            )
        }
    };

    if predicate_prec(p) < parent_prec {
        format!("({body})")
    } else {
        body
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_str;

    /// Idempotency check : parse, print, re-parse, print again ;
    /// the two printed strings must match. This catches lossy
    /// printers (info dropped between AST and string), ambiguous
    /// grammars (different ASTs printing the same string), and
    /// non-canonical print output (parens grow each round).
    fn assert_roundtrip(input: &str) {
        let ast1 = parse_str(input).expect("first parse");
        let s1 = print(&ast1);
        let ast2 =
            parse_str(&s1).unwrap_or_else(|e| panic!("re-parse of printed output {s1:?}: {e}"));
        let s2 = print(&ast2);
        assert_eq!(s1, s2, "print is not idempotent under round-trip");
    }

    // ----- Statement-level -----

    #[test]
    fn roundtrip_checkpoint() {
        assert_roundtrip("CHECKPOINT");
    }

    #[test]
    fn roundtrip_vacuum() {
        assert_roundtrip("VACUUM vectors");
    }

    #[test]
    fn roundtrip_insert_with_positional_params() {
        assert_roundtrip("INSERT INTO vectors (id, embedding, metadata) VALUES ($1, $2, $3)");
    }

    #[test]
    fn roundtrip_insert_with_named_params() {
        assert_roundtrip(
            "INSERT INTO vectors (id, embedding, metadata) VALUES ($id, $embed, $meta)",
        );
    }

    #[test]
    fn roundtrip_insert_batch_form() {
        assert_roundtrip("INSERT INTO vectors (id, embedding, metadata) VALUES $1");
    }

    #[test]
    fn roundtrip_update_simple() {
        assert_roundtrip("UPDATE vectors SET metadata = $1 WHERE id = $2");
    }

    #[test]
    fn roundtrip_update_with_subscript() {
        assert_roundtrip("UPDATE vectors SET metadata['priority'] = 'high' WHERE id = $1");
    }

    #[test]
    fn roundtrip_update_with_multiple_assignments() {
        assert_roundtrip(
            "UPDATE vectors SET metadata['a'] = 'x', metadata['b'] = 'y' WHERE id = $1",
        );
    }

    #[test]
    fn roundtrip_delete_by_id() {
        assert_roundtrip("DELETE FROM vectors WHERE id = $1");
    }

    #[test]
    fn roundtrip_delete_with_compound_predicate() {
        assert_roundtrip("DELETE FROM vectors WHERE category = 'archived' AND year < 2020");
    }

    #[test]
    fn roundtrip_select_star() {
        assert_roundtrip("SELECT * FROM vectors");
    }

    #[test]
    fn roundtrip_select_basic_projection() {
        assert_roundtrip("SELECT id, metadata FROM vectors");
    }

    #[test]
    fn roundtrip_select_with_field_projection() {
        assert_roundtrip("SELECT category, year FROM vectors");
    }

    #[test]
    fn roundtrip_select_with_alias() {
        assert_roundtrip("SELECT embedding <-> $query AS distance FROM vectors");
    }

    #[test]
    fn roundtrip_select_count_star() {
        assert_roundtrip("SELECT COUNT(*) FROM vectors WHERE category = 'docs'");
    }

    #[test]
    fn roundtrip_select_with_order_by_field_desc() {
        assert_roundtrip("SELECT id FROM vectors ORDER BY year DESC");
    }

    #[test]
    fn roundtrip_select_with_order_by_distance_asc() {
        assert_roundtrip("SELECT id FROM vectors ORDER BY embedding <-> $1 ASC");
    }

    #[test]
    fn roundtrip_select_with_limit() {
        assert_roundtrip("SELECT id FROM vectors LIMIT 100");
    }

    #[test]
    fn roundtrip_select_full_hybrid_query() {
        assert_roundtrip(
            "SELECT id, embedding <-> $1 AS distance, metadata FROM vectors \
             WHERE category = 'docs' AND year >= 2024 \
             ORDER BY embedding <-> $1 ASC LIMIT 10",
        );
    }

    #[test]
    fn roundtrip_create_index_with_name() {
        assert_roundtrip("CREATE INDEX idx_cat ON vectors USING HASH (category)");
    }

    #[test]
    fn roundtrip_create_index_without_name() {
        assert_roundtrip("CREATE INDEX ON vectors USING BTREE (year)");
    }

    #[test]
    fn roundtrip_drop_index() {
        assert_roundtrip("DROP INDEX idx_cat ON vectors");
    }

    // ----- Predicate shapes -----

    #[test]
    fn roundtrip_predicate_in_list() {
        assert_roundtrip("DELETE FROM vectors WHERE category IN ('a', 'b', 'c')");
    }

    #[test]
    fn roundtrip_predicate_between() {
        assert_roundtrip("DELETE FROM vectors WHERE score BETWEEN 0.5 AND 1.0");
    }

    #[test]
    fn roundtrip_predicate_is_null() {
        assert_roundtrip("DELETE FROM vectors WHERE category IS NULL");
    }

    #[test]
    fn roundtrip_predicate_is_not_null() {
        assert_roundtrip("DELETE FROM vectors WHERE category IS NOT NULL");
    }

    #[test]
    fn roundtrip_predicate_array_contains() {
        assert_roundtrip("DELETE FROM vectors WHERE tags @> 'rust'");
    }

    #[test]
    fn roundtrip_predicate_distance_threshold() {
        assert_roundtrip("DELETE FROM vectors WHERE embedding <-> $1 < 0.5");
    }

    // ----- Predicate combinators + precedence -----

    #[test]
    fn roundtrip_predicate_simple_and() {
        assert_roundtrip("DELETE FROM vectors WHERE a = 1 AND b = 2");
    }

    #[test]
    fn roundtrip_predicate_simple_or() {
        assert_roundtrip("DELETE FROM vectors WHERE a = 1 OR b = 2");
    }

    #[test]
    fn roundtrip_predicate_not_atom() {
        assert_roundtrip("DELETE FROM vectors WHERE NOT a = 1");
    }

    #[test]
    fn roundtrip_predicate_and_inside_or_no_parens() {
        // Natural precedence : `a AND b OR c` parses as `(a AND b) OR c`,
        // print preserves the same precedence WITHOUT adding parens.
        assert_roundtrip("DELETE FROM vectors WHERE a = 1 AND b = 2 OR c = 3");
    }

    #[test]
    fn roundtrip_predicate_or_inside_and_needs_parens() {
        // The printer MUST add parens here : without them, the meaning
        // would flip from `(OR) AND c` to `a AND (OR)`. The round-trip
        // check is the canary.
        assert_roundtrip("DELETE FROM vectors WHERE (a = 1 OR b = 2) AND c = 3");
    }

    #[test]
    fn roundtrip_predicate_not_around_combinator_needs_parens() {
        assert_roundtrip("DELETE FROM vectors WHERE NOT (a = 1 AND b = 2)");
    }

    #[test]
    fn roundtrip_predicate_three_term_and_flattens() {
        assert_roundtrip("DELETE FROM vectors WHERE a = 1 AND b = 2 AND c = 3");
    }

    // ----- Literals -----

    #[test]
    fn roundtrip_literal_negative_integer() {
        assert_roundtrip("DELETE FROM vectors WHERE delta = -5");
    }

    #[test]
    fn roundtrip_literal_float() {
        assert_roundtrip("DELETE FROM vectors WHERE score = 2.5");
    }

    #[test]
    fn roundtrip_literal_whole_float_keeps_decimal() {
        // 5.0 as f64 must print with the `.0` suffix so re-parse
        // doesn't degrade it to i64.
        assert_roundtrip("DELETE FROM vectors WHERE score = 5.0");
    }

    #[test]
    fn roundtrip_literal_boolean() {
        assert_roundtrip("DELETE FROM vectors WHERE pinned = TRUE");
    }

    #[test]
    fn roundtrip_literal_null() {
        assert_roundtrip("DELETE FROM vectors WHERE category = NULL");
    }
}

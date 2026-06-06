//! Pest-driven parser : `String` -> [`AstStatement`].

use pest::Parser;
use pest::iterators::Pair;

use crate::ast::{
    AstAssignment, AstCreateIndex, AstDelete, AstDistance, AstDropIndex, AstExpr, AstFieldRef,
    AstInsert, AstInsertSource, AstLiteral, AstOrderBy, AstPredicate, AstProjection, AstQuery,
    AstStatement, AstUpdate, AstVacuum, CmpOp, DistanceOp, IndexMethod, OrderDir, ParamRef,
};
use crate::error::KovaQueryError;

// pest_derive emits a `Rule` enum + parse impl at module scope ;
// none of those items carry doc comments. Wrap in a private submodule
// so the `missing_docs` lint can be allowed without polluting the
// rest of the crate.
mod grammar {
    #![allow(missing_docs)]
    use pest_derive::Parser;

    #[derive(Parser)]
    #[grammar = "grammar.pest"]
    pub struct KqlParser;
}

use grammar::{KqlParser, Rule};

/// Parse a single KQL statement from a string into its AST.
///
/// Semantic failures (unknown field, type mismatch, etc.) are not
/// detected here ; they belong to the binder.
///
/// # Errors
///
/// Returns [`KovaQueryError::Parse`] with line/column for any input
/// the grammar rejects.
pub fn parse_str(input: &str) -> Result<AstStatement, KovaQueryError> {
    let mut pairs =
        KqlParser::parse(Rule::program, input).map_err(|e| KovaQueryError::Parse(e.to_string()))?;

    // program -> statement -> <one statement variant>
    let program = pairs.next().expect("program rule present on Ok");
    let statement = program
        .into_inner()
        .next()
        .expect("program has a statement child");
    let inner = statement
        .into_inner()
        .next()
        .expect("statement has exactly one variant child");

    match inner.as_rule() {
        Rule::checkpoint_stmt => Ok(AstStatement::Checkpoint),
        Rule::vacuum_stmt => Ok(AstStatement::Vacuum(parse_vacuum(inner))),
        Rule::insert_stmt => Ok(AstStatement::Insert(parse_insert(inner))),
        Rule::update_stmt => Ok(AstStatement::Update(parse_update(inner))),
        Rule::delete_stmt => Ok(AstStatement::Delete(parse_delete(inner))),
        Rule::select_stmt => Ok(AstStatement::Select(parse_select(inner))),
        Rule::create_index_stmt => Ok(AstStatement::CreateIndex(parse_create_index(inner))),
        Rule::drop_index_stmt => Ok(AstStatement::DropIndex(parse_drop_index(inner))),
        rule => unreachable!("unexpected statement variant: {rule:?}"),
    }
}

/// Build an [`AstQuery`] from a `select_stmt` pair.
///
/// Grammar : `select_stmt = { ^"SELECT" ~ select_list ~ ^"FROM" ~
/// table_ref ~ (^"WHERE" ~ predicate)? ~ order_by_clause? ~
/// limit_clause? }`. Required children come first ; the optional
/// clauses (`predicate`, `order_by_clause`, `limit_clause`) follow
/// in any combination. Dispatch on `as_rule()` to pick them up.
fn parse_select(pair: Pair<Rule>) -> AstQuery {
    let mut inner = pair.into_inner();
    let projection = parse_select_list(inner.next().expect("select_stmt has select_list"));
    let from_table = parse_table_ref(inner.next().expect("select_stmt has table_ref"));

    let mut predicate = None;
    let mut order_by = Vec::new();
    let mut limit = None;
    for child in inner {
        match child.as_rule() {
            Rule::predicate => predicate = Some(parse_predicate(child)),
            Rule::order_by_clause => order_by = parse_order_by_clause(child),
            Rule::limit_clause => limit = Some(parse_limit_clause(child)),
            rule => unreachable!("unexpected select_stmt child: {rule:?}"),
        }
    }

    AstQuery {
        projection,
        from_table,
        predicate,
        order_by,
        limit,
    }
}

/// Build the projection list from a `select_list` pair. Either a
/// single `Wildcard` element or one item per `select_item` child.
fn parse_select_list(pair: Pair<Rule>) -> Vec<AstProjection> {
    pair.into_inner()
        .map(|child| match child.as_rule() {
            Rule::wildcard_proj => AstProjection::Wildcard,
            Rule::select_item => parse_select_item(child),
            rule => unreachable!("unexpected select_list child: {rule:?}"),
        })
        .collect()
}

/// Build a single [`AstProjection`] from a `select_item` pair.
///
/// First visible child is the expression (distance / COUNT(*) /
/// identifier) ; the optional second child is the alias identifier.
fn parse_select_item(pair: Pair<Rule>) -> AstProjection {
    let mut inner = pair.into_inner();
    let first = inner.next().expect("select_item has expression");

    let base = match first.as_rule() {
        Rule::distance_expr => AstProjection::DistanceExpr(parse_distance_expr(first)),
        Rule::count_star => AstProjection::CountStar,
        Rule::identifier => classify_field_projection(first.as_str()),
        rule => unreachable!("unexpected select_item child: {rule:?}"),
    };

    if let Some(alias_pair) = inner.next() {
        AstProjection::Aliased(Box::new(base), alias_pair.as_str().to_string())
    } else {
        base
    }
}

/// Route bare identifiers to the typed [`AstProjection`] variants
/// for the magic column names. Case-insensitive : `id`, `ID`, `Id`
/// all become [`AstProjection::Id`]. Same for `metadata`.
fn classify_field_projection(name: &str) -> AstProjection {
    if name.eq_ignore_ascii_case("id") {
        AstProjection::Id
    } else if name.eq_ignore_ascii_case("metadata") {
        AstProjection::Metadata
    } else {
        AstProjection::Field(name.to_string())
    }
}

/// Collect ORDER BY items from an `order_by_clause` pair.
fn parse_order_by_clause(pair: Pair<Rule>) -> Vec<AstOrderBy> {
    pair.into_inner().map(parse_order_by_item).collect()
}

/// Build a single [`AstOrderBy`] from an `order_by_item` pair.
///
/// First visible child is the expression (distance / identifier) ;
/// the optional second child is the direction.
fn parse_order_by_item(pair: Pair<Rule>) -> AstOrderBy {
    let mut inner = pair.into_inner();
    let first = inner.next().expect("order_by_item has expression");
    let dir = inner.next().map_or(OrderDir::Asc, |dir_pair| {
        if dir_pair.as_str().eq_ignore_ascii_case("desc") {
            OrderDir::Desc
        } else {
            OrderDir::Asc
        }
    });
    match first.as_rule() {
        Rule::distance_expr => AstOrderBy::Distance(parse_distance_expr(first), dir),
        Rule::identifier => AstOrderBy::Field(first.as_str().to_string(), dir),
        rule => unreachable!("unexpected order_by_item child: {rule:?}"),
    }
}

/// Extract the integer count from a `limit_clause` pair.
fn parse_limit_clause(pair: Pair<Rule>) -> u64 {
    pair.into_inner()
        .next()
        .expect("limit_clause has integer")
        .as_str()
        .parse()
        .expect("integer parses as u64 by grammar")
}

/// Build an [`AstUpdate`] from an `update_stmt` pair.
///
/// Grammar : `update_stmt = { ^"UPDATE" ~ table_ref ~ ^"SET" ~
/// set_assignment ~ ("," ~ set_assignment)* ~ ^"WHERE" ~ predicate }`.
/// Visible children : `table_ref`, one or more `set_assignment`s, and
/// finally `predicate`. We dispatch on each child's rule to collect
/// the assignment list and pull the predicate out.
fn parse_update(pair: Pair<Rule>) -> AstUpdate {
    let mut inner = pair.into_inner();
    let table = parse_table_ref(inner.next().expect("update_stmt has table_ref"));

    let mut assignments = Vec::new();
    let mut predicate = None;
    for child in inner {
        match child.as_rule() {
            Rule::set_assignment => assignments.push(parse_set_assignment(child)),
            Rule::predicate => predicate = Some(parse_predicate(child)),
            rule => unreachable!("unexpected update_stmt child: {rule:?}"),
        }
    }

    AstUpdate {
        table,
        assignments,
        predicate: predicate.expect("update_stmt has predicate by grammar"),
    }
}

/// Build an [`AstAssignment`] from a `set_assignment` pair.
///
/// Grammar : `set_assignment = { identifier ~ ("[" ~ string_literal
/// ~ "]")? ~ "=" ~ atom_value }`. The second visible child is either
/// `string_literal` (subscript form) or `atom_value` (whole-field
/// form) ; dispatch on rule to tell them apart.
fn parse_set_assignment(pair: Pair<Rule>) -> AstAssignment {
    let mut inner = pair.into_inner();
    let field = inner
        .next()
        .expect("set_assignment has identifier")
        .as_str()
        .to_string();
    let next = inner.next().expect("set_assignment has more children");

    let (subscript, value_pair) = if matches!(next.as_rule(), Rule::string_literal) {
        let sub = parse_string_literal(&next);
        let val = inner
            .next()
            .expect("set_assignment has atom_value after subscript");
        (Some(sub), val)
    } else {
        (None, next)
    };

    AstAssignment {
        field,
        subscript,
        value: parse_atom_value(value_pair),
    }
}

/// Build an [`AstDelete`] from a `delete_stmt` pair.
///
/// Grammar : `delete_stmt = { ^"DELETE" ~ ^"FROM" ~ table_ref ~
/// ^"WHERE" ~ predicate }`. The keywords are silent literals ;
/// visible children in order : `table_ref`, `predicate`.
fn parse_delete(pair: Pair<Rule>) -> AstDelete {
    let mut inner = pair.into_inner();
    let table = parse_table_ref(inner.next().expect("delete_stmt has table_ref"));
    let predicate = parse_predicate(inner.next().expect("delete_stmt has predicate"));
    AstDelete { table, predicate }
}

/// Extract the table name from a `table_ref` pair.
///
/// `table_ref = { identifier }` : one child.
fn parse_table_ref(pair: Pair<Rule>) -> String {
    pair.into_inner()
        .next()
        .expect("table_ref has an identifier child")
        .as_str()
        .to_string()
}

/// Build an [`AstVacuum`] from a `vacuum_stmt` pair.
///
/// Grammar : `vacuum_stmt = { ^"VACUUM" ~ table_ref }`. The literal
/// `VACUUM` doesn't produce a pair, so the only child is `table_ref`.
fn parse_vacuum(pair: Pair<Rule>) -> AstVacuum {
    let table_ref = pair
        .into_inner()
        .next()
        .expect("vacuum_stmt has a table_ref child");
    AstVacuum {
        table: parse_table_ref(table_ref),
    }
}

/// Build an [`AstInsert`] from an `insert_stmt` pair.
///
/// Grammar : `insert_stmt = { ^"INSERT" ~ ^"INTO" ~ table_ref ~
/// "(" ~ column_list ~ ")" ~ ^"VALUES" ~ values_clause }`. Visible
/// children in order : `table_ref`, `column_list`, `values_clause`.
fn parse_insert(pair: Pair<Rule>) -> AstInsert {
    let mut inner = pair.into_inner();
    let table = parse_table_ref(inner.next().expect("insert_stmt has table_ref"));
    let columns = parse_column_list(inner.next().expect("insert_stmt has column_list"));
    let source = parse_values_clause(inner.next().expect("insert_stmt has values_clause"));
    AstInsert {
        table,
        columns,
        source,
    }
}

/// Extract column names from a `column_list` pair.
fn parse_column_list(pair: Pair<Rule>) -> Vec<String> {
    pair.into_inner().map(|p| p.as_str().to_string()).collect()
}

/// Dispatch a `values_clause` to either explicit rows or a batch param.
fn parse_values_clause(pair: Pair<Rule>) -> AstInsertSource {
    let inner = pair.into_inner().next().expect("values_clause has a child");
    match inner.as_rule() {
        Rule::row_tuple => AstInsertSource::Rows(vec![parse_row_tuple(inner)]),
        Rule::param => AstInsertSource::Param(parse_param(inner)),
        rule => unreachable!("unexpected values_clause child: {rule:?}"),
    }
}

/// Extract a row of values from a `row_tuple` pair.
fn parse_row_tuple(pair: Pair<Rule>) -> Vec<AstExpr> {
    pair.into_inner().map(parse_row_value).collect()
}

/// Build an [`AstExpr`] from a `row_value` pair.
fn parse_row_value(pair: Pair<Rule>) -> AstExpr {
    let inner = pair.into_inner().next().expect("row_value has a child");
    match inner.as_rule() {
        Rule::param => AstExpr::Param(parse_param(inner)),
        rule => unreachable!("unexpected row_value child: {rule:?}"),
    }
}

/// Build a [`ParamRef`] from a `param` pair (positional or named).
fn parse_param(pair: Pair<Rule>) -> ParamRef {
    let inner = pair.into_inner().next().expect("param has a child");
    match inner.as_rule() {
        Rule::positional_param => {
            let integer = inner
                .into_inner()
                .next()
                .expect("positional_param has integer");
            let n: u32 = integer
                .as_str()
                .parse()
                .expect("integer matches u32 by grammar");
            ParamRef::Positional(n)
        }
        Rule::named_param => {
            let identifier = inner
                .into_inner()
                .next()
                .expect("named_param has identifier");
            ParamRef::Named(identifier.as_str().to_string())
        }
        rule => unreachable!("unexpected param child: {rule:?}"),
    }
}

/// Build an [`AstCreateIndex`] from a `create_index_stmt` pair.
///
/// Grammar : `create_index_stmt = { ^"CREATE" ~ ^"INDEX" ~ identifier?
/// ~ ^"ON" ~ table_ref ~ ^"USING" ~ index_method ~ "(" ~ identifier ~ ")" }`.
///
/// When the optional name is absent the first visible child is
/// `table_ref` ; when present it's an `identifier` followed by
/// `table_ref`. Dispatch on `as_rule()`.
fn parse_create_index(pair: Pair<Rule>) -> AstCreateIndex {
    let mut inner = pair.into_inner();
    let mut next = inner.next().expect("create_index_stmt has children");

    let name = if matches!(next.as_rule(), Rule::identifier) {
        let n = next.as_str().to_string();
        next = inner.next().expect("table_ref after optional name");
        Some(n)
    } else {
        None
    };

    let table = parse_table_ref(next);
    let method = parse_index_method(&inner.next().expect("create_index_stmt has index_method"));
    let field = inner
        .next()
        .expect("create_index_stmt has field identifier")
        .as_str()
        .to_string();

    AstCreateIndex {
        name,
        table,
        method,
        field,
    }
}

/// Map an `index_method` pair to the typed enum.
fn parse_index_method(pair: &Pair<Rule>) -> IndexMethod {
    match pair.as_str().to_uppercase().as_str() {
        "HASH" => IndexMethod::Hash,
        "BTREE" => IndexMethod::Btree,
        "INVERTED" => IndexMethod::Inverted,
        other => unreachable!("unknown index method: {other}"),
    }
}

/// Build an [`AstDropIndex`] from a `drop_index_stmt` pair.
fn parse_drop_index(pair: Pair<Rule>) -> AstDropIndex {
    let mut inner = pair.into_inner();
    let name = inner
        .next()
        .expect("drop_index_stmt has name identifier")
        .as_str()
        .to_string();
    let table = parse_table_ref(inner.next().expect("drop_index_stmt has table_ref"));
    AstDropIndex { name, table }
}

// =========================================================================
// Predicate subtree
// =========================================================================
//
// Walks the precedence-climbing grammar (predicate -> or_expr ->
// and_expr -> not_expr -> atom_or_parens -> atom). Boolean nodes
// (`And`, `Or`) are flattened : a single-child or_expr/and_expr
// returns its child directly instead of wrapping in a 1-element
// combinator. This keeps the produced predicate tree canonical.

/// Entry point. `predicate -> or_expr`.
fn parse_predicate(pair: Pair<Rule>) -> AstPredicate {
    parse_or_expr(
        pair.into_inner()
            .next()
            .expect("predicate has or_expr child"),
    )
}

/// `or_expr = { and_expr ~ (^"OR" ~ and_expr)* }`. Multiple children
/// produce an [`AstPredicate::Or`] ; a single child collapses through.
fn parse_or_expr(pair: Pair<Rule>) -> AstPredicate {
    let children: Vec<AstPredicate> = pair.into_inner().map(parse_and_expr).collect();
    if children.len() == 1 {
        children.into_iter().next().expect("len checked")
    } else {
        AstPredicate::Or(children)
    }
}

/// `and_expr = { not_expr ~ (^"AND" ~ not_expr)* }`. Multiple children
/// produce an [`AstPredicate::And`] ; a single child collapses through.
fn parse_and_expr(pair: Pair<Rule>) -> AstPredicate {
    let children: Vec<AstPredicate> = pair.into_inner().map(parse_not_expr).collect();
    if children.len() == 1 {
        children.into_iter().next().expect("len checked")
    } else {
        AstPredicate::And(children)
    }
}

/// `not_expr = { ^"NOT" ~ not_expr | atom_or_parens }`. The `^"NOT"`
/// literal is silent, so we dispatch on the rule of the (single)
/// visible child : if it's another `not_expr`, this was the NOT
/// branch ; otherwise it's an `atom_or_parens`.
fn parse_not_expr(pair: Pair<Rule>) -> AstPredicate {
    let inner = pair.into_inner().next().expect("not_expr has a child");
    match inner.as_rule() {
        Rule::not_expr => AstPredicate::Not(Box::new(parse_not_expr(inner))),
        Rule::atom_or_parens => parse_atom_or_parens(inner),
        rule => unreachable!("unexpected not_expr child: {rule:?}"),
    }
}

/// `atom_or_parens = { "(" ~ predicate ~ ")" | atom }`. Either
/// descends back into a parenthesised predicate or dispatches an atom.
fn parse_atom_or_parens(pair: Pair<Rule>) -> AstPredicate {
    let inner = pair
        .into_inner()
        .next()
        .expect("atom_or_parens has a child");
    match inner.as_rule() {
        Rule::predicate => parse_predicate(inner),
        Rule::atom => parse_atom(inner),
        rule => unreachable!("unexpected atom_or_parens child: {rule:?}"),
    }
}

/// `atom = { distance_threshold | between_atom | in_atom |
/// is_null_atom | array_contains_atom | comparison_atom }`.
fn parse_atom(pair: Pair<Rule>) -> AstPredicate {
    let inner = pair.into_inner().next().expect("atom has a child");
    match inner.as_rule() {
        Rule::distance_threshold => parse_distance_threshold(inner),
        Rule::between_atom => parse_between_atom(inner),
        Rule::in_atom => parse_in_atom(inner),
        Rule::is_null_atom => parse_is_null_atom(inner),
        Rule::array_contains_atom => parse_array_contains_atom(inner),
        Rule::comparison_atom => parse_comparison_atom(inner),
        rule => unreachable!("unexpected atom child: {rule:?}"),
    }
}

/// `field_ref = { identifier ~ ("[" ~ string_literal ~ "]")? }`.
/// Builds an [`AstFieldRef`] capturing the field name plus the
/// optional bracketed subscript. Shared by every atom kind so the
/// language stays uniform.
fn parse_field_ref(pair: Pair<Rule>) -> AstFieldRef {
    let mut inner = pair.into_inner();
    let name = inner
        .next()
        .expect("field_ref has identifier")
        .as_str()
        .to_string();
    let subscript = inner.next().map(|p| parse_string_literal(&p));
    AstFieldRef { name, subscript }
}

/// `comparison_atom = { field_ref ~ comparison_op ~ atom_value }`.
/// Splits `=` into [`AstPredicate::Eq`] ; all other ops go to
/// [`AstPredicate::Cmp`].
fn parse_comparison_atom(pair: Pair<Rule>) -> AstPredicate {
    let mut inner = pair.into_inner();
    let field = parse_field_ref(inner.next().expect("comparison_atom has field_ref"));
    let op = parse_cmp_op(&inner.next().expect("comparison_atom has comparison_op"));
    let value = parse_atom_value(inner.next().expect("comparison_atom has atom_value"));
    if matches!(op, CmpOp::Eq) {
        AstPredicate::Eq(field, value)
    } else {
        AstPredicate::Cmp(field, op, value)
    }
}

/// `in_atom = { field_ref ~ ^"IN" ~ "(" ~ literal ~ ("," ~ literal)* ~ ")" }`.
fn parse_in_atom(pair: Pair<Rule>) -> AstPredicate {
    let mut inner = pair.into_inner();
    let field = parse_field_ref(inner.next().expect("in_atom has field_ref"));
    let values: Vec<AstLiteral> = inner.map(parse_literal).collect();
    AstPredicate::In(field, values)
}

/// `between_atom = { field_ref ~ ^"BETWEEN" ~ literal ~ ^"AND" ~ literal }`.
fn parse_between_atom(pair: Pair<Rule>) -> AstPredicate {
    let mut inner = pair.into_inner();
    let field = parse_field_ref(inner.next().expect("between_atom has field_ref"));
    let lo = parse_literal(inner.next().expect("between_atom has lo literal"));
    let hi = parse_literal(inner.next().expect("between_atom has hi literal"));
    AstPredicate::Between(field, lo, hi)
}

/// `is_null_atom = { field_ref ~ ^"IS" ~ is_null_negation? ~ ^"NULL" }`.
/// The optional `is_null_negation` is the only visible second child ;
/// presence detects the `IS NOT NULL` form.
fn parse_is_null_atom(pair: Pair<Rule>) -> AstPredicate {
    let mut inner = pair.into_inner();
    let field = parse_field_ref(inner.next().expect("is_null_atom has field_ref"));
    let negated = inner.next().is_some();
    AstPredicate::IsNull(field, negated)
}

/// `array_contains_atom = { field_ref ~ "@>" ~ literal }`.
fn parse_array_contains_atom(pair: Pair<Rule>) -> AstPredicate {
    let mut inner = pair.into_inner();
    let field = parse_field_ref(inner.next().expect("array_contains_atom has field_ref"));
    let value = parse_literal(inner.next().expect("array_contains_atom has literal"));
    AstPredicate::ArrayContains(field, value)
}

/// `distance_threshold = { distance_expr ~ comparison_op ~ number_literal }`.
/// The right side parses as `f32` directly because the planner
/// consumes it as a distance bound.
fn parse_distance_threshold(pair: Pair<Rule>) -> AstPredicate {
    let mut inner = pair.into_inner();
    let distance = parse_distance_expr(inner.next().expect("distance_threshold has distance_expr"));
    let op = parse_cmp_op(&inner.next().expect("distance_threshold has comparison_op"));
    let radius_pair = inner.next().expect("distance_threshold has number_literal");
    let radius: f32 = radius_pair
        .as_str()
        .parse()
        .expect("number_literal parses as f32 by grammar");
    AstPredicate::DistanceThreshold(distance, op, radius)
}

/// `distance_expr = { ^"embedding" ~ distance_op ~ param }`. The
/// `embedding` keyword is silent ; visible children are `distance_op`
/// and `param`.
fn parse_distance_expr(pair: Pair<Rule>) -> AstDistance {
    let mut inner = pair.into_inner();
    let metric = parse_distance_op(&inner.next().expect("distance_expr has distance_op"));
    let param = parse_param(inner.next().expect("distance_expr has param"));
    AstDistance { metric, param }
}

/// Map a `distance_op` pair to the typed enum.
fn parse_distance_op(pair: &Pair<Rule>) -> DistanceOp {
    match pair.as_str() {
        "<->" => DistanceOp::L2,
        "<=>" => DistanceOp::Cosine,
        "<#>" => DistanceOp::InnerProduct,
        other => unreachable!("unknown distance op: {other}"),
    }
}

/// Map a `comparison_op` pair to the typed enum. `!=` and `<>` both
/// produce [`CmpOp::Ne`].
fn parse_cmp_op(pair: &Pair<Rule>) -> CmpOp {
    match pair.as_str() {
        "=" => CmpOp::Eq,
        "<" => CmpOp::Lt,
        "<=" => CmpOp::Le,
        ">" => CmpOp::Gt,
        ">=" => CmpOp::Ge,
        "!=" | "<>" => CmpOp::Ne,
        other => unreachable!("unknown comparison op: {other}"),
    }
}

/// `atom_value = { literal | param }`. Dispatches by rule.
fn parse_atom_value(pair: Pair<Rule>) -> AstExpr {
    let inner = pair.into_inner().next().expect("atom_value has a child");
    match inner.as_rule() {
        Rule::literal => AstExpr::Literal(parse_literal(inner)),
        Rule::param => AstExpr::Param(parse_param(inner)),
        rule => unreachable!("unexpected atom_value child: {rule:?}"),
    }
}

/// `literal = { string_literal | number_literal | boolean_literal | null_literal }`.
fn parse_literal(pair: Pair<Rule>) -> AstLiteral {
    let inner = pair.into_inner().next().expect("literal has a child");
    match inner.as_rule() {
        Rule::string_literal => AstLiteral::String(parse_string_literal(&inner)),
        Rule::number_literal => parse_number_literal(&inner),
        Rule::boolean_literal => AstLiteral::Bool(parse_boolean_literal(&inner)),
        Rule::null_literal => AstLiteral::Null,
        rule => unreachable!("unexpected literal child: {rule:?}"),
    }
}

/// Strip the surrounding single quotes from a `string_literal` pair.
/// The grammar guarantees both quotes are present.
fn parse_string_literal(pair: &Pair<Rule>) -> String {
    let raw = pair.as_str();
    raw[1..raw.len() - 1].to_string()
}

/// Decide [`AstLiteral::I64`] vs [`AstLiteral::F64`] from the source
/// shape : presence of a decimal point splits them.
fn parse_number_literal(pair: &Pair<Rule>) -> AstLiteral {
    let s = pair.as_str();
    if s.contains('.') {
        AstLiteral::F64(s.parse().expect("number_literal parses as f64 by grammar"))
    } else {
        AstLiteral::I64(s.parse().expect("number_literal parses as i64 by grammar"))
    }
}

/// Case-insensitive boolean from the matched text.
fn parse_boolean_literal(pair: &Pair<Rule>) -> bool {
    pair.as_str().eq_ignore_ascii_case("true")
}

/// Test-only entry point : parse a bare predicate string. Wired
/// against the `test_predicate = { SOI ~ predicate ~ EOI }` anchor
/// so the test suite can exercise the predicate subtree before any
/// statement that uses WHERE has landed.
///
/// # Errors
///
/// Returns [`KovaQueryError::Parse`] for any input the predicate
/// grammar rejects.
#[cfg(test)]
fn parse_predicate_str(input: &str) -> Result<AstPredicate, KovaQueryError> {
    let mut pairs = KqlParser::parse(Rule::test_predicate, input)
        .map_err(|e| KovaQueryError::Parse(e.to_string()))?;
    let test_pred = pairs.next().expect("test_predicate present on Ok");
    let predicate = test_pred
        .into_inner()
        .next()
        .expect("test_predicate has predicate child");
    Ok(parse_predicate(predicate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_checkpoint_uppercase() {
        assert!(matches!(
            parse_str("CHECKPOINT"),
            Ok(AstStatement::Checkpoint)
        ));
    }

    #[test]
    fn parses_checkpoint_lowercase() {
        assert!(matches!(
            parse_str("checkpoint"),
            Ok(AstStatement::Checkpoint)
        ));
    }

    #[test]
    fn parses_checkpoint_mixed_case() {
        assert!(matches!(
            parse_str("Checkpoint"),
            Ok(AstStatement::Checkpoint)
        ));
    }

    #[test]
    fn parses_checkpoint_with_trailing_semicolon() {
        assert!(matches!(
            parse_str("CHECKPOINT;"),
            Ok(AstStatement::Checkpoint)
        ));
    }

    #[test]
    fn parses_checkpoint_with_surrounding_whitespace() {
        assert!(matches!(
            parse_str("  CHECKPOINT  "),
            Ok(AstStatement::Checkpoint)
        ));
    }

    #[test]
    fn rejects_empty_input() {
        assert!(matches!(parse_str(""), Err(KovaQueryError::Parse(_))));
    }

    #[test]
    fn rejects_garbage_input() {
        assert!(matches!(
            parse_str("not_a_keyword"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    #[test]
    fn rejects_trailing_garbage_after_checkpoint() {
        assert!(matches!(
            parse_str("CHECKPOINT foo"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    #[test]
    fn parses_vacuum_vectors() {
        let ast = parse_str("VACUUM vectors").expect("parse Ok");
        let AstStatement::Vacuum(AstVacuum { table }) = ast else {
            panic!("expected Vacuum");
        };
        assert_eq!(table, "vectors");
    }

    #[test]
    fn parses_vacuum_is_case_insensitive() {
        let ast = parse_str("vacuum vectors").expect("parse Ok");
        assert!(matches!(
            ast,
            AstStatement::Vacuum(AstVacuum { table }) if table == "vectors"
        ));
    }

    #[test]
    fn parses_vacuum_with_trailing_semicolon() {
        let ast = parse_str("VACUUM vectors;").expect("parse Ok");
        assert!(matches!(ast, AstStatement::Vacuum(_)));
    }

    #[test]
    fn parses_vacuum_preserves_identifier_case() {
        let ast = parse_str("VACUUM MyTable").expect("parse Ok");
        let AstStatement::Vacuum(AstVacuum { table }) = ast else {
            panic!("expected Vacuum");
        };
        assert_eq!(table, "MyTable");
    }

    #[test]
    fn parses_vacuum_accepts_underscores_and_digits_in_identifier() {
        let ast = parse_str("VACUUM my_table_2").expect("parse Ok");
        let AstStatement::Vacuum(AstVacuum { table }) = ast else {
            panic!("expected Vacuum");
        };
        assert_eq!(table, "my_table_2");
    }

    #[test]
    fn rejects_vacuum_without_table() {
        assert!(matches!(parse_str("VACUUM"), Err(KovaQueryError::Parse(_))));
    }

    #[test]
    fn rejects_vacuum_with_digit_leading_identifier() {
        assert!(matches!(
            parse_str("VACUUM 2vectors"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    // ----- INSERT -----

    #[test]
    fn parses_insert_with_positional_params() {
        let ast = parse_str("INSERT INTO vectors (id, embedding, metadata) VALUES ($1, $2, $3)")
            .expect("parse Ok");
        let AstStatement::Insert(AstInsert {
            table,
            columns,
            source,
        }) = ast
        else {
            panic!("expected Insert");
        };
        assert_eq!(table, "vectors");
        assert_eq!(columns, vec!["id", "embedding", "metadata"]);
        let AstInsertSource::Rows(rows) = source else {
            panic!("expected Rows");
        };
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].len(), 3);
        for (i, expr) in rows[0].iter().enumerate() {
            let AstExpr::Param(ParamRef::Positional(n)) = expr else {
                panic!("expected positional param");
            };
            assert_eq!(*n, u32::try_from(i + 1).unwrap());
        }
    }

    #[test]
    fn parses_insert_with_named_params() {
        let ast = parse_str(
            "INSERT INTO vectors (id, embedding, metadata) VALUES ($id, $embedding, $metadata)",
        )
        .expect("parse Ok");
        let AstStatement::Insert(AstInsert { source, .. }) = ast else {
            panic!("expected Insert");
        };
        let AstInsertSource::Rows(rows) = source else {
            panic!("expected Rows");
        };
        let names: Vec<_> = rows[0]
            .iter()
            .map(|e| {
                let AstExpr::Param(ParamRef::Named(s)) = e else {
                    panic!("expected named param");
                };
                s.clone()
            })
            .collect();
        assert_eq!(names, vec!["id", "embedding", "metadata"]);
    }

    #[test]
    fn parses_insert_with_mixed_param_forms() {
        let ast = parse_str("INSERT INTO vectors (id, embedding, metadata) VALUES ($1, $vec, $3)")
            .expect("parse Ok");
        let AstStatement::Insert(AstInsert { source, .. }) = ast else {
            panic!("expected Insert");
        };
        let AstInsertSource::Rows(rows) = source else {
            panic!("expected Rows");
        };
        assert!(matches!(
            rows[0][0],
            AstExpr::Param(ParamRef::Positional(1))
        ));
        assert!(matches!(rows[0][1], AstExpr::Param(ParamRef::Named(ref s)) if s == "vec"));
        assert!(matches!(
            rows[0][2],
            AstExpr::Param(ParamRef::Positional(3))
        ));
    }

    #[test]
    fn parses_insert_batch_form() {
        let ast =
            parse_str("INSERT INTO vectors (id, embedding, metadata) VALUES $1").expect("parse Ok");
        let AstStatement::Insert(AstInsert { source, .. }) = ast else {
            panic!("expected Insert");
        };
        assert!(matches!(
            source,
            AstInsertSource::Param(ParamRef::Positional(1))
        ));
    }

    #[test]
    fn parses_insert_batch_with_named_param() {
        let ast = parse_str("INSERT INTO vectors (id, embedding, metadata) VALUES $batch")
            .expect("parse Ok");
        let AstStatement::Insert(AstInsert { source, .. }) = ast else {
            panic!("expected Insert");
        };
        assert!(matches!(
            source,
            AstInsertSource::Param(ParamRef::Named(ref s)) if s == "batch"
        ));
    }

    #[test]
    fn parses_insert_is_case_insensitive_on_keywords() {
        let ast =
            parse_str("insert into vectors (id, embedding) values ($1, $2)").expect("parse Ok");
        assert!(matches!(ast, AstStatement::Insert(_)));
    }

    #[test]
    fn rejects_insert_without_values_clause() {
        assert!(matches!(
            parse_str("INSERT INTO vectors (id, embedding, metadata)"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    #[test]
    fn rejects_insert_without_column_list() {
        assert!(matches!(
            parse_str("INSERT INTO vectors VALUES ($1)"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    #[test]
    fn rejects_insert_with_whitespace_in_param() {
        // `$ 1` shouldn't parse : param is compound-atomic.
        assert!(matches!(
            parse_str("INSERT INTO vectors (id) VALUES ($ 1)"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    // ----- CREATE INDEX -----

    #[test]
    fn parses_create_index_with_name() {
        let ast =
            parse_str("CREATE INDEX idx_cat ON vectors USING HASH (category)").expect("parse Ok");
        let AstStatement::CreateIndex(AstCreateIndex {
            name,
            table,
            method,
            field,
        }) = ast
        else {
            panic!("expected CreateIndex");
        };
        assert_eq!(name.as_deref(), Some("idx_cat"));
        assert_eq!(table, "vectors");
        assert_eq!(method, IndexMethod::Hash);
        assert_eq!(field, "category");
    }

    #[test]
    fn parses_create_index_without_name() {
        let ast = parse_str("CREATE INDEX ON vectors USING BTREE (year)").expect("parse Ok");
        let AstStatement::CreateIndex(AstCreateIndex {
            name,
            table,
            method,
            field,
        }) = ast
        else {
            panic!("expected CreateIndex");
        };
        assert_eq!(name, None);
        assert_eq!(table, "vectors");
        assert_eq!(method, IndexMethod::Btree);
        assert_eq!(field, "year");
    }

    #[test]
    fn parses_create_index_inverted() {
        let ast = parse_str("CREATE INDEX ON vectors USING INVERTED (tags)").expect("parse Ok");
        assert!(matches!(
            ast,
            AstStatement::CreateIndex(AstCreateIndex {
                method: IndexMethod::Inverted,
                ..
            })
        ));
    }

    #[test]
    fn parses_create_index_method_is_case_insensitive() {
        let ast = parse_str("CREATE INDEX ON vectors USING hash (category)").expect("parse Ok");
        assert!(matches!(
            ast,
            AstStatement::CreateIndex(AstCreateIndex {
                method: IndexMethod::Hash,
                ..
            })
        ));
    }

    #[test]
    fn rejects_create_index_without_using_clause() {
        assert!(matches!(
            parse_str("CREATE INDEX ON vectors (category)"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    #[test]
    fn rejects_create_index_with_unknown_method() {
        assert!(matches!(
            parse_str("CREATE INDEX ON vectors USING BLOOM (category)"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    // ----- DROP INDEX -----

    #[test]
    fn parses_drop_index() {
        let ast = parse_str("DROP INDEX idx_cat ON vectors").expect("parse Ok");
        let AstStatement::DropIndex(AstDropIndex { name, table }) = ast else {
            panic!("expected DropIndex");
        };
        assert_eq!(name, "idx_cat");
        assert_eq!(table, "vectors");
    }

    #[test]
    fn rejects_drop_index_without_name() {
        assert!(matches!(
            parse_str("DROP INDEX ON vectors"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    // ----- Predicates : comparison_atom -----

    #[test]
    fn predicate_eq_with_integer_literal() {
        let p = parse_predicate_str("id = 5").expect("parse Ok");
        let AstPredicate::Eq(field, AstExpr::Literal(AstLiteral::I64(5))) = p else {
            panic!("expected Eq(id, 5), got {p:?}");
        };
        assert_eq!(field.name, "id");
        assert!(field.subscript.is_none());
    }

    #[test]
    fn predicate_eq_with_string_literal() {
        let p = parse_predicate_str("name = 'hello'").expect("parse Ok");
        let AstPredicate::Eq(field, AstExpr::Literal(AstLiteral::String(s))) = p else {
            panic!("expected Eq with string, got {p:?}");
        };
        assert_eq!(field.name, "name");
        assert_eq!(s, "hello");
    }

    #[test]
    fn predicate_eq_with_positional_param() {
        let p = parse_predicate_str("id = $1").expect("parse Ok");
        let AstPredicate::Eq(field, AstExpr::Param(ParamRef::Positional(1))) = p else {
            panic!("expected Eq with param, got {p:?}");
        };
        assert_eq!(field.name, "id");
    }

    #[test]
    fn predicate_eq_with_named_param() {
        let p = parse_predicate_str("id = $target").expect("parse Ok");
        let AstPredicate::Eq(_, AstExpr::Param(ParamRef::Named(s))) = p else {
            panic!("expected Eq with named param, got {p:?}");
        };
        assert_eq!(s, "target");
    }

    #[test]
    fn predicate_eq_with_boolean_literal() {
        let p = parse_predicate_str("pinned = TRUE").expect("parse Ok");
        let AstPredicate::Eq(_, AstExpr::Literal(AstLiteral::Bool(true))) = p else {
            panic!("expected Eq with true, got {p:?}");
        };
    }

    #[test]
    fn predicate_eq_with_null_literal() {
        let p = parse_predicate_str("category = NULL").expect("parse Ok");
        let AstPredicate::Eq(_, AstExpr::Literal(AstLiteral::Null)) = p else {
            panic!("expected Eq with NULL, got {p:?}");
        };
    }

    #[test]
    fn predicate_eq_with_float_literal() {
        let p = parse_predicate_str("score = 2.5").expect("parse Ok");
        let AstPredicate::Eq(_, AstExpr::Literal(AstLiteral::F64(f))) = p else {
            panic!("expected Eq with f64, got {p:?}");
        };
        assert!((f - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn predicate_eq_with_negative_integer() {
        let p = parse_predicate_str("delta = -5").expect("parse Ok");
        let AstPredicate::Eq(_, AstExpr::Literal(AstLiteral::I64(-5))) = p else {
            panic!("expected Eq with -5, got {p:?}");
        };
    }

    #[test]
    fn predicate_cmp_lt() {
        let p = parse_predicate_str("year < 2024").expect("parse Ok");
        assert!(matches!(p, AstPredicate::Cmp(_, CmpOp::Lt, _)));
    }

    #[test]
    fn predicate_cmp_le() {
        let p = parse_predicate_str("year <= 2024").expect("parse Ok");
        assert!(matches!(p, AstPredicate::Cmp(_, CmpOp::Le, _)));
    }

    #[test]
    fn predicate_cmp_gt() {
        let p = parse_predicate_str("year > 2024").expect("parse Ok");
        assert!(matches!(p, AstPredicate::Cmp(_, CmpOp::Gt, _)));
    }

    #[test]
    fn predicate_cmp_ge() {
        let p = parse_predicate_str("year >= 2024").expect("parse Ok");
        assert!(matches!(p, AstPredicate::Cmp(_, CmpOp::Ge, _)));
    }

    #[test]
    fn predicate_cmp_ne_bang() {
        let p = parse_predicate_str("year != 2024").expect("parse Ok");
        assert!(matches!(p, AstPredicate::Cmp(_, CmpOp::Ne, _)));
    }

    #[test]
    fn predicate_cmp_ne_angle() {
        let p = parse_predicate_str("year <> 2024").expect("parse Ok");
        assert!(matches!(p, AstPredicate::Cmp(_, CmpOp::Ne, _)));
    }

    // ----- Predicates : other atom shapes -----

    #[test]
    fn predicate_in_list_with_strings() {
        let p = parse_predicate_str("category IN ('docs', 'specs', 'rfcs')").expect("parse Ok");
        let AstPredicate::In(field, values) = p else {
            panic!("expected In, got {p:?}");
        };
        assert_eq!(field.name, "category");
        assert_eq!(values.len(), 3);
        assert!(matches!(values[0], AstLiteral::String(ref s) if s == "docs"));
    }

    #[test]
    fn predicate_in_list_with_integers() {
        let p = parse_predicate_str("year IN (2023, 2024, 2025)").expect("parse Ok");
        let AstPredicate::In(_, values) = p else {
            panic!("expected In, got {p:?}");
        };
        assert!(matches!(values[0], AstLiteral::I64(2023)));
    }

    #[test]
    fn predicate_between() {
        let p = parse_predicate_str("score BETWEEN 0.5 AND 1.0").expect("parse Ok");
        let AstPredicate::Between(field, AstLiteral::F64(lo), AstLiteral::F64(hi)) = p else {
            panic!("expected Between, got {p:?}");
        };
        assert_eq!(field.name, "score");
        assert!((lo - 0.5).abs() < f64::EPSILON);
        assert!((hi - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn predicate_is_null() {
        let p = parse_predicate_str("category IS NULL").expect("parse Ok");
        let AstPredicate::IsNull(field, negated) = p else {
            panic!("expected IsNull, got {p:?}");
        };
        assert_eq!(field.name, "category");
        assert!(!negated);
    }

    #[test]
    fn predicate_is_not_null() {
        let p = parse_predicate_str("category IS NOT NULL").expect("parse Ok");
        let AstPredicate::IsNull(_, negated) = p else {
            panic!("expected IsNull, got {p:?}");
        };
        assert!(negated);
    }

    #[test]
    fn predicate_array_contains() {
        let p = parse_predicate_str("tags @> 'rust'").expect("parse Ok");
        let AstPredicate::ArrayContains(field, AstLiteral::String(value)) = p else {
            panic!("expected ArrayContains, got {p:?}");
        };
        assert_eq!(field.name, "tags");
        assert_eq!(value, "rust");
    }

    // ----- Predicates : subscripted field references -----

    /// `WHERE attrs['country'] = 'IN'` parses with the subscript
    /// captured on the field side.
    #[test]
    fn predicate_eq_with_subscripted_field() {
        let p = parse_predicate_str("attrs['country'] = 'IN'").expect("parse Ok");
        let AstPredicate::Eq(field, AstExpr::Literal(AstLiteral::String(s))) = p else {
            panic!("expected Eq, got {p:?}");
        };
        assert_eq!(field.name, "attrs");
        assert_eq!(field.subscript.as_deref(), Some("country"));
        assert_eq!(s, "IN");
    }

    /// Subscripts work on every atom kind, not just `Eq`. Spot-check
    /// `Cmp`, `In`, `Between`, `IsNull`, `@>`.
    #[test]
    fn predicate_subscripts_work_across_all_atoms() {
        let cases = [
            "attrs['score'] > 0.5",
            "attrs['cat'] IN ('a', 'b')",
            "attrs['year'] BETWEEN 2000 AND 2024",
            "attrs['phone'] IS NOT NULL",
            "attrs['tags'] @> 'rust'",
        ];
        for q in cases {
            let p =
                parse_predicate_str(q).unwrap_or_else(|e| panic!("parse failed for `{q}` : {e:?}"));
            let (AstPredicate::Cmp(field, _, _)
            | AstPredicate::In(field, _)
            | AstPredicate::Between(field, _, _)
            | AstPredicate::IsNull(field, _)
            | AstPredicate::ArrayContains(field, _)) = &p
            else {
                panic!("unexpected variant for `{q}` : {p:?}");
            };
            assert_eq!(field.name, "attrs", "wrong name for `{q}`");
            assert!(
                field.subscript.is_some(),
                "subscript dropped for `{q}` : {p:?}"
            );
        }
    }

    // ----- Predicates : distance threshold -----

    #[test]
    fn predicate_distance_threshold_l2() {
        let p = parse_predicate_str("embedding <-> $1 < 0.5").expect("parse Ok");
        let AstPredicate::DistanceThreshold(dist, op, radius) = p else {
            panic!("expected DistanceThreshold, got {p:?}");
        };
        assert_eq!(dist.metric, DistanceOp::L2);
        assert!(matches!(dist.param, ParamRef::Positional(1)));
        assert_eq!(op, CmpOp::Lt);
        assert!((radius - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn predicate_distance_threshold_cosine() {
        let p = parse_predicate_str("embedding <=> $query <= 0.3").expect("parse Ok");
        let AstPredicate::DistanceThreshold(dist, _, _) = p else {
            panic!("expected DistanceThreshold, got {p:?}");
        };
        assert_eq!(dist.metric, DistanceOp::Cosine);
        assert!(matches!(dist.param, ParamRef::Named(ref s) if s == "query"));
    }

    #[test]
    fn predicate_distance_threshold_inner_product() {
        let p = parse_predicate_str("embedding <#> $1 > -0.1").expect("parse Ok");
        let AstPredicate::DistanceThreshold(dist, _, radius) = p else {
            panic!("expected DistanceThreshold, got {p:?}");
        };
        assert_eq!(dist.metric, DistanceOp::InnerProduct);
        assert!((radius - (-0.1)).abs() < f32::EPSILON);
    }

    // ----- Predicates : boolean combinators -----

    #[test]
    fn predicate_and_two_terms() {
        let p = parse_predicate_str("a = 1 AND b = 2").expect("parse Ok");
        let AstPredicate::And(children) = p else {
            panic!("expected And, got {p:?}");
        };
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn predicate_or_two_terms() {
        let p = parse_predicate_str("a = 1 OR b = 2").expect("parse Ok");
        let AstPredicate::Or(children) = p else {
            panic!("expected Or, got {p:?}");
        };
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn predicate_and_flattens_three_terms() {
        let p = parse_predicate_str("a = 1 AND b = 2 AND c = 3").expect("parse Ok");
        let AstPredicate::And(children) = p else {
            panic!("expected And, got {p:?}");
        };
        assert_eq!(children.len(), 3, "AND chain should flatten");
    }

    #[test]
    fn predicate_not_single_atom() {
        let p = parse_predicate_str("NOT a = 1").expect("parse Ok");
        let AstPredicate::Not(inner) = p else {
            panic!("expected Not, got {p:?}");
        };
        assert!(matches!(*inner, AstPredicate::Eq(_, _)));
    }

    #[test]
    fn predicate_double_negation() {
        let p = parse_predicate_str("NOT NOT a = 1").expect("parse Ok");
        let AstPredicate::Not(outer) = p else {
            panic!("expected outer Not, got {p:?}");
        };
        assert!(matches!(*outer, AstPredicate::Not(_)));
    }

    #[test]
    fn predicate_precedence_and_binds_tighter_than_or() {
        // `a = 1 AND b = 2 OR c = 3` parses as `(a AND b) OR c`.
        let p = parse_predicate_str("a = 1 AND b = 2 OR c = 3").expect("parse Ok");
        let AstPredicate::Or(children) = p else {
            panic!("expected top-level Or, got {p:?}");
        };
        assert_eq!(children.len(), 2);
        assert!(
            matches!(children[0], AstPredicate::And(_)),
            "left of OR should be the AND group"
        );
        assert!(matches!(children[1], AstPredicate::Eq(_, _)));
    }

    #[test]
    fn predicate_parens_override_precedence() {
        // `(a = 1 OR b = 2) AND c = 3` parses as And(Or, Eq).
        let p = parse_predicate_str("(a = 1 OR b = 2) AND c = 3").expect("parse Ok");
        let AstPredicate::And(children) = p else {
            panic!("expected top-level And, got {p:?}");
        };
        assert!(matches!(children[0], AstPredicate::Or(_)));
        assert!(matches!(children[1], AstPredicate::Eq(_, _)));
    }

    #[test]
    fn predicate_single_atom_does_not_wrap_in_combinators() {
        // A bare atom should NOT come out as `And([atom])` or `Or([atom])`.
        let p = parse_predicate_str("x = 1").expect("parse Ok");
        assert!(matches!(p, AstPredicate::Eq(_, _)));
    }

    // ----- Predicates : error paths -----

    #[test]
    fn rejects_predicate_missing_operator() {
        assert!(matches!(
            parse_predicate_str("x"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    #[test]
    fn rejects_predicate_unbalanced_parens() {
        assert!(matches!(
            parse_predicate_str("(x = 1"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    #[test]
    fn rejects_keyword_as_field_name() {
        // `AND` is a reserved keyword ; can't be an identifier.
        assert!(matches!(
            parse_predicate_str("AND = 1"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    // ----- DELETE -----

    #[test]
    fn parses_delete_by_id_equality() {
        let ast = parse_str("DELETE FROM vectors WHERE id = $1").expect("parse Ok");
        let AstStatement::Delete(AstDelete { table, predicate }) = ast else {
            panic!("expected Delete");
        };
        assert_eq!(table, "vectors");
        let AstPredicate::Eq(field, AstExpr::Param(ParamRef::Positional(1))) = predicate else {
            panic!("expected Eq(id, $1), got {predicate:?}");
        };
        assert_eq!(field.name, "id");
    }

    #[test]
    fn parses_delete_with_compound_predicate() {
        let ast = parse_str("DELETE FROM vectors WHERE category = 'archived' AND year < 2020")
            .expect("parse Ok");
        let AstStatement::Delete(AstDelete { predicate, .. }) = ast else {
            panic!("expected Delete");
        };
        let AstPredicate::And(children) = predicate else {
            panic!("expected And, got {predicate:?}");
        };
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn parses_delete_with_in_list() {
        let ast =
            parse_str("DELETE FROM vectors WHERE category IN ('old', 'archived', 'deprecated')")
                .expect("parse Ok");
        let AstStatement::Delete(AstDelete { predicate, .. }) = ast else {
            panic!("expected Delete");
        };
        let AstPredicate::In(field, values) = predicate else {
            panic!("expected In, got {predicate:?}");
        };
        assert_eq!(field.name, "category");
        assert_eq!(values.len(), 3);
    }

    #[test]
    fn parses_delete_with_is_not_null() {
        let ast = parse_str("DELETE FROM vectors WHERE deleted_at IS NOT NULL").expect("parse Ok");
        let AstStatement::Delete(AstDelete { predicate, .. }) = ast else {
            panic!("expected Delete");
        };
        assert!(matches!(predicate, AstPredicate::IsNull(_, true)));
    }

    #[test]
    fn parses_delete_with_trailing_semicolon() {
        let ast = parse_str("DELETE FROM vectors WHERE id = $1;").expect("parse Ok");
        assert!(matches!(ast, AstStatement::Delete(_)));
    }

    #[test]
    fn parses_delete_is_case_insensitive_on_keywords() {
        let ast = parse_str("delete from vectors where id = $1").expect("parse Ok");
        assert!(matches!(ast, AstStatement::Delete(_)));
    }

    #[test]
    fn rejects_delete_without_where_clause() {
        // Mandatory-WHERE is the safety guard against accidental
        // table-wide deletes.
        assert!(matches!(
            parse_str("DELETE FROM vectors"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    #[test]
    fn rejects_delete_without_from_keyword() {
        assert!(matches!(
            parse_str("DELETE vectors WHERE id = $1"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    #[test]
    fn rejects_delete_without_table() {
        assert!(matches!(
            parse_str("DELETE FROM WHERE id = $1"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    // ----- UPDATE -----

    #[test]
    fn parses_update_whole_metadata_replace() {
        let ast = parse_str("UPDATE vectors SET metadata = $1 WHERE id = $2").expect("parse Ok");
        let AstStatement::Update(AstUpdate {
            table,
            assignments,
            predicate,
        }) = ast
        else {
            panic!("expected Update");
        };
        assert_eq!(table, "vectors");
        assert_eq!(assignments.len(), 1);
        let AstAssignment {
            field,
            subscript,
            value,
        } = &assignments[0];
        assert_eq!(field, "metadata");
        assert_eq!(subscript.as_deref(), None);
        assert!(matches!(value, AstExpr::Param(ParamRef::Positional(1))));
        assert!(matches!(predicate, AstPredicate::Eq(_, _)));
    }

    #[test]
    fn parses_update_with_subscript_patch() {
        let ast = parse_str("UPDATE vectors SET metadata['priority'] = 'high' WHERE id = $1")
            .expect("parse Ok");
        let AstStatement::Update(AstUpdate { assignments, .. }) = ast else {
            panic!("expected Update");
        };
        let AstAssignment {
            field,
            subscript,
            value,
        } = &assignments[0];
        assert_eq!(field, "metadata");
        assert_eq!(subscript.as_deref(), Some("priority"));
        assert!(matches!(
            value,
            AstExpr::Literal(AstLiteral::String(s)) if s == "high"
        ));
    }

    #[test]
    fn parses_update_with_multiple_assignments() {
        let ast =
            parse_str("UPDATE vectors SET metadata['a'] = 'x', metadata['b'] = 'y' WHERE id = $1")
                .expect("parse Ok");
        let AstStatement::Update(AstUpdate { assignments, .. }) = ast else {
            panic!("expected Update");
        };
        assert_eq!(assignments.len(), 2);
        assert_eq!(assignments[0].subscript.as_deref(), Some("a"));
        assert_eq!(assignments[1].subscript.as_deref(), Some("b"));
    }

    #[test]
    fn parses_update_is_case_insensitive_on_keywords() {
        let ast = parse_str("update vectors set metadata = $1 where id = $2").expect("parse Ok");
        assert!(matches!(ast, AstStatement::Update(_)));
    }

    #[test]
    fn rejects_update_without_set_clause() {
        assert!(matches!(
            parse_str("UPDATE vectors WHERE id = $1"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    #[test]
    fn rejects_update_without_where_clause() {
        assert!(matches!(
            parse_str("UPDATE vectors SET metadata = $1"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    #[test]
    fn rejects_update_with_empty_set_list() {
        assert!(matches!(
            parse_str("UPDATE vectors SET WHERE id = $1"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    // ----- SELECT -----

    #[test]
    fn parses_select_star() {
        let ast = parse_str("SELECT * FROM vectors").expect("parse Ok");
        let AstStatement::Select(q) = ast else {
            panic!("expected Select");
        };
        assert_eq!(q.projection.len(), 1);
        assert!(matches!(q.projection[0], AstProjection::Wildcard));
        assert_eq!(q.from_table, "vectors");
        assert!(q.predicate.is_none());
        assert!(q.order_by.is_empty());
        assert_eq!(q.limit, None);
    }

    #[test]
    fn parses_select_id_and_metadata_route_to_typed_variants() {
        let ast = parse_str("SELECT id, metadata FROM vectors").expect("parse Ok");
        let AstStatement::Select(q) = ast else {
            panic!("expected Select");
        };
        assert!(matches!(q.projection[0], AstProjection::Id));
        assert!(matches!(q.projection[1], AstProjection::Metadata));
    }

    #[test]
    fn parses_select_regular_field_uses_field_variant() {
        let ast = parse_str("SELECT category, year FROM vectors").expect("parse Ok");
        let AstStatement::Select(q) = ast else {
            panic!("expected Select");
        };
        let names: Vec<String> = q
            .projection
            .iter()
            .map(|p| match p {
                AstProjection::Field(s) => s.clone(),
                other => panic!("expected Field, got {other:?}"),
            })
            .collect();
        assert_eq!(names, vec!["category", "year"]);
    }

    #[test]
    fn parses_select_distance_expression_with_alias() {
        let ast =
            parse_str("SELECT embedding <-> $query AS distance FROM vectors").expect("parse Ok");
        let AstStatement::Select(q) = ast else {
            panic!("expected Select");
        };
        let AstProjection::Aliased(inner, alias) = &q.projection[0] else {
            panic!("expected Aliased");
        };
        assert_eq!(alias, "distance");
        let AstProjection::DistanceExpr(dist) = inner.as_ref() else {
            panic!("expected DistanceExpr inside Aliased");
        };
        assert_eq!(dist.metric, DistanceOp::L2);
        assert!(matches!(dist.param, ParamRef::Named(ref s) if s == "query"));
    }

    #[test]
    fn parses_select_count_star() {
        let ast =
            parse_str("SELECT COUNT(*) FROM vectors WHERE category = 'docs'").expect("parse Ok");
        let AstStatement::Select(q) = ast else {
            panic!("expected Select");
        };
        assert!(matches!(q.projection[0], AstProjection::CountStar));
        assert!(q.predicate.is_some());
    }

    #[test]
    fn parses_select_with_where_clause() {
        let ast = parse_str("SELECT id FROM vectors WHERE id = $1").expect("parse Ok");
        let AstStatement::Select(q) = ast else {
            panic!("expected Select");
        };
        assert!(matches!(q.predicate, Some(AstPredicate::Eq(_, _))));
    }

    #[test]
    fn parses_select_with_order_by_field_desc() {
        let ast = parse_str("SELECT id FROM vectors ORDER BY year DESC").expect("parse Ok");
        let AstStatement::Select(q) = ast else {
            panic!("expected Select");
        };
        assert_eq!(q.order_by.len(), 1);
        let AstOrderBy::Field(name, dir) = &q.order_by[0] else {
            panic!("expected Field ordering");
        };
        assert_eq!(name, "year");
        assert_eq!(*dir, OrderDir::Desc);
    }

    #[test]
    fn parses_select_with_order_by_distance_default_asc() {
        let ast = parse_str("SELECT id FROM vectors ORDER BY embedding <-> $1").expect("parse Ok");
        let AstStatement::Select(q) = ast else {
            panic!("expected Select");
        };
        let AstOrderBy::Distance(dist, dir) = &q.order_by[0] else {
            panic!("expected Distance ordering");
        };
        assert_eq!(dist.metric, DistanceOp::L2);
        assert_eq!(*dir, OrderDir::Asc, "missing direction defaults to Asc");
    }

    #[test]
    fn parses_select_with_multiple_order_by_items() {
        let ast =
            parse_str("SELECT id FROM vectors ORDER BY year DESC, score ASC").expect("parse Ok");
        let AstStatement::Select(q) = ast else {
            panic!("expected Select");
        };
        assert_eq!(q.order_by.len(), 2);
    }

    #[test]
    fn parses_select_with_limit() {
        let ast = parse_str("SELECT id FROM vectors LIMIT 100").expect("parse Ok");
        let AstStatement::Select(q) = ast else {
            panic!("expected Select");
        };
        assert_eq!(q.limit, Some(100));
    }

    #[test]
    fn parses_select_full_hybrid_query() {
        // The canonical hybrid kNN + predicate + ranking query.
        let ast = parse_str(
            "SELECT id, embedding <-> $1 AS distance, metadata FROM vectors \
             WHERE category = 'docs' AND year >= 2024 \
             ORDER BY embedding <-> $1 LIMIT 10",
        )
        .expect("parse Ok");
        let AstStatement::Select(q) = ast else {
            panic!("expected Select");
        };
        assert_eq!(q.projection.len(), 3);
        assert!(matches!(q.projection[0], AstProjection::Id));
        assert!(matches!(q.projection[1], AstProjection::Aliased(_, _)));
        assert!(matches!(q.projection[2], AstProjection::Metadata));
        assert_eq!(q.from_table, "vectors");
        assert!(matches!(q.predicate, Some(AstPredicate::And(_))));
        assert_eq!(q.order_by.len(), 1);
        assert_eq!(q.limit, Some(10));
    }

    #[test]
    fn parses_select_is_case_insensitive_on_keywords() {
        let ast = parse_str("select id from vectors where id = $1 order by year desc limit 5")
            .expect("parse Ok");
        assert!(matches!(ast, AstStatement::Select(_)));
    }

    #[test]
    fn rejects_select_without_from_clause() {
        assert!(matches!(
            parse_str("SELECT id"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    #[test]
    fn rejects_select_with_empty_projection() {
        assert!(matches!(
            parse_str("SELECT FROM vectors"),
            Err(KovaQueryError::Parse(_))
        ));
    }

    #[test]
    fn rejects_select_with_non_integer_limit() {
        assert!(matches!(
            parse_str("SELECT id FROM vectors LIMIT 1.5"),
            Err(KovaQueryError::Parse(_))
        ));
    }
}

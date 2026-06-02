//! Pest-driven parser : `String` -> [`AstStatement`].

use pest::Parser;
use pest::iterators::Pair;

use crate::ast::{
    AstCreateIndex, AstDropIndex, AstExpr, AstInsert, AstInsertSource, AstStatement, AstVacuum,
    IndexMethod, ParamRef,
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
        Rule::create_index_stmt => Ok(AstStatement::CreateIndex(parse_create_index(inner))),
        Rule::drop_index_stmt => Ok(AstStatement::DropIndex(parse_drop_index(inner))),
        rule => unreachable!("unexpected statement variant: {rule:?}"),
    }
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
}

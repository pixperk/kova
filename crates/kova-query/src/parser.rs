//! Pest-driven parser : `String` -> [`AstStatement`].

use pest::Parser;

use crate::ast::AstStatement;
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
        rule => unreachable!("unexpected statement variant: {rule:?}"),
    }
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
}

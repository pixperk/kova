//! Pest-driven parser : `String` -> [`AstStatement`].
//!
//! The grammar lives in `grammar.pest`. Productions are added
//! incrementally as parser sub-milestones land. Each production must
//! ship with at least one happy-path test and one error-path test
//! before the next one is added.

use crate::ast::AstStatement;
use crate::error::KovaQueryError;

/// Parse a single KQL statement from a string into its AST.
///
/// Returns [`KovaQueryError::Parse`] with line/column on syntax
/// failure. Semantic failures (unknown field, type mismatch, etc.)
/// are not detected here ; they belong to the binder.
///
/// # Errors
///
/// Returns [`KovaQueryError::Parse`] for any input the grammar
/// rejects.
pub fn parse_str(_input: &str) -> Result<AstStatement, KovaQueryError> {
    Err(KovaQueryError::Parse(
        "parser not yet implemented : grammar productions land in M1.1 steps 2+".into(),
    ))
}

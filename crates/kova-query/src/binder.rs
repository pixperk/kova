//! AST -> [`LogicalStatement`] : field resolution, type checks,
//! predicate normalisation, and the hard semantic rejects (embedding
//! update, v2-only statements, distance ordering direction, etc.).
//!
//! The binder is stateless in v1 because the schema is inferred ;
//! v2 grows a context carrying the strict-schema registry.

use crate::ast::AstStatement;
use crate::error::KovaQueryError;
use crate::logical::LogicalStatement;

/// Bind an [`AstStatement`] into a [`LogicalStatement`].
///
/// # Errors
///
/// Returns [`KovaQueryError::Bind`] for any semantic violation :
/// unknown field, type mismatch, embedding update, v2-only DDL,
/// distance ordering with `DESC`, wildcard in a list, etc.
// `ast` is consumed once real binders land and start moving fields
// into LogicalStatement payloads ; today CHECKPOINT is the only arm
// that does any real work, so clippy thinks the parameter is wasted.
#[allow(clippy::needless_pass_by_value)]
pub fn bind(ast: AstStatement) -> Result<LogicalStatement, KovaQueryError> {
    match ast {
        AstStatement::Checkpoint => Ok(LogicalStatement::Checkpoint),

        // Filled in as each binder lands. Explicit arms (rather than
        // a `_` catchall) so the compiler complains the moment a new
        // AST variant is added without a binder.
        AstStatement::Vacuum(_) => unimplemented(StatementKind::Vacuum),
        AstStatement::Insert(_) => unimplemented(StatementKind::Insert),
        AstStatement::Update(_) => unimplemented(StatementKind::Update),
        AstStatement::Delete(_) => unimplemented(StatementKind::Delete),
        AstStatement::Select(_) => unimplemented(StatementKind::Select),
        AstStatement::CreateIndex(_) => unimplemented(StatementKind::CreateIndex),
        AstStatement::DropIndex(_) => unimplemented(StatementKind::DropIndex),
    }
}

/// Shape of the not-yet-implemented stub so the error message stays
/// consistent across statement variants.
#[derive(Debug, Clone, Copy)]
enum StatementKind {
    Vacuum,
    Insert,
    Update,
    Delete,
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

    /// Until other binders land, every non-CHECKPOINT statement
    /// reports a clean "not yet implemented" Bind error instead of
    /// panicking. This guards against a future refactor that
    /// accidentally turns the dispatch into a `todo!()`.
    #[test]
    fn unimplemented_variants_return_bind_error() {
        let ast = parse_str("VACUUM vectors").expect("parse Ok");
        let err = bind(ast).expect_err("expected Bind error");
        assert!(
            matches!(err, KovaQueryError::Bind(_)),
            "expected Bind, got {err:?}"
        );
    }
}

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
pub fn bind(_ast: AstStatement) -> Result<LogicalStatement, KovaQueryError> {
    Err(KovaQueryError::Bind("binder not yet implemented".into()))
}

//! AST -> [`LogicalStatement`] : field resolution, type checks,
//! predicate normalisation, and the hard semantic rejects (embedding
//! update, v2-only statements, distance ordering direction, etc.).
//!
//! The binder is stateless in v1 because the schema is inferred ;
//! v2 grows a context carrying the strict-schema registry.

use crate::ast::{AstStatement, AstVacuum};
use crate::error::KovaQueryError;
use crate::logical::{LogicalStatement, LogicalVacuum};

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
        AstStatement::Vacuum(v) => bind_vacuum(v),

        // Filled in as each binder lands. Explicit arms (rather than
        // a `_` catchall) so the compiler complains the moment a new
        // AST variant is added without a binder.
        AstStatement::Insert(_) => unimplemented(StatementKind::Insert),
        AstStatement::Update(_) => unimplemented(StatementKind::Update),
        AstStatement::Delete(_) => unimplemented(StatementKind::Delete),
        AstStatement::Select(_) => unimplemented(StatementKind::Select),
        AstStatement::CreateIndex(_) => unimplemented(StatementKind::CreateIndex),
        AstStatement::DropIndex(_) => unimplemented(StatementKind::DropIndex),
    }
}

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

/// Shape of the not-yet-implemented stub so the error message stays
/// consistent across statement variants. Shrinks as each statement
/// binder lands ; the whole helper goes away when the last variant
/// is wired up.
#[derive(Debug, Clone, Copy)]
enum StatementKind {
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

    /// Every statement type without a real binder yet must report a
    /// clean Bind error, not panic. INSERT is the chosen probe today
    /// because VACUUM moved out of that bucket in step 3.
    #[test]
    fn unimplemented_variants_return_bind_error() {
        let ast = parse_str("INSERT INTO vectors (id, embedding, metadata) VALUES ($1, $2, $3)")
            .expect("parse Ok");
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
        // Binder doesn't case-fold table names ; the executor sees
        // exactly what the user wrote and matches it against whatever
        // Shard catalog it's dispatching against.
        let ast = parse_str("VACUUM MyShard").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Vacuum(LogicalVacuum { table }) = logical else {
            panic!("expected Vacuum");
        };
        assert_eq!(table, "MyShard");
    }

    #[test]
    fn binds_vacuum_accepts_arbitrary_table_name() {
        // The binder doesn't gate against any catalog ; any
        // grammatically-valid identifier flows through. Unknown-table
        // rejection is a runtime concern (executor, see Shard
        // dispatcher) not a bind concern.
        let ast = parse_str("VACUUM products").expect("parse Ok");
        let logical = bind(ast).expect("bind Ok");
        let LogicalStatement::Vacuum(LogicalVacuum { table }) = logical else {
            panic!("expected Vacuum");
        };
        assert_eq!(table, "products");
    }
}

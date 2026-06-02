//! Abstract syntax tree for parsed KQL input.
//!
//! The AST is the boundary between parsing concerns (syntax) and
//! planner concerns (semantics). Built by [`crate::parser::parse_str`]
//! from Pest pairs ; consumed by the binder to produce a
//! `LogicalStatement` (later module).
//!
//! The AST is *permissive* : it accepts every shape the grammar can
//! produce, even semantically wrong ones. "Unknown field" and "type
//! mismatch" are binder errors, not parser errors. Keeping the AST
//! permissive keeps the parser context-free.

/// Top-level parsed KQL statement.
#[derive(Debug, Clone)]
pub enum AstStatement {
    /// `SELECT ... FROM vectors ...`
    Select(AstQuery),
    /// `INSERT INTO vectors (...) VALUES ...`
    Insert(AstInsert),
    /// `UPDATE vectors SET ... WHERE ...`
    Update(AstUpdate),
    /// `DELETE FROM vectors WHERE ...`
    Delete(AstDelete),
    /// `VACUUM vectors`
    Vacuum(AstVacuum),
    /// `CHECKPOINT`
    Checkpoint,
    /// `CREATE INDEX ...` (v2 ; binder rejects this in v1).
    CreateIndex(AstCreateIndex),
    /// `DROP INDEX ...` (v2 ; binder rejects this in v1).
    DropIndex(AstDropIndex),
}

// ----- Statement-shaped stubs -----
//
// Empty structs for now. Each grows fields as its grammar production
// lands : AstQuery in the SELECT step, AstInsert in the INSERT step,
// and so on. The point of stubbing all of them up front is so the
// `AstStatement` enum is complete from day one and downstream code
// (binder, planner) can pattern-match exhaustively as we go.

/// `SELECT` statement payload. Populated when the SELECT production
/// lands.
#[derive(Debug, Clone)]
pub struct AstQuery;

/// `INSERT` statement payload. Populated when the INSERT production
/// lands.
#[derive(Debug, Clone)]
pub struct AstInsert;

/// `UPDATE` statement payload. Populated when the UPDATE production
/// lands.
#[derive(Debug, Clone)]
pub struct AstUpdate;

/// `DELETE` statement payload. Populated when the DELETE production
/// lands.
#[derive(Debug, Clone)]
pub struct AstDelete;

/// `VACUUM` statement payload (target table name).
#[derive(Debug, Clone)]
pub struct AstVacuum;

/// `CREATE INDEX` statement payload. v2 only.
#[derive(Debug, Clone)]
pub struct AstCreateIndex;

/// `DROP INDEX` statement payload. v2 only.
#[derive(Debug, Clone)]
pub struct AstDropIndex;

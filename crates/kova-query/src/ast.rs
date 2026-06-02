//! Abstract syntax tree for parsed KQL input.
//!
//! The AST is the boundary between parsing and binding. It captures
//! every shape the grammar accepts, including semantically invalid
//! ones (unknown field, type mismatch). Those failures belong to the
//! binder ; keeping the AST permissive keeps the parser context-free.

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
    /// `CREATE INDEX ...`
    CreateIndex(AstCreateIndex),
    /// `DROP INDEX ...`
    DropIndex(AstDropIndex),
}

/// `SELECT` statement payload.
#[derive(Debug, Clone)]
pub struct AstQuery;

/// `INSERT` statement payload.
#[derive(Debug, Clone)]
pub struct AstInsert;

/// `UPDATE` statement payload.
#[derive(Debug, Clone)]
pub struct AstUpdate;

/// `DELETE` statement payload.
#[derive(Debug, Clone)]
pub struct AstDelete;

/// `VACUUM` statement payload (target table name).
#[derive(Debug, Clone)]
pub struct AstVacuum;

/// `CREATE INDEX` statement payload.
#[derive(Debug, Clone)]
pub struct AstCreateIndex;

/// `DROP INDEX` statement payload.
#[derive(Debug, Clone)]
pub struct AstDropIndex;

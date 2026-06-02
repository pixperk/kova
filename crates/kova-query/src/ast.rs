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
pub struct AstInsert {
    /// Target table.
    pub table: String,
    /// Column names in the explicit column list.
    pub columns: Vec<String>,
    /// Source of values to insert.
    pub source: AstInsertSource,
}

/// Where INSERT rows come from.
#[derive(Debug, Clone)]
pub enum AstInsertSource {
    /// Explicit row tuples : `VALUES (a, b, c), ...`. v1 accepts the
    /// single-row form only ; batches go through [`AstInsertSource::Param`].
    Rows(Vec<Vec<AstExpr>>),
    /// Batch parameter : `VALUES $1`, where `$1` is bound by the
    /// caller to an array of `(id, embedding, metadata)` tuples.
    Param(ParamRef),
}

/// Value expression. Currently only parameter references ; literal
/// support lands when a use case forces it.
#[derive(Debug, Clone)]
pub enum AstExpr {
    /// `$1` or `$name`.
    Param(ParamRef),
}

/// Parameter reference. Positional or named ; the binder resolves
/// named refs into positional ordinals so the executor only ever
/// sees `Positional`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamRef {
    /// `$N` : 1-based ordinal as written in the source.
    Positional(u32),
    /// `$name`.
    Named(String),
}

/// `UPDATE` statement payload.
#[derive(Debug, Clone)]
pub struct AstUpdate;

/// `DELETE` statement payload.
#[derive(Debug, Clone)]
pub struct AstDelete;

/// `VACUUM` statement payload.
#[derive(Debug, Clone)]
pub struct AstVacuum {
    /// Name of the table to vacuum.
    pub table: String,
}

/// `CREATE INDEX` statement payload.
#[derive(Debug, Clone)]
pub struct AstCreateIndex {
    /// Index name. `None` when grammar form is `CREATE INDEX ON ...`
    /// without an explicit name ; the binder synthesises one later.
    pub name: Option<String>,
    /// Target table.
    pub table: String,
    /// Index method (HASH / BTREE / INVERTED).
    pub method: IndexMethod,
    /// Indexed field name.
    pub field: String,
}

/// `DROP INDEX` statement payload.
#[derive(Debug, Clone)]
pub struct AstDropIndex {
    /// Name of the index to drop.
    pub name: String,
    /// Target table.
    pub table: String,
}

/// Index method specifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexMethod {
    /// Hash index : `=`, `IN`, existence.
    Hash,
    /// `BTree` index : `<`, `<=`, `>`, `>=`, `BETWEEN`.
    Btree,
    /// Inverted index : `@>` array containment.
    Inverted,
}

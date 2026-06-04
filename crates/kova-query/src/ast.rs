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
pub struct AstQuery {
    /// Projection list. `vec![AstProjection::Wildcard]` for `SELECT *`,
    /// otherwise one entry per explicit `select_item`.
    pub projection: Vec<AstProjection>,
    /// Target table from the `FROM` clause.
    pub from_table: String,
    /// Optional WHERE-clause predicate.
    pub predicate: Option<AstPredicate>,
    /// Optional ORDER BY items. Empty when the clause is absent.
    pub order_by: Vec<AstOrderBy>,
    /// Optional LIMIT.
    pub limit: Option<u64>,
}

/// One element of a `SELECT` projection list.
#[derive(Debug, Clone)]
pub enum AstProjection {
    /// `*` : all columns. The parser produces a single-element
    /// `vec![Wildcard]` ; the binder enforces "wildcard cannot
    /// appear alongside other items."
    Wildcard,
    /// `COUNT(*)` : the only aggregate KQL accepts.
    CountStar,
    /// Bare `id` keyword (case-insensitive). The row primary key.
    Id,
    /// Bare `metadata` keyword (case-insensitive). The whole bag.
    Metadata,
    /// `embedding <op> $param` : a distance expression.
    DistanceExpr(AstDistance),
    /// Any other identifier : a metadata field name.
    Field(String),
    /// `<projection> AS alias`. Wraps any of the above with a name.
    Aliased(Box<AstProjection>, String),
}

/// One item in an `ORDER BY` clause.
#[derive(Debug, Clone)]
pub enum AstOrderBy {
    /// Order by a distance expression : `ORDER BY embedding <-> $1`.
    Distance(AstDistance, OrderDir),
    /// Order by a metadata field : `ORDER BY year DESC`.
    Field(String, OrderDir),
}

/// Sort direction. Parser defaults to `Asc` when the user omits it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderDir {
    /// Ascending. Default for all order keys.
    Asc,
    /// Descending.
    Desc,
}

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

/// Value expression. Used wherever a position can hold either a
/// parameter binding or an inline literal (INSERT row values, atom
/// right-hand sides, etc.).
#[derive(Debug, Clone)]
pub enum AstExpr {
    /// `$1` or `$name`.
    Param(ParamRef),
    /// Inline literal : string, number, boolean, null.
    Literal(AstLiteral),
}

/// Literal value parsed from KQL source. The binder type-checks
/// these against field types when applicable ; the parser only
/// guarantees the value lexed cleanly as the listed variant.
#[derive(Debug, Clone, PartialEq)]
pub enum AstLiteral {
    /// Single-quoted string : `'hello'`.
    String(String),
    /// Integer literal without a decimal point.
    I64(i64),
    /// Floating-point literal containing a decimal point.
    F64(f64),
    /// `TRUE` or `FALSE`, case-insensitive.
    Bool(bool),
    /// `NULL`.
    Null,
}

/// WHERE-clause predicate. Built recursively from atoms and boolean
/// combinators.
#[derive(Debug, Clone)]
pub enum AstPredicate {
    /// `field = value`.
    Eq(String, AstExpr),
    /// `field <op> value` for ops other than `=`.
    Cmp(String, CmpOp, AstExpr),
    /// `field IN (lit, lit, ...)`.
    In(String, Vec<AstLiteral>),
    /// `field BETWEEN lo AND hi`.
    Between(String, AstLiteral, AstLiteral),
    /// `field IS NULL` (`false`) or `field IS NOT NULL` (`true`).
    IsNull(String, bool),
    /// `field @> value` : array containment.
    ArrayContains(String, AstLiteral),
    /// Boolean AND of two or more child predicates. Flattened :
    /// `(a AND b) AND c` parses as `And([a, b, c])`.
    And(Vec<AstPredicate>),
    /// Boolean OR of two or more child predicates. Flattened.
    Or(Vec<AstPredicate>),
    /// Boolean NOT. Single child.
    Not(Box<AstPredicate>),
    /// `embedding <op> $q <cmp> radius`. The right side is parsed as
    /// `f32` directly because the planner consumes it as a distance
    /// bound ; parameter binding on the right is rejected at the
    /// binder.
    DistanceThreshold(AstDistance, CmpOp, f32),
}

/// Distance expression : `embedding <op> $param`.
#[derive(Debug, Clone)]
pub struct AstDistance {
    /// Metric operator (`L2` / `Cosine` / `InnerProduct`).
    pub metric: DistanceOp,
    /// Query vector parameter binding.
    pub param: ParamRef,
}

/// Distance metric operator parsed from `<->` / `<=>` / `<#>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceOp {
    /// `<->` : Euclidean (L2) distance.
    L2,
    /// `<=>` : cosine distance.
    Cosine,
    /// `<#>` : (negated) inner product.
    InnerProduct,
}

/// Comparison operator for predicate atoms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    /// `=` (equality ; the parser also routes this to `AstPredicate::Eq`).
    Eq,
    /// `<` (strictly less than).
    Lt,
    /// `<=` (less than or equal).
    Le,
    /// `>` (strictly greater than).
    Gt,
    /// `>=` (greater than or equal).
    Ge,
    /// `!=` or `<>` (not equal).
    Ne,
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
pub struct AstUpdate {
    /// Target table.
    pub table: String,
    /// One or more `SET field = value` (or `SET field['key'] = value`)
    /// assignments. Grammar enforces at least one.
    pub assignments: Vec<AstAssignment>,
    /// Required WHERE-clause predicate. The grammar rejects UPDATE
    /// without a WHERE.
    pub predicate: AstPredicate,
}

/// One assignment in an `UPDATE ... SET ...` clause.
#[derive(Debug, Clone)]
pub struct AstAssignment {
    /// Target field name.
    pub field: String,
    /// Optional string subscript : `Some("key")` for `field['key']
    /// = value`, `None` for `field = value`. The binder uses this
    /// to distinguish whole-bag replace from in-place patch.
    pub subscript: Option<String>,
    /// Right-hand side value (literal or parameter).
    pub value: AstExpr,
}

/// `DELETE` statement payload.
#[derive(Debug, Clone)]
pub struct AstDelete {
    /// Target table.
    pub table: String,
    /// Required WHERE-clause predicate. The grammar rejects DELETE
    /// without a WHERE.
    pub predicate: AstPredicate,
}

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

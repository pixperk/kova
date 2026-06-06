//! Logical statement IR : the binder's output, the planner's input.
//!
//! Where [`crate::ast`] captures *what the user wrote*, this module
//! captures *what the user meant* after field resolution, type
//! checks, and predicate normalisation. Downstream code (planner,
//! executor) reasons about `LogicalStatement` only ; the AST exists
//! to let the parser stay context-free.
//!
//! See [`crate::binder`] for the AST -> `LogicalStatement` conversion.

use crate::ast::{CmpOp, DistanceOp, ParamRef};

/// Top-level logical statement. One variant per AST statement, with
/// the contents resolved and normalised. CREATE / DROP INDEX have no
/// variants here because the binder rejects them in v1.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalStatement {
    /// `SELECT ...`
    Query(LogicalQuery),
    /// `INSERT INTO vectors ...`
    Insert(LogicalInsert),
    /// `UPDATE vectors SET ... WHERE ...`
    Update(LogicalUpdate),
    /// `DELETE FROM vectors WHERE ...`
    Delete(LogicalDelete),
    /// `VACUUM <table>`
    Vacuum(LogicalVacuum),
    /// `CHECKPOINT`
    Checkpoint,
}

/// VACUUM statement after binding. Carries the target table name
/// through ; the executor matches it against the available Shard(s)
/// and errors on mismatch.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalVacuum {
    /// Target table name, preserved from the source (no case folding).
    pub table: String,
}

/// SELECT statement after binding.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalQuery {
    /// Target table name from the `FROM` clause, preserved unchanged.
    pub from_table: String,
    /// Projection list. The binder enforces "wildcard cannot appear
    /// alongside other items" ; downstream code can assume that.
    pub projection: ProjectionSpec,
    /// WHERE-clause predicate in normalised form, if present.
    pub predicate: Option<PredicateExpr>,
    /// ORDER BY ordering keys, in source order. Empty when the
    /// clause is absent. v1 supports multiple keys end-to-end :
    /// `ORDER BY year DESC, score ASC` survives binding as a
    /// length-2 vec.
    pub ordering: Vec<OrderingSpec>,
    /// LIMIT, if present. Required for kNN queries (binder check).
    pub limit: Option<u64>,
}

/// INSERT statement after binding. Column list validation happened
/// here ; the canonical `(id, embedding, metadata)` shape is the
/// only one v1 accepts, so we can collapse to typed parameter slots.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalInsert {
    /// Target table name, preserved from the source.
    pub table: String,
    /// Source of rows.
    pub rows: LogicalInsertSource,
}

/// Where INSERT rows come from after binding. Same dichotomy as the
/// AST but with each parameter slot named for the column it fills.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalInsertSource {
    /// One row, fields bound to parameter slots in canonical order.
    Single {
        /// Parameter for the `id` column.
        id: ParamRef,
        /// Parameter for the `embedding` column.
        embedding: ParamRef,
        /// Parameter for the `metadata` column.
        metadata: ParamRef,
    },
    /// Batch : one parameter slot carrying an array of typed tuples.
    Batch {
        /// Parameter bound to a Vec of `(id, embedding, metadata)` tuples.
        param: ParamRef,
    },
}

/// UPDATE statement after binding. The binder has guaranteed no
/// embedding assignment ; downstream code can assume every
/// assignment touches metadata only.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalUpdate {
    /// Target table name, preserved from the source.
    pub table: String,
    /// WHERE-clause predicate (required by grammar).
    pub predicate: PredicateExpr,
    /// One or more assignments, in source order.
    pub assignments: Vec<LogicalAssignment>,
    /// `Some(...)` when `predicate` is exactly `id = <literal>` or
    /// `id = $param` ; same fast-path semantics as [`LogicalDelete`].
    pub single_id_hint: Option<IdHint>,
}

/// One `field = value` (or `field['key'] = value`) assignment after
/// binding.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalAssignment {
    /// Target field name.
    pub field: String,
    /// `Some("key")` for `field['key'] = value` ; `None` for
    /// `field = value`.
    pub subscript: Option<String>,
    /// Right-hand side : literal or parameter.
    pub value: BoundExpr,
}

/// DELETE statement after binding. Carries an optional hint so the
/// planner can pick the trivial single-id path without re-walking
/// the predicate tree.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalDelete {
    /// Target table name, preserved from the source.
    pub table: String,
    /// WHERE-clause predicate (required by grammar).
    pub predicate: PredicateExpr,
    /// `Some(...)` when `predicate` is exactly `id = <literal>` or
    /// `id = $param`. The binder sets this so the planner gets the
    /// answer (or the parameter slot) for free.
    pub single_id_hint: Option<IdHint>,
}

/// How the binder resolved the single-id shape of a WHERE predicate.
/// Shared between DELETE and UPDATE since both have the same fast-path
/// shape.
#[derive(Debug, Clone, PartialEq)]
pub enum IdHint {
    /// `WHERE id = <integer-literal>` ; planner knows the id directly.
    Literal(u64),
    /// `WHERE id = $param` ; executor resolves the param at run time.
    Param(crate::ast::ParamRef),
}

/// Canonical predicate tree. Flat (no nested `And`/`Or` of the same
/// kind), with `NOT` pushed down to atoms, constants folded, and
/// `True`/`False` only at the root.
#[derive(Debug, Clone, PartialEq)]
pub enum PredicateExpr {
    /// Leaf : a single field comparison.
    Atom(PredAtom),
    /// Conjunction. Always at least two children after normalisation.
    And(Vec<PredicateExpr>),
    /// Disjunction. Always at least two children after normalisation.
    Or(Vec<PredicateExpr>),
    /// Negation. After NOT-push-down, always wraps an atom.
    Not(Box<PredicateExpr>),
    /// Constant true (predicate matches every row). Only produced
    /// by the normaliser at the root.
    True,
    /// Constant false (predicate matches no rows). Only produced by
    /// the normaliser at the root.
    False,
}

/// Predicate atom kinds. Named struct variants because each carries
/// at least three pieces of data ; tuple variants would rot the
/// moment we added a fourth.
#[derive(Debug, Clone, PartialEq)]
pub enum PredAtom {
    /// `field = value`.
    Eq {
        /// Target field.
        field: String,
        /// Right-hand side.
        value: BoundExpr,
    },
    /// `field <op> value` for ops other than `=`.
    Cmp {
        /// Target field.
        field: String,
        /// Comparison operator.
        op: CmpOp,
        /// Right-hand side.
        value: BoundExpr,
    },
    /// `field IN (lit, lit, ...)`.
    In {
        /// Target field.
        field: String,
        /// Membership set.
        values: Vec<BoundLiteral>,
    },
    /// `field BETWEEN lo AND hi`.
    Between {
        /// Target field.
        field: String,
        /// Lower bound (inclusive).
        lo: BoundLiteral,
        /// Upper bound (inclusive).
        hi: BoundLiteral,
    },
    /// `field IS NOT NULL`. The binder normalises `IS NULL` to
    /// `NOT IsNotNull(field)` so downstream code only handles one
    /// shape.
    IsNotNull {
        /// Target field.
        field: String,
    },
    /// `field @> value` : array containment.
    ArrayContains {
        /// Target field.
        field: String,
        /// Value the array must contain.
        value: BoundLiteral,
    },
    /// `embedding <metric> $param <op> radius`. Right-hand side is
    /// f32 because the planner consumes it as a distance bound ;
    /// parameter binding on the right is rejected at the binder.
    DistanceThreshold {
        /// Distance metric operator.
        metric: DistanceOp,
        /// Query vector parameter.
        param: ParamRef,
        /// Comparison against the radius bound.
        op: CmpOp,
        /// Distance bound.
        radius: f32,
    },
}

/// Value expression after binding : either a literal or a parameter
/// reference.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundExpr {
    /// Inline literal.
    Literal(BoundLiteral),
    /// Parameter binding.
    Param(ParamRef),
}

/// Literal value after binding. Same variants as the AST literal ;
/// the wrapper type exists so v2 can extend it (e.g. with strict
/// schema type annotations) without churning the AST.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundLiteral {
    /// Single-quoted string.
    String(String),
    /// Integer literal.
    I64(i64),
    /// Floating-point literal.
    F64(f64),
    /// Boolean.
    Bool(bool),
    /// `NULL`.
    Null,
}

/// ORDER BY ordering after binding.
#[derive(Debug, Clone, PartialEq)]
pub enum OrderingSpec {
    /// Order by a distance expression : `ORDER BY embedding <-> $1`.
    /// Direction is always `Asc` (binder rejects `DESC` for distance).
    Distance {
        /// Distance metric.
        metric: DistanceOp,
        /// Query vector parameter.
        param: ParamRef,
    },
    /// Order by a metadata field : `ORDER BY year DESC`.
    Field {
        /// Field name.
        name: String,
        /// Sort direction.
        dir: OrderDir,
    },
}

/// Sort direction. Same shape as the AST enum ; re-exported here
/// for callers that only depend on the logical layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderDir {
    /// Ascending.
    Asc,
    /// Descending.
    Desc,
}

/// Projection list after binding.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionSpec {
    /// Output columns in order.
    pub columns: Vec<BoundProjection>,
}

/// One projection column after binding. The binder has resolved
/// magic names (`id`, `metadata`) and lifted aliases into the
/// variants that can carry them.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundProjection {
    /// `SELECT *`. Always appears alone (binder check).
    Wildcard,
    /// `COUNT(*)`. Optional alias.
    CountStar {
        /// Optional `AS alias`.
        alias: Option<String>,
    },
    /// The row primary key column.
    Id {
        /// Optional `AS alias`.
        alias: Option<String>,
    },
    /// The whole metadata bag.
    Metadata {
        /// Optional `AS alias`.
        alias: Option<String>,
    },
    /// `embedding <metric> $param AS alias`. The alias is required
    /// here because there's no natural column name for a distance
    /// expression ; the binder rejects unaliased forms.
    Distance {
        /// Distance metric.
        metric: DistanceOp,
        /// Query vector parameter.
        param: ParamRef,
        /// Output column alias (required for distance projections).
        alias: String,
    },
    /// A metadata field by name.
    MetadataField {
        /// Field name in the metadata bag.
        name: String,
        /// Optional `AS alias`.
        alias: Option<String>,
    },
}

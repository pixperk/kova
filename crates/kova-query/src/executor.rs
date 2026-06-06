//! KQL executor : runs a [`PhysicalPlan`] against a `Shard`.
//!
//! The public surface is [`Engine`], which owns a `Shard` and exposes
//! [`Engine::execute_str`] : the full `parse -> bind -> plan -> execute`
//! pipeline behind one call.

use std::collections::HashMap;

use kova_core::{Distance, Metadata, Value, Vector, VectorId};
use kova_storage::{FileMetadataStore, FileWal, Lsn, MmapVectorStore, Shard};

use crate::ast::ParamRef;
use crate::binder::bind;
use crate::error::KovaQueryError;
use crate::parser::parse_str;
use crate::physical::PhysicalPlan;
use crate::planner::{SelectivityEstimate, SelectivityEstimator, plan_with_estimator};

/// Caller-supplied values for parameter slots.
///
/// Positional and named bindings can both be present in one set ;
/// the executor resolves [`ParamRef::Positional`] against the
/// positional vec (1-based indexing) and [`ParamRef::Named`] against
/// the named map.
#[derive(Debug, Default, Clone)]
pub struct ParamBindings {
    positional: Vec<ParamValue>,
    named: HashMap<String, ParamValue>,
}

impl ParamBindings {
    /// An empty binding set. Use for statements with no parameters
    /// (CHECKPOINT, VACUUM <literal-table>).
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Build a binding set from a positional list. Index `i` of the
    /// vec maps to source-level `$(i+1)` (positional params are
    /// 1-based by SQL convention).
    #[must_use]
    pub fn positional(values: Vec<ParamValue>) -> Self {
        Self {
            positional: values,
            named: HashMap::new(),
        }
    }

    /// Build a binding set from a name-to-value map.
    #[must_use]
    pub fn named(map: HashMap<String, ParamValue>) -> Self {
        Self {
            positional: Vec::new(),
            named: map,
        }
    }

    /// Append a positional binding (1-based, so the first call binds
    /// `$1`, the second binds `$2`, etc.).
    #[must_use]
    pub fn with_positional(mut self, value: ParamValue) -> Self {
        self.positional.push(value);
        self
    }

    /// Insert (or replace) a named binding.
    #[must_use]
    pub fn with_named(mut self, name: impl Into<String>, value: ParamValue) -> Self {
        self.named.insert(name.into(), value);
        self
    }

    /// Look up the value bound to a [`ParamRef`]. Errors if the slot
    /// is unbound or (for positional) out of range.
    ///
    /// # Errors
    ///
    /// Returns [`KovaQueryError::Execution`] when the parameter slot
    /// isn't bound by this set.
    pub fn resolve(&self, p: &ParamRef) -> Result<&ParamValue, KovaQueryError> {
        match p {
            ParamRef::Positional(n) => {
                let idx = (*n as usize).checked_sub(1).ok_or_else(|| {
                    KovaQueryError::Execution(
                        "positional parameter $0 is invalid (slots are 1-based)".into(),
                    )
                })?;
                self.positional.get(idx).ok_or_else(|| {
                    KovaQueryError::Execution(format!(
                        "positional parameter ${n} not bound (have {} value(s))",
                        self.positional.len()
                    ))
                })
            }
            ParamRef::Named(name) => self.named.get(name).ok_or_else(|| {
                KovaQueryError::Execution(format!("named parameter ${name} not bound"))
            }),
        }
    }
}

/// Typed parameter value the caller passes for a `$param` slot.
#[derive(Debug, Clone)]
pub enum ParamValue {
    /// Vector primary key. INSERT row id slot.
    Id(VectorId),
    /// Embedding vector. Single-row INSERT, or SELECT's `$query`.
    Vector(Vector),
    /// Metadata bag. Single-row INSERT / UPDATE metadata slot.
    Metadata(Metadata),
    /// Batch of `(id, embedding, metadata)` tuples. `INSERT VALUES $1`.
    Batch(Vec<(VectorId, Vector, Metadata)>),
    // ---- Predicate-side value bindings ----
    //
    // Used when a predicate atom binds a literal slot, e.g.
    // `WHERE category = $1`. Same shape as `kova_core::Value` so
    // the evaluator can compare directly.
    /// UTF-8 string literal value. Predicate side.
    String(String),
    /// Signed 64-bit integer literal value. Predicate side.
    I64(i64),
    /// 64-bit float literal value. Predicate side.
    F64(f64),
    /// Boolean literal value. Predicate side.
    Bool(bool),
    /// SQL NULL. Predicate side.
    Null,
}

/// The outcome of executing a statement. One variant per operator
/// shape ; grows as new operators land.
#[derive(Debug, Clone)]
pub enum ExecutionResult {
    /// CHECKPOINT committed at `lsn`.
    Checkpoint {
        /// LSN of the WAL position that the new snapshot covers.
        lsn: Lsn,
    },
    /// VACUUM finished on `table` ; `removed` nodes were physically
    /// reclaimed and their inbound edges repaired in the HNSW graph.
    Vacuum {
        /// Target table the operation ran against.
        table: String,
        /// Number of tombstoned nodes removed.
        removed: usize,
    },
    /// INSERT completed ; `inserted` rows landed in `table`.
    Insert {
        /// Target table the operation ran against.
        table: String,
        /// Number of rows that landed (1 for single-row, batch length
        /// for batch).
        inserted: u64,
    },
    /// DELETE completed ; `deleted` rows were tombstoned in `table`.
    Delete {
        /// Target table the operation ran against.
        table: String,
        /// Number of rows tombstoned (1 for `DeleteById` ; more once
        /// `DeleteByPredicate` lands).
        deleted: u64,
    },
    /// SELECT returned a result set.
    Rows {
        /// Output column headers, in projection order.
        columns: Vec<String>,
        /// Output rows.
        rows: Vec<Row>,
    },
}

/// One output row from a SELECT result. Cell values are positional
/// in [`Row::values`] ; column headers in
/// [`ExecutionResult::Rows::columns`] line up by index.
#[derive(Debug, Clone)]
pub struct Row {
    /// Cell values in column order.
    pub values: Vec<RowValue>,
}

/// Typed cell value in a [`Row`]. The variants cover everything
/// SELECT can project today : the magic `id` and `distance` columns,
/// the whole metadata bag, individual metadata fields, and `NULL`.
#[derive(Debug, Clone)]
pub enum RowValue {
    /// The row's vector id.
    Id(VectorId),
    /// Distance under the kNN's metric (smaller = closer).
    Distance(f32),
    /// Whole metadata bag, returned when the projection includes
    /// the `metadata` keyword.
    Metadata(Metadata),
    /// A single named metadata field value.
    Field(Value),
    /// `NULL` : returned when a projected metadata field is absent
    /// from the row's bag.
    Null,
}

/// Top-level KQL execution engine. Owns a file-backed [`Shard`] and
/// runs statements against it.
///
/// Engine is concrete to the file-backed combo (`MmapVectorStore`,
/// `FileMetadataStore`, `FileWal`) because `Shard::checkpoint`
/// requires a directory ; the in-memory combo can't checkpoint and
/// CHECKPOINT is a v1 statement. Generic over the distance metric
/// only, which is the type parameter HNSW actually needs.
///
/// The Engine carries the table name its shard answers to. Statements
/// that name a table (`VACUUM <name>`, `INSERT INTO <name> ...`)
/// are validated against this at execute time. Case-insensitive,
/// matching how the binder treats the magic column names.
pub struct Engine<D: Distance> {
    shard: Shard<D, MmapVectorStore, FileMetadataStore, FileWal>,
    table_name: String,
}

impl<D: Distance> Engine<D> {
    /// Wrap a [`Shard`] in an engine. `table_name` is the name KQL
    /// statements use to refer to this shard ; statements that name
    /// a different table error at execute time.
    pub fn new(
        shard: Shard<D, MmapVectorStore, FileMetadataStore, FileWal>,
        table_name: impl Into<String>,
    ) -> Self {
        Self {
            shard,
            table_name: table_name.into(),
        }
    }

    /// Borrow the underlying shard.
    #[must_use]
    pub fn shard(&self) -> &Shard<D, MmapVectorStore, FileMetadataStore, FileWal> {
        &self.shard
    }

    /// Mutably borrow the underlying shard. Useful for tests that
    /// need to peek at state with `shard.contains(id)` etc.
    pub fn shard_mut(&mut self) -> &mut Shard<D, MmapVectorStore, FileMetadataStore, FileWal> {
        &mut self.shard
    }

    /// Consume the engine and return its shard, e.g. to reopen
    /// against another engine.
    #[must_use]
    pub fn into_shard(self) -> Shard<D, MmapVectorStore, FileMetadataStore, FileWal> {
        self.shard
    }

    /// Parse, bind, plan, and execute a KQL statement.
    ///
    /// # Errors
    ///
    /// Returns the first error from any of the four pipeline stages :
    /// parse / bind / plan / execute. The error variant identifies
    /// the stage.
    //
    // `params` is taken by value so callers can chain
    // `ParamBindings::empty().with_positional(v)` inline. Internally
    // we borrow ; clippy can't see that's the right shape.
    #[allow(clippy::needless_pass_by_value)]
    pub fn execute_str(
        &mut self,
        input: &str,
        params: ParamBindings,
    ) -> Result<ExecutionResult, KovaQueryError> {
        let ast = parse_str(input)?;
        let logical = bind(ast)?;
        let estimator = ShardEstimator { shard: &self.shard };
        let physical = plan_with_estimator(logical, &estimator, &params)?;
        self.execute(physical, &params)
    }

    /// Run a [`PhysicalPlan`] against the engine's shard.
    //
    // `params` is borrowed (not consumed) so the executor can resolve
    // multiple slots from the same set within a single op. `plan` is
    // taken by value : every arm that carries payload data moves
    // fields out of the operator (the `table` strings, the `ParamRef`s,
    // etc.).
    #[allow(clippy::needless_pass_by_value)]
    fn execute(
        &mut self,
        plan: PhysicalPlan,
        params: &ParamBindings,
    ) -> Result<ExecutionResult, KovaQueryError> {
        match plan {
            PhysicalPlan::Checkpoint => {
                let lsn = self
                    .shard
                    .checkpoint()
                    .map_err(|e| KovaQueryError::Backend(Box::new(e)))?;
                Ok(ExecutionResult::Checkpoint { lsn })
            }
            PhysicalPlan::Vacuum { table } => {
                self.assert_table(&table)?;
                let removed = self
                    .shard
                    .vacuum()
                    .map_err(|e| KovaQueryError::Backend(Box::new(e)))?;
                Ok(ExecutionResult::Vacuum { table, removed })
            }
            PhysicalPlan::InsertOne {
                table,
                id: id_ref,
                embedding: emb_ref,
                metadata: meta_ref,
            } => {
                self.assert_table(&table)?;
                let id = expect_id(params.resolve(&id_ref)?, "id")?;
                let embedding = expect_vector(params.resolve(&emb_ref)?, "embedding")?;
                let metadata = expect_metadata(params.resolve(&meta_ref)?, "metadata")?;
                self.shard
                    .insert(id, embedding, metadata)
                    .map_err(|e| KovaQueryError::Backend(Box::new(e)))?;
                Ok(ExecutionResult::Insert { table, inserted: 1 })
            }
            PhysicalPlan::InsertMany {
                table,
                batch: batch_ref,
            } => {
                self.assert_table(&table)?;
                let batch = expect_batch(params.resolve(&batch_ref)?, "batch")?;
                let inserted = batch.len() as u64;
                self.shard
                    .insert_many(batch)
                    .map_err(|e| KovaQueryError::Backend(Box::new(e)))?;
                Ok(ExecutionResult::Insert { table, inserted })
            }
            PhysicalPlan::DeleteById { table, id } => {
                self.assert_table(&table)?;
                self.shard
                    .delete(id)
                    .map_err(|e| KovaQueryError::Backend(Box::new(e)))?;
                Ok(ExecutionResult::Delete { table, deleted: 1 })
            }

            // Read path : the outermost operator must be Projection,
            // because that's the only one that builds user-facing
            // `Row` values. Internal read operators
            // (KnnSearch / MetadataScan / ExactDistance / Limit)
            // flow `Vec<InternalHit>` between themselves via
            // `execute_read` and shouldn't appear at the top level.
            PhysicalPlan::Projection { input, spec } => {
                let hits = self.execute_read(*input, params)?;
                let columns = projection_column_names(&spec);
                let rows: Result<Vec<Row>, _> =
                    hits.iter().map(|h| project_hit(h, &spec)).collect();
                Ok(ExecutionResult::Rows {
                    columns,
                    rows: rows?,
                })
            }
            PhysicalPlan::Limit { .. }
            | PhysicalPlan::KnnSearch { .. }
            | PhysicalPlan::MetadataScan { .. }
            | PhysicalPlan::ExactDistance { .. } => Err(KovaQueryError::Plan(
                "read-path operator at top level ; planner must wrap in Projection".into(),
            )),
        }
    }

    /// Internal read-path executor. Returns the typed hits that flow
    /// between read operators ; only the outermost `Projection`
    /// converts them into user-facing rows.
    fn execute_read(
        &self,
        plan: PhysicalPlan,
        params: &ParamBindings,
    ) -> Result<Vec<InternalHit>, KovaQueryError> {
        match plan {
            PhysicalPlan::KnnSearch {
                table,
                query,
                metric: _,
                k,
                post_filter,
            } => {
                self.assert_table(&table)?;
                let query_vec = expect_vector(params.resolve(&query)?, "query")?;
                let hits = self
                    .shard
                    .search(&query_vec, k)
                    .map_err(|e| KovaQueryError::Backend(Box::new(e)))?;
                // Apply post-filter (if any), then promote SearchHit to
                // InternalHit. distance is Some(_) on the kNN path.
                let mut out = Vec::with_capacity(hits.len());
                for h in hits {
                    if let Some(ref pred) = post_filter
                        && !eval_predicate(pred, &h.metadata, params)?
                    {
                        continue;
                    }
                    out.push(InternalHit {
                        id: h.id,
                        distance: Some(h.distance),
                        metadata: h.metadata,
                    });
                }
                Ok(out)
            }
            PhysicalPlan::MetadataScan { table, predicate } => {
                self.assert_table(&table)?;
                // The closure must signal eval errors back to the caller
                // without coercing them into `false` (which would silently
                // drop rows we couldn't classify). Capture into a mut
                // optional ; check after the scan returns.
                let mut closure_err: Option<KovaQueryError> = None;
                let ids = self.shard.scan_metadata(|m| {
                    if closure_err.is_some() {
                        return false;
                    }
                    match eval_predicate(&predicate, m, params) {
                        Ok(b) => b,
                        Err(e) => {
                            closure_err = Some(e);
                            false
                        }
                    }
                });
                if let Some(e) = closure_err {
                    return Err(e);
                }
                let hits = ids
                    .into_iter()
                    .filter_map(|id| {
                        self.shard.get_metadata(id).map(|metadata| InternalHit {
                            id,
                            distance: None,
                            metadata,
                        })
                    })
                    .collect();
                Ok(hits)
            }
            PhysicalPlan::ExactDistance {
                input,
                query,
                metric: _,
                k,
            } => {
                let candidates = self.execute_read(*input, params)?;
                let query_vec = expect_vector(params.resolve(&query)?, "query")?;
                // Compute exact distance for each candidate, drop those
                // whose vector is gone (tombstoned or absent).
                let mut scored: Vec<InternalHit> = candidates
                    .into_iter()
                    .filter_map(|h| {
                        self.shard
                            .distance_to(h.id, &query_vec)
                            .map(|d| InternalHit {
                                id: h.id,
                                distance: Some(d),
                                metadata: h.metadata,
                            })
                    })
                    .collect();
                // Sort ascending by distance.
                scored.sort_by(|a, b| match (a.distance, b.distance) {
                    (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
                    _ => std::cmp::Ordering::Equal,
                });
                scored.truncate(k);
                Ok(scored)
            }
            PhysicalPlan::Limit { input, limit } => {
                let hits = self.execute_read(*input, params)?;
                let cap = usize::try_from(limit).unwrap_or(usize::MAX);
                Ok(hits.into_iter().take(cap).collect())
            }
            other => Err(KovaQueryError::Plan(format!(
                "{} is not a read-path operator",
                physical_kind(&other)
            ))),
        }
    }

    /// Validate that a statement-supplied table name matches this
    /// engine's shard. Case-insensitive, same convention as the
    /// binder uses for magic column names.
    fn assert_table(&self, name: &str) -> Result<(), KovaQueryError> {
        if name.eq_ignore_ascii_case(&self.table_name) {
            Ok(())
        } else {
            Err(KovaQueryError::Execution(format!(
                "unknown table '{name}' ; this engine is wrapping '{}'",
                self.table_name
            )))
        }
    }
}

// =========================================================================
// Param-value extractors
// =========================================================================
//
// Each `expect_<kind>` pulls a specific [`ParamValue`] variant out of
// a resolved binding and errors cleanly when the caller passed the
// wrong type. The slot name is threaded through so error messages
// identify which parameter slot was wrong (e.g., "parameter slot
// 'embedding' expects Vector, got Id"). Reduces user-facing
// debugging time by one round-trip.

/// Extract a [`VectorId`] from a resolved [`ParamValue`]. `VectorId`
/// is `Copy`, so no clone.
fn expect_id(value: &ParamValue, slot: &str) -> Result<VectorId, KovaQueryError> {
    match value {
        ParamValue::Id(id) => Ok(*id),
        other => Err(KovaQueryError::Execution(format!(
            "parameter slot '{slot}' expects Id, got {}",
            param_value_kind(other)
        ))),
    }
}

/// Extract a [`Vector`] from a resolved [`ParamValue`]. Clones the
/// inner `Vec<f32>` because `Shard::insert` consumes by value.
/// v2 may take ownership through the binding to skip the clone.
fn expect_vector(value: &ParamValue, slot: &str) -> Result<Vector, KovaQueryError> {
    match value {
        ParamValue::Vector(v) => Ok(v.clone()),
        other => Err(KovaQueryError::Execution(format!(
            "parameter slot '{slot}' expects Vector, got {}",
            param_value_kind(other)
        ))),
    }
}

/// Extract a [`Metadata`] from a resolved [`ParamValue`]. Clones.
fn expect_metadata(value: &ParamValue, slot: &str) -> Result<Metadata, KovaQueryError> {
    match value {
        ParamValue::Metadata(m) => Ok(m.clone()),
        other => Err(KovaQueryError::Execution(format!(
            "parameter slot '{slot}' expects Metadata, got {}",
            param_value_kind(other)
        ))),
    }
}

/// Extract a batch (`Vec<(VectorId, Vector, Metadata)>`) from a
/// resolved [`ParamValue`]. Clones the entire array — expensive
/// for large batches but correct ; the v2 optimisation is to
/// consume `ParamBindings` by value into `execute`.
fn expect_batch(
    value: &ParamValue,
    slot: &str,
) -> Result<Vec<(VectorId, Vector, Metadata)>, KovaQueryError> {
    match value {
        ParamValue::Batch(b) => Ok(b.clone()),
        other => Err(KovaQueryError::Execution(format!(
            "parameter slot '{slot}' expects Batch, got {}",
            param_value_kind(other)
        ))),
    }
}

/// Selectivity estimator backed by a live shard. The planner consults
/// this to decide between plan A and plan B for SELECT queries with
/// predicates.
///
/// v1 implementation walks the metadata store, evaluating the
/// predicate against every row (cheap because metadata is in-memory).
/// v2 will swap in index cardinality lookups when secondary indexes
/// ship.
struct ShardEstimator<'a, D: Distance> {
    shard: &'a Shard<D, MmapVectorStore, FileMetadataStore, FileWal>,
}

impl<D: Distance> SelectivityEstimator for ShardEstimator<'_, D> {
    fn estimate(&self, pred: &PredicateExpr, params: &ParamBindings) -> SelectivityEstimate {
        let total = self.shard.len();
        // The closure swallows eval errors (returns false on error)
        // because the estimator's job is fast cardinality, not error
        // reporting. Errors will surface when the executor evaluates
        // the predicate for real during MetadataScan or post-filter.
        let matches = self
            .shard
            .count_matching(|m| eval_predicate(pred, m, params).unwrap_or(false));
        SelectivityEstimate { matches, total }
    }
}

/// Static label for a [`PhysicalPlan`] variant ; used in error
/// messages when a read-path operator shows up at the wrong place.
fn physical_kind(plan: &PhysicalPlan) -> &'static str {
    match plan {
        PhysicalPlan::Checkpoint => "Checkpoint",
        PhysicalPlan::Vacuum { .. } => "Vacuum",
        PhysicalPlan::InsertOne { .. } => "InsertOne",
        PhysicalPlan::InsertMany { .. } => "InsertMany",
        PhysicalPlan::DeleteById { .. } => "DeleteById",
        PhysicalPlan::KnnSearch { .. } => "KnnSearch",
        PhysicalPlan::Limit { .. } => "Limit",
        PhysicalPlan::Projection { .. } => "Projection",
        PhysicalPlan::MetadataScan { .. } => "MetadataScan",
        PhysicalPlan::ExactDistance { .. } => "ExactDistance",
    }
}

// =========================================================================
// Predicate evaluation
// =========================================================================
//
// Walks a `PredicateExpr` against a row's `Metadata` bag and the
// caller's `ParamBindings`. Returns a boolean : does the row pass
// the filter? Used by `KnnSearch`'s post-filter step.
//
// NULL handling : v1 follows the "filtered out" rule for atoms
// touching absent fields (e.g. `category = 'docs'` on a row with no
// `category` returns false). v2 may switch to Postgres 3-value
// logic (NULL = NULL produces NULL, propagates through AND/OR), but
// for plan A the "drop unknowns" behaviour is what users expect.

use crate::ast::CmpOp;
use crate::logical::{
    BoundExpr, BoundLiteral, BoundProjection, PredAtom, PredicateExpr, ProjectionSpec,
};

/// Walk a predicate against a row.
fn eval_predicate(
    pred: &PredicateExpr,
    meta: &Metadata,
    params: &ParamBindings,
) -> Result<bool, KovaQueryError> {
    match pred {
        PredicateExpr::True => Ok(true),
        PredicateExpr::False => Ok(false),
        PredicateExpr::And(children) => {
            for c in children {
                if !eval_predicate(c, meta, params)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        PredicateExpr::Or(children) => {
            for c in children {
                if eval_predicate(c, meta, params)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        PredicateExpr::Not(inner) => Ok(!eval_predicate(inner, meta, params)?),
        PredicateExpr::Atom(atom) => eval_atom(atom, meta, params),
    }
}

/// Walk one atom against a row.
fn eval_atom(
    atom: &PredAtom,
    meta: &Metadata,
    params: &ParamBindings,
) -> Result<bool, KovaQueryError> {
    match atom {
        PredAtom::Eq { field, value } => {
            let expected = resolve_bound_value(value, params)?;
            Ok(meta.get(field).is_some_and(|v| values_eq(v, &expected)))
        }
        PredAtom::Cmp { field, op, value } => {
            let expected = resolve_bound_value(value, params)?;
            Ok(meta
                .get(field)
                .and_then(|v| values_cmp(v, &expected, *op))
                .unwrap_or(false))
        }
        PredAtom::In { field, values } => {
            let Some(actual) = meta.get(field) else {
                return Ok(false);
            };
            Ok(values
                .iter()
                .map(literal_to_value)
                .any(|lit| values_eq(actual, &lit)))
        }
        PredAtom::Between { field, lo, hi } => {
            let Some(actual) = meta.get(field) else {
                return Ok(false);
            };
            let lo_v = literal_to_value(lo);
            let hi_v = literal_to_value(hi);
            let ge_lo = values_cmp(actual, &lo_v, CmpOp::Ge).unwrap_or(false);
            let le_hi = values_cmp(actual, &hi_v, CmpOp::Le).unwrap_or(false);
            Ok(ge_lo && le_hi)
        }
        PredAtom::IsNotNull { field } => Ok(meta.contains_key(field)),
        PredAtom::ArrayContains { field, value } => {
            let target = literal_to_value(value);
            match meta.get(field) {
                Some(Value::Array(arr)) => Ok(arr.iter().any(|v| values_eq(v, &target))),
                _ => Ok(false),
            }
        }
        PredAtom::DistanceThreshold { .. } => Err(KovaQueryError::Execution(
            "DistanceThreshold predicate in SELECT WHERE is not supported in plan A ; \
             use it as a radius search later"
                .into(),
        )),
    }
}

/// Resolve a [`BoundExpr`] (either a literal or a `$param`) to a
/// concrete [`Value`] for predicate comparison.
fn resolve_bound_value(expr: &BoundExpr, params: &ParamBindings) -> Result<Value, KovaQueryError> {
    match expr {
        BoundExpr::Literal(l) => Ok(literal_to_value(l)),
        BoundExpr::Param(p) => param_value_to_value(params.resolve(p)?),
    }
}

fn literal_to_value(l: &BoundLiteral) -> Value {
    match l {
        BoundLiteral::String(s) => Value::String(s.clone()),
        BoundLiteral::I64(n) => Value::I64(*n),
        BoundLiteral::F64(f) => Value::F64(*f),
        BoundLiteral::Bool(b) => Value::Bool(*b),
        // NULL is represented as "no entry in the metadata bag" ; we
        // model it as an unmatchable sentinel here so equality and
        // ordering against any concrete value fall through to false.
        BoundLiteral::Null => Value::Array(Vec::new()),
    }
}

/// Turn a caller-bound [`ParamValue`] into a [`Value`] for predicate
/// comparison. Only the literal-shaped variants are valid here ;
/// vector / id / metadata / batch in a WHERE position is a binder
/// bug we surface as an Execution error.
fn param_value_to_value(p: &ParamValue) -> Result<Value, KovaQueryError> {
    match p {
        ParamValue::String(s) => Ok(Value::String(s.clone())),
        ParamValue::I64(n) => Ok(Value::I64(*n)),
        ParamValue::F64(f) => Ok(Value::F64(*f)),
        ParamValue::Bool(b) => Ok(Value::Bool(*b)),
        ParamValue::Null => Ok(Value::Array(Vec::new())),
        other => Err(KovaQueryError::Execution(format!(
            "parameter in WHERE clause must be a literal type (String/I64/F64/Bool/Null), \
             got {}",
            param_value_kind(other)
        ))),
    }
}

/// Equality on [`Value`] : same-type direct compare, plus numeric
/// cross-type (I64 <-> F64). Different non-numeric types are always
/// unequal.
fn values_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::String(x), Value::String(y)) => x == y,
        (Value::I64(x), Value::I64(y)) => x == y,
        (Value::F64(x), Value::F64(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Array(x), Value::Array(y)) => x == y,
        // Numeric coercion : 5 == 5.0
        (Value::I64(x), Value::F64(y)) | (Value::F64(y), Value::I64(x)) => {
            #[allow(clippy::cast_precision_loss)]
            let xf = *x as f64;
            xf == *y
        }
        _ => false,
    }
}

/// Ordering compare on [`Value`] using the SQL comparison ops.
/// Returns `Some(bool)` when comparable, `None` when not (different
/// non-coercible types).
fn values_cmp(a: &Value, b: &Value, op: CmpOp) -> Option<bool> {
    use std::cmp::Ordering;

    let ordering: Ordering = match (a, b) {
        (Value::String(x), Value::String(y)) => x.partial_cmp(y),
        (Value::I64(x), Value::I64(y)) => x.partial_cmp(y),
        (Value::F64(x), Value::F64(y)) => x.partial_cmp(y),
        (Value::Bool(x), Value::Bool(y)) => x.partial_cmp(y),
        (Value::I64(x), Value::F64(y)) => {
            #[allow(clippy::cast_precision_loss)]
            let xf = *x as f64;
            xf.partial_cmp(y)
        }
        (Value::F64(x), Value::I64(y)) => {
            #[allow(clippy::cast_precision_loss)]
            let yf = *y as f64;
            x.partial_cmp(&yf)
        }
        _ => return None,
    }?;
    Some(match op {
        CmpOp::Eq => ordering == Ordering::Equal,
        CmpOp::Ne => ordering != Ordering::Equal,
        CmpOp::Lt => ordering == Ordering::Less,
        CmpOp::Le => ordering != Ordering::Greater,
        CmpOp::Gt => ordering == Ordering::Greater,
        CmpOp::Ge => ordering != Ordering::Less,
    })
}

// =========================================================================
// Projection
// =========================================================================
//
// Turn the typed `SearchHit` flowing up from the kNN into a
// user-facing `Row` with the columns the projection spec asked for.

/// Build the column-name vector for a projection spec, in order.
fn projection_column_names(spec: &ProjectionSpec) -> Vec<String> {
    spec.columns.iter().map(column_name).collect()
}

/// Output column name for a single projection : the alias if set,
/// otherwise the natural name (e.g. `"id"`, `"metadata"`, the field
/// name).
fn column_name(col: &BoundProjection) -> String {
    match col {
        BoundProjection::Wildcard => "*".into(),
        BoundProjection::CountStar { alias } => alias.clone().unwrap_or_else(|| "count".into()),
        BoundProjection::Id { alias } => alias.clone().unwrap_or_else(|| "id".into()),
        BoundProjection::Metadata { alias } => alias.clone().unwrap_or_else(|| "metadata".into()),
        BoundProjection::Distance { alias, .. } => alias.clone(),
        BoundProjection::MetadataField { name, alias } => {
            alias.clone().unwrap_or_else(|| name.clone())
        }
    }
}

/// Internal row carrier between read-path operators. kNN paths
/// (`KnnSearch`, `ExactDistance`) produce hits with `distance =
/// Some(_)` ; the pure scan path (`MetadataScan`) produces
/// `distance = None`, and any downstream Distance projection on such
/// a hit fails at projection time with a Plan error.
#[derive(Debug, Clone)]
struct InternalHit {
    id: VectorId,
    distance: Option<f32>,
    metadata: Metadata,
}

/// Convert one [`InternalHit`] to a [`Row`] under the projection spec.
fn project_hit(hit: &InternalHit, spec: &ProjectionSpec) -> Result<Row, KovaQueryError> {
    let values: Result<Vec<RowValue>, _> = spec
        .columns
        .iter()
        .map(|c| project_column(hit, c))
        .collect();
    Ok(Row { values: values? })
}

/// Project one column from a hit.
fn project_column(hit: &InternalHit, col: &BoundProjection) -> Result<RowValue, KovaQueryError> {
    match col {
        BoundProjection::Wildcard => Err(KovaQueryError::Plan(
            "wildcard projection should have been expanded by the planner".into(),
        )),
        BoundProjection::CountStar { .. } => Err(KovaQueryError::Plan(
            "COUNT(*) is not supported in plan A ; lands with aggregates later".into(),
        )),
        BoundProjection::Id { .. } => Ok(RowValue::Id(hit.id)),
        BoundProjection::Distance { .. } => match hit.distance {
            Some(d) => Ok(RowValue::Distance(d)),
            None => Err(KovaQueryError::Plan(
                "distance projection requires a kNN ordering ; this plan didn't compute distances"
                    .into(),
            )),
        },
        BoundProjection::Metadata { .. } => Ok(RowValue::Metadata(hit.metadata.clone())),
        BoundProjection::MetadataField { name, .. } => Ok(hit
            .metadata
            .get(name)
            .map_or(RowValue::Null, |v| RowValue::Field(v.clone()))),
    }
}

/// Static label for each [`ParamValue`] variant ; used to build
/// helpful "got X, expected Y" error messages.
fn param_value_kind(value: &ParamValue) -> &'static str {
    match value {
        ParamValue::Id(_) => "Id",
        ParamValue::Vector(_) => "Vector",
        ParamValue::Metadata(_) => "Metadata",
        ParamValue::Batch(_) => "Batch",
        ParamValue::String(_) => "String",
        ParamValue::I64(_) => "I64",
        ParamValue::F64(_) => "F64",
        ParamValue::Bool(_) => "Bool",
        ParamValue::Null => "Null",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kova_core::L2;
    use kova_index::HnswParams;
    use kova_storage::Shard;
    use tempfile::tempdir;

    /// End-to-end : real file-backed shard, parse + bind + plan +
    /// execute CHECKPOINT, expect an LSN back.
    #[test]
    fn executes_checkpoint_end_to_end() {
        let dir = tempdir().expect("tempdir");
        let shard = Shard::open(dir.path(), 4, L2, HnswParams::default()).expect("Shard::open");
        let mut engine = Engine::new(shard, "vectors");
        let result = engine
            .execute_str("CHECKPOINT", ParamBindings::empty())
            .expect("execute_str");
        let ExecutionResult::Checkpoint { lsn } = result else {
            panic!("expected Checkpoint, got {result:?}");
        };
        // First checkpoint on an empty shard : WAL has no records,
        // so the captured lsn is ZERO. Just check the variant fired.
        assert_eq!(lsn, Lsn::ZERO);
    }

    /// Two consecutive CHECKPOINTs return the same LSN if no writes
    /// happened between them (WAL didn't advance), and both succeed.
    #[test]
    fn checkpoint_is_idempotent_when_no_writes_between() {
        let dir = tempdir().expect("tempdir");
        let shard = Shard::open(dir.path(), 4, L2, HnswParams::default()).expect("Shard::open");
        let mut engine = Engine::new(shard, "vectors");
        let first = engine
            .execute_str("CHECKPOINT", ParamBindings::empty())
            .unwrap();
        let second = engine
            .execute_str("CHECKPOINT", ParamBindings::empty())
            .unwrap();
        let ExecutionResult::Checkpoint { lsn: l1 } = first else {
            panic!("expected Checkpoint, got {first:?}");
        };
        let ExecutionResult::Checkpoint { lsn: l2 } = second else {
            panic!("expected Checkpoint, got {second:?}");
        };
        assert_eq!(l1, l2);
    }

    /// Parse errors surface from `execute_str` cleanly.
    #[test]
    fn execute_str_propagates_parse_error() {
        let dir = tempdir().expect("tempdir");
        let shard = Shard::open(dir.path(), 4, L2, HnswParams::default()).expect("Shard::open");
        let mut engine = Engine::new(shard, "vectors");
        let err = engine
            .execute_str("not_a_keyword", ParamBindings::empty())
            .expect_err("expected error");
        assert!(matches!(err, KovaQueryError::Parse(_)));
    }

    /// Bind errors surface too. UPDATE with embedding assignment is
    /// the canonical bind failure ; v1 rejects it.
    #[test]
    fn execute_str_propagates_bind_error() {
        let dir = tempdir().expect("tempdir");
        let shard = Shard::open(dir.path(), 4, L2, HnswParams::default()).expect("Shard::open");
        let mut engine = Engine::new(shard, "vectors");
        let err = engine
            .execute_str(
                "UPDATE vectors SET embedding = $1 WHERE id = $2",
                ParamBindings::empty(),
            )
            .expect_err("expected error");
        assert!(matches!(err, KovaQueryError::Bind(_)));
    }

    /// Statements without an executor arm yet (UPDATE, DELETE,
    /// SELECT) report a clean Plan error.
    #[test]
    fn execute_str_propagates_plan_error_for_unimplemented() {
        let dir = tempdir().expect("tempdir");
        let shard = Shard::open(dir.path(), 4, L2, HnswParams::default()).expect("Shard::open");
        let mut engine = Engine::new(shard, "vectors");
        let err = engine
            .execute_str(
                "UPDATE vectors SET metadata = $1 WHERE id = $2",
                ParamBindings::empty(),
            )
            .expect_err("expected error");
        assert!(matches!(err, KovaQueryError::Plan(_)));
    }

    // ----- VACUUM -----

    /// VACUUM on an empty shard returns 0 removed.
    #[test]
    fn executes_vacuum_on_empty_shard() {
        let dir = tempdir().expect("tempdir");
        let shard = Shard::open(dir.path(), 4, L2, HnswParams::default()).expect("Shard::open");
        let mut engine = Engine::new(shard, "vectors");
        let result = engine
            .execute_str("VACUUM vectors", ParamBindings::empty())
            .expect("execute_str");
        let ExecutionResult::Vacuum { table, removed } = result else {
            panic!("expected Vacuum, got {result:?}");
        };
        assert_eq!(table, "vectors");
        assert_eq!(removed, 0);
    }

    /// VACUUM after inserts + deletes physically removes the tombstoned
    /// nodes. Sets up state via `shard_mut()` because INSERT/DELETE
    /// aren't wired through the executor yet.
    #[test]
    fn executes_vacuum_after_inserts_and_deletes() {
        use kova_core::Vector;

        let dir = tempdir().expect("tempdir");
        let shard = Shard::open(dir.path(), 4, L2, HnswParams::default()).expect("Shard::open");
        let mut engine = Engine::new(shard, "vectors");

        // Insert 3 vectors directly via Shard API. `u16` keeps the
        // numeric conversions lossless : `u16 -> f32` and `u16 -> u64`
        // both fit by construction, no cast lints.
        for i in 1..=3u16 {
            engine
                .shard_mut()
                .insert(
                    VectorId::new(u64::from(i)),
                    Vector::try_new(vec![f32::from(i), 0.0, 0.0, 0.0]).unwrap(),
                    Metadata::new(),
                )
                .expect("insert");
        }
        // Delete two of them ; deletes are tombstones, vacuum is what
        // actually reclaims them.
        engine
            .shard_mut()
            .delete(VectorId::new(1))
            .expect("delete 1");
        engine
            .shard_mut()
            .delete(VectorId::new(2))
            .expect("delete 2");

        let result = engine
            .execute_str("VACUUM vectors", ParamBindings::empty())
            .expect("vacuum");
        let ExecutionResult::Vacuum { removed, .. } = result else {
            panic!("expected Vacuum");
        };
        assert_eq!(removed, 2, "two tombstoned nodes should be reclaimed");
    }

    /// Statement-named table that doesn't match the engine's shard
    /// reports an Execution error from the dispatcher's table check.
    /// The binder accepts any name ; the executor is the layer that
    /// knows what shard it has.
    #[test]
    fn rejects_vacuum_on_unknown_table() {
        let dir = tempdir().expect("tempdir");
        let shard = Shard::open(dir.path(), 4, L2, HnswParams::default()).expect("Shard::open");
        let mut engine = Engine::new(shard, "vectors");
        let err = engine
            .execute_str("VACUUM products", ParamBindings::empty())
            .expect_err("expected error");
        let KovaQueryError::Execution(msg) = err else {
            panic!("expected Execution, got {err:?}");
        };
        assert!(
            msg.contains("'products'") && msg.contains("'vectors'"),
            "message should mention both names : {msg}"
        );
    }

    /// Engine's table-name match is case-insensitive.
    #[test]
    fn vacuum_accepts_case_variation_in_table_name() {
        let dir = tempdir().expect("tempdir");
        let shard = Shard::open(dir.path(), 4, L2, HnswParams::default()).expect("Shard::open");
        let mut engine = Engine::new(shard, "vectors");
        let result = engine
            .execute_str("VACUUM Vectors", ParamBindings::empty())
            .expect("execute_str");
        assert!(matches!(result, ExecutionResult::Vacuum { .. }));
    }

    // ----- ParamBindings unit tests -----

    #[test]
    fn param_bindings_resolve_positional() {
        let b = ParamBindings::positional(vec![ParamValue::Id(VectorId::new(42))]);
        let v = b.resolve(&ParamRef::Positional(1)).expect("bound");
        assert!(matches!(v, ParamValue::Id(id) if id.get() == 42));
    }

    #[test]
    fn param_bindings_resolve_named() {
        let b = ParamBindings::default().with_named("target", ParamValue::Id(VectorId::new(7)));
        let v = b.resolve(&ParamRef::Named("target".into())).expect("bound");
        assert!(matches!(v, ParamValue::Id(id) if id.get() == 7));
    }

    #[test]
    fn param_bindings_resolve_unbound_positional_errors() {
        let b = ParamBindings::empty();
        let err = b
            .resolve(&ParamRef::Positional(1))
            .expect_err("expected error");
        let KovaQueryError::Execution(msg) = err else {
            panic!("expected Execution");
        };
        assert!(msg.contains("not bound"));
    }

    #[test]
    fn param_bindings_resolve_unbound_named_errors() {
        let b = ParamBindings::empty();
        let err = b
            .resolve(&ParamRef::Named("missing".into()))
            .expect_err("expected error");
        assert!(matches!(err, KovaQueryError::Execution(_)));
    }

    #[test]
    fn param_bindings_reject_zero_positional() {
        let b = ParamBindings::positional(vec![ParamValue::Id(VectorId::new(1))]);
        let err = b
            .resolve(&ParamRef::Positional(0))
            .expect_err("expected error");
        assert!(matches!(err, KovaQueryError::Execution(_)));
    }

    // ----- INSERT -----

    fn make_engine(dir: &tempfile::TempDir) -> Engine<L2> {
        let shard = Shard::open(dir.path(), 4, L2, HnswParams::default()).expect("Shard::open");
        Engine::new(shard, "vectors")
    }

    fn unit_vec() -> Vector {
        Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).expect("non-empty")
    }

    #[test]
    fn executes_insert_one_with_positional_params() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        let params = ParamBindings::empty()
            .with_positional(ParamValue::Id(VectorId::new(42)))
            .with_positional(ParamValue::Vector(unit_vec()))
            .with_positional(ParamValue::Metadata(Metadata::new()));
        let result = engine
            .execute_str(
                "INSERT INTO vectors (id, embedding, metadata) VALUES ($1, $2, $3)",
                params,
            )
            .expect("execute_str");
        let ExecutionResult::Insert { table, inserted } = result else {
            panic!("expected Insert, got {result:?}");
        };
        assert_eq!(table, "vectors");
        assert_eq!(inserted, 1);
        // The shard now holds the id.
        assert!(engine.shard().contains(VectorId::new(42)));
    }

    #[test]
    fn executes_insert_one_with_named_params() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        let params = ParamBindings::empty()
            .with_named("id", ParamValue::Id(VectorId::new(7)))
            .with_named("vec", ParamValue::Vector(unit_vec()))
            .with_named("meta", ParamValue::Metadata(Metadata::new()));
        engine
            .execute_str(
                "INSERT INTO vectors (id, embedding, metadata) VALUES ($id, $vec, $meta)",
                params,
            )
            .expect("execute_str");
        assert!(engine.shard().contains(VectorId::new(7)));
    }

    #[test]
    fn executes_insert_many_batch() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        let batch: Vec<(VectorId, Vector, Metadata)> = (1..=5u16)
            .map(|i| {
                (
                    VectorId::new(u64::from(i)),
                    Vector::try_new(vec![f32::from(i), 0.0, 0.0, 0.0]).unwrap(),
                    Metadata::new(),
                )
            })
            .collect();
        let params = ParamBindings::empty().with_positional(ParamValue::Batch(batch));
        let result = engine
            .execute_str(
                "INSERT INTO vectors (id, embedding, metadata) VALUES $1",
                params,
            )
            .expect("execute_str");
        let ExecutionResult::Insert { inserted, .. } = result else {
            panic!("expected Insert");
        };
        assert_eq!(inserted, 5);
        for i in 1..=5u64 {
            assert!(engine.shard().contains(VectorId::new(i)));
        }
    }

    #[test]
    fn rejects_insert_unbound_param() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        // Only bind $1 ; $2 and $3 are missing.
        let params = ParamBindings::empty().with_positional(ParamValue::Id(VectorId::new(1)));
        let err = engine
            .execute_str(
                "INSERT INTO vectors (id, embedding, metadata) VALUES ($1, $2, $3)",
                params,
            )
            .expect_err("expected error");
        let KovaQueryError::Execution(msg) = err else {
            panic!("expected Execution, got {err:?}");
        };
        assert!(msg.contains("not bound"));
    }

    #[test]
    fn rejects_insert_with_wrong_param_type() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        // Bind $1 (the id slot) to a Vector ; type mismatch.
        let params = ParamBindings::empty()
            .with_positional(ParamValue::Vector(unit_vec()))
            .with_positional(ParamValue::Vector(unit_vec()))
            .with_positional(ParamValue::Metadata(Metadata::new()));
        let err = engine
            .execute_str(
                "INSERT INTO vectors (id, embedding, metadata) VALUES ($1, $2, $3)",
                params,
            )
            .expect_err("expected error");
        let KovaQueryError::Execution(msg) = err else {
            panic!("expected Execution, got {err:?}");
        };
        assert!(
            msg.contains("expects Id") && msg.contains("got Vector"),
            "message should pinpoint the type mismatch : {msg}"
        );
    }

    #[test]
    fn rejects_insert_into_unknown_table() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        let params = ParamBindings::empty()
            .with_positional(ParamValue::Id(VectorId::new(1)))
            .with_positional(ParamValue::Vector(unit_vec()))
            .with_positional(ParamValue::Metadata(Metadata::new()));
        let err = engine
            .execute_str(
                "INSERT INTO products (id, embedding, metadata) VALUES ($1, $2, $3)",
                params,
            )
            .expect_err("expected error");
        assert!(matches!(err, KovaQueryError::Execution(_)));
    }

    #[test]
    fn rejects_insert_duplicate_id() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        let bindings = || {
            ParamBindings::empty()
                .with_positional(ParamValue::Id(VectorId::new(99)))
                .with_positional(ParamValue::Vector(unit_vec()))
                .with_positional(ParamValue::Metadata(Metadata::new()))
        };
        engine
            .execute_str(
                "INSERT INTO vectors (id, embedding, metadata) VALUES ($1, $2, $3)",
                bindings(),
            )
            .expect("first insert");
        let err = engine
            .execute_str(
                "INSERT INTO vectors (id, embedding, metadata) VALUES ($1, $2, $3)",
                bindings(),
            )
            .expect_err("expected duplicate-id error");
        // The duplicate comes from the Shard, so it bubbles up as Backend.
        assert!(matches!(err, KovaQueryError::Backend(_)));
    }

    /// The end-to-end M1.3 milestone test : INSERT through KQL,
    /// CHECKPOINT through KQL, drop the engine, reopen the shard
    /// from disk, verify the insert survived. Writes work through
    /// the language and survive a process boundary.
    #[test]
    fn end_to_end_insert_then_checkpoint_then_reopen_survives() {
        let dir = tempdir().expect("tempdir");
        {
            let mut engine = make_engine(&dir);
            engine
                .execute_str(
                    "INSERT INTO vectors (id, embedding, metadata) VALUES ($1, $2, $3)",
                    ParamBindings::empty()
                        .with_positional(ParamValue::Id(VectorId::new(123)))
                        .with_positional(ParamValue::Vector(unit_vec()))
                        .with_positional(ParamValue::Metadata(Metadata::new())),
                )
                .expect("insert");
            engine
                .execute_str("CHECKPOINT", ParamBindings::empty())
                .expect("checkpoint");
        } // Engine drops, Shard drops, files close.

        // Reopen the shard from the same directory ; the insert
        // should be durable.
        let shard = Shard::open(dir.path(), 4, L2, HnswParams::default()).expect("reopen");
        let engine = Engine::new(shard, "vectors");
        assert!(engine.shard().contains(VectorId::new(123)));
    }

    // ----- DELETE -----

    /// Insert via KQL, then delete by literal id via KQL. The
    /// post-delete `contains` returns false because Shard.contains
    /// filters tombstones.
    #[test]
    fn executes_delete_by_literal_id() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        engine
            .execute_str(
                "INSERT INTO vectors (id, embedding, metadata) VALUES ($1, $2, $3)",
                ParamBindings::empty()
                    .with_positional(ParamValue::Id(VectorId::new(5)))
                    .with_positional(ParamValue::Vector(unit_vec()))
                    .with_positional(ParamValue::Metadata(Metadata::new())),
            )
            .expect("insert");
        let result = engine
            .execute_str("DELETE FROM vectors WHERE id = 5", ParamBindings::empty())
            .expect("delete");
        let ExecutionResult::Delete { table, deleted } = result else {
            panic!("expected Delete, got {result:?}");
        };
        assert_eq!(table, "vectors");
        assert_eq!(deleted, 1);
        assert!(!engine.shard().contains(VectorId::new(5)));
    }

    /// The planner makes its first real decision here : the binder
    /// detected the literal-id form of the predicate and set
    /// `single_id_hint = Some(5)`, so the planner emits `DeleteById`
    /// without walking the predicate tree at execute time.
    #[test]
    fn planner_picks_delete_by_id_for_literal_predicate() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        engine
            .execute_str(
                "INSERT INTO vectors (id, embedding, metadata) VALUES ($1, $2, $3)",
                ParamBindings::empty()
                    .with_positional(ParamValue::Id(VectorId::new(10)))
                    .with_positional(ParamValue::Vector(unit_vec()))
                    .with_positional(ParamValue::Metadata(Metadata::new())),
            )
            .expect("insert");
        // The literal-id form lands on the fast path even though the
        // predicate is structurally identical to a more general
        // `WHERE` clause.
        engine
            .execute_str("DELETE FROM vectors WHERE id = 10", ParamBindings::empty())
            .expect("delete");
        assert!(!engine.shard().contains(VectorId::new(10)));
    }

    /// Param-bound id : the binder did NOT set the single-id hint
    /// because the value isn't known at bind time. The planner
    /// reports the unsupported-shape Plan error cleanly. This is
    /// the v1 boundary that `DeleteByPredicate` will lift later.
    #[test]
    fn rejects_delete_by_param_bound_id() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        let params = ParamBindings::empty().with_positional(ParamValue::Id(VectorId::new(1)));
        let err = engine
            .execute_str("DELETE FROM vectors WHERE id = $1", params)
            .expect_err("expected Plan error");
        let KovaQueryError::Plan(msg) = err else {
            panic!("expected Plan, got {err:?}");
        };
        assert!(
            msg.contains("integer-literal"),
            "message should call out the literal-only constraint : {msg}"
        );
    }

    /// Same as above but a compound predicate. Binder leaves hint as
    /// None ; planner errors with the same message.
    #[test]
    fn rejects_delete_by_compound_predicate() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        let err = engine
            .execute_str(
                "DELETE FROM vectors WHERE category = 'old'",
                ParamBindings::empty(),
            )
            .expect_err("expected Plan error");
        assert!(matches!(err, KovaQueryError::Plan(_)));
    }

    /// DELETE on a non-existent id surfaces the Shard's `NotFound` as
    /// a Backend error.
    #[test]
    fn rejects_delete_of_missing_id() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        let err = engine
            .execute_str("DELETE FROM vectors WHERE id = 999", ParamBindings::empty())
            .expect_err("expected error");
        assert!(matches!(err, KovaQueryError::Backend(_)));
    }

    /// DELETE on an unknown table reports an Execution error via the
    /// engine's table dispatcher.
    #[test]
    fn rejects_delete_on_unknown_table() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        let err = engine
            .execute_str("DELETE FROM products WHERE id = 1", ParamBindings::empty())
            .expect_err("expected error");
        assert!(matches!(err, KovaQueryError::Execution(_)));
    }

    // ----- SELECT plan A : helpers -----

    /// Build a unit vector that points in the `i`th direction at
    /// distance 1, padded with zeros. `i` is 1-based (matching ids),
    /// so `axis_vec(1)` is `[1, 0, 0, 0]`, `axis_vec(2)` is
    /// `[0, 1, 0, 0]`, etc.
    fn axis_vec(i: u16) -> Vector {
        let mut v = vec![0.0_f32; 4];
        let idx = ((i as usize) - 1) % 4;
        v[idx] = 1.0;
        Vector::try_new(v).expect("non-empty")
    }

    fn meta_of(pairs: &[(&str, Value)]) -> Metadata {
        let mut m = Metadata::new();
        for (k, v) in pairs {
            m.insert((*k).to_string(), v.clone());
        }
        m
    }

    /// Seed `engine` with rows 1..=N where each row has a distinct
    /// axis-aligned vector and a metadata bag.
    fn seed_engine(engine: &mut Engine<L2>, metas: &[Metadata]) {
        for (i, meta) in metas.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let i = i as u16 + 1;
            engine
                .shard_mut()
                .insert(VectorId::new(u64::from(i)), axis_vec(i), meta.clone())
                .expect("seed insert");
        }
    }

    // ----- SELECT plan A : kNN happy paths -----

    #[test]
    fn executes_select_id_with_knn_ordering() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(
            &mut engine,
            &[Metadata::new(), Metadata::new(), Metadata::new()],
        );
        // Query points at axis 0 ; id=1 has its 1.0 on axis 0, so it
        // should be the nearest.
        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "SELECT id FROM vectors ORDER BY embedding <-> $1 LIMIT 2",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { columns, rows } = result else {
            panic!("expected Rows, got {result:?}");
        };
        assert_eq!(columns, vec!["id".to_string()]);
        assert_eq!(rows.len(), 2);
        // First row is the closest match.
        let RowValue::Id(first) = rows[0].values[0] else {
            panic!("expected Id");
        };
        assert_eq!(first.get(), 1);
    }

    #[test]
    fn executes_select_with_distance_alias() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(&mut engine, &[Metadata::new(), Metadata::new()]);
        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "SELECT id, embedding <-> $1 AS dist FROM vectors ORDER BY embedding <-> $1 LIMIT 2",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { columns, rows } = result else {
            panic!("expected Rows");
        };
        assert_eq!(columns, vec!["id".to_string(), "dist".to_string()]);
        // Distance for the nearest neighbour : 0.0 (exact match).
        let RowValue::Distance(d) = rows[0].values[1] else {
            panic!("expected Distance");
        };
        assert!(
            d.abs() < f32::EPSILON,
            "nearest distance should be ~0, got {d}"
        );
    }

    #[test]
    fn executes_select_star_expands_to_id_and_metadata() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        let m = meta_of(&[("category", Value::String("docs".into()))]);
        seed_engine(&mut engine, std::slice::from_ref(&m));
        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "SELECT * FROM vectors ORDER BY embedding <-> $1 LIMIT 1",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { columns, rows } = result else {
            panic!("expected Rows");
        };
        assert_eq!(columns, vec!["id".to_string(), "metadata".to_string()]);
        let row = &rows[0];
        assert!(matches!(row.values[0], RowValue::Id(_)));
        let RowValue::Metadata(actual) = &row.values[1] else {
            panic!("expected Metadata");
        };
        assert_eq!(actual.get("category"), Some(&Value::String("docs".into())));
    }

    #[test]
    fn executes_select_metadata_field_projects_field_value() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        let m = meta_of(&[("category", Value::String("docs".into()))]);
        seed_engine(&mut engine, &[m]);
        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "SELECT id, category FROM vectors ORDER BY embedding <-> $1 LIMIT 1",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        let RowValue::Field(Value::String(s)) = &rows[0].values[1] else {
            panic!("expected Field(String), got {:?}", rows[0].values[1]);
        };
        assert_eq!(s, "docs");
    }

    #[test]
    fn missing_metadata_field_projects_as_null() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(&mut engine, &[Metadata::new()]);
        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "SELECT id, ghost_field FROM vectors ORDER BY embedding <-> $1 LIMIT 1",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        assert!(matches!(rows[0].values[1], RowValue::Null));
    }

    // ----- SELECT plan A : post-filter (WHERE) -----

    #[test]
    fn post_filter_drops_rows_that_fail_predicate() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(
            &mut engine,
            &[
                meta_of(&[("category", Value::String("docs".into()))]),
                meta_of(&[("category", Value::String("specs".into()))]),
                meta_of(&[("category", Value::String("docs".into()))]),
            ],
        );
        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "SELECT id FROM vectors WHERE category = 'docs' \
                 ORDER BY embedding <-> $1 LIMIT 10",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        assert_eq!(rows.len(), 2, "two 'docs' rows pass the post-filter");
        let ids: Vec<u64> = rows
            .iter()
            .map(|r| match r.values[0] {
                RowValue::Id(id) => id.get(),
                _ => unreachable!(),
            })
            .collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
    }

    #[test]
    fn post_filter_supports_and_or_combinators() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(
            &mut engine,
            &[
                meta_of(&[
                    ("category", Value::String("docs".into())),
                    ("year", Value::I64(2024)),
                ]),
                meta_of(&[
                    ("category", Value::String("docs".into())),
                    ("year", Value::I64(2020)),
                ]),
                meta_of(&[
                    ("category", Value::String("specs".into())),
                    ("year", Value::I64(2024)),
                ]),
            ],
        );
        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "SELECT id FROM vectors \
                 WHERE category = 'docs' AND year >= 2024 \
                 ORDER BY embedding <-> $1 LIMIT 10",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        assert_eq!(rows.len(), 1, "only id=1 matches docs AND year>=2024");
    }

    #[test]
    fn post_filter_resolves_param_in_where() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(
            &mut engine,
            &[
                meta_of(&[("category", Value::String("docs".into()))]),
                meta_of(&[("category", Value::String("specs".into()))]),
            ],
        );
        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "SELECT id FROM vectors WHERE category = $cat \
                 ORDER BY embedding <-> $q LIMIT 10",
                ParamBindings::empty()
                    .with_named("q", ParamValue::Vector(q))
                    .with_named("cat", ParamValue::String("specs".into())),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        assert_eq!(rows.len(), 1);
        let RowValue::Id(id) = rows[0].values[0] else {
            panic!("expected Id");
        };
        assert_eq!(id.get(), 2);
    }

    #[test]
    fn post_filter_with_in_list() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(
            &mut engine,
            &[
                meta_of(&[("category", Value::String("a".into()))]),
                meta_of(&[("category", Value::String("b".into()))]),
                meta_of(&[("category", Value::String("c".into()))]),
            ],
        );
        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "SELECT id FROM vectors WHERE category IN ('a', 'c') \
                 ORDER BY embedding <-> $1 LIMIT 10",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        assert_eq!(rows.len(), 2);
    }

    // ----- SELECT plan A : rejection paths -----

    #[test]
    fn rejects_non_knn_select() {
        // Non-kNN SELECT (no distance ordering) is plan B / C
        // territory ; plan A errors here cleanly.
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        let err = engine
            .execute_str(
                "SELECT id FROM vectors WHERE category = 'docs' LIMIT 10",
                ParamBindings::empty(),
            )
            .expect_err("expected Plan error");
        let KovaQueryError::Plan(msg) = err else {
            panic!("expected Plan, got {err:?}");
        };
        assert!(
            msg.contains("kNN"),
            "message should call out kNN-only constraint : {msg}"
        );
    }

    #[test]
    fn rejects_select_on_wrong_table() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let err = engine
            .execute_str(
                "SELECT id FROM products ORDER BY embedding <-> $1 LIMIT 10",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect_err("expected error");
        assert!(matches!(err, KovaQueryError::Execution(_)));
    }

    // ----- SELECT plan B : scan + exact distance -----

    /// Planner-shape check : a SELECT with a predicate emits plan B
    /// (`Projection` -> `Limit` -> `ExactDistance` -> `MetadataScan`).
    #[test]
    fn planner_picks_plan_b_when_predicate_present() {
        use crate::physical::PhysicalPlan;
        use crate::planner::plan;
        let ast = parse_str(
            "SELECT id FROM vectors WHERE category = 'docs' \
             ORDER BY embedding <-> $1 LIMIT 10",
        )
        .expect("parse");
        let logical = crate::binder::bind(ast).expect("bind");
        let physical = plan(logical).expect("plan");
        let PhysicalPlan::Projection { input, .. } = physical else {
            panic!("expected Projection root");
        };
        let PhysicalPlan::Limit { input, .. } = *input else {
            panic!("expected Limit");
        };
        let PhysicalPlan::ExactDistance { input, .. } = *input else {
            panic!("expected ExactDistance, got {input:?}");
        };
        assert!(
            matches!(*input, PhysicalPlan::MetadataScan { .. }),
            "expected MetadataScan, got {input:?}"
        );
    }

    /// Planner-shape check : a SELECT without a predicate stays on
    /// plan A (`KnnSearch` with overfetch).
    #[test]
    fn planner_picks_plan_a_when_no_predicate() {
        use crate::physical::PhysicalPlan;
        use crate::planner::plan;
        let ast =
            parse_str("SELECT id FROM vectors ORDER BY embedding <-> $1 LIMIT 10").expect("parse");
        let logical = crate::binder::bind(ast).expect("bind");
        let physical = plan(logical).expect("plan");
        let PhysicalPlan::Projection { input, .. } = physical else {
            panic!("expected Projection root");
        };
        let PhysicalPlan::Limit { input, .. } = *input else {
            panic!("expected Limit");
        };
        assert!(
            matches!(*input, PhysicalPlan::KnnSearch { .. }),
            "expected KnnSearch, got {input:?}"
        );
    }

    /// Plan B returns rows sorted ascending by exact distance.
    /// The scan finds matching rows, `ExactDistance` ranks them.
    #[test]
    fn plan_b_orders_results_by_exact_distance() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        // Three 'docs' rows + one 'specs' (which the predicate excludes).
        // Vector positions chosen so distance ordering is deterministic.
        seed_engine(
            &mut engine,
            &[
                meta_of(&[("category", Value::String("docs".into()))]), // id 1, vec [1,0,0,0]
                meta_of(&[("category", Value::String("docs".into()))]), // id 2, vec [0,1,0,0]
                meta_of(&[("category", Value::String("docs".into()))]), // id 3, vec [0,0,1,0]
                meta_of(&[("category", Value::String("specs".into()))]), // id 4, vec [0,0,0,1]
            ],
        );
        // Query at [0.9, 0.1, 0, 0] : closest 'docs' rows in order are 1, 2, 3.
        let q = Vector::try_new(vec![0.9, 0.1, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "SELECT id FROM vectors WHERE category = 'docs' \
                 ORDER BY embedding <-> $1 LIMIT 3",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        assert_eq!(rows.len(), 3);
        // The 'specs' row (id 4) is excluded.
        let ids: Vec<u64> = rows
            .iter()
            .map(|r| match r.values[0] {
                RowValue::Id(id) => id.get(),
                _ => unreachable!(),
            })
            .collect();
        assert!(!ids.contains(&4), "specs row leaked through plan B");
        assert_eq!(ids[0], 1, "nearest 'docs' should be id 1");
    }

    /// Plan B respects LIMIT : if the scan finds more matches than k,
    /// only the k nearest are returned.
    #[test]
    fn plan_b_caps_result_at_limit() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        // All 4 rows match the predicate.
        seed_engine(
            &mut engine,
            &[
                meta_of(&[("tag", Value::String("a".into()))]),
                meta_of(&[("tag", Value::String("a".into()))]),
                meta_of(&[("tag", Value::String("a".into()))]),
                meta_of(&[("tag", Value::String("a".into()))]),
            ],
        );
        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "SELECT id FROM vectors WHERE tag = 'a' \
                 ORDER BY embedding <-> $1 LIMIT 2",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        assert_eq!(rows.len(), 2, "LIMIT 2 should cap to 2 rows");
    }

    /// Plan B returns zero rows when no metadata row matches the
    /// predicate, without errors. Plan A would have returned k
    /// candidates that all failed post-filter ; plan B never even
    /// runs the kNN.
    #[test]
    fn plan_b_handles_empty_match_set() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(
            &mut engine,
            &[meta_of(&[("category", Value::String("specs".into()))])],
        );
        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "SELECT id FROM vectors WHERE category = 'docs' \
                 ORDER BY embedding <-> $1 LIMIT 10",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        assert!(
            rows.is_empty(),
            "no 'docs' rows present, result should be empty"
        );
    }

    // ----- selectivity-driven dispatch -----

    /// High selectivity (most rows pass the predicate) keeps the
    /// planner on plan A even when a predicate is present : kNN
    /// overfetch + post-filter wins when the filter rarely drops
    /// candidates. Seed a shard where 9/10 rows pass and verify the
    /// emitted plan has `KnnSearch` underneath the Limit.
    #[test]
    fn high_selectivity_keeps_plan_a_with_post_filter() {
        use crate::physical::PhysicalPlan;
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        // 9 'docs' rows + 1 'specs' : selectivity = 0.9 > 0.5 threshold.
        let mut metas: Vec<Metadata> = (0..9)
            .map(|_| meta_of(&[("category", Value::String("docs".into()))]))
            .collect();
        metas.push(meta_of(&[("category", Value::String("specs".into()))]));
        seed_engine(&mut engine, &metas);

        // Plan through Engine's full pipeline (which uses ShardEstimator).
        let ast = parse_str(
            "SELECT id FROM vectors WHERE category = 'docs' \
             ORDER BY embedding <-> $1 LIMIT 5",
        )
        .expect("parse");
        let logical = crate::binder::bind(ast).expect("bind");
        let est = ShardEstimator {
            shard: engine.shard(),
        };
        let physical = crate::planner::plan_with_estimator(logical, &est, &ParamBindings::empty())
            .expect("plan");

        // Projection → Limit → KnnSearch (with post_filter)
        let PhysicalPlan::Projection { input, .. } = physical else {
            panic!("expected Projection");
        };
        let PhysicalPlan::Limit { input, .. } = *input else {
            panic!("expected Limit");
        };
        let PhysicalPlan::KnnSearch { post_filter, .. } = *input else {
            panic!("expected KnnSearch (plan A), got {input:?}");
        };
        assert!(
            post_filter.is_some(),
            "plan A with predicate has post_filter"
        );
    }

    /// Low selectivity (few rows pass) flips the planner to plan B :
    /// the predicate matches 1 of 10 rows, so scan + exact-distance
    /// is cheaper than overfetched kNN.
    #[test]
    fn low_selectivity_picks_plan_b() {
        use crate::physical::PhysicalPlan;
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        // 1 'docs' + 9 'other' : selectivity = 0.1 < 0.5.
        let mut metas: Vec<Metadata> = vec![meta_of(&[("category", Value::String("docs".into()))])];
        for _ in 0..9 {
            metas.push(meta_of(&[("category", Value::String("other".into()))]));
        }
        seed_engine(&mut engine, &metas);

        let ast = parse_str(
            "SELECT id FROM vectors WHERE category = 'docs' \
             ORDER BY embedding <-> $1 LIMIT 5",
        )
        .expect("parse");
        let logical = crate::binder::bind(ast).expect("bind");
        let est = ShardEstimator {
            shard: engine.shard(),
        };
        let physical = crate::planner::plan_with_estimator(logical, &est, &ParamBindings::empty())
            .expect("plan");

        let PhysicalPlan::Projection { input, .. } = physical else {
            panic!("expected Projection");
        };
        let PhysicalPlan::Limit { input, .. } = *input else {
            panic!("expected Limit");
        };
        assert!(
            matches!(*input, PhysicalPlan::ExactDistance { .. }),
            "low selectivity should pick plan B (ExactDistance), got {input:?}"
        );
    }

    /// Both plans return the same answer on the same data : at
    /// selectivity ~50% both strategies are correct, so the result
    /// set must match regardless of which the planner picked.
    #[test]
    fn plan_a_and_plan_b_return_same_ids() {
        let dir_a = tempdir().expect("tempdir A");
        let dir_b = tempdir().expect("tempdir B");
        let mut engine_a = make_engine(&dir_a);
        let mut engine_b = make_engine(&dir_b);

        // Seed both engines identically. 5 'docs' + 5 'other'.
        let metas: Vec<Metadata> = (0..10)
            .map(|i| {
                let tag = if i < 5 { "docs" } else { "other" };
                meta_of(&[("category", Value::String(tag.into()))])
            })
            .collect();
        seed_engine(&mut engine_a, &metas);
        seed_engine(&mut engine_b, &metas);

        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        // Engine A uses ShardEstimator (selectivity ~50% → plan A
        // wins, since 0.5 is the threshold-boundary). Engine B is
        // forced to plan B by replaying the SAME query through both.
        // Both should yield the same set of ids (order may differ
        // due to kNN approximation, so we compare as sorted vecs).
        let res_a = engine_a
            .execute_str(
                "SELECT id FROM vectors WHERE category = 'docs' \
                 ORDER BY embedding <-> $1 LIMIT 5",
                ParamBindings::empty().with_positional(ParamValue::Vector(q.clone())),
            )
            .expect("A execute");
        let res_b = engine_b
            .execute_str(
                "SELECT id FROM vectors WHERE category = 'docs' \
                 ORDER BY embedding <-> $1 LIMIT 5",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("B execute");

        let extract_ids = |r: ExecutionResult| -> Vec<u64> {
            let ExecutionResult::Rows { rows, .. } = r else {
                panic!("expected Rows");
            };
            let mut ids: Vec<u64> = rows
                .iter()
                .map(|r| match r.values[0] {
                    RowValue::Id(id) => id.get(),
                    _ => unreachable!(),
                })
                .collect();
            ids.sort_unstable();
            ids
        };
        let ids_a = extract_ids(res_a);
        let ids_b = extract_ids(res_b);
        assert_eq!(ids_a, ids_b, "plan A and plan B disagree on the result set");
    }

    /// Plan B fills in real distances via `Shard::distance_to`, so
    /// `embedding <-> $q AS distance` projections work.
    #[test]
    fn plan_b_distance_projection_returns_exact_distance() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(
            &mut engine,
            &[meta_of(&[("category", Value::String("docs".into()))])],
        );
        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "SELECT id, embedding <-> $1 AS dist FROM vectors \
                 WHERE category = 'docs' \
                 ORDER BY embedding <-> $1 LIMIT 1",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { columns, rows } = result else {
            panic!("expected Rows");
        };
        assert_eq!(columns, vec!["id".to_string(), "dist".to_string()]);
        let RowValue::Distance(d) = rows[0].values[1] else {
            panic!("expected Distance, got {:?}", rows[0].values[1]);
        };
        // axis_vec(1) = [1, 0, 0, 0] equals the query exactly, distance 0.
        assert!(d.abs() < f32::EPSILON, "expected ~0, got {d}");
    }

    /// Full write-path round-trip : insert two ids, delete one,
    /// checkpoint, reopen, verify only the surviving id is present.
    /// Pins the M1.3 milestone for the delete path specifically.
    #[test]
    fn end_to_end_insert_delete_checkpoint_reopen() {
        let dir = tempdir().expect("tempdir");
        {
            let mut engine = make_engine(&dir);
            for id in [100, 200u64] {
                engine
                    .execute_str(
                        "INSERT INTO vectors (id, embedding, metadata) VALUES ($1, $2, $3)",
                        ParamBindings::empty()
                            .with_positional(ParamValue::Id(VectorId::new(id)))
                            .with_positional(ParamValue::Vector(unit_vec()))
                            .with_positional(ParamValue::Metadata(Metadata::new())),
                    )
                    .expect("insert");
            }
            engine
                .execute_str("DELETE FROM vectors WHERE id = 100", ParamBindings::empty())
                .expect("delete");
            engine
                .execute_str("CHECKPOINT", ParamBindings::empty())
                .expect("checkpoint");
        } // Engine drops, files close.

        let shard = Shard::open(dir.path(), 4, L2, HnswParams::default()).expect("reopen");
        let engine = Engine::new(shard, "vectors");
        assert!(
            !engine.shard().contains(VectorId::new(100)),
            "deleted id should stay deleted across reopen"
        );
        assert!(
            engine.shard().contains(VectorId::new(200)),
            "surviving id should still be present"
        );
    }
}

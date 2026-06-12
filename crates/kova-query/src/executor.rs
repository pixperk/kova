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
    /// UPDATE completed ; `updated` rows' metadata bags were rewritten
    /// in `table`.
    Update {
        /// Target table the operation ran against.
        table: String,
        /// Number of rows whose metadata was replaced.
        updated: u64,
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
                id,
                embedding,
                metadata,
            } => self.exec_insert_one(table, &id, &embedding, &metadata, params),
            PhysicalPlan::InsertMany { table, batch } => {
                self.exec_insert_many(table, &batch, params)
            }
            PhysicalPlan::DeleteById { table, id } => {
                self.assert_table(&table)?;
                self.shard
                    .delete(id)
                    .map_err(|e| KovaQueryError::Backend(Box::new(e)))?;
                Ok(ExecutionResult::Delete { table, deleted: 1 })
            }
            PhysicalPlan::DeleteByParamId { table, id_param } => {
                self.assert_table(&table)?;
                let id = expect_id(params.resolve(&id_param)?, "id")?;
                self.shard
                    .delete(id)
                    .map_err(|e| KovaQueryError::Backend(Box::new(e)))?;
                Ok(ExecutionResult::Delete { table, deleted: 1 })
            }
            PhysicalPlan::DeleteByPredicate { table, predicate } => {
                self.exec_delete_by_predicate(&table, &predicate, params)
            }
            PhysicalPlan::DeleteByRadius {
                table,
                query,
                metric: _,
                radius,
                inclusive,
                post_filter,
            } => {
                let op = RadiusOp {
                    query: &query,
                    radius,
                    inclusive,
                    post_filter: post_filter.as_ref(),
                };
                self.exec_delete_by_radius(&table, &op, params)
            }
            PhysicalPlan::UpdateById { .. }
            | PhysicalPlan::UpdateByParamId { .. }
            | PhysicalPlan::UpdateByPredicate { .. }
            | PhysicalPlan::UpdateByRadius { .. } => self.dispatch_update(plan, params),

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
            PhysicalPlan::Count {
                table,
                predicate,
                column_name,
            } => self.exec_count(&table, predicate, column_name, params),
            PhysicalPlan::Limit { .. }
            | PhysicalPlan::KnnSearch { .. }
            | PhysicalPlan::MetadataScan { .. }
            | PhysicalPlan::ExactDistance { .. }
            | PhysicalPlan::RadiusSearch { .. }
            | PhysicalPlan::FilteredKnnSearch { .. } => Err(KovaQueryError::Plan(
                "read-path operator at top level ; planner must wrap in Projection".into(),
            )),
        }
    }

    /// Single-row INSERT : resolve the three parameter slots into
    /// concrete values and dispatch to `Shard::insert`.
    fn exec_insert_one(
        &mut self,
        table: String,
        id_ref: &crate::ast::ParamRef,
        emb_ref: &crate::ast::ParamRef,
        meta_ref: &crate::ast::ParamRef,
        params: &ParamBindings,
    ) -> Result<ExecutionResult, KovaQueryError> {
        self.assert_table(&table)?;
        let id = expect_id(params.resolve(id_ref)?, "id")?;
        let embedding = expect_vector(params.resolve(emb_ref)?, "embedding")?;
        let metadata = expect_metadata(params.resolve(meta_ref)?, "metadata")?;
        self.shard
            .insert(id, embedding, metadata)
            .map_err(|e| KovaQueryError::Backend(Box::new(e)))?;
        Ok(ExecutionResult::Insert { table, inserted: 1 })
    }

    /// Batch INSERT : resolve the single batch param into a Vec of
    /// tuples and dispatch to `Shard::insert_many`.
    fn exec_insert_many(
        &mut self,
        table: String,
        batch_ref: &crate::ast::ParamRef,
        params: &ParamBindings,
    ) -> Result<ExecutionResult, KovaQueryError> {
        self.assert_table(&table)?;
        let batch = expect_batch(params.resolve(batch_ref)?, "batch")?;
        let inserted = batch.len() as u64;
        self.shard
            .insert_many(batch)
            .map_err(|e| KovaQueryError::Backend(Box::new(e)))?;
        Ok(ExecutionResult::Insert { table, inserted })
    }

    /// COUNT(*) : either `Shard::len()` (no predicate, fast path) or
    /// `Shard::count_matching` with an evaluated predicate. Returns a
    /// one-row, one-column `Rows` result.
    fn exec_count(
        &self,
        table: &str,
        predicate: Option<PredicateExpr>,
        column_name: String,
        params: &ParamBindings,
    ) -> Result<ExecutionResult, KovaQueryError> {
        self.assert_table(table)?;
        let count = match predicate {
            None => self.shard.len(),
            Some(pred) => count_matching_with_predicate(&self.shard, &pred, params)?,
        };
        let count_i64 = i64::try_from(count).unwrap_or(i64::MAX);
        let row = Row {
            values: vec![RowValue::Field(Value::I64(count_i64))],
        };
        Ok(ExecutionResult::Rows {
            columns: vec![column_name],
            rows: vec![row],
        })
    }

    /// Filtered-kNN read-path arm. Plan C : threads the predicate
    /// into the HNSW walk via `Shard::search_filtered`. The closure-
    /// error-capture pattern propagates any predicate-eval failure
    /// instead of silently dropping rows.
    fn exec_filtered_knn(
        &self,
        table: &str,
        query: &crate::ast::ParamRef,
        k: usize,
        filter: &PredicateExpr,
        params: &ParamBindings,
    ) -> Result<Vec<InternalHit>, KovaQueryError> {
        self.assert_table(table)?;
        let query_vec = expect_vector(params.resolve(query)?, "query")?;
        let mut closure_err: Option<KovaQueryError> = None;
        let hits = self
            .shard
            .search_filtered(&query_vec, k, |m| {
                if closure_err.is_some() {
                    return false;
                }
                match eval_predicate(filter, m, params) {
                    Ok(b) => b,
                    Err(e) => {
                        closure_err = Some(e);
                        false
                    }
                }
            })
            .map_err(|e| KovaQueryError::Backend(Box::new(e)))?;
        if let Some(e) = closure_err {
            return Err(e);
        }
        Ok(hits
            .into_iter()
            .map(|h| InternalHit {
                id: h.id,
                distance: Some(h.distance),
                metadata: h.metadata,
            })
            .collect())
    }

    /// Dispatch the four UPDATE variants to their respective exec
    /// helpers. Kept out of [`Self::execute`] to hold the dispatch
    /// table under its line cap.
    fn dispatch_update(
        &mut self,
        plan: PhysicalPlan,
        params: &ParamBindings,
    ) -> Result<ExecutionResult, KovaQueryError> {
        match plan {
            PhysicalPlan::UpdateById {
                table,
                id,
                assignments,
            } => self.exec_update_by_id(&table, id, &assignments, params),
            PhysicalPlan::UpdateByParamId {
                table,
                id_param,
                assignments,
            } => {
                let id = expect_id(params.resolve(&id_param)?, "id")?;
                self.exec_update_by_id(&table, id, &assignments, params)
            }
            PhysicalPlan::UpdateByPredicate {
                table,
                predicate,
                assignments,
            } => self.exec_update_by_predicate(&table, &predicate, &assignments, params),
            PhysicalPlan::UpdateByRadius {
                table,
                query,
                metric: _,
                radius,
                inclusive,
                post_filter,
                assignments,
            } => {
                let op = RadiusOp {
                    query: &query,
                    radius,
                    inclusive,
                    post_filter: post_filter.as_ref(),
                };
                self.exec_update_by_radius(&table, &op, &assignments, params)
            }
            other => Err(KovaQueryError::Execution(format!(
                "dispatch_update called with non-update plan : {}",
                physical_kind(&other)
            ))),
        }
    }

    /// UPDATE-by-id arm. Fetches the current metadata bag (errors if
    /// the id is missing or tombstoned), applies each assignment to a
    /// fresh copy, and dispatches the resulting bag to
    /// `Shard::update_metadata`.
    fn exec_update_by_id(
        &mut self,
        table: &str,
        id: VectorId,
        assignments: &[LogicalAssignment],
        params: &ParamBindings,
    ) -> Result<ExecutionResult, KovaQueryError> {
        self.assert_table(table)?;
        let mut bag = self.shard.get_metadata(id).ok_or_else(|| {
            KovaQueryError::Execution(format!(
                "UPDATE target id {id:?} not found in metadata store"
            ))
        })?;
        apply_assignments(&mut bag, assignments, params)?;
        let updated = self
            .shard
            .update_metadata(std::iter::once((id, bag)))
            .map_err(|e| KovaQueryError::Backend(Box::new(e)))?;
        Ok(ExecutionResult::Update {
            table: table.to_string(),
            updated: updated as u64,
        })
    }

    /// UPDATE-by-predicate arm. Scans metadata for ids whose bag
    /// passes `predicate`, builds the `(id, new_bag)` pairs in memory,
    /// then dispatches the whole batch to `Shard::update_metadata`
    /// for one WAL group-commit. Uses the closure-error-capture
    /// pattern to propagate predicate-eval failures.
    fn exec_update_by_predicate(
        &mut self,
        table: &str,
        predicate: &PredicateExpr,
        assignments: &[LogicalAssignment],
        params: &ParamBindings,
    ) -> Result<ExecutionResult, KovaQueryError> {
        self.assert_table(table)?;
        let ids = ids_matching(&self.shard, predicate, params)?;
        let staged = self.build_updates_from_ids(&ids, assignments, params)?;
        let written = self
            .shard
            .update_metadata(staged)
            .map_err(|e| KovaQueryError::Backend(Box::new(e)))?;
        Ok(ExecutionResult::Update {
            table: table.to_string(),
            updated: written as u64,
        })
    }

    /// UPDATE-by-radius arm. Runs the radius walk to produce the
    /// in-ball hit set, applies strict-boundary and post-filter drops,
    /// then writes the assignment-mutated bags via
    /// `Shard::update_metadata`. Reuses the metadata bag carried by
    /// each `SearchHit` instead of re-fetching from the store.
    fn exec_update_by_radius(
        &mut self,
        table: &str,
        op: &RadiusOp<'_>,
        assignments: &[LogicalAssignment],
        params: &ParamBindings,
    ) -> Result<ExecutionResult, KovaQueryError> {
        self.assert_table(table)?;
        let query_vec = expect_vector(params.resolve(op.query)?, "query")?;
        let hits = self
            .shard
            .search_radius(&query_vec, op.radius)
            .map_err(|e| KovaQueryError::Backend(Box::new(e)))?;
        let mut staged: Vec<(VectorId, Metadata)> = Vec::with_capacity(hits.len());
        for h in hits {
            if !op.inclusive && h.distance >= op.radius {
                continue;
            }
            if let Some(pred) = op.post_filter
                && !eval_predicate(pred, &h.metadata, params)?
            {
                continue;
            }
            let mut bag = h.metadata;
            apply_assignments(&mut bag, assignments, params)?;
            staged.push((h.id, bag));
        }
        let written = self
            .shard
            .update_metadata(staged)
            .map_err(|e| KovaQueryError::Backend(Box::new(e)))?;
        Ok(ExecutionResult::Update {
            table: table.to_string(),
            updated: written as u64,
        })
    }

    /// Fetch the current bag for each id and apply `assignments` to a
    /// copy. Returns the staged batch ready for
    /// `Shard::update_metadata`. Used by the predicate path where the
    /// id-producer (`scan_metadata`) doesn't carry bags.
    fn build_updates_from_ids(
        &self,
        ids: &[VectorId],
        assignments: &[LogicalAssignment],
        params: &ParamBindings,
    ) -> Result<Vec<(VectorId, Metadata)>, KovaQueryError> {
        let mut updates: Vec<(VectorId, Metadata)> = Vec::with_capacity(ids.len());
        for &id in ids {
            let Some(mut bag) = self.shard.get_metadata(id) else {
                return Err(KovaQueryError::Execution(format!(
                    "id {id:?} disappeared from metadata store between scan and update"
                )));
            };
            apply_assignments(&mut bag, assignments, params)?;
            updates.push((id, bag));
        }
        Ok(updates)
    }

    /// DELETE-by-radius write-path arm. Runs the radius walk to
    /// produce the in-ball id set, drops boundary hits when the user
    /// wrote `<` (strict), applies any `post_filter` residue against
    /// each hit's metadata, then dispatches the survivors to
    /// `Shard::delete_many` for one WAL group-commit.
    fn exec_delete_by_radius(
        &mut self,
        table: &str,
        op: &RadiusOp<'_>,
        params: &ParamBindings,
    ) -> Result<ExecutionResult, KovaQueryError> {
        self.assert_table(table)?;
        let query_vec = expect_vector(params.resolve(op.query)?, "query")?;
        let hits = self
            .shard
            .search_radius(&query_vec, op.radius)
            .map_err(|e| KovaQueryError::Backend(Box::new(e)))?;
        let mut ids: Vec<VectorId> = Vec::with_capacity(hits.len());
        for h in hits {
            if !op.inclusive && h.distance >= op.radius {
                continue;
            }
            if let Some(pred) = op.post_filter
                && !eval_predicate(pred, &h.metadata, params)?
            {
                continue;
            }
            ids.push(h.id);
        }
        let deleted = self
            .shard
            .delete_many(ids)
            .map_err(|e| KovaQueryError::Backend(Box::new(e)))?;
        Ok(ExecutionResult::Delete {
            table: table.to_string(),
            deleted: deleted as u64,
        })
    }

    /// DELETE-by-predicate write-path arm. Scans the metadata for ids
    /// whose bag passes `predicate`, then dispatches the whole id set
    /// to `Shard::delete_many` for one batched WAL commit. The
    /// closure-error-capture pattern propagates any predicate-eval
    /// failure instead of silently dropping rows.
    fn exec_delete_by_predicate(
        &mut self,
        table: &str,
        predicate: &PredicateExpr,
        params: &ParamBindings,
    ) -> Result<ExecutionResult, KovaQueryError> {
        self.assert_table(table)?;
        let ids = ids_matching(&self.shard, predicate, params)?;
        let deleted = self
            .shard
            .delete_many(ids)
            .map_err(|e| KovaQueryError::Backend(Box::new(e)))?;
        Ok(ExecutionResult::Delete {
            table: table.to_string(),
            deleted: deleted as u64,
        })
    }

    /// Metadata-scan read-path arm. Walks every live row's metadata,
    /// keeps the ones the predicate accepts, and emits `InternalHit`s
    /// with `distance: None` (the downstream `ExactDistance` fills it).
    /// Uses the closure-error-capture pattern so predicate-eval errors
    /// surface instead of getting swallowed as `false`.
    fn exec_metadata_scan(
        &self,
        table: &str,
        predicate: &PredicateExpr,
        params: &ParamBindings,
    ) -> Result<Vec<InternalHit>, KovaQueryError> {
        self.assert_table(table)?;

        // Ask the catalog first. Three outcomes : the full predicate
        // is index-evaluable (Full), part of it is (Hybrid, with a
        // residue we still evaluate per-row), or none of it is
        // (Fallback to the scan path below).
        let catalog = self.shard.catalog();
        match crate::index_eval::try_index_eval(predicate, catalog, params) {
            crate::index_eval::IndexEval::Full(bitmap) => Ok(bitmap_to_hits(&self.shard, &bitmap)),
            crate::index_eval::IndexEval::Hybrid {
                candidates,
                residue,
            } => {
                let mut out = Vec::new();
                for raw_id in &candidates {
                    let id = VectorId::from(raw_id);
                    let Some(meta) = self.shard.get_metadata(id) else {
                        // Tombstoned between catalog snapshot and now,
                        // or never had metadata. Either way, skip.
                        continue;
                    };
                    if !eval_predicate(&residue, &meta, params)? {
                        continue;
                    }
                    out.push(InternalHit {
                        id,
                        distance: None,
                        metadata: meta,
                    });
                }
                Ok(out)
            }
            crate::index_eval::IndexEval::Fallback => {
                let mut closure_err: Option<KovaQueryError> = None;
                let ids = self.shard.scan_metadata(|m| {
                    if closure_err.is_some() {
                        return false;
                    }
                    match eval_predicate(predicate, m, params) {
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
        }
    }

    /// Radius-search read-path arm. Drops boundary hits when the user
    /// wrote `<` (strict), and applies `post_filter` against each hit's
    /// metadata when a residue predicate was attached by the planner.
    fn exec_radius_search(
        &self,
        table: &str,
        query: &crate::ast::ParamRef,
        radius: f32,
        inclusive: bool,
        post_filter: Option<&PredicateExpr>,
        params: &ParamBindings,
    ) -> Result<Vec<InternalHit>, KovaQueryError> {
        self.assert_table(table)?;
        let query_vec = expect_vector(params.resolve(query)?, "query")?;
        let hits = self
            .shard
            .search_radius(&query_vec, radius)
            .map_err(|e| KovaQueryError::Backend(Box::new(e)))?;
        let mut out = Vec::with_capacity(hits.len());
        for h in hits {
            if !inclusive && h.distance >= radius {
                continue;
            }
            if let Some(pred) = post_filter
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
                self.exec_metadata_scan(&table, &predicate, params)
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
            PhysicalPlan::FilteredKnnSearch {
                table,
                query,
                metric: _,
                k,
                filter,
            } => self.exec_filtered_knn(&table, &query, k, &filter, params),
            PhysicalPlan::RadiusSearch {
                table,
                query,
                metric: _,
                radius,
                inclusive,
                post_filter,
            } => self.exec_radius_search(
                &table,
                &query,
                radius,
                inclusive,
                post_filter.as_ref(),
                params,
            ),
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
/// resolved [`ParamValue`]. Clones the entire array , expensive
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

/// Materialise an id bitmap (from the catalog's `lookup`) into the
/// `Vec<InternalHit>` shape `exec_metadata_scan` returns. Drops ids
/// the metadata store no longer has a bag for (tombstoned between
/// the catalog state and now), so the live-id semantics match the
/// scan path.
fn bitmap_to_hits<D: Distance>(
    shard: &Shard<D, MmapVectorStore, FileMetadataStore, FileWal>,
    bitmap: &roaring::RoaringTreemap,
) -> Vec<InternalHit> {
    let mut out = Vec::with_capacity(usize::try_from(bitmap.len()).unwrap_or(0));
    for raw_id in bitmap {
        let id = VectorId::from(raw_id);
        if let Some(meta) = shard.get_metadata(id) {
            out.push(InternalHit {
                id,
                distance: None,
                metadata: meta,
            });
        }
    }
    out
}

/// Helper for `Count` : count live rows matching a predicate. Asks
/// the catalog first ; Full hits return cardinality in O(1) without
/// touching any metadata bag, Hybrid hits walk only the indexed
/// candidates and evaluate the residue per-row, Fallback runs the
/// original closure-driven scan.
fn count_matching_with_predicate<D: Distance>(
    shard: &Shard<D, MmapVectorStore, FileMetadataStore, FileWal>,
    pred: &PredicateExpr,
    params: &ParamBindings,
) -> Result<usize, KovaQueryError> {
    match crate::index_eval::try_index_eval(pred, shard.catalog(), params) {
        crate::index_eval::IndexEval::Full(bitmap) => {
            // Catalog tracks every mutation synchronously, so the
            // bitmap's cardinality IS the exact count of live
            // matching rows. No per-row work needed.
            Ok(usize::try_from(bitmap.len()).unwrap_or(usize::MAX))
        }
        crate::index_eval::IndexEval::Hybrid {
            candidates,
            residue,
        } => {
            let mut count: usize = 0;
            for raw_id in &candidates {
                let id = VectorId::from(raw_id);
                let Some(meta) = shard.get_metadata(id) else {
                    continue;
                };
                if eval_predicate(&residue, &meta, params)? {
                    count += 1;
                }
            }
            Ok(count)
        }
        crate::index_eval::IndexEval::Fallback => {
            let mut closure_err: Option<KovaQueryError> = None;
            let count = shard.count_matching(|m| {
                if closure_err.is_some() {
                    return false;
                }
                match eval_predicate(pred, m, params) {
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
            Ok(count)
        }
    }
}

/// Helper for the predicate-driven write paths
/// (DELETE-by-predicate, UPDATE-by-predicate) : produce the live id
/// set the predicate matches. Same three-way dispatch as
/// [`count_matching_with_predicate`] ; the difference is the return
/// type. Full hits materialise the bitmap into a `Vec<VectorId>`,
/// Hybrid hits iterate candidates and residue-filter, Fallback runs
/// the closure-driven `scan_metadata` path.
fn ids_matching<D: Distance>(
    shard: &Shard<D, MmapVectorStore, FileMetadataStore, FileWal>,
    pred: &PredicateExpr,
    params: &ParamBindings,
) -> Result<Vec<VectorId>, KovaQueryError> {
    match crate::index_eval::try_index_eval(pred, shard.catalog(), params) {
        crate::index_eval::IndexEval::Full(bitmap) => {
            // Bitmap holds only live ids the indexes have observed,
            // matching the predicate exactly. Materialise to the
            // shape `delete_many` / `update_metadata` expect.
            Ok(bitmap.iter().map(VectorId::from).collect())
        }
        crate::index_eval::IndexEval::Hybrid {
            candidates,
            residue,
        } => {
            let mut out = Vec::new();
            for raw_id in &candidates {
                let id = VectorId::from(raw_id);
                let Some(meta) = shard.get_metadata(id) else {
                    continue;
                };
                if eval_predicate(&residue, &meta, params)? {
                    out.push(id);
                }
            }
            Ok(out)
        }
        crate::index_eval::IndexEval::Fallback => {
            let mut closure_err: Option<KovaQueryError> = None;
            let ids = shard.scan_metadata(|m| {
                if closure_err.is_some() {
                    return false;
                }
                match eval_predicate(pred, m, params) {
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
            Ok(ids)
        }
    }
}

/// Borrowed bundle of the radius-shape fields shared between the
/// radius search / delete / update arms. Keeps the per-arm helper
/// signatures from blowing past the argument-count limit.
struct RadiusOp<'a> {
    query: &'a crate::ast::ParamRef,
    radius: f32,
    inclusive: bool,
    post_filter: Option<&'a PredicateExpr>,
}

/// Mutate `bag` in place by applying each assignment in source order.
///
/// Two assignment shapes :
///
/// - `SET field = value` : insert `value` at `field` (overwrites
///   whatever was there, regardless of type).
/// - `SET field['key'] = value` : look up `field` ; if absent, create
///   an empty [`Value::Map`] ; if present but not a `Map`, error.
///   Insert `value` at `key` inside the map.
///
/// Used by all UPDATE arms (single-id, predicate, radius).
fn apply_assignments(
    bag: &mut Metadata,
    assignments: &[LogicalAssignment],
    params: &ParamBindings,
) -> Result<(), KovaQueryError> {
    for a in assignments {
        let value = resolve_bound_value_for_assignment(&a.value, params)?;
        match &a.subscript {
            None => {
                bag.insert(a.field.clone(), value);
            }
            Some(key) => {
                // Resolve the nested map : create one if the field is
                // missing, error if it's present but not a Map. We
                // can't blindly overwrite a non-Map field because that
                // would silently drop user data.
                let slot = bag
                    .entry(a.field.clone())
                    .or_insert_with(|| Value::Map(std::collections::HashMap::new()));
                let Value::Map(inner) = slot else {
                    return Err(KovaQueryError::Execution(format!(
                        "subscripted assignment to field '{}' : expected a Map value, \
                         found a different type ; refusing to overwrite",
                        a.field
                    )));
                };
                inner.insert(key.clone(), value);
            }
        }
    }
    Ok(())
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
        PhysicalPlan::DeleteByParamId { .. } => "DeleteByParamId",
        PhysicalPlan::DeleteByPredicate { .. } => "DeleteByPredicate",
        PhysicalPlan::DeleteByRadius { .. } => "DeleteByRadius",
        PhysicalPlan::UpdateById { .. } => "UpdateById",
        PhysicalPlan::UpdateByParamId { .. } => "UpdateByParamId",
        PhysicalPlan::UpdateByPredicate { .. } => "UpdateByPredicate",
        PhysicalPlan::UpdateByRadius { .. } => "UpdateByRadius",
        PhysicalPlan::KnnSearch { .. } => "KnnSearch",
        PhysicalPlan::Limit { .. } => "Limit",
        PhysicalPlan::Projection { .. } => "Projection",
        PhysicalPlan::MetadataScan { .. } => "MetadataScan",
        PhysicalPlan::ExactDistance { .. } => "ExactDistance",
        PhysicalPlan::Count { .. } => "Count",
        PhysicalPlan::RadiusSearch { .. } => "RadiusSearch",
        PhysicalPlan::FilteredKnnSearch { .. } => "FilteredKnnSearch",
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
    BoundExpr, BoundLiteral, BoundProjection, FieldRef, LogicalAssignment, PredAtom, PredicateExpr,
    ProjectionSpec,
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

/// Resolve a [`FieldRef`] against a row's metadata bag. Bare
/// references key into the bag directly ; subscripted references
/// expect a `Value::Map` at the top-level field and key into it.
/// Returns `None` when any step misses, matching the SQL "predicate
/// is false on NULL" convention.
fn lookup_field_value<'a>(field: &FieldRef, meta: &'a Metadata) -> Option<&'a Value> {
    let top = meta.get(&field.name)?;
    match &field.subscript {
        None => Some(top),
        Some(key) => match top {
            Value::Map(inner) => inner.get(key),
            _ => None,
        },
    }
}

/// `IS NOT NULL` form of [`lookup_field_value`] : reports presence
/// without borrowing.
fn field_is_present(field: &FieldRef, meta: &Metadata) -> bool {
    lookup_field_value(field, meta).is_some()
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
            Ok(lookup_field_value(field, meta).is_some_and(|v| values_eq(v, &expected)))
        }
        PredAtom::Cmp { field, op, value } => {
            let expected = resolve_bound_value(value, params)?;
            Ok(lookup_field_value(field, meta)
                .and_then(|v| values_cmp(v, &expected, *op))
                .unwrap_or(false))
        }
        PredAtom::In { field, values } => {
            let Some(actual) = lookup_field_value(field, meta) else {
                return Ok(false);
            };
            Ok(values
                .iter()
                .map(literal_to_value)
                .any(|lit| values_eq(actual, &lit)))
        }
        PredAtom::Between { field, lo, hi } => {
            let Some(actual) = lookup_field_value(field, meta) else {
                return Ok(false);
            };
            let lo_v = literal_to_value(lo);
            let hi_v = literal_to_value(hi);
            let ge_lo = values_cmp(actual, &lo_v, CmpOp::Ge).unwrap_or(false);
            let le_hi = values_cmp(actual, &hi_v, CmpOp::Le).unwrap_or(false);
            Ok(ge_lo && le_hi)
        }
        PredAtom::IsNotNull { field } => Ok(field_is_present(field, meta)),
        PredAtom::ArrayContains { field, value } => {
            let target = literal_to_value(value);
            match lookup_field_value(field, meta) {
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

/// Resolve a [`BoundExpr`] in an assignment-RHS position. Wider than
/// the predicate version : also accepts `ParamValue::Metadata` by
/// wrapping it in [`Value::Map`] so callers can write `SET attrs = $1`
/// where `$1` is a whole metadata bag.
fn resolve_bound_value_for_assignment(
    expr: &BoundExpr,
    params: &ParamBindings,
) -> Result<Value, KovaQueryError> {
    match expr {
        BoundExpr::Literal(l) => Ok(literal_to_value(l)),
        BoundExpr::Param(p) => match params.resolve(p)? {
            ParamValue::Metadata(m) => Ok(Value::Map(m.clone())),
            other => param_value_to_value(other),
        },
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
        // Structural equality on nested bags. Cross-type comparisons
        // (Map vs anything else) fall through to false, matching the
        // policy for every other variant.
        (Value::Map(x), Value::Map(y)) => x == y,
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

    /// Plan errors propagate cleanly through `execute_str` instead of
    /// panicking deeper in the pipeline. Uses `OR` containing a
    /// distance-threshold (rejected at the planner because the union
    /// operator hasn't shipped) as the exemplar.
    #[test]
    fn execute_str_propagates_plan_error_for_unimplemented() {
        let dir = tempdir().expect("tempdir");
        let shard = Shard::open(dir.path(), 4, L2, HnswParams::default()).expect("Shard::open");
        let mut engine = Engine::new(shard, "vectors");
        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let err = engine
            .execute_str(
                "DELETE FROM vectors WHERE embedding <-> $1 < 0.5 OR tag = 'docs'",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
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

    /// Param-bound id : binder sets the `Param` hint, planner emits
    /// `DeleteByParamId`, executor resolves the param at run time and
    /// dispatches to `Shard::delete` like the literal path. Same fast
    /// path semantics : exactly one row tombstoned, no metadata scan.
    #[test]
    fn delete_by_param_bound_id_tombstones_the_resolved_id() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(&mut engine, &[meta_of(&[]), meta_of(&[])]);
        let params = ParamBindings::empty().with_positional(ParamValue::Id(VectorId::new(1)));
        let result = engine
            .execute_str("DELETE FROM vectors WHERE id = $1", params)
            .expect("execute_str");
        let ExecutionResult::Delete { deleted, .. } = result else {
            panic!("expected Delete, got {result:?}");
        };
        assert_eq!(deleted, 1);
        assert!(!engine.shard().contains(VectorId::new(1)));
        assert!(engine.shard().contains(VectorId::new(2)));
    }

    /// Named-param variant of the above. Catches a bug where positional
    /// vs named resolution diverges.
    #[test]
    fn delete_by_named_param_id() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(&mut engine, &[meta_of(&[]), meta_of(&[])]);
        let params = ParamBindings::empty().with_named("target", ParamValue::Id(VectorId::new(2)));
        let result = engine
            .execute_str("DELETE FROM vectors WHERE id = $target", params)
            .expect("execute_str");
        let ExecutionResult::Delete { deleted, .. } = result else {
            panic!("expected Delete");
        };
        assert_eq!(deleted, 1);
        assert!(engine.shard().contains(VectorId::new(1)));
        assert!(!engine.shard().contains(VectorId::new(2)));
    }

    /// Param-bound id with a wrong-typed value surfaces a clear
    /// Execution error from `expect_id`.
    #[test]
    fn delete_by_param_bound_id_wrong_type_errors() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(&mut engine, &[meta_of(&[])]);
        let params = ParamBindings::empty().with_positional(ParamValue::I64(1));
        let err = engine
            .execute_str("DELETE FROM vectors WHERE id = $1", params)
            .expect_err("expected Execution error");
        assert!(matches!(err, KovaQueryError::Execution(_)));
    }

    /// Compound predicate on a metadata field routes through
    /// `DeleteByPredicate` and tombstones every matching row.
    #[test]
    fn delete_by_compound_predicate_deletes_matching_rows() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(
            &mut engine,
            &[
                meta_of(&[("category", Value::String("old".into()))]),
                meta_of(&[("category", Value::String("new".into()))]),
                meta_of(&[("category", Value::String("old".into()))]),
            ],
        );
        let result = engine
            .execute_str(
                "DELETE FROM vectors WHERE category = 'old'",
                ParamBindings::empty(),
            )
            .expect("execute_str");
        let ExecutionResult::Delete { deleted, .. } = result else {
            panic!("expected Delete, got {result:?}");
        };
        assert_eq!(deleted, 2);
        assert!(!engine.shard().contains(VectorId::new(1)));
        assert!(engine.shard().contains(VectorId::new(2)));
        assert!(!engine.shard().contains(VectorId::new(3)));
    }

    /// DELETE matching nothing returns `deleted = 0` cleanly. No WAL
    /// activity, no errors.
    #[test]
    fn delete_by_predicate_no_matches_returns_zero() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(
            &mut engine,
            &[meta_of(&[("category", Value::String("a".into()))])],
        );
        let result = engine
            .execute_str(
                "DELETE FROM vectors WHERE category = 'nonexistent'",
                ParamBindings::empty(),
            )
            .expect("execute_str");
        let ExecutionResult::Delete { deleted, .. } = result else {
            panic!("expected Delete");
        };
        assert_eq!(deleted, 0);
        assert!(engine.shard().contains(VectorId::new(1)));
    }

    /// `DELETE WHERE embedding <-> $q < r` routes through the radius
    /// operator : every id within the ball is tombstoned, ids outside
    /// it survive. Same semantics as `SELECT ... WHERE dist < r`
    /// applied as a write.
    #[test]
    fn delete_by_radius_tombstones_in_ball() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        // 8 axis-aligned vectors : ids 1,5 -> e_0 ; 2,6 -> e_1 ; etc.
        let metas: Vec<Metadata> = (0..8).map(|_| Metadata::new()).collect();
        seed_engine(&mut engine, &metas);

        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "DELETE FROM vectors WHERE embedding <-> $1 < 0.5",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Delete { deleted, .. } = result else {
            panic!("expected Delete");
        };
        // Only ids 1 and 5 sit on e_0 (distance 0 to the query).
        assert_eq!(deleted, 2);
        assert!(!engine.shard().contains(VectorId::new(1)));
        assert!(!engine.shard().contains(VectorId::new(5)));
        assert!(engine.shard().contains(VectorId::new(2)));
        assert!(engine.shard().contains(VectorId::new(6)));
    }

    /// AND-residue on a radius DELETE peels off as a post-filter.
    /// Only ids that are both in-ball AND match the residue get
    /// tombstoned.
    #[test]
    fn delete_by_radius_with_and_residue_applies_post_filter() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        // ids 1, 5 -> e_0. Tag id 1 as 'docs', id 5 as 'other'.
        let metas = vec![
            meta_of(&[("tag", Value::String("docs".into()))]), // id 1
            meta_of(&[("tag", Value::String("other".into()))]), // id 2
            meta_of(&[("tag", Value::String("other".into()))]), // id 3
            meta_of(&[("tag", Value::String("other".into()))]), // id 4
            meta_of(&[("tag", Value::String("other".into()))]), // id 5
        ];
        seed_engine(&mut engine, &metas);

        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "DELETE FROM vectors \
                 WHERE embedding <-> $1 < 0.5 AND tag = 'docs'",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Delete { deleted, .. } = result else {
            panic!("expected Delete");
        };
        // id 1 is in-ball AND tag='docs'. id 5 is in-ball but tag='other'.
        assert_eq!(deleted, 1);
        assert!(!engine.shard().contains(VectorId::new(1)));
        assert!(engine.shard().contains(VectorId::new(5)));
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

    // ----- UPDATE : single-id fast paths -----

    /// `UPDATE ... SET field = 'literal' WHERE id = N` mutates exactly
    /// the targeted row's metadata bag. Other rows untouched.
    #[test]
    fn update_by_literal_id_replaces_field() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(
            &mut engine,
            &[
                meta_of(&[("category", Value::String("old".into()))]),
                meta_of(&[("category", Value::String("untouched".into()))]),
            ],
        );
        let result = engine
            .execute_str(
                "UPDATE vectors SET category = 'new' WHERE id = 1",
                ParamBindings::empty(),
            )
            .expect("execute_str");
        let ExecutionResult::Update { updated, .. } = result else {
            panic!("expected Update, got {result:?}");
        };
        assert_eq!(updated, 1);
        assert_eq!(
            engine
                .shard()
                .get_metadata(VectorId::new(1))
                .unwrap()
                .get("category"),
            Some(&Value::String("new".into()))
        );
        assert_eq!(
            engine
                .shard()
                .get_metadata(VectorId::new(2))
                .unwrap()
                .get("category"),
            Some(&Value::String("untouched".into()))
        );
    }

    /// `WHERE id = $1` resolves the param at execute time and dispatches
    /// to the same fast path.
    #[test]
    fn update_by_param_bound_id_sets_field() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(
            &mut engine,
            &[meta_of(&[("category", Value::String("old".into()))])],
        );
        let params = ParamBindings::empty().with_positional(ParamValue::Id(VectorId::new(1)));
        let result = engine
            .execute_str("UPDATE vectors SET category = 'new' WHERE id = $1", params)
            .expect("execute_str");
        let ExecutionResult::Update { updated, .. } = result else {
            panic!("expected Update");
        };
        assert_eq!(updated, 1);
        assert_eq!(
            engine
                .shard()
                .get_metadata(VectorId::new(1))
                .unwrap()
                .get("category"),
            Some(&Value::String("new".into()))
        );
    }

    /// UPDATE on a non-existent id surfaces as an Execution error.
    #[test]
    fn update_unknown_id_errors() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(&mut engine, &[meta_of(&[])]);
        let err = engine
            .execute_str(
                "UPDATE vectors SET tag = 'x' WHERE id = 999",
                ParamBindings::empty(),
            )
            .expect_err("expected error");
        // No metadata bag for id 999 ; surfaces as Execution.
        assert!(matches!(err, KovaQueryError::Execution(_)));
    }

    /// Param-bound assignment value : `SET tag = $1`. Should resolve
    /// the param to a `Value` and write it.
    #[test]
    fn update_with_param_bound_assignment_value() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(&mut engine, &[meta_of(&[])]);
        let params =
            ParamBindings::empty().with_positional(ParamValue::String("from-param".into()));
        let result = engine
            .execute_str("UPDATE vectors SET tag = $1 WHERE id = 1", params)
            .expect("execute_str");
        let ExecutionResult::Update { updated, .. } = result else {
            panic!("expected Update");
        };
        assert_eq!(updated, 1);
        assert_eq!(
            engine
                .shard()
                .get_metadata(VectorId::new(1))
                .unwrap()
                .get("tag"),
            Some(&Value::String("from-param".into()))
        );
    }

    /// Multiple SET clauses : each assignment lands. Sequential
    /// application order is the source order.
    #[test]
    fn update_with_multiple_assignments() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(
            &mut engine,
            &[meta_of(&[("a", Value::I64(1)), ("b", Value::I64(2))])],
        );
        let result = engine
            .execute_str(
                "UPDATE vectors SET a = 10, b = 20, c = 30 WHERE id = 1",
                ParamBindings::empty(),
            )
            .expect("execute_str");
        let ExecutionResult::Update { updated, .. } = result else {
            panic!("expected Update");
        };
        assert_eq!(updated, 1);
        let bag = engine.shard().get_metadata(VectorId::new(1)).unwrap();
        assert_eq!(bag.get("a"), Some(&Value::I64(10)));
        assert_eq!(bag.get("b"), Some(&Value::I64(20)));
        assert_eq!(bag.get("c"), Some(&Value::I64(30)));
    }

    /// Predicate UPDATE : scan metadata for matching ids, apply the
    /// assignments to each bag, write the batch via one
    /// `Shard::update_metadata` call.
    #[test]
    fn update_by_predicate_rewrites_matching_rows() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(
            &mut engine,
            &[
                meta_of(&[("category", Value::String("old".into()))]),
                meta_of(&[("category", Value::String("new".into()))]),
                meta_of(&[("category", Value::String("old".into()))]),
            ],
        );
        let result = engine
            .execute_str(
                "UPDATE vectors SET category = 'archived' WHERE category = 'old'",
                ParamBindings::empty(),
            )
            .expect("execute_str");
        let ExecutionResult::Update { updated, .. } = result else {
            panic!("expected Update, got {result:?}");
        };
        assert_eq!(updated, 2);
        // ids 1 and 3 had category='old' -> now 'archived' ;
        // id 2 had 'new' and stays unchanged.
        assert_eq!(
            engine
                .shard()
                .get_metadata(VectorId::new(1))
                .unwrap()
                .get("category"),
            Some(&Value::String("archived".into()))
        );
        assert_eq!(
            engine
                .shard()
                .get_metadata(VectorId::new(2))
                .unwrap()
                .get("category"),
            Some(&Value::String("new".into()))
        );
        assert_eq!(
            engine
                .shard()
                .get_metadata(VectorId::new(3))
                .unwrap()
                .get("category"),
            Some(&Value::String("archived".into()))
        );
    }

    /// Predicate UPDATE that matches no rows returns `updated = 0`
    /// cleanly. No WAL activity (`Shard::update_metadata` early-returns
    /// on empty input).
    #[test]
    fn update_by_predicate_no_matches_returns_zero() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(
            &mut engine,
            &[meta_of(&[("category", Value::String("docs".into()))])],
        );
        let result = engine
            .execute_str(
                "UPDATE vectors SET category = 'archived' WHERE category = 'nonexistent'",
                ParamBindings::empty(),
            )
            .expect("execute_str");
        let ExecutionResult::Update { updated, .. } = result else {
            panic!("expected Update");
        };
        assert_eq!(updated, 0);
        // Original row untouched.
        assert_eq!(
            engine
                .shard()
                .get_metadata(VectorId::new(1))
                .unwrap()
                .get("category"),
            Some(&Value::String("docs".into()))
        );
    }

    /// Radius UPDATE : every id in the ball gets its assignments
    /// applied. Out-of-ball rows untouched.
    #[test]
    fn update_by_radius_rewrites_in_ball() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        // 8 axis-aligned ; ids 1, 5 sit on e_0.
        let metas: Vec<Metadata> = (0..8)
            .map(|_| meta_of(&[("tag", Value::String("old".into()))]))
            .collect();
        seed_engine(&mut engine, &metas);

        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "UPDATE vectors SET tag = 'new' WHERE embedding <-> $1 < 0.5",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Update { updated, .. } = result else {
            panic!("expected Update");
        };
        assert_eq!(updated, 2);
        assert_eq!(
            engine
                .shard()
                .get_metadata(VectorId::new(1))
                .unwrap()
                .get("tag"),
            Some(&Value::String("new".into()))
        );
        assert_eq!(
            engine
                .shard()
                .get_metadata(VectorId::new(5))
                .unwrap()
                .get("tag"),
            Some(&Value::String("new".into()))
        );
        // id 2 is at e_1 (distance sqrt(2) > 0.5), should stay 'old'.
        assert_eq!(
            engine
                .shard()
                .get_metadata(VectorId::new(2))
                .unwrap()
                .get("tag"),
            Some(&Value::String("old".into()))
        );
    }

    /// Radius UPDATE with AND-residue : only ids that are in-ball AND
    /// pass the residue get updated.
    #[test]
    fn update_by_radius_with_and_residue_applies_post_filter() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        // ids 1, 5 -> e_0. id 1 tagged 'docs', id 5 tagged 'other'.
        let metas = vec![
            meta_of(&[("tag", Value::String("docs".into()))]),
            meta_of(&[("tag", Value::String("other".into()))]),
            meta_of(&[("tag", Value::String("other".into()))]),
            meta_of(&[("tag", Value::String("other".into()))]),
            meta_of(&[("tag", Value::String("other".into()))]),
        ];
        seed_engine(&mut engine, &metas);

        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "UPDATE vectors SET status = 'archived' \
                 WHERE embedding <-> $1 < 0.5 AND tag = 'docs'",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Update { updated, .. } = result else {
            panic!("expected Update");
        };
        assert_eq!(updated, 1);
        assert_eq!(
            engine
                .shard()
                .get_metadata(VectorId::new(1))
                .unwrap()
                .get("status"),
            Some(&Value::String("archived".into()))
        );
        // id 5 was in-ball but failed the residue, no 'status' added.
        assert_eq!(
            engine
                .shard()
                .get_metadata(VectorId::new(5))
                .unwrap()
                .get("status"),
            None
        );
    }

    /// OR containing a distance-threshold atom is rejected at the
    /// planner exactly like DELETE.
    #[test]
    fn update_by_or_distance_is_rejected() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(&mut engine, &[meta_of(&[])]);
        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let err = engine
            .execute_str(
                "UPDATE vectors SET tag = 'x' \
                 WHERE embedding <-> $1 < 0.5 OR tag = 'docs'",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect_err("expected Plan error");
        assert!(matches!(err, KovaQueryError::Plan(_)));
    }

    /// Subscripted assignment on a field that doesn't exist yet :
    /// `apply_assignments` creates an empty Map at that field and
    /// then writes the keyed value into it.
    #[test]
    fn update_subscript_creates_map_when_field_missing() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(&mut engine, &[meta_of(&[])]);
        engine
            .execute_str(
                "UPDATE vectors SET attrs['author'] = 'alice' WHERE id = 1",
                ParamBindings::empty(),
            )
            .expect("execute_str");
        let bag = engine.shard().get_metadata(VectorId::new(1)).unwrap();
        let Some(Value::Map(inner)) = bag.get("attrs") else {
            panic!("expected attrs to be a Map, got {:?}", bag.get("attrs"));
        };
        assert_eq!(inner.get("author"), Some(&Value::String("alice".into())));
    }

    /// Subscripted assignment on a field that's already a Map merges
    /// the new key in without dropping existing keys.
    #[test]
    fn update_subscript_merges_into_existing_map() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        let mut inner = Metadata::new();
        inner.insert("author".into(), Value::String("alice".into()));
        let mut bag = Metadata::new();
        bag.insert("attrs".into(), Value::Map(inner));
        seed_engine(&mut engine, &[bag]);

        engine
            .execute_str(
                "UPDATE vectors SET attrs['priority'] = 1 WHERE id = 1",
                ParamBindings::empty(),
            )
            .expect("execute_str");
        let bag = engine.shard().get_metadata(VectorId::new(1)).unwrap();
        let Some(Value::Map(map)) = bag.get("attrs") else {
            panic!("expected attrs to be a Map");
        };
        assert_eq!(map.get("author"), Some(&Value::String("alice".into())));
        assert_eq!(map.get("priority"), Some(&Value::I64(1)));
    }

    /// Subscripted assignment against a non-Map field errors loudly
    /// rather than silently overwriting the existing value.
    #[test]
    fn update_subscript_on_non_map_field_errors() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(
            &mut engine,
            &[meta_of(&[("attrs", Value::String("flat".into()))])],
        );
        let err = engine
            .execute_str(
                "UPDATE vectors SET attrs['key'] = 'x' WHERE id = 1",
                ParamBindings::empty(),
            )
            .expect_err("expected Execution error");
        assert!(matches!(err, KovaQueryError::Execution(_)));
        // Original value untouched.
        let bag = engine.shard().get_metadata(VectorId::new(1)).unwrap();
        assert_eq!(bag.get("attrs"), Some(&Value::String("flat".into())));
    }

    /// `SET attrs = $1` where `$1` is a `ParamValue::Metadata`
    /// resolves to a `Value::Map` and replaces the field. The
    /// roadmap exemplar that earlier required a deferred decision
    /// now works straight through.
    #[test]
    fn update_with_metadata_param_assigns_map() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(&mut engine, &[meta_of(&[])]);
        let mut new_attrs = Metadata::new();
        new_attrs.insert("country".into(), Value::String("IN".into()));
        new_attrs.insert("verified".into(), Value::Bool(true));
        let params =
            ParamBindings::empty().with_positional(ParamValue::Metadata(new_attrs.clone()));
        engine
            .execute_str("UPDATE vectors SET attrs = $1 WHERE id = 1", params)
            .expect("execute_str");
        let bag = engine.shard().get_metadata(VectorId::new(1)).unwrap();
        let Some(Value::Map(map)) = bag.get("attrs") else {
            panic!("expected attrs to be a Map");
        };
        assert_eq!(map, &new_attrs);
    }

    /// Subscripted update survives WAL replay : the
    /// `Record::UpdateMetadata` round-trip preserves nested Map
    /// values.
    #[test]
    fn update_subscript_persists_across_reopen() {
        let dir = tempdir().expect("tempdir");
        {
            let mut engine = make_engine(&dir);
            seed_engine(&mut engine, &[meta_of(&[])]);
            engine
                .execute_str(
                    "UPDATE vectors SET attrs['author'] = 'alice' WHERE id = 1",
                    ParamBindings::empty(),
                )
                .expect("execute_str");
        }
        let shard = Shard::open(dir.path(), 4, L2, HnswParams::default()).expect("Shard::open");
        let bag = shard.get_metadata(VectorId::new(1)).unwrap();
        let Some(Value::Map(map)) = bag.get("attrs") else {
            panic!("expected attrs to be a Map after replay");
        };
        assert_eq!(map.get("author"), Some(&Value::String("alice".into())));
    }

    // ----- subscripted-predicate end-to-end -----

    /// SELECT with a subscripted predicate finds rows whose nested
    /// Map value matches. Plan A's metadata post-filter runs the
    /// evaluator's subscript navigation.
    #[test]
    fn select_with_subscripted_predicate_finds_matching_rows() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        let mut attrs_in = Metadata::new();
        attrs_in.insert("country".into(), Value::String("IN".into()));
        let mut attrs_us = Metadata::new();
        attrs_us.insert("country".into(), Value::String("US".into()));
        seed_engine(
            &mut engine,
            &[
                meta_of(&[("attrs", Value::Map(attrs_in))]),
                meta_of(&[("attrs", Value::Map(attrs_us))]),
            ],
        );

        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "SELECT id FROM vectors WHERE attrs['country'] = 'IN' \
                 ORDER BY embedding <-> $1 LIMIT 5",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        let ids: Vec<u64> = rows
            .iter()
            .map(|r| match r.values[0] {
                RowValue::Id(id) => id.get(),
                _ => panic!("expected Id"),
            })
            .collect();
        assert!(ids.contains(&1), "id 1 (attrs.country=IN) should match");
        assert!(
            !ids.contains(&2),
            "id 2 (attrs.country=US) should not match"
        );
    }

    /// Subscripted predicate on a missing nested key returns no
    /// matches, no error. Same semantics as a top-level missing field.
    #[test]
    fn select_with_subscripted_predicate_no_match_returns_empty() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        let mut attrs = Metadata::new();
        attrs.insert("country".into(), Value::String("IN".into()));
        seed_engine(&mut engine, &[meta_of(&[("attrs", Value::Map(attrs))])]);

        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "SELECT id FROM vectors WHERE attrs['missing'] = 'x' \
                 ORDER BY embedding <-> $1 LIMIT 5",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        assert!(rows.is_empty());
    }

    /// Subscripted predicate against a non-Map field is treated as
    /// "no match" (same as bare predicates against a missing field).
    /// We don't want a runtime error here ; the row just doesn't
    /// satisfy.
    #[test]
    fn select_with_subscripted_predicate_on_non_map_is_no_match() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(
            &mut engine,
            &[meta_of(&[("attrs", Value::String("flat".into()))])],
        );
        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "SELECT id FROM vectors WHERE attrs['country'] = 'IN' \
                 ORDER BY embedding <-> $1 LIMIT 5",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        assert!(rows.is_empty());
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

    /// `SELECT WHERE pred LIMIT k` (no ORDER BY) goes through the
    /// scan-and-limit bypass : MetadataScan(pred) wrapped in a Limit.
    /// Order isn't promised but the predicate must be honoured.
    #[test]
    fn scan_and_limit_returns_predicate_matches() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        // 3 'docs' + 7 'other'.
        let mut metas: Vec<Metadata> = (0..3)
            .map(|_| meta_of(&[("category", Value::String("docs".into()))]))
            .collect();
        for _ in 0..7 {
            metas.push(meta_of(&[("category", Value::String("other".into()))]));
        }
        seed_engine(&mut engine, &metas);

        let result = engine
            .execute_str(
                "SELECT id FROM vectors WHERE category = 'docs' LIMIT 10",
                ParamBindings::empty(),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        // First 3 seeded rows are the 'docs' ones (ids 1, 2, 3).
        assert!(!rows.is_empty(), "should return at least one match");
        for r in &rows {
            let RowValue::Id(id) = r.values[0] else {
                panic!("expected Id");
            };
            assert!(
                id.get() <= 3,
                "row {} doesn't carry the 'docs' tag",
                id.get()
            );
        }
    }

    /// `LIMIT` truncates the scan output to at most the requested
    /// count, regardless of how many rows match the predicate.
    #[test]
    fn scan_and_limit_caps_result_size() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        // 8 'docs' ; LIMIT 3 should cap.
        let metas: Vec<Metadata> = (0..8)
            .map(|_| meta_of(&[("category", Value::String("docs".into()))]))
            .collect();
        seed_engine(&mut engine, &metas);

        let result = engine
            .execute_str(
                "SELECT id FROM vectors WHERE category = 'docs' LIMIT 3",
                ParamBindings::empty(),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        assert_eq!(rows.len(), 3, "LIMIT 3 should cap result size");
    }

    /// `LIMIT` without WHERE is rejected : an unbounded slice-scan of
    /// arbitrary rows is a foot-gun the planner refuses to serve.
    #[test]
    fn rejects_limit_without_where() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(&mut engine, &[Metadata::new(), Metadata::new()]);
        let err = engine
            .execute_str("SELECT id FROM vectors LIMIT 5", ParamBindings::empty())
            .expect_err("expected Plan error");
        let KovaQueryError::Plan(msg) = err else {
            panic!("expected Plan, got {err:?}");
        };
        assert!(
            msg.contains("WHERE"),
            "message should mention WHERE requirement : {msg}"
        );
    }

    /// `SELECT` with no ORDER BY and no LIMIT and no WHERE still
    /// errors : the kNN-shape path catches this case, since there's
    /// nothing for the scan-and-limit bypass to grab.
    #[test]
    fn rejects_select_without_ordering_or_limit() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        let err = engine
            .execute_str(
                "SELECT id FROM vectors WHERE category = 'docs'",
                ParamBindings::empty(),
            )
            .expect_err("expected Plan error");
        let KovaQueryError::Plan(msg) = err else {
            panic!("expected Plan, got {err:?}");
        };
        assert!(
            msg.contains("kNN"),
            "message should call out the kNN-only constraint : {msg}"
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

    /// Planner-shape check : a SELECT without a predicate stays on
    /// plan A (`KnnSearch` with overfetch). The estimator is never
    /// consulted in the no-predicate branch, so this test only needs
    /// any estimator that compiles.
    #[test]
    fn planner_picks_plan_a_when_no_predicate() {
        use crate::physical::PhysicalPlan;
        use crate::planner::plan_with_estimator;
        let dir = tempdir().expect("tempdir");
        let engine = make_engine(&dir);
        let est = ShardEstimator {
            shard: engine.shard(),
        };
        let ast =
            parse_str("SELECT id FROM vectors ORDER BY embedding <-> $1 LIMIT 10").expect("parse");
        let logical = crate::binder::bind(ast).expect("bind");
        let physical = plan_with_estimator(logical, &est, &ParamBindings::empty()).expect("plan");
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

        // Projection -> Limit -> KnnSearch (with post_filter)
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
        // 1 'docs' + 29 'other' : selectivity ≈ 0.033 < 0.05 (plan B band).
        let mut metas: Vec<Metadata> = vec![meta_of(&[("category", Value::String("docs".into()))])];
        for _ in 0..29 {
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

    /// Mid-range selectivity (in `[PLAN_B_UPPER, PLAN_A_LOWER)`) routes
    /// to plan C : the filter threads into the kNN walk, no overfetch,
    /// no metadata scan. Verifies the planner emits
    /// `FilteredKnnSearch` under the `Limit`.
    #[test]
    fn mid_selectivity_picks_plan_c() {
        use crate::physical::PhysicalPlan;
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        // 2 'docs' + 8 'other' : selectivity = 0.2, in the plan C band.
        let mut metas: Vec<Metadata> = (0..2)
            .map(|_| meta_of(&[("category", Value::String("docs".into()))]))
            .collect();
        for _ in 0..8 {
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
            matches!(*input, PhysicalPlan::FilteredKnnSearch { .. }),
            "mid selectivity should pick plan C (FilteredKnnSearch), got {input:?}"
        );
    }

    /// Plan C end-to-end : run a SELECT in the mid-selectivity band
    /// Decision-grid sweep : drive the planner with a deterministic
    /// estimator across the full selectivity range and assert each
    /// fraction maps to the expected inner operator. Uses a fake
    /// estimator so we don't pay seeding cost per cell.
    #[test]
    fn planner_decision_grid_across_selectivity() {
        use crate::physical::PhysicalPlan;
        use crate::planner::{SelectivityEstimate, SelectivityEstimator};

        struct Const(f64);
        impl SelectivityEstimator for Const {
            fn estimate(
                &self,
                _pred: &crate::logical::PredicateExpr,
                _params: &ParamBindings,
            ) -> SelectivityEstimate {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let matches = (self.0 * 1000.0) as usize;
                SelectivityEstimate {
                    matches,
                    total: 1000,
                }
            }
        }

        #[derive(Debug)]
        enum Expect {
            PlanA,
            PlanB,
            PlanC,
        }

        let cases: &[(f64, Expect)] = &[
            (0.001, Expect::PlanB),
            (0.04, Expect::PlanB),
            (0.05, Expect::PlanC), // PLAN_B_UPPER boundary : at threshold = plan C
            (0.10, Expect::PlanC),
            (0.30, Expect::PlanC),
            (0.49, Expect::PlanC),
            (0.50, Expect::PlanA), // PLAN_A_LOWER boundary : at threshold = plan A
            (0.80, Expect::PlanA),
            (1.00, Expect::PlanA),
        ];

        for (sel, expected) in cases {
            let ast = parse_str(
                "SELECT id FROM vectors WHERE category = 'docs' \
                 ORDER BY embedding <-> $1 LIMIT 5",
            )
            .expect("parse");
            let logical = crate::binder::bind(ast).expect("bind");
            let plan =
                crate::planner::plan_with_estimator(logical, &Const(*sel), &ParamBindings::empty())
                    .expect("plan");
            let PhysicalPlan::Projection { input, .. } = plan else {
                panic!("expected Projection at sel={sel}");
            };
            let PhysicalPlan::Limit { input, .. } = *input else {
                panic!("expected Limit at sel={sel}");
            };
            let matches = matches!(
                (expected, input.as_ref()),
                (Expect::PlanA, PhysicalPlan::KnnSearch { .. })
                    | (Expect::PlanB, PhysicalPlan::ExactDistance { .. })
                    | (Expect::PlanC, PhysicalPlan::FilteredKnnSearch { .. })
            );
            assert!(
                matches,
                "selectivity {sel} expected {expected:?}, got operator {input:?}"
            );
        }
    }

    /// and check the returned ids actually pass the predicate. This
    /// hits the filter-threaded HNSW walk via `Shard::search_filtered`.
    #[test]
    fn plan_c_end_to_end_returns_only_passing_ids() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        // 3 'docs' + 7 'other' : selectivity = 0.3, plan C band.
        let mut metas: Vec<Metadata> = (0..3)
            .map(|_| meta_of(&[("category", Value::String("docs".into()))]))
            .collect();
        for _ in 0..7 {
            metas.push(meta_of(&[("category", Value::String("other".into()))]));
        }
        seed_engine(&mut engine, &metas);

        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "SELECT id FROM vectors WHERE category = 'docs' \
                 ORDER BY embedding <-> $1 LIMIT 3",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        let ids: Vec<u64> = rows
            .iter()
            .map(|r| match r.values[0] {
                RowValue::Id(id) => id.get(),
                _ => panic!("expected Id"),
            })
            .collect();
        // First three seeded rows are the 'docs' ones (ids 1, 2, 3).
        for id in &ids {
            assert!(
                *id <= 3,
                "plan C returned id {id} which doesn't carry 'docs' tag"
            );
        }
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

        // Engine A uses ShardEstimator (selectivity ~50% -> plan A
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

    // ----- COUNT(*) -----

    #[test]
    fn count_star_on_empty_shard_returns_zero() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        let result = engine
            .execute_str("SELECT COUNT(*) FROM vectors", ParamBindings::empty())
            .expect("execute_str");
        let ExecutionResult::Rows { columns, rows } = result else {
            panic!("expected Rows");
        };
        assert_eq!(columns, vec!["count".to_string()]);
        assert_eq!(rows.len(), 1);
        let RowValue::Field(Value::I64(n)) = &rows[0].values[0] else {
            panic!("expected I64 Field, got {:?}", rows[0].values[0]);
        };
        assert_eq!(*n, 0);
    }

    #[test]
    fn count_star_returns_total_live_rows() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(
            &mut engine,
            &[Metadata::new(), Metadata::new(), Metadata::new()],
        );
        let result = engine
            .execute_str("SELECT COUNT(*) FROM vectors", ParamBindings::empty())
            .expect("execute_str");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        let RowValue::Field(Value::I64(n)) = rows[0].values[0] else {
            panic!("expected I64");
        };
        assert_eq!(n, 3);
    }

    #[test]
    fn count_star_with_predicate_counts_only_matching() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(
            &mut engine,
            &[
                meta_of(&[("category", Value::String("docs".into()))]),
                meta_of(&[("category", Value::String("specs".into()))]),
                meta_of(&[("category", Value::String("docs".into()))]),
                meta_of(&[("category", Value::String("docs".into()))]),
            ],
        );
        let result = engine
            .execute_str(
                "SELECT COUNT(*) FROM vectors WHERE category = 'docs'",
                ParamBindings::empty(),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        let RowValue::Field(Value::I64(n)) = rows[0].values[0] else {
            panic!("expected I64");
        };
        assert_eq!(n, 3, "three 'docs' rows match");
    }

    #[test]
    fn count_star_with_alias_uses_alias_as_column_name() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(&mut engine, &[Metadata::new(), Metadata::new()]);
        let result = engine
            .execute_str("SELECT COUNT(*) AS n FROM vectors", ParamBindings::empty())
            .expect("execute_str");
        let ExecutionResult::Rows { columns, .. } = result else {
            panic!("expected Rows");
        };
        assert_eq!(columns, vec!["n".to_string()]);
    }

    #[test]
    fn count_star_after_delete_excludes_tombstones() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(
            &mut engine,
            &[Metadata::new(), Metadata::new(), Metadata::new()],
        );
        engine.shard_mut().delete(VectorId::new(2)).expect("delete");
        let result = engine
            .execute_str("SELECT COUNT(*) FROM vectors", ParamBindings::empty())
            .expect("execute_str");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        let RowValue::Field(Value::I64(n)) = rows[0].values[0] else {
            panic!("expected I64");
        };
        assert_eq!(n, 2, "tombstoned id 2 should not count");
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

    // ----- RadiusSearch -----

    /// `WHERE embedding <-> $q < r` with no ORDER BY and no LIMIT
    /// becomes a [`PhysicalPlan::RadiusSearch`]. Verifies the planner
    /// takes the bypass and the executor runs the operator end-to-end.
    #[test]
    fn radius_search_returns_ids_within_radius() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        // 8 axis-aligned vectors : ids 1,5 -> e_0 ; 2,6 -> e_1 ; etc.
        let metas: Vec<Metadata> = (0..8).map(|_| Metadata::new()).collect();
        seed_engine(&mut engine, &metas);

        // Query = e_0. Distance to e_0 ids is 0, to other-axis ids is sqrt(2).
        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "SELECT id FROM vectors WHERE embedding <-> $1 < 0.5",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        let mut ids: Vec<u64> = rows
            .iter()
            .map(|r| match r.values[0] {
                RowValue::Id(id) => id.get(),
                _ => panic!("expected Id"),
            })
            .collect();
        ids.sort_unstable();
        // Only ids 1 and 5 sit on e_0.
        assert_eq!(ids, vec![1, 5]);
    }

    /// Strict (`<`) versus inclusive (`<=`) : the executor drops
    /// boundary hits for `<` so users get the semantic they wrote.
    #[test]
    fn radius_search_strict_drops_boundary_hits() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(&mut engine, &[Metadata::new(), Metadata::new()]);
        // id 1 at e_0, id 2 at e_1. Query at e_0 : distances 0 and sqrt(2).
        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        // Inclusive : exact boundary at sqrt(2) keeps id 2.
        let inclusive = engine
            .execute_str(
                "SELECT id FROM vectors WHERE embedding <-> $1 <= 1.4142135",
                ParamBindings::empty().with_positional(ParamValue::Vector(q.clone())),
            )
            .expect("inclusive");
        let ExecutionResult::Rows { rows, .. } = inclusive else {
            panic!("expected Rows");
        };
        assert_eq!(rows.len(), 2);

        // Strict against the same boundary should drop id 2.
        let strict = engine
            .execute_str(
                "SELECT id FROM vectors WHERE embedding <-> $1 < 1.4142135",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("strict");
        let ExecutionResult::Rows { rows, .. } = strict else {
            panic!("expected Rows");
        };
        let ids: Vec<u64> = rows
            .iter()
            .map(|r| match r.values[0] {
                RowValue::Id(id) => id.get(),
                _ => panic!("expected Id"),
            })
            .collect();
        assert_eq!(ids, vec![1]);
    }

    /// A non-distance atom alongside the radius gets peeled off and
    /// applied as a `post_filter` after the radius walk.
    #[test]
    fn radius_search_with_and_residue_applies_post_filter() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        // ids 1,5 -> e_0. Tag one 'docs', the other 'other'.
        let metas = vec![
            meta_of(&[("tag", Value::String("docs".into()))]), // id 1
            meta_of(&[("tag", Value::String("other".into()))]), // id 2
            meta_of(&[("tag", Value::String("other".into()))]), // id 3
            meta_of(&[("tag", Value::String("other".into()))]), // id 4
            meta_of(&[("tag", Value::String("other".into()))]), // id 5
        ];
        seed_engine(&mut engine, &metas);

        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = engine
            .execute_str(
                "SELECT id FROM vectors \
                 WHERE embedding <-> $1 < 0.5 AND tag = 'docs'",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect("execute_str");
        let ExecutionResult::Rows { rows, .. } = result else {
            panic!("expected Rows");
        };
        // Without the tag filter we'd get ids 1 and 5. The post_filter
        // narrows it to id 1.
        assert_eq!(rows.len(), 1);
        let RowValue::Id(got) = rows[0].values[0] else {
            panic!("expected Id");
        };
        assert_eq!(got.get(), 1);
    }

    /// `embedding <-> $1 < r OR tag = 'a'` is rejected by the planner :
    /// the Union operator that would implement this lands in a later
    /// milestone, so v1 fails loud rather than silently misinterpreting.
    #[test]
    fn radius_search_with_or_distance_is_rejected() {
        let dir = tempdir().expect("tempdir");
        let mut engine = make_engine(&dir);
        seed_engine(&mut engine, &[Metadata::new(), Metadata::new()]);

        let q = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let err = engine
            .execute_str(
                "SELECT id FROM vectors \
                 WHERE embedding <-> $1 < 0.5 OR tag = 'docs'",
                ParamBindings::empty().with_positional(ParamValue::Vector(q)),
            )
            .expect_err("OR-with-distance should not plan");
        assert!(
            matches!(err, KovaQueryError::Plan(ref m) if m.contains("Union") || m.contains("distance-threshold")),
            "expected Plan error mentioning the rejected radius-OR shape, got {err:?}"
        );
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

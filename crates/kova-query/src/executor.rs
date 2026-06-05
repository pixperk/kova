//! KQL executor : runs a [`PhysicalPlan`] against a `Shard`.
//!
//! The public surface is [`Engine`], which owns a `Shard` and exposes
//! [`Engine::execute_str`] : the full `parse -> bind -> plan -> execute`
//! pipeline behind one call.

use std::collections::HashMap;

use kova_core::{Distance, Metadata, Vector, VectorId};
use kova_storage::{FileMetadataStore, FileWal, Lsn, MmapVectorStore, Shard};

use crate::ast::ParamRef;
use crate::binder::bind;
use crate::error::KovaQueryError;
use crate::parser::parse_str;
use crate::physical::PhysicalPlan;
use crate::planner::plan;

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
    /// Vector primary key.
    Id(VectorId),
    /// Embedding vector. For single-row INSERT.
    Vector(Vector),
    /// Metadata bag. For single-row INSERT / UPDATE.
    Metadata(Metadata),
    /// Batch of `(id, embedding, metadata)` tuples. For `VALUES $1`
    /// batch INSERT.
    Batch(Vec<(VectorId, Vector, Metadata)>),
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
        let physical = plan(logical)?;
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

/// Static label for each [`ParamValue`] variant ; used to build
/// helpful "got X, expected Y" error messages.
fn param_value_kind(value: &ParamValue) -> &'static str {
    match value {
        ParamValue::Id(_) => "Id",
        ParamValue::Vector(_) => "Vector",
        ParamValue::Metadata(_) => "Metadata",
        ParamValue::Batch(_) => "Batch",
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

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
    // taken by value : today only CHECKPOINT (carries no payload) is
    // implemented ; future arms move fields out of the operator
    // payload, which is why the by-value shape is right.
    #[allow(clippy::needless_pass_by_value)]
    fn execute(
        &mut self,
        plan: PhysicalPlan,
        _params: &ParamBindings,
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

    /// Statements without an executor arm yet (INSERT, UPDATE, etc.)
    /// report a clean Plan error.
    #[test]
    fn execute_str_propagates_plan_error_for_unimplemented() {
        let dir = tempdir().expect("tempdir");
        let shard = Shard::open(dir.path(), 4, L2, HnswParams::default()).expect("Shard::open");
        let mut engine = Engine::new(shard, "vectors");
        let err = engine
            .execute_str(
                "INSERT INTO vectors (id, embedding, metadata) VALUES ($1, $2, $3)",
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
}

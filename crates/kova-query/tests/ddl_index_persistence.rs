//! Persistence tests for the DDL surface : CREATE INDEX / DROP INDEX
//! survive a close-reopen cycle **without** requiring a checkpoint
//! in between.
//!
//! The previous rule was "DDL is transient until the next
//! checkpoint" (same contract as `add_*_index`). Once DDL goes
//! through the WAL via `Record::CreateIndex` / `Record::DropIndex`,
//! reopen replays the records and reapplies them to the catalog ;
//! the index lives.
//!
//! These tests pin the new contract.

use kova_core::{L2, Metadata, Value, Vector, VectorId};
use kova_index::HnswParams;
use kova_query::{Engine, ExecutionResult, ParamBindings, ParamValue};
use kova_storage::Shard;
use tempfile::tempdir;

fn open_shard(
    dir: &std::path::Path,
) -> Shard<L2, kova_storage::MmapVectorStore, kova_storage::FileMetadataStore, kova_storage::FileWal>
{
    Shard::open(dir, 2, L2, HnswParams::default()).unwrap()
}

fn v(data: Vec<f32>) -> Vector {
    Vector::try_new(data).unwrap()
}

fn meta(pairs: &[(&str, Value)]) -> Metadata {
    let mut m = Metadata::new();
    for (k, val) in pairs {
        m.insert((*k).to_string(), val.clone());
    }
    m
}

fn s(x: &str) -> Value {
    Value::String(x.into())
}

fn insert(engine: &mut Engine<L2>, id: u64, category: &str) {
    engine
        .execute_str(
            "INSERT INTO vectors (id, embedding, metadata) VALUES ($1, $2, $3)",
            ParamBindings::empty()
                .with_positional(ParamValue::Id(VectorId::new(id)))
                .with_positional(ParamValue::Vector(v(vec![
                    f32::from(u16::try_from(id).unwrap()),
                    0.0,
                ])))
                .with_positional(ParamValue::Metadata(meta(&[("category", s(category))]))),
        )
        .unwrap();
}

fn count_docs(engine: &mut Engine<L2>) -> i64 {
    let result = engine
        .execute_str(
            "SELECT COUNT(*) FROM vectors WHERE category = 'docs'",
            ParamBindings::empty(),
        )
        .unwrap();
    let ExecutionResult::Rows { rows, .. } = result else {
        panic!("expected Rows");
    };
    let kova_query::RowValue::Field(Value::I64(n)) = &rows[0].values[0] else {
        panic!("expected I64");
    };
    *n
}

#[test]
fn create_index_survives_close_without_checkpoint() {
    let dir = tempdir().unwrap();

    // Session 1 : insert rows, create the index, close WITHOUT
    // calling CHECKPOINT.
    {
        let mut engine = Engine::new(open_shard(dir.path()), "vectors");
        for n in 0u64..6 {
            let cat = if n % 2 == 0 { "docs" } else { "blog" };
            insert(&mut engine, n, cat);
        }
        engine
            .execute_str(
                "CREATE INDEX idx_cat ON vectors USING HASH (category)",
                ParamBindings::empty(),
            )
            .unwrap();
        // Sanity : the index works in the live session.
        assert_eq!(count_docs(&mut engine), 3);
    }

    // Session 2 : reopen. The catalog file on disk is still the
    // empty pre-DDL one, but the WAL contains the CreateIndex
    // record. Replay should restore the index.
    let mut engine = Engine::new(open_shard(dir.path()), "vectors");
    assert_eq!(
        count_docs(&mut engine),
        3,
        "CREATE INDEX should survive close even without an intervening CHECKPOINT"
    );

    // Verify by name : the registry round-tripped through the WAL too.
    let name = engine.shard().catalog().resolve_name("idx_cat");
    assert!(
        name.is_some(),
        "named index should survive reopen via WAL replay"
    );
}

#[test]
fn drop_index_survives_close_without_checkpoint() {
    let dir = tempdir().unwrap();

    {
        let mut engine = Engine::new(open_shard(dir.path()), "vectors");
        for n in 0u64..6 {
            let cat = if n % 2 == 0 { "docs" } else { "blog" };
            insert(&mut engine, n, cat);
        }
        engine
            .execute_str(
                "CREATE INDEX idx_cat ON vectors USING HASH (category)",
                ParamBindings::empty(),
            )
            .unwrap();
        engine
            .execute_str("DROP INDEX idx_cat ON vectors", ParamBindings::empty())
            .unwrap();
    }

    let engine = Engine::new(open_shard(dir.path()), "vectors");
    assert!(
        engine.shard().catalog().resolve_name("idx_cat").is_none(),
        "dropped index should stay dropped across reopen"
    );
}

#[test]
fn ddl_then_inserts_replay_in_order() {
    // The order matters : CREATE INDEX, then INSERTs. At reopen,
    // replay applies CreateIndex first (backfill walks an empty
    // metadata store and adds nothing). Then each Insert calls
    // catalog.on_insert which routes through the new index.
    //
    // Final state : the index sees every post-CREATE insert.
    let dir = tempdir().unwrap();

    {
        let mut engine = Engine::new(open_shard(dir.path()), "vectors");
        engine
            .execute_str(
                "CREATE INDEX idx_cat ON vectors USING HASH (category)",
                ParamBindings::empty(),
            )
            .unwrap();
        for n in 0u64..6 {
            let cat = if n % 2 == 0 { "docs" } else { "blog" };
            insert(&mut engine, n, cat);
        }
    }

    let mut engine = Engine::new(open_shard(dir.path()), "vectors");
    assert_eq!(count_docs(&mut engine), 3);
}

#[test]
fn inserts_then_ddl_backfill_during_replay() {
    // The harder order : INSERTs, then CREATE INDEX. At reopen,
    // replay applies each Insert first (catalog has no index, so
    // on_insert is a no-op), then CreateIndex which backfills the
    // already-loaded metadata.
    let dir = tempdir().unwrap();

    {
        let mut engine = Engine::new(open_shard(dir.path()), "vectors");
        for n in 0u64..6 {
            let cat = if n % 2 == 0 { "docs" } else { "blog" };
            insert(&mut engine, n, cat);
        }
        engine
            .execute_str(
                "CREATE INDEX idx_cat ON vectors USING HASH (category)",
                ParamBindings::empty(),
            )
            .unwrap();
    }

    let mut engine = Engine::new(open_shard(dir.path()), "vectors");
    assert_eq!(
        count_docs(&mut engine),
        3,
        "backfill during CREATE INDEX replay should pick up all preceding rows"
    );
}

#[test]
fn ddl_around_checkpoint_collapses_correctly() {
    // The interesting case : CREATE INDEX, then CHECKPOINT (catalog
    // file persists), then close. At reopen, the catalog file
    // already has the index ; WAL is truncated post-checkpoint and
    // has nothing to replay. The index lives via the catalog file
    // alone, no WAL needed.
    let dir = tempdir().unwrap();

    {
        let mut engine = Engine::new(open_shard(dir.path()), "vectors");
        for n in 0u64..6 {
            let cat = if n % 2 == 0 { "docs" } else { "blog" };
            insert(&mut engine, n, cat);
        }
        engine
            .execute_str(
                "CREATE INDEX idx_cat ON vectors USING HASH (category)",
                ParamBindings::empty(),
            )
            .unwrap();
        engine
            .execute_str("CHECKPOINT", ParamBindings::empty())
            .unwrap();
    }

    let mut engine = Engine::new(open_shard(dir.path()), "vectors");
    assert_eq!(count_docs(&mut engine), 3);
    assert!(engine.shard().catalog().resolve_name("idx_cat").is_some());
}

#[test]
fn create_then_drop_then_create_with_same_name_survives_reopen() {
    let dir = tempdir().unwrap();

    {
        let mut engine = Engine::new(open_shard(dir.path()), "vectors");
        engine
            .execute_str(
                "CREATE INDEX idx_cat ON vectors USING HASH (category)",
                ParamBindings::empty(),
            )
            .unwrap();
        engine
            .execute_str("DROP INDEX idx_cat ON vectors", ParamBindings::empty())
            .unwrap();
        engine
            .execute_str(
                "CREATE INDEX idx_cat ON vectors USING BTREE (category)",
                ParamBindings::empty(),
            )
            .unwrap();
    }

    let engine = Engine::new(open_shard(dir.path()), "vectors");
    let (_, kind) = engine
        .shard()
        .catalog()
        .resolve_name("idx_cat")
        .expect("named index should survive reopen");
    // The final CREATE was BTREE, so that's what survives.
    assert_eq!(kind, kova_meta_index::IndexKind::Btree);
}

#[test]
fn duplicate_create_after_reopen_still_errors() {
    // Strict registration must survive the reopen too : reopening
    // a shard with an index, then trying to CREATE the same name
    // again, must error loudly.
    let dir = tempdir().unwrap();

    {
        let mut engine = Engine::new(open_shard(dir.path()), "vectors");
        engine
            .execute_str(
                "CREATE INDEX idx_cat ON vectors USING HASH (category)",
                ParamBindings::empty(),
            )
            .unwrap();
    }

    let mut engine = Engine::new(open_shard(dir.path()), "vectors");
    let err = engine
        .execute_str(
            "CREATE INDEX idx_cat ON vectors USING HASH (category)",
            ParamBindings::empty(),
        )
        .unwrap_err();
    let msg = format!("{err}");
    // Walk the source chain to find the inner message.
    let mut cur: Option<&dyn std::error::Error> = std::error::Error::source(&err);
    let mut found = msg.contains("idx_cat") && msg.contains("already exists");
    while let Some(e) = cur {
        let s = format!("{e}");
        if s.contains("idx_cat") && s.contains("already exists") {
            found = true;
        }
        cur = std::error::Error::source(e);
    }
    assert!(found, "expected duplicate-name error after reopen");
}

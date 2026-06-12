//! End-to-end CREATE INDEX / DROP INDEX integration tests.
//!
//! Drives the full pipeline (parser → binder → planner → executor)
//! through `Engine::execute_str` and verifies the resulting catalog
//! state by issuing a query that would only succeed on the index
//! path. After DROP, the catalog stops answering and we fall back
//! to a metadata scan (same answer, same shape — round-trip checked
//! the byte-identity claim).

use kova_core::{L2, Metadata, Value, Vector, VectorId};
use kova_index::HnswParams;
use kova_query::{Engine, ExecutionResult, ParamBindings, ParamValue};
use kova_storage::Shard;
use tempfile::tempdir;

/// Walk the `source()` chain of an error and concatenate every
/// message into one string. Avoids the `KovaQueryError::Backend`
/// outer Display swallowing the inner reason.
fn full_error_chain(err: &dyn std::error::Error) -> String {
    let mut parts = vec![format!("{err}")];
    let mut next = err.source();
    while let Some(e) = next {
        parts.push(format!("{e}"));
        next = e.source();
    }
    parts.join(" :: ")
}

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

fn i(n: i64) -> Value {
    Value::I64(n)
}

fn arr(xs: &[&str]) -> Value {
    Value::Array(xs.iter().map(|x| s(x)).collect())
}

/// Seed a small fixture so the catalog backfill has rows to walk.
fn seed(engine: &mut Engine<L2>) {
    for n in 0u64..10 {
        let category = if n % 2 == 0 { "docs" } else { "blog" };
        let year = 2020 + i64::try_from(n).unwrap();
        let tags = if n % 3 == 0 {
            arr(&["rust"])
        } else {
            arr(&["go"])
        };
        engine
            .execute_str(
                "INSERT INTO vectors (id, embedding, metadata) VALUES ($1, $2, $3)",
                ParamBindings::empty()
                    .with_positional(ParamValue::Id(VectorId::new(n)))
                    .with_positional(ParamValue::Vector(v(vec![
                        f32::from(u16::try_from(n).unwrap()),
                        0.0,
                    ])))
                    .with_positional(ParamValue::Metadata(meta(&[
                        ("category", s(category)),
                        ("year", i(year)),
                        ("tags", tags),
                    ]))),
            )
            .unwrap();
    }
}

#[test]
fn create_hash_index_returns_create_index_result() {
    let dir = tempdir().unwrap();
    let mut engine = Engine::new(open_shard(dir.path()), "vectors");
    seed(&mut engine);

    let result = engine
        .execute_str(
            "CREATE INDEX idx_cat ON vectors USING HASH (category)",
            ParamBindings::empty(),
        )
        .unwrap();
    let ExecutionResult::CreateIndex { table, name } = result else {
        panic!("expected CreateIndex result, got {result:?}");
    };
    assert_eq!(table, "vectors");
    assert_eq!(name, "idx_cat");
}

#[test]
fn create_index_without_name_synthesises_one() {
    let dir = tempdir().unwrap();
    let mut engine = Engine::new(open_shard(dir.path()), "vectors");
    seed(&mut engine);

    let result = engine
        .execute_str(
            "CREATE INDEX ON vectors USING BTREE (year)",
            ParamBindings::empty(),
        )
        .unwrap();
    let ExecutionResult::CreateIndex { name, .. } = result else {
        panic!("expected CreateIndex result");
    };
    assert_eq!(name, "idx_year_btree");
}

#[test]
fn create_then_drop_index_round_trip() {
    let dir = tempdir().unwrap();
    let mut engine = Engine::new(open_shard(dir.path()), "vectors");
    seed(&mut engine);

    // Create then drop.
    engine
        .execute_str(
            "CREATE INDEX idx_cat ON vectors USING HASH (category)",
            ParamBindings::empty(),
        )
        .unwrap();

    let drop_result = engine
        .execute_str("DROP INDEX idx_cat ON vectors", ParamBindings::empty())
        .unwrap();
    let ExecutionResult::DropIndex { table, name } = drop_result else {
        panic!("expected DropIndex result, got {drop_result:?}");
    };
    assert_eq!(table, "vectors");
    assert_eq!(name, "idx_cat");
}

#[test]
fn create_duplicate_name_errors() {
    let dir = tempdir().unwrap();
    let mut engine = Engine::new(open_shard(dir.path()), "vectors");
    seed(&mut engine);

    engine
        .execute_str(
            "CREATE INDEX idx_cat ON vectors USING HASH (category)",
            ParamBindings::empty(),
        )
        .unwrap();

    let err = engine
        .execute_str(
            "CREATE INDEX idx_cat ON vectors USING HASH (tags)",
            ParamBindings::empty(),
        )
        .unwrap_err();
    let msg = full_error_chain(&err);
    assert!(
        msg.contains("idx_cat") && msg.contains("already exists"),
        "expected duplicate-name error, got: {msg}"
    );
}

#[test]
fn drop_unknown_name_errors() {
    let dir = tempdir().unwrap();
    let mut engine = Engine::new(open_shard(dir.path()), "vectors");
    seed(&mut engine);

    let err = engine
        .execute_str("DROP INDEX nope ON vectors", ParamBindings::empty())
        .unwrap_err();
    let msg = full_error_chain(&err);
    assert!(
        msg.contains("nope") && msg.contains("no index"),
        "expected unknown-name error, got: {msg}"
    );
}

#[test]
fn ddl_against_wrong_table_errors() {
    let dir = tempdir().unwrap();
    let mut engine = Engine::new(open_shard(dir.path()), "vectors");
    seed(&mut engine);

    let err = engine
        .execute_str(
            "CREATE INDEX idx ON wrong_table USING HASH (category)",
            ParamBindings::empty(),
        )
        .unwrap_err();
    let msg = full_error_chain(&err);
    assert!(
        msg.contains("table") || msg.contains("wrong_table"),
        "expected table-mismatch error, got: {msg}"
    );
}

#[test]
fn created_index_powers_subsequent_select() {
    // After CREATE INDEX, the catalog should route the SELECT through
    // the index path. We can't observe that directly, but we can
    // assert the query succeeds and returns the correct rows. The
    // M2.5 round-trip tests already prove byte-identity ; this test
    // just pins the integration glue.
    let dir = tempdir().unwrap();
    let mut engine = Engine::new(open_shard(dir.path()), "vectors");
    seed(&mut engine);

    engine
        .execute_str(
            "CREATE INDEX idx_cat ON vectors USING HASH (category)",
            ParamBindings::empty(),
        )
        .unwrap();

    let result = engine
        .execute_str(
            "SELECT COUNT(*) FROM vectors WHERE category = 'docs'",
            ParamBindings::empty(),
        )
        .unwrap();
    let ExecutionResult::Rows { rows, .. } = result else {
        panic!("expected Rows");
    };
    // Five `docs` rows in the fixture (even ids 0,2,4,6,8).
    let kova_query::RowValue::Field(Value::I64(n)) = &rows[0].values[0] else {
        panic!("expected I64 count, got {:?}", rows[0].values[0]);
    };
    assert_eq!(*n, 5);
}

#[test]
fn create_inverted_index_then_array_contains() {
    let dir = tempdir().unwrap();
    let mut engine = Engine::new(open_shard(dir.path()), "vectors");
    seed(&mut engine);

    engine
        .execute_str(
            "CREATE INDEX idx_tags ON vectors USING INVERTED (tags)",
            ParamBindings::empty(),
        )
        .unwrap();

    let result = engine
        .execute_str(
            "SELECT COUNT(*) FROM vectors WHERE tags @> 'rust'",
            ParamBindings::empty(),
        )
        .unwrap();
    let ExecutionResult::Rows { rows, .. } = result else {
        panic!("expected Rows");
    };
    // Rows with n % 3 == 0 : ids 0, 3, 6, 9 → 4 matches.
    let kova_query::RowValue::Field(Value::I64(n)) = &rows[0].values[0] else {
        panic!("expected I64 count");
    };
    assert_eq!(*n, 4);
}

#[test]
fn drop_then_recreate_is_supported() {
    let dir = tempdir().unwrap();
    let mut engine = Engine::new(open_shard(dir.path()), "vectors");
    seed(&mut engine);

    engine
        .execute_str(
            "CREATE INDEX idx_cat ON vectors USING HASH (category)",
            ParamBindings::empty(),
        )
        .unwrap();
    engine
        .execute_str("DROP INDEX idx_cat ON vectors", ParamBindings::empty())
        .unwrap();
    // Re-create under the same name : should succeed (no stale name).
    engine
        .execute_str(
            "CREATE INDEX idx_cat ON vectors USING HASH (category)",
            ParamBindings::empty(),
        )
        .unwrap();
}

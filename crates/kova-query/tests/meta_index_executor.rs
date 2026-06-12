//! Integration tests for the executor's `MetadataScan` index path.
//!
//! Strategy : run the same KQL query twice on the same dataset, once
//! WITHOUT any indexes registered (forces fallback to `shard.scan_metadata`)
//! and once WITH indexes registered (engages the catalog-driven path).
//! Both runs must produce identical id sets. The point of the slice
//! isn't to change semantics ; it's to compute the same answer faster.

use std::collections::BTreeSet;

use kova_core::{L2, Metadata, Value, Vector, VectorId};
use kova_index::HnswParams;
use kova_query::{Engine, ExecutionResult, ParamBindings, ParamValue, RowValue};
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

fn i(n: i64) -> Value {
    Value::I64(n)
}

fn arr(xs: &[&str]) -> Value {
    Value::Array(xs.iter().map(|x| s(x)).collect())
}

/// Seed `engine` with the canonical fixture used across the tests
/// in this file : 20 rows, three indexable fields. Categories
/// alternate even/odd, years cycle 2020..2026, tags rotate.
fn seed(engine: &mut Engine<L2>) {
    for n in 0u64..20 {
        let category = if n % 2 == 0 { "docs" } else { "blog" };
        let year = 2020 + i64::try_from(n % 7).unwrap();
        let tags = if n % 3 == 0 {
            arr(&["rust", "async"])
        } else if n % 3 == 1 {
            arr(&["go"])
        } else {
            arr(&["python", "ml"])
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

/// Extract just the id set from a `SELECT id ...` result. Order is
/// implementation-defined for the scan-and-limit path ; we compare
/// as sets.
fn id_set(result: ExecutionResult) -> BTreeSet<u64> {
    let ExecutionResult::Rows { rows, .. } = result else {
        panic!("expected Rows, got {result:?}");
    };
    rows.into_iter()
        .map(|row| {
            let RowValue::Id(id) = row.values[0] else {
                panic!("expected first column to be Id");
            };
            id.get()
        })
        .collect()
}

/// Run the same query twice : once without any indexes (scan path)
/// and once with the given indexes registered (index path). Assert
/// both produce the same id set.
fn round_trip(sql: &str, params: ParamBindings, index_setup: fn(&mut Engine<L2>)) {
    // --- Run A : no indexes (fallback path) ---
    let dir_a = tempdir().unwrap();
    let mut engine_a = Engine::new(open_shard(dir_a.path()), "vectors");
    seed(&mut engine_a);
    let a = id_set(engine_a.execute_str(sql, params.clone()).unwrap());

    // --- Run B : indexes registered (catalog path) ---
    let dir_b = tempdir().unwrap();
    let mut engine_b = Engine::new(open_shard(dir_b.path()), "vectors");
    seed(&mut engine_b);
    index_setup(&mut engine_b);
    let b = id_set(engine_b.execute_str(sql, params).unwrap());

    assert_eq!(a, b, "scan path vs index path diverged for SQL: {sql}");
    assert!(
        !a.is_empty(),
        "fixture should produce non-empty result for {sql}"
    );
}

fn install_hash_on_category(engine: &mut Engine<L2>) {
    engine.shard_mut().add_hash_index("category").unwrap();
}

fn install_btree_on_year(engine: &mut Engine<L2>) {
    engine.shard_mut().add_btree_index("year").unwrap();
}

fn install_inverted_on_tags(engine: &mut Engine<L2>) {
    engine.shard_mut().add_inverted_index("tags").unwrap();
}

fn install_all_three(engine: &mut Engine<L2>) {
    engine.shard_mut().add_hash_index("category").unwrap();
    engine.shard_mut().add_btree_index("year").unwrap();
    engine.shard_mut().add_inverted_index("tags").unwrap();
}

// ---- Pure scan-and-limit shapes (no kNN ORDER BY) ----

#[test]
fn scan_and_limit_eq_on_hash_index_matches_fallback() {
    round_trip(
        "SELECT id FROM vectors WHERE category = 'docs' LIMIT 100",
        ParamBindings::empty(),
        install_hash_on_category,
    );
}

#[test]
fn scan_and_limit_range_on_btree_index_matches_fallback() {
    round_trip(
        "SELECT id FROM vectors WHERE year > 2023 LIMIT 100",
        ParamBindings::empty(),
        install_btree_on_year,
    );
}

#[test]
fn scan_and_limit_between_on_btree_index_matches_fallback() {
    round_trip(
        "SELECT id FROM vectors WHERE year BETWEEN 2022 AND 2024 LIMIT 100",
        ParamBindings::empty(),
        install_btree_on_year,
    );
}

#[test]
fn scan_and_limit_array_contains_on_inverted_index_matches_fallback() {
    round_trip(
        "SELECT id FROM vectors WHERE tags @> 'rust' LIMIT 100",
        ParamBindings::empty(),
        install_inverted_on_tags,
    );
}

#[test]
fn scan_and_limit_in_clause_on_hash_index_matches_fallback() {
    round_trip(
        "SELECT id FROM vectors WHERE category IN ('docs', 'blog') LIMIT 100",
        ParamBindings::empty(),
        install_hash_on_category,
    );
}

#[test]
fn scan_and_limit_is_not_null_on_hash_index_matches_fallback() {
    round_trip(
        "SELECT id FROM vectors WHERE category IS NOT NULL LIMIT 100",
        ParamBindings::empty(),
        install_hash_on_category,
    );
}

#[test]
fn scan_and_limit_ne_on_hash_index_matches_fallback() {
    round_trip(
        "SELECT id FROM vectors WHERE category != 'docs' LIMIT 100",
        ParamBindings::empty(),
        install_hash_on_category,
    );
}

// ---- AND composition ----

#[test]
fn and_two_indexes_full_path_matches_fallback() {
    round_trip(
        "SELECT id FROM vectors WHERE category = 'docs' AND year >= 2023 LIMIT 100",
        ParamBindings::empty(),
        |e| {
            e.shard_mut().add_hash_index("category").unwrap();
            e.shard_mut().add_btree_index("year").unwrap();
        },
    );
}

#[test]
fn and_three_indexes_full_path_matches_fallback() {
    round_trip(
        "SELECT id FROM vectors \
         WHERE category = 'docs' AND year >= 2022 AND tags @> 'rust' \
         LIMIT 100",
        ParamBindings::empty(),
        install_all_three,
    );
}

#[test]
fn and_hybrid_indexed_plus_unindexed_matches_fallback() {
    // `year` indexed ; `category` is NOT indexed -> hybrid path
    round_trip(
        "SELECT id FROM vectors WHERE year > 2022 AND category = 'docs' LIMIT 100",
        ParamBindings::empty(),
        install_btree_on_year,
    );
}

// ---- OR composition ----

#[test]
fn or_two_indexes_full_path_matches_fallback() {
    round_trip(
        "SELECT id FROM vectors WHERE category = 'docs' OR category = 'blog' LIMIT 100",
        ParamBindings::empty(),
        install_hash_on_category,
    );
}

#[test]
fn or_with_one_unindexed_branch_falls_back_to_scan() {
    // The OR can't be index-evaluated because one branch is unindexed.
    // The fallback path must still produce the right answer.
    round_trip(
        "SELECT id FROM vectors WHERE category = 'docs' OR year > 2024 LIMIT 100",
        ParamBindings::empty(),
        install_hash_on_category,
    );
}

// ---- NOT shape (currently falls back) ----

#[test]
fn not_predicate_falls_back_to_scan() {
    round_trip(
        "SELECT id FROM vectors WHERE NOT (category = 'docs') LIMIT 100",
        ParamBindings::empty(),
        install_hash_on_category,
    );
}

// ---- DELETE-by-predicate (still uses Shard::scan_metadata today ; verifies the index path
//      doesn't accidentally break the unrelated DML scan paths) ----

#[test]
fn delete_by_predicate_still_works_with_indexes_registered() {
    let dir = tempdir().unwrap();
    let mut engine = Engine::new(open_shard(dir.path()), "vectors");
    seed(&mut engine);
    engine.shard_mut().add_hash_index("category").unwrap();

    let result = engine
        .execute_str(
            "DELETE FROM vectors WHERE category = 'blog'",
            ParamBindings::empty(),
        )
        .unwrap();
    let ExecutionResult::Delete { deleted, .. } = result else {
        panic!("expected Delete result");
    };
    assert_eq!(deleted, 10);
}

// ---- Indexes added AFTER inserts (backfill path) ----

#[test]
fn index_registered_after_inserts_still_matches_fallback() {
    let dir = tempdir().unwrap();
    let mut engine = Engine::new(open_shard(dir.path()), "vectors");
    seed(&mut engine);
    // Register AFTER inserts ; backfill walks the metadata store.
    engine.shard_mut().add_hash_index("category").unwrap();

    let dir_b = tempdir().unwrap();
    let mut engine_b = Engine::new(open_shard(dir_b.path()), "vectors");
    seed(&mut engine_b);

    let with_idx = id_set(
        engine
            .execute_str(
                "SELECT id FROM vectors WHERE category = 'docs' LIMIT 100",
                ParamBindings::empty(),
            )
            .unwrap(),
    );
    let no_idx = id_set(
        engine_b
            .execute_str(
                "SELECT id FROM vectors WHERE category = 'docs' LIMIT 100",
                ParamBindings::empty(),
            )
            .unwrap(),
    );
    assert_eq!(with_idx, no_idx);
}

// ---- Empty / no-match cases ----

#[test]
fn no_match_on_indexed_field_returns_empty() {
    let dir = tempdir().unwrap();
    let mut engine = Engine::new(open_shard(dir.path()), "vectors");
    seed(&mut engine);
    engine.shard_mut().add_hash_index("category").unwrap();

    let result = engine
        .execute_str(
            "SELECT id FROM vectors WHERE category = 'no-such-category' LIMIT 100",
            ParamBindings::empty(),
        )
        .unwrap();
    let ExecutionResult::Rows { rows, .. } = result else {
        panic!("expected Rows");
    };
    assert!(rows.is_empty());
}

// ---- COUNT round-trip ----

/// Extract a `COUNT(*)` result as `i64`.
fn count_value(result: ExecutionResult) -> i64 {
    let ExecutionResult::Rows { rows, .. } = result else {
        panic!("expected Rows, got {result:?}");
    };
    assert_eq!(rows.len(), 1, "COUNT must return exactly one row");
    let RowValue::Field(Value::I64(n)) = rows[0].values[0] else {
        panic!("expected I64 count cell, got {:?}", rows[0].values[0]);
    };
    n
}

/// Run the same COUNT query with and without indexes registered.
/// Assert the count is identical.
fn count_round_trip(sql: &str, params: ParamBindings, index_setup: fn(&mut Engine<L2>)) {
    let dir_a = tempdir().unwrap();
    let mut engine_a = Engine::new(open_shard(dir_a.path()), "vectors");
    seed(&mut engine_a);
    let a = count_value(engine_a.execute_str(sql, params.clone()).unwrap());

    let dir_b = tempdir().unwrap();
    let mut engine_b = Engine::new(open_shard(dir_b.path()), "vectors");
    seed(&mut engine_b);
    index_setup(&mut engine_b);
    let b = count_value(engine_b.execute_str(sql, params).unwrap());

    assert_eq!(a, b, "scan path vs index path diverged for COUNT: {sql}");
    assert!(a > 0, "fixture should produce non-zero count for {sql}");
}

#[test]
fn count_eq_on_hash_index_matches_fallback() {
    count_round_trip(
        "SELECT COUNT(*) FROM vectors WHERE category = 'docs'",
        ParamBindings::empty(),
        install_hash_on_category,
    );
}

#[test]
fn count_range_on_btree_index_matches_fallback() {
    count_round_trip(
        "SELECT COUNT(*) FROM vectors WHERE year >= 2024",
        ParamBindings::empty(),
        install_btree_on_year,
    );
}

#[test]
fn count_array_contains_on_inverted_index_matches_fallback() {
    count_round_trip(
        "SELECT COUNT(*) FROM vectors WHERE tags @> 'rust'",
        ParamBindings::empty(),
        install_inverted_on_tags,
    );
}

#[test]
fn count_and_chain_full_path_matches_fallback() {
    count_round_trip(
        "SELECT COUNT(*) FROM vectors WHERE category = 'docs' AND year >= 2022 AND tags @> 'rust'",
        ParamBindings::empty(),
        install_all_three,
    );
}

#[test]
fn count_hybrid_path_matches_fallback() {
    // year indexed ; category isn't -> Hybrid in the count helper.
    count_round_trip(
        "SELECT COUNT(*) FROM vectors WHERE year > 2022 AND category = 'docs'",
        ParamBindings::empty(),
        install_btree_on_year,
    );
}

#[test]
fn count_no_predicate_unchanged_with_indexes() {
    // `COUNT(*)` with no WHERE goes through `shard.len()` directly,
    // not through count_matching_with_predicate. The index install
    // is a no-op for this path ; we test it anyway to pin the
    // behaviour against future refactors.
    count_round_trip(
        "SELECT COUNT(*) FROM vectors",
        ParamBindings::empty(),
        install_all_three,
    );
}

// ---- DELETE-by-predicate round-trip ----

/// Extract a DELETE result as the deleted count + the surviving id
/// set (via a follow-up SELECT). Returns `(deleted_count,
/// surviving_ids)`.
fn delete_and_survivors(
    engine: &mut Engine<L2>,
    delete_sql: &str,
    params: ParamBindings,
) -> (u64, BTreeSet<u64>) {
    let result = engine.execute_str(delete_sql, params).unwrap();
    let ExecutionResult::Delete { deleted, .. } = result else {
        panic!("expected Delete result, got {result:?}");
    };
    let survivors = id_set(
        engine
            .execute_str(
                "SELECT id FROM vectors WHERE id IS NOT NULL LIMIT 1000",
                ParamBindings::empty(),
            )
            .unwrap_or_else(|_| {
                // `id IS NOT NULL` isn't a thing ; use a scan-and-limit
                // bypass via a different shape. We fall back to walking
                // every id by listing all categories the fixture knows.
                engine
                    .execute_str(
                        "SELECT id FROM vectors WHERE category IN ('docs', 'blog') LIMIT 1000",
                        ParamBindings::empty(),
                    )
                    .unwrap()
            }),
    );
    (deleted, survivors)
}

fn delete_round_trip(sql: &str, params: ParamBindings, index_setup: fn(&mut Engine<L2>)) {
    let dir_a = tempdir().unwrap();
    let mut engine_a = Engine::new(open_shard(dir_a.path()), "vectors");
    seed(&mut engine_a);
    let (del_a, surv_a) = delete_and_survivors(&mut engine_a, sql, params.clone());

    let dir_b = tempdir().unwrap();
    let mut engine_b = Engine::new(open_shard(dir_b.path()), "vectors");
    seed(&mut engine_b);
    index_setup(&mut engine_b);
    let (del_b, surv_b) = delete_and_survivors(&mut engine_b, sql, params);

    assert_eq!(del_a, del_b, "delete count diverged for: {sql}");
    assert_eq!(surv_a, surv_b, "survivor set diverged for: {sql}");
    assert!(
        del_a > 0,
        "fixture should delete a non-zero count for {sql}"
    );
}

#[test]
fn delete_eq_on_hash_index_matches_fallback() {
    delete_round_trip(
        "DELETE FROM vectors WHERE category = 'docs'",
        ParamBindings::empty(),
        install_hash_on_category,
    );
}

#[test]
fn delete_range_on_btree_index_matches_fallback() {
    delete_round_trip(
        "DELETE FROM vectors WHERE year >= 2024",
        ParamBindings::empty(),
        install_btree_on_year,
    );
}

#[test]
fn delete_and_chain_full_path_matches_fallback() {
    delete_round_trip(
        "DELETE FROM vectors WHERE category = 'blog' AND year < 2023",
        ParamBindings::empty(),
        |e| {
            e.shard_mut().add_hash_index("category").unwrap();
            e.shard_mut().add_btree_index("year").unwrap();
        },
    );
}

#[test]
fn delete_hybrid_path_matches_fallback() {
    delete_round_trip(
        "DELETE FROM vectors WHERE year < 2023 AND category = 'docs'",
        ParamBindings::empty(),
        install_btree_on_year,
    );
}

// ---- UPDATE-by-predicate round-trip ----

/// After an UPDATE, query the rows where the assigned value lives
/// and return their id set. Tests that the right ids got mutated.
fn update_and_after_set(
    engine: &mut Engine<L2>,
    update_sql: &str,
    follow_up_sql: &str,
    params: ParamBindings,
) -> (u64, BTreeSet<u64>) {
    let result = engine.execute_str(update_sql, params.clone()).unwrap();
    let ExecutionResult::Update { updated, .. } = result else {
        panic!("expected Update result, got {result:?}");
    };
    let after = id_set(engine.execute_str(follow_up_sql, params).unwrap());
    (updated, after)
}

fn update_round_trip(
    update_sql: &str,
    follow_up_sql: &str,
    params: ParamBindings,
    index_setup: fn(&mut Engine<L2>),
) {
    let dir_a = tempdir().unwrap();
    let mut engine_a = Engine::new(open_shard(dir_a.path()), "vectors");
    seed(&mut engine_a);
    let (upd_a, after_a) =
        update_and_after_set(&mut engine_a, update_sql, follow_up_sql, params.clone());

    let dir_b = tempdir().unwrap();
    let mut engine_b = Engine::new(open_shard(dir_b.path()), "vectors");
    seed(&mut engine_b);
    index_setup(&mut engine_b);
    let (upd_b, after_b) = update_and_after_set(&mut engine_b, update_sql, follow_up_sql, params);

    assert_eq!(upd_a, upd_b, "update count diverged for: {update_sql}");
    assert_eq!(
        after_a, after_b,
        "post-update id set diverged for: {update_sql}"
    );
    assert!(
        upd_a > 0,
        "fixture should update a non-zero count for {update_sql}"
    );
}

#[test]
fn update_eq_on_hash_index_matches_fallback() {
    update_round_trip(
        // Update : reassign category for all blogs to a fresh value.
        "UPDATE vectors SET category = 'updated' WHERE category = 'blog'",
        // Follow-up : list ids that now have category = 'updated'.
        "SELECT id FROM vectors WHERE category = 'updated' LIMIT 100",
        ParamBindings::empty(),
        install_hash_on_category,
    );
}

#[test]
fn update_range_on_btree_index_matches_fallback() {
    update_round_trip(
        "UPDATE vectors SET priority = 9 WHERE year >= 2024",
        // priority isn't a fixture field ; use scan-and-limit via the
        // year predicate to recover the affected ids.
        "SELECT id FROM vectors WHERE year >= 2024 LIMIT 100",
        ParamBindings::empty(),
        install_btree_on_year,
    );
}

#[test]
fn update_hybrid_path_matches_fallback() {
    update_round_trip(
        "UPDATE vectors SET reviewed = 1 WHERE year >= 2023 AND category = 'docs'",
        "SELECT id FROM vectors WHERE year >= 2023 AND category = 'docs' LIMIT 100",
        ParamBindings::empty(),
        install_btree_on_year,
    );
}

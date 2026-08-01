//! Plan A must not be dispatched when it cannot fill the LIMIT.
//!
//! Plan A walks the HNSW for `k * KNN_OVERFETCH` candidates,
//! post-filters them, and returns the survivors **without retrying**.
//! Its expected yield is therefore `k * OVERFETCH * s`, which falls
//! below the LIMIT whenever `s < 1 / OVERFETCH` (0.25).
//!
//! Measured before the fix (`examples/validate_cost_model.rs`,
//! `dim=16 n=10_000`, rows returned for `LIMIT 10`) :
//!
//! ```text
//!   s=0.001  ->  A 0    B 10   C 10        (10 rows match)
//!   s=0.01   ->  A 0    B 10   C 10
//!   s=0.05   ->  A 0    B 10   C 10
//!   s=0.2    ->  A 9    B 10   C 10
//! ```
//!
//! Plan A returned **nothing** for a query with ten valid answers, and
//! the cost model dispatched it because 51 us of nothing beat 369 us of
//! the right answer : `cost_plan_a` is deliberately independent of
//! selectivity, so no cost comparison could see the difference.
//!
//! These tests go through the **full public pipeline**
//! (`Engine::execute_str`), not a forced plan, so they assert what a
//! user actually gets.

use kova_core::{L2, Metadata, Value, Vector, VectorId};
use kova_index::HnswParams;
use kova_query::executor::{Engine, ExecutionResult, ParamBindings, ParamValue};
use kova_storage::Shard;
use rand::{RngExt, SeedableRng, rngs::StdRng};

const DIM: usize = 8;
const N: usize = 2_000;
const SEED: u64 = 0x5747_0000_5747_0000;

/// Build a shard of `N` rows where exactly `matching` of them satisfy
/// `bucket = 0`. Matches are scattered through insertion order.
fn engine_with_selectivity(dir: &tempfile::TempDir, matching: usize) -> Engine<L2> {
    let shard = Shard::open(dir.path(), DIM, L2, HnswParams::default()).expect("open");
    let mut engine = Engine::new(shard, "vectors");
    let mut rng = StdRng::seed_from_u64(SEED);

    // Stride coprime with N spreads the matching rows evenly.
    let stride = 7_919usize;
    // Batch the load : singleton `insert` fsyncs the WAL per row, which
    // made building these fixtures dominate the test's runtime.
    // `insert_many` group-commits the whole batch behind one barrier.
    let batch: Vec<(VectorId, Vector, Metadata)> = (0..N)
        .map(|i| {
            let v: Vec<f32> = (0..DIM).map(|_| rng.random::<f32>()).collect();
            let is_match = (i * stride) % N < matching;
            let mut m = Metadata::new();
            m.insert("bucket".into(), Value::I64(i64::from(!is_match)));
            (
                VectorId::new(i as u64),
                Vector::try_new(v).expect("vector"),
                m,
            )
        })
        .collect();
    engine.shard_mut().insert_many(batch).expect("insert_many");
    engine
}

fn knn_rows(engine: &mut Engine<L2>, limit: usize) -> usize {
    let mut rng = StdRng::seed_from_u64(SEED ^ 0xF00D);
    let q: Vec<f32> = (0..DIM).map(|_| rng.random::<f32>()).collect();
    let result = engine
        .execute_str(
            &format!(
                "SELECT id, embedding <-> $1 AS d FROM vectors \
                 WHERE bucket = 0 ORDER BY embedding <-> $1 LIMIT {limit}"
            ),
            ParamBindings::empty()
                .with_positional(ParamValue::Vector(Vector::try_new(q).expect("query"))),
        )
        .expect("query should succeed");
    match result {
        ExecutionResult::Rows { rows, .. } => rows.len(),
        other => panic!("expected Rows, got {other:?}"),
    }
}

/// The headline case : a selective filter with far more matches than
/// the LIMIT asks for. Every plan *can* answer this completely ; only
/// plan A fails to, and only because it does not retry.
#[test]
fn selective_filter_still_fills_the_limit() {
    for (matching, label) in [
        (2usize, "s=0.001"),
        (20, "s=0.01"),
        (100, "s=0.05"),
        (400, "s=0.2"),
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut engine = engine_with_selectivity(&dir, matching);
        let limit = 10;
        let got = knn_rows(&mut engine, limit);
        let expected = limit.min(matching);
        assert_eq!(
            got, expected,
            "{label}: {matching}/{N} rows match, LIMIT {limit} -> expected {expected} rows, got {got}. \
             A starved plan A was dispatched ; see cost::plan_a_can_satisfy."
        );
    }
}

/// When fewer rows match than the LIMIT asks for, the complete answer
/// is the match count. Plan A returns a fraction of even that.
#[test]
fn limit_larger_than_match_count_returns_every_match() {
    let matching = 20;
    let dir = tempfile::tempdir().expect("tempdir");
    let mut engine = engine_with_selectivity(&dir, matching);
    let got = knn_rows(&mut engine, 500);
    assert_eq!(
        got, matching,
        "{matching}/{N} rows match and LIMIT is 500 -> all {matching} should come back, got {got}"
    );
}

/// A loose filter is plan A's home ground and must keep working : the
/// correctness gate should not push every query off plan A.
#[test]
fn loose_filter_still_fills_the_limit() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut engine = engine_with_selectivity(&dir, N * 9 / 10); // s = 0.9
    assert_eq!(knn_rows(&mut engine, 10), 10);
    assert_eq!(knn_rows(&mut engine, 100), 100);
}

/// A predicate that matches nothing returns nothing : the gate must not
/// invent rows or error.
#[test]
fn unsatisfiable_filter_returns_empty() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut engine = engine_with_selectivity(&dir, 0);
    assert_eq!(knn_rows(&mut engine, 10), 0);
}

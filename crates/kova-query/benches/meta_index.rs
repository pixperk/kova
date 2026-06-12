//! Baseline benchmarks for the IndexCatalog integration.
//!
//! Each query is run twice against identical 10k-row shards :
//! - **scan**    : no indexes registered, paths fall back to
//!   `shard.scan_metadata` / `shard.count_matching` (O(N) every time)
//! - **indexed** : the relevant indexes are registered, paths route
//!   through `IndexCatalog::lookup` / `IndexCatalog::estimate`
//!
//! Same SQL, same fixture, same parser/binder/planner — the only
//! difference is whether the catalog has anything to say. The delta
//! is the M2.5 win.
//!
//! Dataset shape :
//! - 10_000 rows, 16-dim vectors (small enough to bench in seconds)
//! - `category` : one of {"docs", "blog", "code", "wiki"} (4 values,
//!   ~25% selectivity per value)
//! - `year`     : 2015..=2025 (11 values, range queries on this)
//! - `tags`     : array of 1-3 strings drawn from {rust, python, go,
//!   async, ml, web} (~5% per single-tag query)
//! - `priority` : 1..10 (NOT indexed, used to exercise hybrid + scan)

#![allow(missing_docs, clippy::cast_precision_loss)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use kova_core::{L2, Metadata, Value, Vector, VectorId};
use kova_index::HnswParams;
use kova_query::{Engine, ParamBindings, ParamValue};
use kova_storage::Shard;
use tempfile::TempDir;

const ROWS: u64 = 10_000;
const DIM: usize = 16;

/// Deterministic seed : the same `i` always produces the same row.
fn category_for(i: u64) -> &'static str {
    match i % 4 {
        0 => "docs",
        1 => "blog",
        2 => "code",
        _ => "wiki",
    }
}

fn year_for(i: u64) -> i64 {
    2015 + i64::try_from(i % 11).unwrap()
}

fn priority_for(i: u64) -> i64 {
    1 + i64::try_from(i % 10).unwrap()
}

fn tags_for(i: u64) -> Value {
    let pool = ["rust", "python", "go", "async", "ml", "web"];
    let len = 1 + usize::try_from(i % 3).unwrap();
    let mut out = Vec::with_capacity(len);
    for k in 0..len {
        let pick = (usize::try_from(i).unwrap_or(0) + k) % pool.len();
        out.push(Value::String(pool[pick].into()));
    }
    Value::Array(out)
}

fn vector_for(i: u64) -> Vector {
    let mut data = vec![0.0_f32; DIM];
    for (k, slot) in data.iter_mut().enumerate() {
        *slot = (i as f32 + k as f32 * 0.01) / f32::from(u16::try_from(ROWS).unwrap());
    }
    Vector::try_new(data).unwrap()
}

fn meta_for(i: u64) -> Metadata {
    let mut m = Metadata::new();
    m.insert("category".into(), Value::String(category_for(i).into()));
    m.insert("year".into(), Value::I64(year_for(i)));
    m.insert("tags".into(), tags_for(i));
    m.insert("priority".into(), Value::I64(priority_for(i)));
    m
}

/// Build a fresh file-backed shard, seed `ROWS` rows. The
/// `register_indexes` closure either is a no-op (scan variant) or
/// installs the indexes the bench needs (indexed variant).
fn build_engine(register_indexes: impl FnOnce(&mut Engine<L2>)) -> (Engine<L2>, TempDir) {
    let dir = TempDir::new().unwrap();
    let shard = Shard::open(dir.path(), DIM, L2, HnswParams::default()).unwrap();
    let mut engine = Engine::new(shard, "vectors");

    // Use batched INSERT for fixture speed.
    let batch: Vec<_> = (0..ROWS)
        .map(|i| (VectorId::new(i), vector_for(i), meta_for(i)))
        .collect();
    engine
        .execute_str(
            "INSERT INTO vectors (id, embedding, metadata) VALUES $1",
            ParamBindings::empty().with_positional(ParamValue::Batch(batch)),
        )
        .unwrap();

    register_indexes(&mut engine);
    (engine, dir)
}

/// Issue `sql` repeatedly. The result must be consumed by criterion's
/// `black_box` so the compiler doesn't fold the call away.
fn run_query(engine: &mut Engine<L2>, sql: &str) {
    let _ = black_box(
        engine
            .execute_str(black_box(sql), ParamBindings::empty())
            .expect("query"),
    );
}

fn install_hash(engine: &mut Engine<L2>) {
    engine
        .execute_str(
            "CREATE INDEX idx_cat ON vectors USING HASH (category)",
            ParamBindings::empty(),
        )
        .unwrap();
}

fn install_btree(engine: &mut Engine<L2>) {
    engine
        .execute_str(
            "CREATE INDEX idx_year ON vectors USING BTREE (year)",
            ParamBindings::empty(),
        )
        .unwrap();
}

fn install_inverted(engine: &mut Engine<L2>) {
    engine
        .execute_str(
            "CREATE INDEX idx_tags ON vectors USING INVERTED (tags)",
            ParamBindings::empty(),
        )
        .unwrap();
}

fn install_all(engine: &mut Engine<L2>) {
    install_hash(engine);
    install_btree(engine);
    install_inverted(engine);
}

// -----------------------------------------------------------------
// 1. COUNT with single Eq atom (HashIndex)
// -----------------------------------------------------------------
fn bench_count_eq(c: &mut Criterion) {
    let sql = "SELECT COUNT(*) FROM vectors WHERE category = 'docs'";

    let (mut e_scan, _d1) = build_engine(|_| {});
    let (mut e_idx, _d2) = build_engine(install_hash);

    let mut group = c.benchmark_group("count_eq_hash");
    group.bench_function("scan", |b| b.iter(|| run_query(&mut e_scan, sql)));
    group.bench_function("indexed", |b| b.iter(|| run_query(&mut e_idx, sql)));
    group.finish();
}

// -----------------------------------------------------------------
// 2. COUNT with range atom (BTreeIndex)
// -----------------------------------------------------------------
fn bench_count_range(c: &mut Criterion) {
    let sql = "SELECT COUNT(*) FROM vectors WHERE year >= 2022";

    let (mut e_scan, _d1) = build_engine(|_| {});
    let (mut e_idx, _d2) = build_engine(install_btree);

    let mut group = c.benchmark_group("count_range_btree");
    group.bench_function("scan", |b| b.iter(|| run_query(&mut e_scan, sql)));
    group.bench_function("indexed", |b| b.iter(|| run_query(&mut e_idx, sql)));
    group.finish();
}

// -----------------------------------------------------------------
// 3. COUNT with array containment (InvertedIndex)
// -----------------------------------------------------------------
fn bench_count_inverted(c: &mut Criterion) {
    let sql = "SELECT COUNT(*) FROM vectors WHERE tags @> 'rust'";

    let (mut e_scan, _d1) = build_engine(|_| {});
    let (mut e_idx, _d2) = build_engine(install_inverted);

    let mut group = c.benchmark_group("count_array_inverted");
    group.bench_function("scan", |b| b.iter(|| run_query(&mut e_scan, sql)));
    group.bench_function("indexed", |b| b.iter(|| run_query(&mut e_idx, sql)));
    group.finish();
}

// -----------------------------------------------------------------
// 4. COUNT with full-index AND chain (Hash + BTree both used)
// -----------------------------------------------------------------
fn bench_count_and_full(c: &mut Criterion) {
    let sql = "SELECT COUNT(*) FROM vectors \
               WHERE category = 'docs' AND year >= 2020 AND tags @> 'rust'";

    let (mut e_scan, _d1) = build_engine(|_| {});
    let (mut e_idx, _d2) = build_engine(install_all);

    let mut group = c.benchmark_group("count_and_full");
    group.bench_function("scan", |b| b.iter(|| run_query(&mut e_scan, sql)));
    group.bench_function("indexed", |b| b.iter(|| run_query(&mut e_idx, sql)));
    group.finish();
}

// -----------------------------------------------------------------
// 5. COUNT with hybrid (one indexed, one not)
// -----------------------------------------------------------------
fn bench_count_and_hybrid(c: &mut Criterion) {
    // `category` is indexed ; `priority` is NOT (never registered)
    // -> Hybrid : index narrows to category=docs candidates, then
    //    the per-row residue evaluates priority > 5.
    let sql = "SELECT COUNT(*) FROM vectors WHERE category = 'docs' AND priority > 5";

    let (mut e_scan, _d1) = build_engine(|_| {});
    let (mut e_idx, _d2) = build_engine(install_hash);

    let mut group = c.benchmark_group("count_and_hybrid");
    group.bench_function("scan", |b| b.iter(|| run_query(&mut e_scan, sql)));
    group.bench_function("indexed", |b| b.iter(|| run_query(&mut e_idx, sql)));
    group.finish();
}

// -----------------------------------------------------------------
// 6. SELECT id ... LIMIT 100 (materialises hits)
// -----------------------------------------------------------------
fn bench_select_eq_limit(c: &mut Criterion) {
    let sql = "SELECT id FROM vectors WHERE category = 'docs' LIMIT 100";

    let (mut e_scan, _d1) = build_engine(|_| {});
    let (mut e_idx, _d2) = build_engine(install_hash);

    let mut group = c.benchmark_group("select_eq_limit");
    group.bench_function("scan", |b| b.iter(|| run_query(&mut e_scan, sql)));
    group.bench_function("indexed", |b| b.iter(|| run_query(&mut e_idx, sql)));
    group.finish();
}

criterion_group!(
    benches,
    bench_count_eq,
    bench_count_range,
    bench_count_inverted,
    bench_count_and_full,
    bench_count_and_hybrid,
    bench_select_eq_limit,
);
criterion_main!(benches);

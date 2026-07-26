// Calibration runner : cast-heavy, by-value-PhysicalPlan. These lints
// are appropriate for the lib but noise here.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::needless_pass_by_value,
    clippy::doc_markdown,
    clippy::similar_names
)]

//! Per-machine calibration of [`CostCoefficients`].
//!
//! Microbenches each coefficient the cost model assumes, then prints
//! a struct literal you can paste into `CostCoefficients::default()`
//! or wire into a `Workload` dispatch on the target machine.
//!
//! - `c_distance_per_dim` : direct loop of `L2::distance` at dim=128.
//! - `c_metadata_get`     : direct loop of `MetadataStore::get` on a
//!   50k-row `FileMetadataStore`.
//! - `c_filter_eval`      : derived from plan B at very low
//!   selectivity (scan term dominates).
//! - `c_hnsw_per_visit`   : derived from plan A given the other
//!   three. Some per-execution overhead bleeds into this term ; it's
//!   a first-cut calibration, not microsecond-precise.
//!
//! Run with:
//!   cargo run --release --example calibrate_cost_coefficients --features internal-bench

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

use kova_core::{Distance, L2, Metadata, MetadataStore, Value, Vector, VectorId};
use kova_index::HnswParams;
use kova_query::ast::{DistanceOp, ParamRef};
use kova_query::cost::{CostCoefficients, KNN_OVERFETCH};
use kova_query::executor::{Engine, ParamBindings, ParamValue};
use kova_query::logical::{
    BoundExpr, BoundProjection, FieldRef, PredAtom, PredicateExpr, ProjectionSpec,
};
use kova_query::physical::PhysicalPlan;
use kova_query::planner::internal_bench::{build_plan_a, build_plan_b};
use kova_storage::{FileMetadataStore, Shard};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use tempfile::tempdir;

const SEED: u64 = 0xCA11_BAA1_C057_BEEF;

// Distance microbench.
const DIM_FOR_DIST: usize = 128;
const DIST_SAMPLES: usize = 1_000_000;

// Metadata-get microbench.
const N_FOR_META: usize = 50_000;
const META_SAMPLES: usize = 50_000;

// Plan-derived microbench.
const N_FOR_PLAN: usize = 10_000;
const PLAN_DIM: usize = 16;
const PLAN_K: usize = 10;
const PLAN_BUCKETS: usize = 1_000; // selectivity = 1/1000 = 0.001
const PLAN_WARMUP: usize = 5;
const PLAN_SAMPLES: usize = 50;

fn calibrate_distance_per_dim() -> f64 {
    let mut rng = StdRng::seed_from_u64(SEED);
    let a: Vec<f32> = (0..DIM_FOR_DIST)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect();
    let b: Vec<f32> = (0..DIM_FOR_DIST)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect();
    let av = Vector::try_new(a).unwrap();
    let bv = Vector::try_new(b).unwrap();

    let mut sink = 0.0f32;
    let t0 = Instant::now();
    for _ in 0..DIST_SAMPLES {
        // black_box on the inputs prevents the compiler from hoisting
        // the call or folding the result, which would otherwise reduce
        // the loop to nothing in release mode.
        sink += L2.distance(black_box(&av), black_box(&bv));
    }
    let elapsed = t0.elapsed().as_nanos() as f64;
    black_box(sink);
    elapsed / (DIST_SAMPLES as f64 * DIM_FOR_DIST as f64)
}

/// Build the fixture both metadata microbenches share, plus the
/// randomised id list they probe it with.
fn metadata_fixture() -> (tempfile::TempDir, FileMetadataStore, Vec<VectorId>) {
    let dir = tempdir().unwrap();
    let mut store = FileMetadataStore::open(dir.path().join("metadata.bin")).unwrap();

    for i in 0..N_FOR_META {
        let mut m = Metadata::new();
        m.insert("k".into(), Value::String(format!("v{i}")));
        m.insert("n".into(), Value::I64(i as i64));
        store.put(VectorId::new(i as u64), m).unwrap();
    }

    let mut rng = StdRng::seed_from_u64(SEED ^ 0x77);
    let ids: Vec<VectorId> = (0..META_SAMPLES)
        .map(|_| VectorId::new(rng.random::<u64>() % N_FOR_META as u64))
        .collect();

    (dir, store, ids)
}

/// Measure both metadata accessors against **one** fixture.
///
/// Returns `(c_metadata_get, c_metadata_peek)` :
///
/// - `get` is the **cloning** accessor. Plans A and B pay it, because
///   both materialise owned bags into their results
///   (`SearchHit.metadata`, `InternalHit.metadata`).
/// - `with_metadata` is the **borrowing** accessor. Plan C pays it per
///   visited graph node, because its filter closure only reads the bag.
///
/// They must be measured on the same store : the whole point is the
/// *ratio* between them, and two fixtures would let allocator state or
/// page-cache warmth leak into the comparison.
///
/// Sharing the fixture is also what keeps this runner usable.
/// `FileMetadataStore::put` rewrites the entire file per call, so
/// building an `N_FOR_META`-row store is quadratic and dominates the
/// whole calibration — building it twice roughly doubled the runtime
/// for no extra information.
fn calibrate_metadata_accessors() -> (f64, f64) {
    let (_dir, store, ids) = metadata_fixture();

    // --- get : lookup + deep clone ---
    let mut sink = 0usize;
    let t0 = Instant::now();
    for id in &ids {
        if let Some(m) = store.get(*id) {
            sink += m.len();
        }
    }
    let get_ns = t0.elapsed().as_nanos() as f64 / META_SAMPLES as f64;
    black_box(sink);

    // --- with_metadata : lookup only ---
    let mut sink = 0usize;
    let t0 = Instant::now();
    for id in &ids {
        if let Some(n) = store.with_metadata(*id, HashMap::len) {
            sink += n;
        }
    }
    let peek_ns = t0.elapsed().as_nanos() as f64 / META_SAMPLES as f64;
    black_box(sink);

    (get_ns, peek_ns)
}

// Visit count comes from the model itself (`cost::internal_bench`),
// never a local copy : `c_hnsw_per_visit` is derived by dividing a
// measured latency by this count, so a formula that drifts from the
// one `cost_plan_a` uses yields a coefficient that is wrong by
// exactly the ratio between them.
use kova_query::cost::internal_bench::hnsw_visits;

fn run_median_ns(engine: &mut Engine<L2>, plan: PhysicalPlan, params: &ParamBindings) -> f64 {
    for _ in 0..PLAN_WARMUP {
        let _ = engine.execute_plan(plan.clone(), params);
    }
    let mut samples: Vec<u128> = Vec::with_capacity(PLAN_SAMPLES);
    for _ in 0..PLAN_SAMPLES {
        let t0 = Instant::now();
        let _ = engine.execute_plan(plan.clone(), params);
        samples.push(t0.elapsed().as_nanos());
    }
    samples.sort_unstable();
    samples[samples.len() / 2] as f64
}

// Solve for (c_filter, c_hnsw) from a real shard's plan A and plan B
// medians, given c_meta and c_dist.
fn calibrate_plans(c_meta: f64, c_dist: f64) -> (f64, f64) {
    let dir = tempdir().unwrap();
    let mut shard = Shard::open(dir.path(), PLAN_DIM, L2, HnswParams::default()).unwrap();
    let mut rng = StdRng::seed_from_u64(SEED ^ 0xAA);
    for i in 0..N_FOR_PLAN {
        let v: Vec<f32> = (0..PLAN_DIM)
            .map(|_| rng.random::<f32>() * 2.0 - 1.0)
            .collect();
        let mut m = Metadata::new();
        m.insert("bucket".into(), Value::I64((i % PLAN_BUCKETS) as i64));
        shard
            .insert(VectorId::new(i as u64), Vector::try_new(v).unwrap(), m)
            .unwrap();
    }
    shard.checkpoint().unwrap();

    let qv: Vec<f32> = (0..PLAN_DIM)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect();
    let params = ParamBindings::positional(vec![
        ParamValue::Vector(Vector::try_new(qv).unwrap()),
        ParamValue::I64(0),
    ]);

    let mut engine = Engine::new(shard, "vectors");
    let pred = PredicateExpr::Atom(PredAtom::Eq {
        field: FieldRef::plain("bucket"),
        value: BoundExpr::Param(ParamRef::Positional(2)),
    });
    let projection = ProjectionSpec {
        columns: vec![
            BoundProjection::Id { alias: None },
            BoundProjection::Metadata { alias: None },
        ],
    };
    let wrap = |inner: PhysicalPlan| PhysicalPlan::Projection {
        input: Box::new(inner),
        spec: projection.clone(),
    };

    let plan_a = wrap(build_plan_a(
        "vectors".into(),
        ParamRef::Positional(1),
        DistanceOp::L2,
        PLAN_K,
        Some(pred.clone()),
        PLAN_K as u64,
    ));
    let plan_b = wrap(build_plan_b(
        "vectors".into(),
        pred,
        ParamRef::Positional(1),
        DistanceOp::L2,
        PLAN_K,
        PLAN_K as u64,
    ));

    let t_b_ns = run_median_ns(&mut engine, plan_b, &params);
    let t_a_ns = run_median_ns(&mut engine, plan_a, &params);

    // Cost B = n * c_filter + matches * (c_meta + dim * c_dist)
    // matches ≈ max(1, selectivity * n) = max(1, 10) = 10
    let matches = (N_FOR_PLAN as f64 / PLAN_BUCKETS as f64).max(1.0);
    let per_match = c_meta + PLAN_DIM as f64 * c_dist;
    let c_filter = (t_b_ns - matches * per_match) / N_FOR_PLAN as f64;

    // Cost A = visits * (c_hnsw + dim * c_dist) + overfetch * (c_meta + c_filter)
    let overfetch = (PLAN_K * KNN_OVERFETCH) as f64;
    let visits = hnsw_visits(PLAN_K * KNN_OVERFETCH, N_FOR_PLAN);
    let post_filter = overfetch * (c_meta + c_filter);
    let per_visit_total = (t_a_ns - post_filter) / visits;
    let c_hnsw = per_visit_total - PLAN_DIM as f64 * c_dist;

    println!("       (plan B median = {t_b_ns:.0} ns, plan A median = {t_a_ns:.0} ns)");
    (c_filter.max(0.0), c_hnsw.max(0.0))
}

fn main() {
    println!("Calibrating CostCoefficients on this machine.");
    println!();

    println!(
        "[1/3] c_distance_per_dim : L2 distance loop, dim={DIM_FOR_DIST}, {DIST_SAMPLES} samples"
    );
    let c_dist = calibrate_distance_per_dim();
    println!("       -> {c_dist:.3} ns/dim");
    println!();

    println!(
        "[2/3] c_metadata_get + c_metadata_peek : {N_FOR_META} rows, {META_SAMPLES} samples each"
    );
    let (c_meta, c_peek) = calibrate_metadata_accessors();
    println!("       -> get  (clone)  = {c_meta:.0} ns");
    println!("       -> peek (borrow) = {c_peek:.0} ns");
    println!(
        "       -> borrowing is {:.1}x cheaper  <- what 0A.2 bought plan C",
        c_meta / c_peek.max(0.001)
    );
    println!();

    println!(
        "[3/3] c_filter_eval, c_hnsw_per_visit : derived from plan B + plan A at n={N_FOR_PLAN}, dim={PLAN_DIM}"
    );
    let (c_filter, c_hnsw) = calibrate_plans(c_meta, c_dist);
    println!("       -> c_filter_eval     = {c_filter:.0} ns");
    println!("       -> c_hnsw_per_visit  = {c_hnsw:.0} ns");
    println!();

    println!("Calibrated CostCoefficients for this machine :");
    println!();
    println!("    Self {{");
    println!("        c_hnsw_per_visit:   {c_hnsw:.1},");
    println!("        c_distance_per_dim: {c_dist:.2},");
    println!("        c_metadata_get:     {c_meta:.1},");
    println!("        c_metadata_peek:    {c_peek:.1},");
    println!("        c_filter_eval:      {c_filter:.1},");
    println!("    }}");
    println!();

    let d = CostCoefficients::default();
    println!("Default vs measured (ratio = measured / default) :");
    println!(
        "  c_hnsw_per_visit   default={:>7.1}  measured={c_hnsw:>7.1}  ratio={:.2}x",
        d.c_hnsw_per_visit,
        c_hnsw / d.c_hnsw_per_visit
    );
    println!(
        "  c_distance_per_dim default={:>7.2}  measured={c_dist:>7.2}  ratio={:.2}x",
        d.c_distance_per_dim,
        c_dist / d.c_distance_per_dim
    );
    println!(
        "  c_metadata_get     default={:>7.1}  measured={c_meta:>7.1}  ratio={:.2}x",
        d.c_metadata_get,
        c_meta / d.c_metadata_get
    );
    println!(
        "  c_metadata_peek    default={:>7.1}  measured={c_peek:>7.1}  ratio={:.2}x",
        d.c_metadata_peek,
        c_peek / d.c_metadata_peek
    );
    println!(
        "  c_filter_eval      default={:>7.1}  measured={c_filter:>7.1}  ratio={:.2}x",
        d.c_filter_eval,
        c_filter / d.c_filter_eval
    );
    println!();
    println!("Paste the struct above into cost.rs::Default impl, then re-run");
    println!("validate_cost_model to confirm regret drops.");
}

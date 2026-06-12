// Validation harness : cast-heavy, loop-index-heavy, by-value-PhysicalPlan.
// These lints are appropriate for the lib but noise here.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::needless_pass_by_value,
    clippy::needless_range_loop,
    clippy::doc_markdown,
    clippy::similar_names
)]

//! Cost model validation harness.
//!
//! Sweeps (dim, n, selectivity) and force-runs plans A / B / C on each
//! cell. Compares the cost model's predicted winner to the measured
//! winner. Prints a confusion matrix, mean regret, and a list of
//! disagreement cells.
//!
//! Run with:
//!   cargo run --release --example validate_cost_model --features internal-bench

use std::time::Instant;

use kova_core::{L2, Metadata, Value, Vector, VectorId};
use kova_index::HnswParams;
use kova_query::ast::{DistanceOp, ParamRef};
use kova_query::cost::{
    CostCoefficients, PlanKind, Workload, cost_plan_a, cost_plan_b, cost_plan_c, dispatch_via_cost,
};
use kova_query::executor::{Engine, ParamBindings, ParamValue};
use kova_query::logical::{
    BoundExpr, BoundProjection, FieldRef, PredAtom, PredicateExpr, ProjectionSpec,
};
use kova_query::physical::PhysicalPlan;
use kova_query::planner::internal_bench::{build_plan_a, build_plan_b, build_plan_c};
use kova_storage::{FileMetadataStore, FileWal, MmapVectorStore, Shard};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use tempfile::tempdir;

const DIMS: &[usize] = &[16, 128, 1536];
const NS: &[usize] = &[1_000, 10_000];
const SELS: &[f64] = &[0.001, 0.01, 0.05, 0.2, 0.5];
const K: usize = 10;
const WARMUP: usize = 3;
const SAMPLES: usize = 25;
const SEED: u64 = 0xC057_C057_C057_F00D;

type TestShard = Shard<L2, MmapVectorStore, FileMetadataStore, FileWal>;

struct CellResult {
    dim: usize,
    n: usize,
    s: f64,
    measured_micros: [f64; 3],
    #[allow(dead_code)] // kept for debugging when tuning coefficients
    predicted_cost: [f64; 3],
    measured_winner: PlanKind,
    predicted_winner: PlanKind,
}

fn id_metadata_projection() -> ProjectionSpec {
    ProjectionSpec {
        columns: vec![
            BoundProjection::Id { alias: None },
            BoundProjection::Metadata { alias: None },
        ],
    }
}

fn wrap(plan: PhysicalPlan) -> PhysicalPlan {
    PhysicalPlan::Projection {
        input: Box::new(plan),
        spec: id_metadata_projection(),
    }
}

fn predicate_eq_bucket_param() -> PredicateExpr {
    PredicateExpr::Atom(PredAtom::Eq {
        field: FieldRef::plain("bucket"),
        value: BoundExpr::Param(ParamRef::Positional(2)),
    })
}

fn build_shard(dir: &std::path::Path, dim: usize, n: usize, buckets: usize) -> TestShard {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut shard = Shard::open(dir, dim, L2, HnswParams::default()).unwrap();
    for i in 0..n {
        let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let bucket = (i % buckets) as i64;
        let mut m = Metadata::new();
        m.insert("bucket".into(), Value::I64(bucket));
        shard
            .insert(VectorId::new(i as u64), Vector::try_new(v).unwrap(), m)
            .unwrap();
    }
    shard.checkpoint().unwrap();
    shard
}

fn time_plan(
    label: &str,
    engine: &mut Engine<L2>,
    plan: PhysicalPlan,
    params: &ParamBindings,
) -> f64 {
    for _ in 0..WARMUP {
        if let Err(e) = engine.execute_plan(plan.clone(), params) {
            eprintln!("[{label}] execute error: {e:?}");
            return f64::INFINITY;
        }
    }
    let mut samples: Vec<u128> = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let t0 = Instant::now();
        if let Err(e) = engine.execute_plan(plan.clone(), params) {
            eprintln!("[{label}] execute error mid-sample: {e:?}");
            return f64::INFINITY;
        }
        samples.push(t0.elapsed().as_nanos());
    }
    samples.sort_unstable();
    // Return microseconds with sub-microsecond resolution.
    (samples[samples.len() / 2] as f64) / 1000.0
}

fn argmin3(xs: [f64; 3]) -> PlanKind {
    let mut best = 0;
    for i in 1..3 {
        if xs[i] < xs[best] {
            best = i;
        }
    }
    [PlanKind::A, PlanKind::B, PlanKind::C][best]
}

fn main() {
    let coeffs = CostCoefficients::default();
    let mut results = Vec::new();

    for &dim in DIMS {
        for &n in NS {
            for &s in SELS {
                let buckets = (1.0 / s).round() as usize;
                let dir = tempdir().unwrap();
                let shard = build_shard(dir.path(), dim, n, buckets);

                let mut rng = StdRng::seed_from_u64(SEED ^ (dim as u64));
                let qv: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
                let params = ParamBindings::positional(vec![
                    ParamValue::Vector(Vector::try_new(qv).unwrap()),
                    ParamValue::I64(0),
                ]);

                let mut engine = Engine::new(shard, "vectors");
                let pred = predicate_eq_bucket_param();

                let plan_a = wrap(build_plan_a(
                    "vectors".into(),
                    ParamRef::Positional(1),
                    DistanceOp::L2,
                    K,
                    Some(pred.clone()),
                    K as u64,
                ));
                let plan_b = wrap(build_plan_b(
                    "vectors".into(),
                    pred.clone(),
                    ParamRef::Positional(1),
                    DistanceOp::L2,
                    K,
                    K as u64,
                ));
                let plan_c = wrap(build_plan_c(
                    "vectors".into(),
                    pred,
                    ParamRef::Positional(1),
                    DistanceOp::L2,
                    K,
                    K as u64,
                ));

                let cell = format!("dim={dim} n={n} s={s}");
                let m_a = time_plan(&format!("{cell} A"), &mut engine, plan_a, &params);
                let m_b = time_plan(&format!("{cell} B"), &mut engine, plan_b, &params);
                let m_c = time_plan(&format!("{cell} C"), &mut engine, plan_c, &params);
                let measured = [m_a, m_b, m_c];

                let w = Workload {
                    selectivity: s,
                    user_k: K,
                    total_rows: n,
                    dim,
                };
                let predicted = [
                    cost_plan_a(&w, &coeffs),
                    cost_plan_b(&w, &coeffs),
                    cost_plan_c(&w, &coeffs),
                ];

                results.push(CellResult {
                    dim,
                    n,
                    s,
                    measured_micros: measured,
                    predicted_cost: predicted,
                    measured_winner: argmin3(measured),
                    predicted_winner: dispatch_via_cost(&w, &coeffs),
                });
                println!(
                    "  dim={dim:4} n={n:6} s={s:.3}  A={m_a:7.0}us  B={m_b:7.0}us  C={m_c:7.0}us"
                );
            }
        }
    }
    print_report(&results);
}

fn print_report(rs: &[CellResult]) {
    let mut conf = [[0usize; 3]; 3];
    for r in rs {
        conf[r.predicted_winner as usize][r.measured_winner as usize] += 1;
    }
    println!("\nConfusion matrix (rows=predicted, cols=actual):");
    println!("                A      B      C");
    let names = ["A", "B", "C"];
    for i in 0..3 {
        print!("predicted {}    ", names[i]);
        for j in 0..3 {
            print!(" {:5}", conf[i][j]);
        }
        println!();
    }

    let mut total_regret = 0.0;
    for r in rs {
        let pred = r.measured_micros[r.predicted_winner as usize];
        let best = r.measured_micros[r.measured_winner as usize];
        total_regret += pred / best.max(1.0);
    }
    let avg_regret = total_regret / rs.len() as f64;
    println!("\nMean regret = {avg_regret:.3}  (1.000 = perfect)");

    println!("\nDisagreement cells:");
    println!("  dim     n      s   predicted  actual   A(us)    B(us)    C(us)    regret");
    for r in rs {
        if r.predicted_winner != r.measured_winner {
            let pred = r.measured_micros[r.predicted_winner as usize];
            let best = r.measured_micros[r.measured_winner as usize];
            println!(
                "  {:4} {:6}  {:.3}    {:?}     {:?}    {:7.0}  {:7.0}  {:7.0}    {:.2}x",
                r.dim,
                r.n,
                r.s,
                r.predicted_winner,
                r.measured_winner,
                r.measured_micros[0],
                r.measured_micros[1],
                r.measured_micros[2],
                pred / best,
            );
        }
    }
}

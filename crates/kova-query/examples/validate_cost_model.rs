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
use kova_query::executor::{Engine, ExecutionResult, ParamBindings, ParamValue};
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
/// Result-count sweep. Plan C's cost advantage over plan A comes from
/// `ef_C = max(2k, 50)` vs `ef_A = max(8k, 50)` (A overfetches to
/// `k * KNN_OVERFETCH`), so the gap only opens up as `k` grows. A grid
/// fixed at k=10 never lets C win ; that's why this axis exists.
const KS: &[usize] = &[10, 50, 100, 500];
const WARMUP: usize = 3;
const SAMPLES: usize = 25;
const SEED: u64 = 0xC057_C057_C057_F00D;

type TestShard = Shard<L2, MmapVectorStore, FileMetadataStore, FileWal>;

/// Read a comma-separated axis override out of the environment, falling
/// back to the compiled-in default.
///
/// The full grid is `DIMS x NS x SELS x KS` and takes tens of minutes,
/// almost all of it in HNSW construction. Overriding the axes lets a
/// targeted question ("what happens above s=0.5?") be answered with a
/// handful of cells instead of a whole re-sweep :
///
/// ```sh
/// KOVA_SELS=0.7,0.9 KOVA_NS=10000 KOVA_KS=50,100,500 \
///   cargo run --release --example validate_cost_model --features internal-bench
/// ```
fn axis<T: std::str::FromStr>(var: &str, default: &[T]) -> Vec<T>
where
    T: Copy,
{
    let parsed: Option<Vec<T>> = std::env::var(var).ok().map(|s| {
        s.split(',')
            .filter_map(|x| x.trim().parse::<T>().ok())
            .collect()
    });
    match parsed {
        Some(v) if !v.is_empty() => v,
        _ => default.to_vec(),
    }
}

struct CellResult {
    dim: usize,
    n: usize,
    s: f64,
    k: usize,
    measured_micros: [f64; 3],
    /// Rows each plan actually returned. Compare against `k`.
    returned: [usize; 3],
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

/// Median latency in microseconds, plus **how many rows the plan
/// actually returned**.
///
/// The row count is not decoration. Plan A overfetches `k * OVERFETCH`
/// candidates, post-filters, and returns whatever survives — with no
/// retry. At low selectivity that means a `LIMIT k` query can come back
/// with far fewer than `k` rows, and the cost model deliberately does
/// not charge for it (`cost_plan_a` has no selectivity term, on the
/// grounds that starvation is a recall concern rather than a latency
/// one). Nothing then went on to measure the recall. This does.
fn time_plan(
    label: &str,
    engine: &mut Engine<L2>,
    plan: PhysicalPlan,
    params: &ParamBindings,
) -> (f64, usize) {
    let mut returned = 0usize;
    for _ in 0..WARMUP {
        match engine.execute_plan(plan.clone(), params) {
            Ok(ExecutionResult::Rows { rows, .. }) => returned = rows.len(),
            Ok(other) => {
                eprintln!("[{label}] expected Rows, got {other:?}");
                return (f64::INFINITY, 0);
            }
            Err(e) => {
                eprintln!("[{label}] execute error: {e:?}");
                return (f64::INFINITY, 0);
            }
        }
    }
    let mut samples: Vec<u128> = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let t0 = Instant::now();
        if let Err(e) = engine.execute_plan(plan.clone(), params) {
            eprintln!("[{label}] execute error mid-sample: {e:?}");
            return (f64::INFINITY, returned);
        }
        samples.push(t0.elapsed().as_nanos());
    }
    samples.sort_unstable();
    // Microseconds with sub-microsecond resolution.
    ((samples[samples.len() / 2] as f64) / 1000.0, returned)
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

    let dims = axis::<usize>("KOVA_DIMS", DIMS);
    let ns = axis::<usize>("KOVA_NS", NS);
    let sels = axis::<f64>("KOVA_SELS", SELS);
    let ks = axis::<usize>("KOVA_KS", KS);
    println!("grid: dims={dims:?} ns={ns:?} sels={sels:?} ks={ks:?}");

    for &dim in &dims {
        for &n in &ns {
            for &s in &sels {
                // The shard depends on (dim, n, s) but NOT on k, so it is
                // built once per cell and reused across the whole k sweep.
                // Building per-k would quadruple the (dominant) HNSW
                // construction cost for identical data.
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

                for &k in &ks {
                    let pred = predicate_eq_bucket_param();

                    let plan_a = wrap(build_plan_a(
                        "vectors".into(),
                        ParamRef::Positional(1),
                        DistanceOp::L2,
                        k,
                        Some(pred.clone()),
                        k as u64,
                    ));
                    let plan_b = wrap(build_plan_b(
                        "vectors".into(),
                        pred.clone(),
                        ParamRef::Positional(1),
                        DistanceOp::L2,
                        k,
                        k as u64,
                    ));
                    let plan_c = wrap(build_plan_c(
                        "vectors".into(),
                        pred,
                        ParamRef::Positional(1),
                        DistanceOp::L2,
                        k,
                        k as u64,
                    ));

                    let cell = format!("dim={dim} n={n} s={s} k={k}");
                    let (m_a, r_a) = time_plan(&format!("{cell} A"), &mut engine, plan_a, &params);
                    let (m_b, r_b) = time_plan(&format!("{cell} B"), &mut engine, plan_b, &params);
                    let (m_c, r_c) = time_plan(&format!("{cell} C"), &mut engine, plan_c, &params);
                    let measured = [m_a, m_b, m_c];
                    let returned = [r_a, r_b, r_c];

                    let w = Workload {
                        selectivity: s,
                        user_k: k,
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
                        k,
                        measured_micros: measured,
                        returned,
                        predicted_cost: predicted,
                        measured_winner: argmin3(measured),
                        predicted_winner: dispatch_via_cost(&w, &coeffs),
                    });
                    println!(
                        "  dim={dim:4} n={n:6} s={s:.3} k={k:4}  \
                         A={m_a:8.0}us/{r_a:<4} B={m_b:8.0}us/{r_b:<4} C={m_c:8.0}us/{r_c:<4} \
                         (us/rows, want {k})"
                    );
                }
            }
        }
    }
    print_report(&results);
}

/// Confusion matrix + mean regret over an arbitrary subset of cells.
/// Factored out so the per-k breakdown can reuse it.
fn confusion_and_regret(rs: &[&CellResult]) -> ([[usize; 3]; 3], f64) {
    let mut conf = [[0usize; 3]; 3];
    let mut total_regret = 0.0;
    for r in rs {
        conf[r.predicted_winner as usize][r.measured_winner as usize] += 1;
        let pred = r.measured_micros[r.predicted_winner as usize];
        let best = r.measured_micros[r.measured_winner as usize];
        total_regret += pred / best.max(1.0);
    }
    let avg = if rs.is_empty() {
        0.0
    } else {
        total_regret / rs.len() as f64
    };
    (conf, avg)
}

fn print_confusion(conf: [[usize; 3]; 3]) {
    println!("                A      B      C");
    let names = ["A", "B", "C"];
    for i in 0..3 {
        print!("predicted {}    ", names[i]);
        for j in 0..3 {
            print!(" {:5}", conf[i][j]);
        }
        println!();
    }
}

fn print_report(rs: &[CellResult]) {
    let all: Vec<&CellResult> = rs.iter().collect();

    // ---- Overall ----
    let (conf, avg_regret) = confusion_and_regret(&all);
    println!("\n=== OVERALL ({} cells) ===", rs.len());
    println!("\nConfusion matrix (rows=predicted, cols=actual):");
    print_confusion(conf);
    println!("\nMean regret = {avg_regret:.3}  (1.000 = perfect)");

    // ---- Per-k breakdown : the whole point of the k sweep ----
    //
    // Plan C's `ef` advantage over plan A scales with k, so a matrix
    // collapsed across k hides whether C has a band at all.
    println!("\n=== PER-k BREAKDOWN ===");
    for &k in KS {
        let subset: Vec<&CellResult> = rs.iter().filter(|r| r.k == k).collect();
        let (conf, regret) = confusion_and_regret(&subset);
        let c_measured_wins = subset
            .iter()
            .filter(|r| r.measured_winner == PlanKind::C)
            .count();
        let c_dispatched = subset
            .iter()
            .filter(|r| r.predicted_winner == PlanKind::C)
            .count();
        println!(
            "\n--- k={k} ({} cells) : regret {regret:.3} | C measured-best in {c_measured_wins}, \
             C dispatched in {c_dispatched} ---",
            subset.len()
        );
        print_confusion(conf);
    }

    // ---- Result completeness : does each plan actually return k rows? ----
    //
    // The cost model scores latency only. Plan A can return fewer rows
    // than the LIMIT asked for (overfetch + post-filter, no retry) while
    // plan C cannot (its termination gate requires a full results heap).
    // A dispatch that is optimal on latency can therefore be handing
    // back a short answer, and nothing in the confusion matrix or the
    // regret figure would show it.
    println!("\n=== RESULT COMPLETENESS : rows returned vs k ===");
    println!("   dim      n      s     k     A      B      C   dispatched  short?");
    let mut a_short = 0;
    let mut dispatched_short = 0;
    for r in rs {
        let want = r.k.min(r.n);
        let short_a = r.returned[0] < want;
        let short_dispatched = r.returned[r.predicted_winner as usize] < want;
        if short_a {
            a_short += 1;
        }
        if short_dispatched {
            dispatched_short += 1;
        }
        // Only print rows where something came up short ; the complete
        // ones are the uninteresting majority.
        if short_a || short_dispatched {
            println!(
                "  {:4} {:6}  {:.3} {:5} {:5} {:6} {:6}      {:?}      {}",
                r.dim,
                r.n,
                r.s,
                r.k,
                r.returned[0],
                r.returned[1],
                r.returned[2],
                r.predicted_winner,
                if short_dispatched {
                    "DISPATCHED PLAN SHORT"
                } else {
                    "(plan A only)"
                },
            );
        }
    }
    println!(
        "\n  plan A returned < k in {a_short}/{} cells ; \
         the DISPATCHED plan did in {dispatched_short}/{}",
        rs.len(),
        rs.len()
    );

    // ---- Where plan C actually wins, measured ----
    println!("\n=== CELLS WHERE PLAN C IS MEASURED FASTEST ===");
    println!("   dim      n      s     k    A(us)     B(us)     C(us)   C vs best-other");
    let mut c_wins = 0;
    for r in rs {
        if r.measured_winner == PlanKind::C {
            c_wins += 1;
            let best_other = r.measured_micros[0].min(r.measured_micros[1]);
            println!(
                "  {:4} {:6}  {:.3} {:5}  {:8.0}  {:8.0}  {:8.0}   {:.2}x faster",
                r.dim,
                r.n,
                r.s,
                r.k,
                r.measured_micros[0],
                r.measured_micros[1],
                r.measured_micros[2],
                best_other / r.measured_micros[2].max(1.0),
            );
        }
    }
    if c_wins == 0 {
        println!("  (none — plan C was never the fastest plan on any cell)");
    }

    // ---- Disagreements ----
    println!("\n=== DISAGREEMENT CELLS ===");
    println!("   dim      n      s     k  predicted  actual    A(us)     B(us)     C(us)   regret");
    for r in rs {
        if r.predicted_winner != r.measured_winner {
            let pred = r.measured_micros[r.predicted_winner as usize];
            let best = r.measured_micros[r.measured_winner as usize];
            println!(
                "  {:4} {:6}  {:.3} {:5}    {:?}      {:?}    {:8.0}  {:8.0}  {:8.0}   {:.2}x",
                r.dim,
                r.n,
                r.s,
                r.k,
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

// Replay harness : cast-heavy by nature, and the embedded cell table
// is formatted for reading as a grid rather than by rustfmt.
#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    // `main` is a sequence of report sections ; splitting it would
    // scatter the output format across the file.
    clippy::too_many_lines,
    clippy::needless_range_loop
)]

//! Replay a recorded `validate_cost_model` run through the *current*
//! cost model, without re-measuring anything.
//!
//! The measurements below are fixed observations : the harness forces
//! each plan and times it, so they do not depend on what the dispatcher
//! would have chosen. Only the *predictions* change when the model
//! changes. That makes this the right loop for model work : seconds
//! instead of the ~60-90 minutes a full re-sweep costs.
//!
//! Re-measure only when the **executor** changes. Regenerate this file
//! with `scratchpad/gen_probe.py <sweep-output>`.
//!
//! Run with:
//!   cargo run --release --example cost_probe --features internal-bench

use kova_query::cost::{
    CostCoefficients, PlanKind, Workload, cost_plan_a, cost_plan_b, cost_plan_c, dispatch_via_cost,
};

/// One measured cell : dim, n, selectivity, k, then per-plan
/// `(latency_us, rows_returned)` for A / B / C.
struct M(usize, usize, f64, usize, f64, f64, f64, usize, usize, usize);

/// 84 cells, `n=10_000`, corrected per-mille selectivity
/// construction, row counts recorded.
#[rustfmt::skip]
const CELLS: &[M] = &[
    M(16, 10000, 0.001, 10, 81.0, 528.0, 3271.0, 0, 10, 10),
    M(16, 10000, 0.001, 50, 172.0, 337.0, 2970.0, 0, 10, 10),
    M(16, 10000, 0.001, 100, 317.0, 335.0, 2942.0, 1, 10, 10),
    M(16, 10000, 0.001, 500, 946.0, 338.0, 2948.0, 3, 10, 10),
    M(16, 10000, 0.01, 10, 125.0, 647.0, 2191.0, 0, 10, 10),
    M(16, 10000, 0.01, 50, 179.0, 372.0, 1819.0, 0, 50, 50),
    M(16, 10000, 0.01, 100, 318.0, 362.0, 3004.0, 2, 100, 100),
    M(16, 10000, 0.01, 500, 955.0, 362.0, 2969.0, 23, 100, 100),
    M(16, 10000, 0.05, 10, 114.0, 704.0, 966.0, 0, 10, 10),
    M(16, 10000, 0.05, 50, 216.0, 484.0, 782.0, 7, 50, 50),
    M(16, 10000, 0.05, 100, 332.0, 412.0, 1138.0, 15, 100, 100),
    M(16, 10000, 0.05, 500, 962.0, 437.0, 3039.0, 90, 500, 500),
    M(16, 10000, 0.2, 10, 140.0, 938.0, 297.0, 9, 10, 10),
    M(16, 10000, 0.2, 50, 239.0, 722.0, 242.0, 40, 50, 50),
    M(16, 10000, 0.2, 100, 362.0, 614.0, 402.0, 84, 100, 100),
    M(16, 10000, 0.2, 500, 973.0, 612.0, 1256.0, 385, 500, 500),
    M(16, 10000, 0.5, 10, 112.0, 1460.0, 135.0, 10, 10, 10),
    M(16, 10000, 0.5, 50, 224.0, 1155.0, 113.0, 50, 50, 50),
    M(16, 10000, 0.5, 100, 355.0, 1031.0, 207.0, 100, 100, 100),
    M(16, 10000, 0.5, 500, 1070.0, 976.0, 677.0, 500, 500, 500),
    M(16, 10000, 0.7, 10, 116.0, 1738.0, 93.0, 10, 10, 10),
    M(16, 10000, 0.7, 50, 208.0, 1326.0, 85.0, 50, 50, 50),
    M(16, 10000, 0.7, 100, 334.0, 1265.0, 164.0, 100, 100, 100),
    M(16, 10000, 0.7, 500, 1016.0, 1262.0, 543.0, 500, 500, 500),
    M(16, 10000, 0.9, 10, 120.0, 2029.0, 68.0, 10, 10, 10),
    M(16, 10000, 0.9, 50, 203.0, 1546.0, 64.0, 50, 50, 50),
    M(16, 10000, 0.9, 100, 312.0, 1519.0, 120.0, 100, 100, 100),
    M(16, 10000, 0.9, 500, 994.0, 1546.0, 464.0, 500, 500, 500),
    M(128, 10000, 0.001, 10, 293.0, 544.0, 4421.0, 0, 10, 10),
    M(128, 10000, 0.001, 50, 492.0, 344.0, 4069.0, 0, 10, 10),
    M(128, 10000, 0.001, 100, 743.0, 343.0, 4063.0, 0, 10, 10),
    M(128, 10000, 0.001, 500, 1652.0, 346.0, 4058.0, 4, 10, 10),
    M(128, 10000, 0.01, 10, 263.0, 573.0, 3592.0, 1, 10, 10),
    M(128, 10000, 0.01, 50, 687.0, 435.0, 3966.0, 2, 50, 50),
    M(128, 10000, 0.01, 100, 771.0, 369.0, 4540.0, 3, 100, 100),
    M(128, 10000, 0.01, 500, 1722.0, 368.0, 4275.0, 27, 100, 100),
    M(128, 10000, 0.05, 10, 280.0, 616.0, 1609.0, 2, 10, 10),
    M(128, 10000, 0.05, 50, 495.0, 445.0, 1337.0, 10, 50, 50),
    M(128, 10000, 0.05, 100, 733.0, 435.0, 1780.0, 20, 100, 100),
    M(128, 10000, 0.05, 500, 1627.0, 460.0, 4509.0, 97, 500, 500),
    M(128, 10000, 0.2, 10, 265.0, 981.0, 662.0, 9, 10, 10),
    M(128, 10000, 0.2, 50, 544.0, 706.0, 544.0, 42, 50, 50),
    M(128, 10000, 0.2, 100, 730.0, 688.0, 902.0, 78, 100, 100),
    M(128, 10000, 0.2, 500, 1663.0, 709.0, 1983.0, 391, 500, 500),
    M(128, 10000, 0.5, 10, 265.0, 1747.0, 303.0, 10, 10, 10),
    M(128, 10000, 0.5, 50, 517.0, 1356.0, 277.0, 50, 50, 50),
    M(128, 10000, 0.5, 100, 893.0, 1317.0, 539.0, 100, 100, 100),
    M(128, 10000, 0.5, 500, 1730.0, 1296.0, 1434.0, 500, 500, 500),
    M(128, 10000, 0.7, 10, 333.0, 2105.0, 251.0, 10, 10, 10),
    M(128, 10000, 0.7, 50, 505.0, 1750.0, 228.0, 50, 50, 50),
    M(128, 10000, 0.7, 100, 747.0, 1658.0, 365.0, 100, 100, 100),
    M(128, 10000, 0.7, 500, 1736.0, 1790.0, 1192.0, 500, 500, 500),
    M(128, 10000, 0.9, 10, 245.0, 2510.0, 177.0, 10, 10, 10),
    M(128, 10000, 0.9, 50, 561.0, 2637.0, 174.0, 50, 50, 50),
    M(128, 10000, 0.9, 100, 806.0, 2589.0, 305.0, 100, 100, 100),
    M(128, 10000, 0.9, 500, 1670.0, 2190.0, 967.0, 500, 500, 500),
    M(1536, 10000, 0.001, 10, 1126.0, 356.0, 11305.0, 0, 10, 10),
    M(1536, 10000, 0.001, 50, 2844.0, 346.0, 11321.0, 0, 10, 10),
    M(1536, 10000, 0.001, 100, 4284.0, 342.0, 11346.0, 0, 10, 10),
    M(1536, 10000, 0.001, 500, 7864.0, 345.0, 11363.0, 1, 10, 10),
    M(1536, 10000, 0.01, 10, 1113.0, 422.0, 9537.0, 1, 10, 10),
    M(1536, 10000, 0.01, 50, 2984.0, 421.0, 9403.0, 1, 50, 50),
    M(1536, 10000, 0.01, 100, 4210.0, 430.0, 11189.0, 3, 100, 99),
    M(1536, 10000, 0.01, 500, 7712.0, 428.0, 11092.0, 13, 100, 99),
    M(1536, 10000, 0.05, 10, 1086.0, 726.0, 7217.0, 2, 10, 10),
    M(1536, 10000, 0.05, 50, 2729.0, 742.0, 7079.0, 5, 50, 50),
    M(1536, 10000, 0.05, 100, 4141.0, 746.0, 8166.0, 17, 100, 100),
    M(1536, 10000, 0.05, 500, 7684.0, 765.0, 11120.0, 80, 500, 499),
    M(1536, 10000, 0.2, 10, 1178.0, 2153.0, 4365.0, 6, 10, 10),
    M(1536, 10000, 0.2, 50, 3015.0, 2225.0, 3877.0, 35, 50, 50),
    M(1536, 10000, 0.2, 100, 4723.0, 2452.0, 6639.0, 73, 100, 100),
    M(1536, 10000, 0.2, 500, 10733.0, 2850.0, 9775.0, 389, 500, 500),
    M(1536, 10000, 0.5, 10, 1115.0, 4584.0, 2022.0, 10, 10, 10),
    M(1536, 10000, 0.5, 50, 2724.0, 4596.0, 2036.0, 50, 50, 50),
    M(1536, 10000, 0.5, 100, 4233.0, 4716.0, 3113.0, 100, 100, 100),
    M(1536, 10000, 0.5, 500, 7841.0, 4621.0, 6658.0, 500, 500, 500),
    M(1536, 10000, 0.7, 10, 1168.0, 6786.0, 1419.0, 10, 10, 10),
    M(1536, 10000, 0.7, 50, 2744.0, 6836.0, 1407.0, 50, 50, 50),
    M(1536, 10000, 0.7, 100, 4187.0, 6579.0, 2301.0, 100, 100, 100),
    M(1536, 10000, 0.7, 500, 7944.0, 6834.0, 5826.0, 500, 500, 500),
    M(1536, 10000, 0.9, 10, 1133.0, 8593.0, 956.0, 10, 10, 10),
    M(1536, 10000, 0.9, 50, 2730.0, 8519.0, 969.0, 50, 50, 50),
    M(1536, 10000, 0.9, 100, 4207.0, 8388.0, 1801.0, 100, 100, 100),
    M(1536, 10000, 0.9, 500, 7755.0, 8616.0, 5144.0, 500, 500, 500),
];

impl M {
    fn dim(&self) -> usize {
        self.0
    }
    fn n(&self) -> usize {
        self.1
    }
    fn s(&self) -> f64 {
        self.2
    }
    fn k(&self) -> usize {
        self.3
    }
    fn latency(&self) -> [f64; 3] {
        [self.4, self.5, self.6]
    }
    fn rows(&self) -> [usize; 3] {
        [self.7, self.8, self.9]
    }

    /// Rows a correct plan must return : you cannot return more than
    /// match, and should not return fewer than the LIMIT when they exist.
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn complete_answer(&self) -> usize {
        self.k().min((self.s() * self.n() as f64).round() as usize)
    }

    fn workload(&self) -> Workload {
        Workload {
            selectivity: self.s(),
            user_k: self.k(),
            total_rows: self.n(),
            dim: self.dim(),
        }
    }
}

/// The plan a perfect dispatcher should have chosen : **the fastest
/// plan that actually answered the query.**
///
/// Ranking on latency alone reproduces the very bug this investigation
/// found. At `s=0.001` plan A runs in 51 us returning **zero rows**
/// while plan B takes 369 us returning all ten matches ; a plain argmin
/// calls A the winner, so a dispatcher that correctly refuses A is
/// scored as wrong and charged `369/51 = 7.2x` regret for it. Measured
/// across this grid, that mis-scoring hits 28% of cells at a mean 2.4x.
fn fastest_correct_plan(m: &M) -> PlanKind {
    let kinds = [PlanKind::A, PlanKind::B, PlanKind::C];
    let (lat, rows, complete) = (m.latency(), m.rows(), m.complete_answer());
    let mut best: Option<usize> = None;
    for i in 0..3 {
        if rows[i] < complete {
            continue;
        }
        if best.is_none_or(|b| lat[i] < lat[b]) {
            best = Some(i);
        }
    }
    best.map_or(PlanKind::A, |i| kinds[i])
}

struct Score {
    correct: usize,
    regret: f64,
    c_dispatched: usize,
    c_right: usize,
    starved_dispatches: usize,
}

fn evaluate(coeffs: &CostCoefficients) -> Score {
    let (mut correct, mut total_regret) = (0, 0.0);
    let (mut c_dispatched, mut c_right, mut starved) = (0, 0, 0);
    for m in CELLS {
        let predicted = dispatch_via_cost(&m.workload(), coeffs);
        let actual = fastest_correct_plan(m);
        let lat = m.latency();
        if predicted == actual {
            correct += 1;
        }
        if predicted == PlanKind::C {
            c_dispatched += 1;
            if actual == PlanKind::C {
                c_right += 1;
            }
        }
        // Did the dispatched plan actually answer the query?
        if m.rows()[predicted as usize] < m.complete_answer() {
            starved += 1;
        }
        total_regret += lat[predicted as usize] / lat[actual as usize].max(1.0);
    }
    Score {
        correct,
        regret: total_regret / CELLS.len() as f64,
        c_dispatched,
        c_right,
        starved_dispatches: starved,
    }
}

fn calibrated() -> CostCoefficients {
    CostCoefficients {
        c_hnsw_per_visit: 116.4,
        c_distance_per_dim: 0.10,
        c_metadata_get: 249.5,
        c_metadata_peek: 10.4,
        c_filter_eval: 53.3,
    }
}

/// Same machine, but pricing plan C's per-visit metadata access at the
/// *cloning* rate, i.e. the engine as it was before the borrowing
/// accessor landed.
fn calibrated_before_borrow() -> CostCoefficients {
    let c = calibrated();
    CostCoefficients {
        c_metadata_peek: c.c_metadata_get,
        ..c
    }
}

fn main() {
    let n = CELLS.len();
    let c_wins: Vec<&M> = CELLS
        .iter()
        .filter(|m| fastest_correct_plan(m) == PlanKind::C)
        .collect();
    println!(
        "{n} measured cells ; plan C is the fastest *correct* plan in {}\n",
        c_wins.len()
    );

    println!("=== Dispatch quality ===");
    println!("  scenario                          correct   regret   C disp (right)   starved");
    for (label, coeffs) in [
        ("shipped defaults (x86)", CostCoefficients::default()),
        (
            "calibrated, C priced as cloning",
            calibrated_before_borrow(),
        ),
        ("calibrated, C borrows (now)", calibrated()),
    ] {
        let s = evaluate(&coeffs);
        println!(
            "  {label:32} {:3}/{n:<3}   {:.3}   {:6} ({})      {}",
            s.correct, s.regret, s.c_dispatched, s.c_right, s.starved_dispatches
        );
    }
    println!("\n  `starved` = cells where the DISPATCHED plan returned fewer rows");
    println!("  than min(k, matching). Should be 0.");

    println!("\n=== Plan C win rate by selectivity (fastest correct plan) ===");
    let mut sels: Vec<f64> = CELLS.iter().map(M::s).collect();
    sels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sels.dedup();
    for s in sels {
        let cells: Vec<&M> = CELLS.iter().filter(|m| (m.s() - s).abs() < 1e-9).collect();
        let wins = cells
            .iter()
            .filter(|m| fastest_correct_plan(m) == PlanKind::C)
            .count();
        let disp = cells
            .iter()
            .filter(|m| dispatch_via_cost(&m.workload(), &calibrated()) == PlanKind::C)
            .count();
        println!(
            "  s={s:<6}  C fastest {wins}/{}   C dispatched {disp}/{}",
            cells.len(),
            cells.len()
        );
    }

    println!("\n=== Cells where plan C is the fastest correct plan ===");
    println!("   dim      s      k       A       B       C   vs best correct   dispatched?");
    for m in &c_wins {
        let lat = m.latency();
        // Compare against the fastest *correct* alternative. Using
        // min(A, B) would credit plan A's latency in cells where A
        // returned an incomplete answer, understating C's real margin.
        let complete = m.complete_answer();
        let other = (0..2)
            .filter(|&i| m.rows()[i] >= complete)
            .map(|i| lat[i])
            .fold(f64::INFINITY, f64::min);
        let picked = dispatch_via_cost(&m.workload(), &calibrated());
        println!(
            "  {:4} {:6} {:6} {:7.0} {:7.0} {:7.0}    {:.2}x   {}",
            m.dim(),
            m.s(),
            m.k(),
            lat[0],
            lat[1],
            lat[2],
            other / lat[2].max(1.0),
            if picked == PlanKind::C { "yes" } else { "NO" },
        );
    }

    println!("\n=== Result completeness (all cells) ===");
    let mut short = [0usize; 3];
    for m in CELLS {
        let complete = m.complete_answer();
        for i in 0..3 {
            if m.rows()[i] < complete {
                short[i] += 1;
            }
        }
    }
    println!("  cells where the plan returned fewer rows than min(k, matching):");
    println!("    plan A : {}/{n}", short[0]);
    println!("    plan B : {}/{n}", short[1]);
    println!("    plan C : {}/{n}", short[2]);

    println!("\n=== One cell in detail : dim=16 s=0.5 k=50 ===");
    if let Some(m) = CELLS
        .iter()
        .find(|m| m.dim() == 16 && (m.s() - 0.5).abs() < 1e-9 && m.k() == 50)
    {
        for (label, coeffs) in [
            ("C priced as cloning", calibrated_before_borrow()),
            ("C borrows (now)", calibrated()),
        ] {
            let w = m.workload();
            println!(
                "  {label:20}  A={:9.0}  B={:9.0}  C={:9.0}  picks {:?}",
                cost_plan_a(&w, &coeffs),
                cost_plan_b(&w, &coeffs),
                cost_plan_c(&w, &coeffs),
                dispatch_via_cost(&w, &coeffs),
            );
        }
    }
}

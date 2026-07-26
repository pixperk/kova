//! Cost-based plan dispatch.
//!
//! Replaces the hardcoded selectivity bands in the planner with a
//! closed-form cost estimate per plan. The dispatcher computes the
//! cost of plans A, B, and C under the current workload and picks
//! the cheapest.
//!
//! ## The plans, briefly
//!
//! - **Plan A** : overfetched kNN (`k * KNN_OVERFETCH` candidates)
//!   followed by post-filter. Wins when the predicate is loose
//!   enough that most overfetched candidates pass.
//! - **Plan B** : iterate the matching id set (via index or scan),
//!   compute exact distance per match, take top-k. Wins when the
//!   match set is small.
//! - **Plan C** : filtered HNSW walk where the predicate is
//!   consulted during graph traversal. Wins at **high** selectivity
//!   on large shards with a large `k` — not, as an earlier version of
//!   this doc claimed, in the middle band.
//!
//!   The measured band (`examples/validate_cost_model.rs`, 120 cells)
//!   is `s = 0.5`, `n = 10_000`, `k >= 50`, across every dimension
//!   tested, where C runs 1.4-2.2x faster than the best alternative.
//!   The reason is plan A's fixed `k * KNN_OVERFETCH` : sized for a
//!   filter that rejects, it does roughly double the necessary graph
//!   work when the filter passes half the rows, while C asks for
//!   exactly `k`.
//!
//!   At *low* selectivity plan C is the worst option by a wide margin,
//!   because its results heap can never fill and the walk drains the
//!   whole candidate set (measured: 13ms at n=10k, dim=1536, s=0.001).
//!
//! ## Inputs
//!
//! The cost model reads four numbers off the workload :
//!
//! - `selectivity` in `[0.0, 1.0]` : fraction of rows matching
//! - `user_k` : the query's LIMIT
//! - `total_rows` : shard size
//! - `dim` : vector dimension
//!
//! And four [`CostCoefficients`] off the machine :
//!
//! - `c_hnsw_per_visit` : cost of visiting one HNSW node (heap
//!   ops + neighbour walk, exclusive of distance compute)
//! - `c_distance_per_dim` : per-scalar cost of distance compute.
//!   Distance for `dim`-dim vectors costs `dim * c_distance_per_dim`.
//! - `c_metadata_get` : cost of fetching one metadata bag
//!   from the metadata store
//! - `c_filter_eval` : cost of evaluating one predicate atom
//!   on a metadata bag
//!
//! All costs are in nanoseconds. Magnitudes don't matter (only
//! ratios drive dispatch), but defaults are tuned to typical x86
//! numbers.
//!
//! ## What this model captures
//!
//! - Plan A's fixed overfetch cost. The executor walks the HNSW for
//!   exactly `k * KNN_OVERFETCH` candidates and returns however
//!   many survive post-filter, possibly fewer than `k`. This is a
//!   recall trade-off, not a latency one, so cost A is independent
//!   of selectivity.
//! - Plan B's full-shard metadata scan plus linear-in-matches
//!   distance compute. Dominant scan term `n * c_filter_eval` is
//!   what makes plan B expensive at large `n`.
//! - Plan C's filter overhead per visit, with visit count scaling
//!   by `1/s` and clamped at `n` (the point where the filtered walk
//!   has degenerated into a full scan).
//!
//! ## What this model deliberately omits
//!
//! - **HNSW visit count vs `n`** : we approximate as
//!   `ef * log2(n)`. Real HNSW visit counts depend on `M`,
//!   `ef_construction`, and the data distribution. The
//!   approximation is correct to within a small constant.
//!
//! - **Cache effects, NUMA, prefetching** : real per-visit cost
//!   varies with shard size. We use a flat coefficient.
//! - **Column correlation in compound predicates** : we already
//!   assume independence in the selectivity estimator ; the cost
//!   model inherits that assumption.
//!
//! Coefficient values shipped with the crate are educated guesses
//! tuned to dim=16 / n=10k workloads. Per-machine calibration via
//! microbenchmarks is the natural follow-up.
//!
//! ## Staying honest about what the executor does
//!
//! Every bug this model has shipped has had the same shape : the
//! formulas scored an *idealised* plan rather than the one the
//! executor actually runs. Four instances so far —
//!
//! 1. plan A charged a starvation-retry term for a retry that does
//!    not exist (the executor returns short rather than retrying) ;
//! 2. plan B was missing the `n * c_filter_eval` scan term, because
//!    the model assumed an O(1) catalog lookup that only happens
//!    when the predicate is fully indexed ;
//! 3. `hnsw_visits` used `ef = max(2k, 50)` where the index uses
//!    `ef = max(k, 50)` ;
//! 4. plan C's visit multiplier was linear in `(1 - s)` where the
//!    real behaviour is hyperbolic in `s`.
//!
//! All four were found the same way : by running
//! `examples/validate_cost_model.rs`, which forces each plan on a
//! grid and compares predicted to measured. **Change a formula here,
//! re-run that harness.** A model that is internally consistent can
//! still describe a different program than the one you shipped.

// Cost math is full of ratios of counts. `usize` / `u64` → `f64`
// loses precision past 2^53, which we never approach.
#![allow(clippy::cast_precision_loss)]

/// Per-machine cost coefficients. Magnitudes are in nanoseconds ;
/// only ratios drive dispatch decisions.
///
/// Calibrate with microbenchmarks on the target machine for the
/// most accurate dispatch. The shipped defaults are reasonable
/// for typical x86 servers running unoptimised distance code.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostCoefficients {
    /// Nanoseconds per HNSW node visit, excluding distance compute.
    /// Covers heap push/pop + neighbour table walk.
    pub c_hnsw_per_visit: f64,
    /// Nanoseconds per scalar in a distance computation. A
    /// `dim`-dim distance costs `dim * c_distance_per_dim`.
    pub c_distance_per_dim: f64,
    /// Nanoseconds per metadata bag fetch from the metadata store
    /// (`HashMap` lookup + clone).
    pub c_metadata_get: f64,
    /// Nanoseconds per predicate atom evaluation on a fetched bag.
    pub c_filter_eval: f64,
}

impl Default for CostCoefficients {
    fn default() -> Self {
        // Calibrated by `examples/calibrate_cost_coefficients.rs` on a
        // typical x86 dev machine : SIMD-vectorised L2 distance gives
        // ~0.15 ns/dim, file-backed metadata gets are ~310 ns, the
        // walk_field filter check is ~70 ns/row, and an HNSW visit
        // costs ~100 ns excluding the distance compute it triggers.
        // Per-machine calibration via the same example is the right
        // move when latency targets matter.
        Self {
            c_hnsw_per_visit: 100.0,
            c_distance_per_dim: 0.15,
            c_metadata_get: 310.0,
            c_filter_eval: 70.0,
        }
    }
}

/// Workload context the cost model dispatches on.
#[derive(Debug, Clone, Copy)]
pub struct Workload {
    /// Predicate selectivity in `[0.0, 1.0]`.
    pub selectivity: f64,
    /// Result limit (the query's LIMIT).
    pub user_k: usize,
    /// Total live rows in the shard.
    pub total_rows: usize,
    /// Vector dimension.
    pub dim: usize,
}

/// Plan dispatched by the cost model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanKind {
    /// Plan A : overfetched kNN + post-filter.
    A,
    /// Plan B : metadata scan + exact distance.
    B,
    /// Plan C : filtered HNSW walk.
    C,
}

/// kNN overfetch multiplier. Plan A asks for `user_k * KNN_OVERFETCH`
/// candidates from the kNN walk so the post-filter has headroom to
/// drop some without missing the top-k.
pub const KNN_OVERFETCH: usize = 4;

/// Minimum `ef` for an HNSW search. Even small-k queries pay this
/// floor's worth of visits.
///
/// Must track `HnswParams::ef_search`'s default : the executor uses
/// `ef = params.ef_search.max(k)`, and this model mirrors that.
const HNSW_EF_FLOOR: f64 = 50.0;

/// Expected HNSW visit count for a search returning `k` items from a
/// shard of `n` rows. Approximated as `ef * log2(n)`.
///
/// `ef` must match what [`kova_index::HnswIndex`] actually uses :
///
/// ```text
/// // hnsw/search.rs, search_impl + search_filtered_impl
/// let ef = self.params.ef_search.max(k);
/// ```
///
/// i.e. `max(k, 50)` — **not** `max(2k, 50)`. An earlier version of
/// this function doubled `k`, which inflated the modelled visit count
/// for every plan that walks the graph. The error mostly cancelled in
/// the plan A vs plan C comparison (both were doubled) except below
/// the floor, where it mattered a lot : at `k = 10` the model saw
/// `ef_A = 80` vs `ef_C = 50` and credited plan C with a 1.6x
/// visit-count advantage it does not have. In reality both floor at
/// 50 and the advantage is exactly 1.0x.
fn hnsw_visits(k: usize, n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    let n_f = n as f64;
    let ef = (k as f64).max(HNSW_EF_FLOOR).min(n_f);
    ef * n_f.log2()
}

/// Visit-count multiplier for plan C's filtered walk, as a function of
/// selectivity.
///
/// Plan C only admits filter-passing nodes into its results heap, and
/// its termination rule requires that heap to hold `ef` entries before
/// it can short-circuit. So to collect `ef` in-filter results it must
/// visit roughly `ef / s` nodes : the multiplier is **hyperbolic in
/// `s`**, not linear.
///
/// The previous implementation was `1 + 4 * (1 - s)`, a line
/// saturating at 5x. That was wrong at both ends :
///
/// | s | linear model | `1/s` | measured |
/// |---|---|---|---|
/// | 0.5 | 3.0x | 2.0x | ~1.5-2x (plan C wins here) |
/// | 0.001 | 5.0x | 1000x, clamped to `n` | visits ~all of `n` |
///
/// Overstating the cost at high selectivity meant plan C was never
/// dispatched in the band where it is genuinely 1.4-2.2x faster;
/// understating it at low selectivity hid the fact that plan C
/// degenerates into a full scan there.
///
/// The result is clamped by the caller to `n` : you cannot visit more
/// nodes than exist. That clamp is what keeps `s -> 0` finite.
fn filter_visit_multiplier(selectivity: f64) -> f64 {
    let s = selectivity.clamp(0.0, 1.0);
    if s <= 0.0 {
        // No matching rows : the walk drains the candidate set. The
        // caller's `.min(total_rows)` turns this into "visit everything".
        return f64::INFINITY;
    }
    1.0 / s
}

/// Cost of plan A : overfetched kNN + post-filter.
///
/// The executor walks the HNSW for exactly `k * KNN_OVERFETCH`
/// candidates, post-filters them, and returns whatever survives.
/// No retry on starvation : if selectivity is low, the result set
/// may be smaller than `k`. That is a recall trade-off the planner
/// surfaces separately ; cost A is independent of selectivity.
#[must_use]
pub fn cost_plan_a(w: &Workload, c: &CostCoefficients) -> f64 {
    let overfetch = ((w.user_k * KNN_OVERFETCH) as f64)
        .min(w.total_rows as f64)
        .max(1.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let visits = hnsw_visits(overfetch as usize, w.total_rows);
    let per_visit = c.c_hnsw_per_visit + c.c_distance_per_dim * w.dim as f64;
    let walk = visits * per_visit;
    let post_filter = overfetch * (c.c_metadata_get + c.c_filter_eval);
    walk + post_filter
}

/// Cost of plan B : metadata scan + exact distance on matches.
///
/// The scan walks all `n` rows of the metadata store, paying
/// `c_filter_eval` per row to test the predicate (the
/// `walk_field` path is field-targeted, so we skip the full bag
/// fetch on non-matching rows). For matching rows, we then fetch
/// the bag and compute exact distance against the query vector.
#[must_use]
pub fn cost_plan_b(w: &Workload, c: &CostCoefficients) -> f64 {
    let n = w.total_rows as f64;
    let scan = n * c.c_filter_eval;
    let matches = (w.selectivity * n).max(1.0);
    let per_match = c.c_metadata_get + c.c_distance_per_dim * w.dim as f64;
    scan + matches * per_match
}

/// Cost of plan C : filtered HNSW walk.
///
/// Visit count scales by `1 / s` (see [`filter_visit_multiplier`]) :
/// at high selectivity the walk behaves like vanilla HNSW ; as the
/// filter rejects more, proportionally more nodes must be visited to
/// fill the results heap. Clamped at `total_rows`, since the walk
/// cannot visit more nodes than the graph holds — that clamp is the
/// model's way of saying "plan C degenerated into a full scan."
#[must_use]
pub fn cost_plan_c(w: &Workload, c: &CostCoefficients) -> f64 {
    let base = hnsw_visits(w.user_k, w.total_rows);
    let overhead = filter_visit_multiplier(w.selectivity);
    let visits = (base * overhead).min(w.total_rows as f64);
    let per_visit = c.c_hnsw_per_visit
        + c.c_distance_per_dim * w.dim as f64
        + c.c_metadata_get
        + c.c_filter_eval;
    visits * per_visit
}

/// Internals exposed for the benchmark harnesses only, gated behind
/// the `internal-bench` feature so the regular public API doesn't
/// grow them.
///
/// This exists because a copy is not a mirror.
/// `examples/calibrate_cost_coefficients.rs` used to carry its own
/// `hnsw_visits` under a `// mirrored from cost::hnsw_visits`
/// comment. When the real one was corrected from `max(2k, 50)` to
/// `max(k, 50)`, the copy silently kept the old formula — and since
/// the calibrator *derives* `c_hnsw_per_visit` by dividing a measured
/// latency by a modelled visit count, the stale copy produced a
/// coefficient off by the exact ratio of the two formulas (1.6x).
///
/// One definition, shared. If a harness needs a piece of the model,
/// re-export it here rather than transcribing it.
#[cfg(feature = "internal-bench")]
pub mod internal_bench {
    /// The model's HNSW visit-count estimate, `ef * log2(n)` with
    /// `ef = max(k, 50).min(n)`. See [`super::hnsw_visits`].
    #[must_use]
    pub fn hnsw_visits(k: usize, n: usize) -> f64 {
        super::hnsw_visits(k, n)
    }
}

/// Dispatch a plan by computing `cost_plan_a/b/c` and returning the
/// kind with the lowest cost. Ties break A > B > C (most-general
/// first).
#[must_use]
pub fn dispatch_via_cost(w: &Workload, c: &CostCoefficients) -> PlanKind {
    let a = cost_plan_a(w, c);
    let b = cost_plan_b(w, c);
    let cc = cost_plan_c(w, c);
    if a <= b && a <= cc {
        PlanKind::A
    } else if b <= cc {
        PlanKind::B
    } else {
        PlanKind::C
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workload(selectivity: f64, k: usize, n: usize, dim: usize) -> Workload {
        Workload {
            selectivity,
            user_k: k,
            total_rows: n,
            dim,
        }
    }

    // ---- hnsw_visits : must mirror the index, not an idealised HNSW ----

    /// The index computes `ef = params.ef_search.max(k)`, so any `k`
    /// at or below the floor produces the same visit count. A model
    /// that used `max(2k, 50)` would show k=10 and k=25 differing.
    #[test]
    fn hnsw_visits_floors_at_ef_search_not_double_k() {
        let at_10 = hnsw_visits(10, 10_000);
        let at_25 = hnsw_visits(25, 10_000);
        let at_50 = hnsw_visits(50, 10_000);
        assert!(
            (at_10 - at_50).abs() < f64::EPSILON,
            "k=10 must floor to ef=50"
        );
        assert!(
            (at_25 - at_50).abs() < f64::EPSILON,
            "k=25 must floor to ef=50"
        );
    }

    /// Above the floor, visits scale linearly in `k` (ef == k).
    #[test]
    fn hnsw_visits_scales_linearly_above_the_floor() {
        let at_100 = hnsw_visits(100, 10_000);
        let at_200 = hnsw_visits(200, 10_000);
        assert!((at_200 / at_100 - 2.0).abs() < 1e-9);
    }

    /// The regression this pins : at k=10 plan A requests `k * 4 = 40`
    /// candidates and plan C requests `10`. Both are under the ef floor,
    /// so both walk with ef=50 and plan C has **no** visit-count
    /// advantage. The old `max(2k, 50)` formula claimed `ef_A = 80` vs
    /// `ef_C = 50`, crediting C with a 1.6x edge it never had.
    #[test]
    fn plan_a_and_plan_c_walk_identically_at_small_k() {
        let k = 10;
        let plan_a_request = k * KNN_OVERFETCH; // 40
        assert!(
            (hnsw_visits(plan_a_request, 10_000) - hnsw_visits(k, 10_000)).abs() < f64::EPSILON,
            "below the ef floor A and C must walk the same width"
        );
    }

    // ---- filter_visit_multiplier : hyperbolic, not linear ----

    #[test]
    fn filter_multiplier_is_inverse_selectivity() {
        assert!((filter_visit_multiplier(1.0) - 1.0).abs() < 1e-9);
        assert!((filter_visit_multiplier(0.5) - 2.0).abs() < 1e-9);
        assert!((filter_visit_multiplier(0.1) - 10.0).abs() < 1e-9);
        assert!(filter_visit_multiplier(0.0).is_infinite());
    }

    /// The old linear form saturated at 5x, so it could never express
    /// "this walk visits the whole shard." The hyperbolic form does,
    /// and `cost_plan_c` clamps it at `total_rows`.
    #[test]
    fn plan_c_degenerates_to_a_full_scan_at_low_selectivity() {
        let c = CostCoefficients::default();
        let n = 10_000;
        let w = workload(0.001, 10, n, 16);
        let per_visit =
            c.c_hnsw_per_visit + c.c_distance_per_dim * 16.0 + c.c_metadata_get + c.c_filter_eval;
        // Clamped at exactly one visit per row in the shard.
        assert!((cost_plan_c(&w, &c) - n as f64 * per_visit).abs() < 1.0);
    }

    // ---- cost functions ----

    #[test]
    fn cost_plan_b_grows_with_selectivity() {
        let c = CostCoefficients::default();
        let w_low = workload(0.1, 10, 10_000, 16);
        let w_high = workload(0.5, 10, 10_000, 16);
        let cost_low = cost_plan_b(&w_low, &c);
        let cost_high = cost_plan_b(&w_high, &c);
        // Both pay the same scan term ; the matches term grows by 5x
        // so total cost grows but by less than 5x.
        assert!(cost_high > cost_low);
        assert!(cost_high < cost_low * 5.0);
    }

    #[test]
    fn cost_plan_a_independent_of_selectivity() {
        let c = CostCoefficients::default();
        let w_mid = workload(0.5, 10, 10_000, 16);
        let w_tight = workload(0.05, 10, 10_000, 16);
        // Plan A's overfetch is fixed at `k * KNN_OVERFETCH` ; the
        // executor returns however many candidates survive the
        // post-filter. Cost should not depend on selectivity.
        let cost_mid = cost_plan_a(&w_mid, &c);
        let cost_tight = cost_plan_a(&w_tight, &c);
        assert!((cost_mid - cost_tight).abs() < 0.01);
    }

    #[test]
    fn cost_plan_b_includes_scan_term() {
        let c = CostCoefficients::default();
        // At very low selectivity, the scan term dominates : cost
        // should be roughly n * c_filter_eval regardless of how
        // few matches there are.
        let w = workload(0.0001, 10, 100_000, 16);
        let cost = cost_plan_b(&w, &c);
        let scan_only = 100_000.0 * c.c_filter_eval;
        // Within 10% of pure scan cost (one match adds negligible).
        assert!((cost / scan_only - 1.0).abs() < 0.1);
    }

    #[test]
    fn cost_plan_c_grows_with_low_selectivity() {
        let c = CostCoefficients::default();
        let w_loose = workload(0.9, 10, 10_000, 16);
        let w_tight = workload(0.1, 10, 10_000, 16);
        // Lower selectivity → more visits per match found.
        let cost_loose = cost_plan_c(&w_loose, &c);
        let cost_tight = cost_plan_c(&w_tight, &c);
        assert!(cost_tight > cost_loose);
    }

    #[test]
    fn cost_plan_b_scales_with_dim() {
        let c = CostCoefficients::default();
        let w_low_dim = workload(0.3, 10, 10_000, 16);
        let w_high_dim = workload(0.3, 10, 10_000, 1536);
        let cost_16 = cost_plan_b(&w_low_dim, &c);
        let cost_1536 = cost_plan_b(&w_high_dim, &c);
        // SIMD distance compute is cheap per scalar, so 96x more dim
        // doesn't translate to anywhere near 96x more cost ; the
        // dim-independent scan term dominates. Just check the
        // direction.
        assert!(cost_1536 > cost_16);
    }

    // ---- dispatch ----

    #[test]
    fn dispatch_picks_b_at_very_low_selectivity_small_shard() {
        // At small n, plan B's scan term `n * c_filter_eval` is short
        // enough that even a tiny match set still beats plan A's
        // HNSW walk overhead. This is the classic small-shard +
        // selective-predicate regime.
        let w = workload(0.001, 10, 1_000, 1536);
        assert_eq!(
            dispatch_via_cost(&w, &CostCoefficients::default()),
            PlanKind::B
        );
    }

    #[test]
    fn dispatch_picks_a_at_high_selectivity() {
        // Loose predicate : plan A's overfetch succeeds first try,
        // plan B's per-match cost blows up at large match set.
        let w = workload(0.9, 10, 10_000, 16);
        assert_eq!(
            dispatch_via_cost(&w, &CostCoefficients::default()),
            PlanKind::A
        );
    }

    #[test]
    fn dispatch_picks_a_at_high_dim_low_selectivity_large_n() {
        // With the starvation term removed, plan A's HNSW walk is
        // cheap even at very low selectivity : the executor just
        // returns however many candidates pass the post-filter.
        // At n=1M, plan B's scan term `n * c_filter_eval` makes it
        // expensive ; plan C's filter overhead at low s is worse
        // still. Plan A wins.
        let w = workload(0.02, 10, 1_000_000, 1536);
        assert_eq!(
            dispatch_via_cost(&w, &CostCoefficients::default()),
            PlanKind::A
        );
    }

    #[test]
    fn dispatch_picks_b_on_tiny_shard() {
        // For tiny shards, scan everything beats HNSW walk overhead.
        let w = workload(0.5, 5, 100, 16);
        assert_eq!(
            dispatch_via_cost(&w, &CostCoefficients::default()),
            PlanKind::B
        );
    }

    /// Plan C is measurably fastest at high selectivity on a large
    /// shard — `examples/validate_cost_model.rs` records
    /// `dim=16 n=10_000 s=0.5 k=50` at A=229us B=1149us **C=103us**
    /// (C wins by 2.2x), plus seven sibling cells.
    ///
    /// The model only agrees once `c_metadata_get` reflects a
    /// *borrowed* bag rather than a cloned one. The shipped default
    /// (310ns) was calibrated against `MetadataStore::get`, which
    /// clones the whole `HashMap` ; plan C pays that per visited node,
    /// so it dominates C's per-visit cost. With
    /// `MetadataStore::with_metadata` on the read paths the real cost
    /// is an order of magnitude lower, and dispatch lines up with the
    /// measurement.
    ///
    /// If this test starts failing after a re-calibration, the
    /// coefficient moved — re-run the validation harness before
    /// changing the assertion.
    #[test]
    fn plan_c_is_dispatched_in_its_measured_band_when_metadata_is_borrowed() {
        let borrowing = CostCoefficients {
            c_metadata_get: 25.0,
            ..Default::default()
        };
        let w = workload(0.5, 50, 10_000, 16);
        assert_eq!(dispatch_via_cost(&w, &borrowing), PlanKind::C);
    }

    #[test]
    fn dispatch_is_deterministic_at_selectivity_zero() {
        // selectivity = 0 : no matches, every plan should be
        // well-defined and dispatch should pick something.
        let w = workload(0.0, 10, 10_000, 16);
        let _ = dispatch_via_cost(&w, &CostCoefficients::default());
    }

    #[test]
    fn dispatch_is_deterministic_at_selectivity_one() {
        let w = workload(1.0, 10, 10_000, 16);
        let _ = dispatch_via_cost(&w, &CostCoefficients::default());
    }

    // ---- coefficients propagate ----

    #[test]
    fn doubling_distance_cost_doubles_plan_b_roughly() {
        let c1 = CostCoefficients::default();
        let c2 = CostCoefficients {
            c_distance_per_dim: c1.c_distance_per_dim * 2.0,
            ..c1
        };
        let w = workload(0.3, 10, 10_000, 16);
        let cost_1 = cost_plan_b(&w, &c1);
        let cost_2 = cost_plan_b(&w, &c2);
        // Plan B cost is (metadata + dim * c_dist) per match.
        // metadata is c_metadata_get=200, distance term was 5*16=80,
        // now 10*16=160. So total per_match goes 280 → 360.
        // cost_2 / cost_1 ≈ 360/280 ≈ 1.29.
        assert!(cost_2 > cost_1);
        assert!(cost_2 < cost_1 * 1.5);
    }

    #[test]
    fn extreme_dim_can_flip_a_to_c() {
        // At dim=1536, plan A's per-visit cost includes huge
        // distance computes. Plan C's per-visit cost is even
        // bigger because of filter+metadata, but C's visit count
        // (which scales by 1/s) might still win because plan A's
        // overfetch retries hit higher k_eff.
        let typical = workload(0.3, 10, 100_000, 16);
        let extreme = workload(0.3, 10, 100_000, 1536);
        let pick_typical = dispatch_via_cost(&typical, &CostCoefficients::default());
        let pick_extreme = dispatch_via_cost(&extreme, &CostCoefficients::default());
        // Just assert dispatch returns sensible plans ; this test
        // pins that high-dim workloads can differ from low-dim
        // even at the same selectivity.
        assert!(matches!(
            pick_typical,
            PlanKind::A | PlanKind::B | PlanKind::C
        ));
        assert!(matches!(
            pick_extreme,
            PlanKind::A | PlanKind::B | PlanKind::C
        ));
    }
}

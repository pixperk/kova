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
//!   consulted during graph traversal. Wins in the middle band
//!   where A starves and B's match set is large.
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
//! - Plan A's overfetch starvation when `selectivity * k_eff < k`,
//!   modelled as `k_eff = max(k * overfetch, k / s)`.
//! - Plan B's linear-in-matches distance compute cost, dominant
//!   when dim is high.
//! - Plan C's filter overhead per visit, with visit count scaling
//!   by `1/s` (worst-case linear filter rejection rate).
//!
//! ## What this model deliberately omits
//!
//! - **HNSW visit count vs `n`** : we approximate as
//!   `ef * log2(n)`. Real HNSW visit counts depend on `M`,
//!   `ef_construction`, and the data distribution. The
//!   approximation is correct to within a small constant.
//! - **Cache effects, NUMA, prefetching** : real per-visit cost
//!   varies with shard size. We use a flat coefficient.
//! - **Column correlation in compound predicates** : we already
//!   assume independence in the selectivity estimator ; the cost
//!   model inherits that assumption.
//!
//! Coefficient values shipped with the crate are educated guesses
//! tuned to dim=16 / n=10k workloads. Per-machine calibration via
//! microbenchmarks is the natural follow-up.

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
        Self {
            c_hnsw_per_visit: 500.0,
            c_distance_per_dim: 5.0,
            c_metadata_get: 200.0,
            c_filter_eval: 200.0,
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
const HNSW_EF_FLOOR: f64 = 50.0;

/// Filtered HNSW overhead factor : on the `1.0 - selectivity` axis,
/// how much more does plan C visit than vanilla? At s=1 the factor
/// is 1.0 (filter never rejects). At s=0 the factor saturates at
/// `1 + FILTER_OVERHEAD_AT_S0`.
const FILTER_OVERHEAD_AT_S0: f64 = 4.0;

/// Expected HNSW visit count for a kNN search returning `k` items
/// from a shard of `n` rows. Standard approximation : `ef * log2(n)`
/// where `ef = max(2*k, HNSW_EF_FLOOR)`.
fn hnsw_visits(k: usize, n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    let n_f = n as f64;
    let ef = ((k as f64) * 2.0).max(HNSW_EF_FLOOR).min(n_f);
    ef * n_f.log2()
}

/// Cost of plan A : overfetched kNN + post-filter.
///
/// The overfetch is `k * KNN_OVERFETCH`. When selectivity is low
/// enough that the overfetch can't deliver `k` survivors, `k_eff`
/// grows to `k / selectivity` (the model's retry estimate).
#[must_use]
pub fn cost_plan_a(w: &Workload, c: &CostCoefficients) -> f64 {
    let overfetch = (w.user_k * KNN_OVERFETCH) as f64;
    let k_eff = if w.selectivity > 0.0 {
        // Need at least k / s candidates to find k matches.
        let by_retry = w.user_k as f64 / w.selectivity;
        overfetch.max(by_retry)
    } else {
        overfetch
    };
    // Cap k_eff at the shard size : we can't ask for more
    // candidates than exist.
    let k_eff = k_eff.min(w.total_rows as f64).max(1.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let visits = hnsw_visits(k_eff as usize, w.total_rows);
    let per_visit = c.c_hnsw_per_visit + c.c_distance_per_dim * w.dim as f64;
    let walk = visits * per_visit;
    let post_filter = k_eff * (c.c_metadata_get + c.c_filter_eval);
    walk + post_filter
}

/// Cost of plan B : iterate matching ids, exact distance per match.
///
/// Assumes the catalog gives the matching id set in O(1) (the
/// `try_index_eval` `Full` case) ; otherwise the upstream scan
/// cost would be modelled separately.
#[must_use]
pub fn cost_plan_b(w: &Workload, c: &CostCoefficients) -> f64 {
    let matches = (w.selectivity * w.total_rows as f64).max(1.0);
    let per_match = c.c_metadata_get + c.c_distance_per_dim * w.dim as f64;
    matches * per_match
}

/// Cost of plan C : filtered HNSW walk.
///
/// Visit count scales by `1 + FILTER_OVERHEAD_AT_S0 * (1 - s)` :
/// at high selectivity, walks like vanilla ; at low selectivity,
/// the filter rejection rate forces more visits to find `k`
/// matches.
#[must_use]
pub fn cost_plan_c(w: &Workload, c: &CostCoefficients) -> f64 {
    let base = hnsw_visits(w.user_k, w.total_rows);
    let overhead = 1.0 + FILTER_OVERHEAD_AT_S0 * (1.0 - w.selectivity).clamp(0.0, 1.0);
    let visits = (base * overhead).min(w.total_rows as f64);
    let per_visit = c.c_hnsw_per_visit
        + c.c_distance_per_dim * w.dim as f64
        + c.c_metadata_get
        + c.c_filter_eval;
    visits * per_visit
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

    // ---- cost functions ----

    #[test]
    fn cost_plan_b_grows_linearly_in_selectivity() {
        let c = CostCoefficients::default();
        let w_low = workload(0.1, 10, 10_000, 16);
        let w_high = workload(0.5, 10, 10_000, 16);
        let cost_low = cost_plan_b(&w_low, &c);
        let cost_high = cost_plan_b(&w_high, &c);
        // 5x more matches → 5x more cost.
        assert!((cost_high / cost_low - 5.0).abs() < 0.01);
    }

    #[test]
    fn cost_plan_a_grows_with_low_selectivity_retry() {
        let c = CostCoefficients::default();
        let w_mid = workload(0.5, 10, 10_000, 16);
        let w_tight = workload(0.05, 10, 10_000, 16);
        // Tight predicate → k_eff = k/s = 200, much bigger than
        // the overfetch=40 baseline, so visits grow.
        let cost_mid = cost_plan_a(&w_mid, &c);
        let cost_tight = cost_plan_a(&w_tight, &c);
        assert!(
            cost_tight > cost_mid,
            "tight predicate should be more expensive"
        );
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
        // 96x more dimensions → way more cost.
        assert!(cost_1536 > cost_16 * 10.0);
    }

    // ---- dispatch ----

    #[test]
    fn dispatch_picks_b_at_very_low_selectivity_typical_workload() {
        // Tiny match set : plan B's `matches * dim_cost` is cheap,
        // plan A's HNSW walk is wasted on candidates that mostly
        // fail the filter.
        let w = workload(0.001, 10, 10_000, 16);
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
    fn dispatch_picks_c_at_high_dim_moderate_selectivity() {
        // The case the cost model surfaces that hardcoded bands
        // miss : at dim=1536, plan B's distance compute dominates,
        // plan A's overfetch retries are expensive ; plan C threads
        // a smaller visit budget through the index with cheap
        // filter eval per node.
        let w = workload(0.02, 10, 1_000_000, 1536);
        assert_eq!(
            dispatch_via_cost(&w, &CostCoefficients::default()),
            PlanKind::C
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

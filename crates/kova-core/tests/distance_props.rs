//! Property tests for the [`Distance`] trait implementations.
//!
//! These complement the inline unit tests in `src/distance.rs`. Where those
//! assert specific known values, these assert *properties* that should hold for
//! any valid input : symmetry, non-negativity, triangle inequality, etc.
//! proptest generates ~256 cases per property by default and shrinks any
//! failure to the minimal counterexample.

#![allow(missing_docs)]

use kova_core::{Cosine, Distance, InnerProduct, L2, Vector};
use proptest::prelude::*;

/// FP slack for "approximately equal" assertions. Bounded input range
/// (`-10..10`) keeps absolute rounding error well under this.
const EPSILON: f32 = 1e-3;

fn vector_strategy(dim: usize) -> impl Strategy<Value = Vector> {
    prop::collection::vec(-10.0_f32..10.0_f32, dim..=dim)
        .prop_map(|data| Vector::try_new(data).expect("strategy yields finite, non-empty"))
}

proptest! {
    // ---------- L2 (a true metric) ----------

    #[test]
    fn l2_self_distance_is_zero(a in vector_strategy(16)) {
        prop_assert!(L2.distance(&a, &a).abs() < EPSILON);
    }

    #[test]
    fn l2_is_symmetric(a in vector_strategy(16), b in vector_strategy(16)) {
        let ab = L2.distance(&a, &b);
        let ba = L2.distance(&b, &a);
        prop_assert!((ab - ba).abs() < EPSILON);
    }

    #[test]
    fn l2_is_non_negative(a in vector_strategy(16), b in vector_strategy(16)) {
        prop_assert!(L2.distance(&a, &b) >= 0.0);
    }

    #[test]
    fn l2_satisfies_triangle_inequality(
        a in vector_strategy(8),
        b in vector_strategy(8),
        c in vector_strategy(8),
    ) {
        let ac = L2.distance(&a, &c);
        let ab_plus_bc = L2.distance(&a, &b) + L2.distance(&b, &c);
        prop_assert!(
            ac <= ab_plus_bc + EPSILON,
            "triangle inequality violated: d(a,c)={ac} > d(a,b)+d(b,c)={ab_plus_bc}",
        );
    }

    // ---------- Cosine (not a true metric : skip triangle) ----------

    #[test]
    fn cosine_is_symmetric(a in vector_strategy(16), b in vector_strategy(16)) {
        let ab = Cosine.distance(&a, &b);
        let ba = Cosine.distance(&b, &a);
        prop_assert!((ab - ba).abs() < EPSILON);
    }

    #[test]
    fn cosine_is_non_negative(a in vector_strategy(16), b in vector_strategy(16)) {
        // tiny FP slack: identical-direction vectors can round to -1e-7 or so
        prop_assert!(Cosine.distance(&a, &b) >= -EPSILON);
    }

    #[test]
    fn cosine_self_distance_is_zero(a in vector_strategy(16)) {
        // Skip the all-zero case : the impl returns 1.0 there by design.
        prop_assume!(a.as_slice().iter().any(|&x| x != 0.0));
        prop_assert!(Cosine.distance(&a, &a).abs() < EPSILON);
    }

    // ---------- InnerProduct (not a metric : only symmetry holds) ----------

    #[test]
    fn inner_product_is_symmetric(
        a in vector_strategy(16),
        b in vector_strategy(16),
    ) {
        let ab = InnerProduct.distance(&a, &b);
        let ba = InnerProduct.distance(&b, &a);
        prop_assert!((ab - ba).abs() < EPSILON);
    }
}

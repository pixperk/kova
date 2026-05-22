//! Distance metrics over [`Vector`].
//!
//! The [`Distance`] trait is the abstraction every index, planner, and benchmark
//! uses to score how close two vectors are. Three concrete implementations live
//! here: [`Cosine`], [`L2`], and [`InnerProduct`]. All return `f32` where
//! **smaller means closer**, matching what HNSW's min-heaps expect.
//!
//! Implementations are SIMD-accelerated via `wide::f32x8` (8-lane f32). Any
//! remainder past the last full 8-lane chunk is folded in scalar. The `wide`
//! crate provides a scalar fallback on platforms without SIMD, so this code
//! compiles and runs everywhere.

use wide::f32x8;

use crate::Vector;

/// Load 8 consecutive f32s from a slice into a SIMD lane.
#[inline]
fn load8(s: &[f32]) -> f32x8 {
    debug_assert!(s.len() >= 8);
    f32x8::new([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
}

/// A distance metric between two [`Vector`]s.
///
/// Implementations must return `f32` where smaller values mean *closer*.
/// Both vectors must have the same dimension; otherwise the result is undefined.
/// Callers (the index layer) are responsible for upholding this invariant :
/// implementations should `debug_assert_eq!(a.dim(), b.dim())` to catch
/// violations in dev builds.
///
/// `Send + Sync + 'static` so the metric can be stored in trait objects shared
/// across threads (HNSW will parallelise over `rayon`, and the server is multi-
/// threaded).
pub trait Distance: Send + Sync + 'static {
    /// Computes the distance between `a` and `b`.
    #[must_use]
    fn distance(&self, a: &Vector, b: &Vector) -> f32;

    /// Stable, lowercase identifier for the metric (`"cosine"`, `"l2"`,
    /// `"inner_product"`). Used in logs, benchmark output, and KQL.
    fn name(&self) -> &'static str;
}

/// Cosine distance: `1 - cos_similarity(a, b)`.
///
/// Range `[0, 2]`. Not a true metric (no triangle inequality). For a zero
/// vector on either side, returns `1.0` instead of `NaN`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Cosine;

/// Euclidean (L2) distance: `sqrt(sum((a_i - b_i)^2))`.
///
/// Range `[0, ∞)`. A true metric : satisfies the triangle inequality.
#[derive(Debug, Clone, Copy, Default)]
pub struct L2;

/// Negated inner product: `-dot(a, b)`.
///
/// Range `(-∞, ∞)`. Not a true metric. Negated so that HNSW's "smaller is
/// closer" convention holds : without the negation, the min-heap would order
/// candidates backwards.
#[derive(Debug, Clone, Copy, Default)]
pub struct InnerProduct;

impl Distance for Cosine {
    fn distance(&self, a: &Vector, b: &Vector) -> f32 {
        debug_assert_eq!(a.dim(), b.dim(), "Cosine: dimension mismatch");

        let xs = a.as_slice();
        let ys = b.as_slice();

        // Single-pass: dot, norm_left^2, norm_right^2 accumulated together.
        let mut dot_acc = f32x8::ZERO;
        let mut left_norm_acc = f32x8::ZERO;
        let mut right_norm_acc = f32x8::ZERO;

        let mut iter_a = xs.chunks_exact(8);
        let mut iter_b = ys.chunks_exact(8);

        for (a_chunk, b_chunk) in iter_a.by_ref().zip(iter_b.by_ref()) {
            let av = load8(a_chunk);
            let bv = load8(b_chunk);
            dot_acc += av * bv;
            left_norm_acc += av * av;
            right_norm_acc += bv * bv;
        }

        let mut dot: f32 = dot_acc.reduce_add();
        let mut left_sq: f32 = left_norm_acc.reduce_add();
        let mut right_sq: f32 = right_norm_acc.reduce_add();

        for (&x, &y) in iter_a.remainder().iter().zip(iter_b.remainder().iter()) {
            dot += x * y;
            left_sq += x * x;
            right_sq += y * y;
        }

        let norm_a = left_sq.sqrt();
        let norm_b = right_sq.sqrt();

        // Treat a zero vector as orthogonal to everything : avoids NaN from 0/0.
        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0;
        }

        1.0 - dot / (norm_a * norm_b)
    }

    fn name(&self) -> &'static str {
        "cosine"
    }
}

impl Distance for L2 {
    fn distance(&self, a: &Vector, b: &Vector) -> f32 {
        debug_assert_eq!(a.dim(), b.dim(), "L2: dimension mismatch");

        let xs = a.as_slice();
        let ys = b.as_slice();

        let mut acc = f32x8::ZERO;
        let mut iter_a = xs.chunks_exact(8);
        let mut iter_b = ys.chunks_exact(8);

        for (a_chunk, b_chunk) in iter_a.by_ref().zip(iter_b.by_ref()) {
            let av = load8(a_chunk);
            let bv = load8(b_chunk);
            let d = av - bv;
            acc += d * d;
        }

        let mut sum: f32 = acc.reduce_add();
        for (&x, &y) in iter_a.remainder().iter().zip(iter_b.remainder().iter()) {
            let d = x - y;
            sum += d * d;
        }

        sum.sqrt()
    }

    fn name(&self) -> &'static str {
        "l2"
    }
}

impl Distance for InnerProduct {
    fn distance(&self, a: &Vector, b: &Vector) -> f32 {
        debug_assert_eq!(a.dim(), b.dim(), "InnerProduct: dimension mismatch");

        let xs = a.as_slice();
        let ys = b.as_slice();

        let mut acc = f32x8::ZERO;
        let mut iter_a = xs.chunks_exact(8);
        let mut iter_b = ys.chunks_exact(8);

        for (a_chunk, b_chunk) in iter_a.by_ref().zip(iter_b.by_ref()) {
            let av = load8(a_chunk);
            let bv = load8(b_chunk);
            acc += av * bv;
        }

        let mut dot: f32 = acc.reduce_add();
        for (&x, &y) in iter_a.remainder().iter().zip(iter_b.remainder().iter()) {
            dot += x * y;
        }

        -dot
    }

    fn name(&self) -> &'static str {
        "inner_product"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    fn v(data: Vec<f32>) -> Vector {
        Vector::try_new(data).expect("test vector should be valid")
    }

    #[test]
    fn names() {
        assert_eq!(Cosine.name(), "cosine");
        assert_eq!(L2.name(), "l2");
        assert_eq!(InnerProduct.name(), "inner_product");
    }

    #[test]
    fn l2_identical_is_zero() {
        let a = v(vec![1.0, 2.0, 3.0]);
        assert!(approx(L2.distance(&a, &a), 0.0));
    }

    #[test]
    fn l2_three_four_five_triangle() {
        let a = v(vec![3.0, 0.0]);
        let b = v(vec![0.0, 4.0]);
        assert!(approx(L2.distance(&a, &b), 5.0));
    }

    #[test]
    fn l2_orthogonal_unit_vectors() {
        let a = v(vec![1.0, 0.0]);
        let b = v(vec![0.0, 1.0]);
        assert!(approx(L2.distance(&a, &b), 2.0_f32.sqrt()));
    }

    #[test]
    fn cosine_identical_is_zero() {
        let a = v(vec![1.0, 2.0, 3.0]);
        assert!(approx(Cosine.distance(&a, &a), 0.0));
    }

    #[test]
    fn cosine_orthogonal_is_one() {
        let a = v(vec![1.0, 0.0]);
        let b = v(vec![0.0, 1.0]);
        assert!(approx(Cosine.distance(&a, &b), 1.0));
    }

    #[test]
    fn cosine_opposite_is_two() {
        let a = v(vec![1.0, 0.0]);
        let b = v(vec![-1.0, 0.0]);
        assert!(approx(Cosine.distance(&a, &b), 2.0));
    }

    #[test]
    fn cosine_zero_vector_returns_one() {
        let zero = v(vec![0.0, 0.0]);
        let nonzero = v(vec![1.0, 1.0]);
        assert!(approx(Cosine.distance(&zero, &nonzero), 1.0));
    }

    #[test]
    fn inner_product_orthogonal_is_zero() {
        let a = v(vec![1.0, 0.0]);
        let b = v(vec![0.0, 1.0]);
        assert!(approx(InnerProduct.distance(&a, &b), 0.0));
    }

    #[test]
    fn inner_product_identical_is_negative_norm_squared() {
        // dot(a, a) = 1 + 4 + 9 = 14, negated => -14
        let a = v(vec![1.0, 2.0, 3.0]);
        assert!(approx(InnerProduct.distance(&a, &a), -14.0));
    }
}

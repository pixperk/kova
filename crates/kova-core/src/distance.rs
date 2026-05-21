//! Distance metrics over [`Vector`].
//!
//! The [`Distance`] trait is the abstraction every index, planner, and benchmark
//! uses to score how close two vectors are. Three concrete implementations live
//! here: [`Cosine`], [`L2`], and [`InnerProduct`]. All return `f32` where
//! **smaller means closer**, matching what HNSW's min-heaps expect.

use crate::Vector;

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

        let dot: f32 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = xs.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = ys.iter().map(|y| y * y).sum::<f32>().sqrt();

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

        a.as_slice()
            .iter()
            .zip(b.as_slice().iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum::<f32>()
            .sqrt()
    }

    fn name(&self) -> &'static str {
        "l2"
    }
}

impl Distance for InnerProduct {
    fn distance(&self, a: &Vector, b: &Vector) -> f32 {
        debug_assert_eq!(a.dim(), b.dim(), "InnerProduct: dimension mismatch");

        let dot: f32 = a
            .as_slice()
            .iter()
            .zip(b.as_slice().iter())
            .map(|(x, y)| x * y)
            .sum();

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

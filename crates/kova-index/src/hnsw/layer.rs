//! Level assignment for new HNSW nodes : the lottery that decides each
//! node's top layer.
//!
//! Implements `floor(-ln(uniform(0,1)) * m_l)`, the canonical level
//! distribution from Malkov & Yashunin (2016). With `m_l = 1 / ln(M)`,
//! ~93% of nodes land on layer 0, ~6% on layer 1, exponentially fewer above.

use rand::{Rng, RngExt};

/// Draws a top layer for a new node, given `m_l = 1 / ln(M)`.
///
/// Implements the formula `floor(-ln(uniform(0,1)) * m_l)`.
///
/// The function uses `1 - rng.random()` to avoid `u = 0` (which would make
/// `ln(u) = -inf`); this gives `u` in `(0, 1]` and a well-defined result.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss, // -ln(u) is non-negative for u in (0, 1]; floor stays >= 0
)]
pub(crate) fn random_level<R: Rng + ?Sized>(m_l: f64, rng: &mut R) -> usize {
    let u: f64 = 1.0 - rng.random::<f64>(); // (0, 1]
    (-u.ln() * m_l).floor() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn deterministic_with_seed() {
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);
        let m_l = 1.0 / 16.0_f64.ln();

        for _ in 0..100 {
            assert_eq!(random_level(m_l, &mut rng1), random_level(m_l, &mut rng2));
        }
    }

    #[test]
    fn distribution_skews_to_zero() {
        let mut rng = StdRng::seed_from_u64(42);
        let m_l = 1.0 / 16.0_f64.ln();
        let mut counts = std::collections::HashMap::<usize, u32>::new();

        for _ in 0..10_000 {
            *counts.entry(random_level(m_l, &mut rng)).or_insert(0) += 1;
        }

        // ~93% at layer 0, exponentially less at higher layers.
        let level_0 = counts.get(&0).copied().unwrap_or(0);
        assert!(
            level_0 > 9_000,
            "level 0 count was {level_0}, expected >9000"
        );

        // Higher levels exist.
        assert!(counts.keys().any(|&k| k >= 2), "no nodes at layer >= 2");
    }

    #[test]
    fn larger_m_l_produces_higher_max_level() {
        let mut rng = StdRng::seed_from_u64(42);
        let small_m_l = 1.0 / 64.0_f64.ln(); // m=64, small m_l
        let large_m_l = 1.0 / 4.0_f64.ln(); // m=4, large m_l

        let max_small: usize = (0..1_000)
            .map(|_| random_level(small_m_l, &mut rng))
            .max()
            .unwrap();
        let max_large: usize = (0..1_000)
            .map(|_| random_level(large_m_l, &mut rng))
            .max()
            .unwrap();

        assert!(
            max_large > max_small,
            "larger m_l should yield deeper graphs (got small={max_small}, large={max_large})"
        );
    }
}

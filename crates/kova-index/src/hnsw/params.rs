//! Tuning parameters for the HNSW index. Defaults from Malkov & Yashunin (2016).

/// Tuning parameters for an HNSW index.
///
/// Fields are `pub` so callers can override after `new`. Use [`Self::new`]
/// to derive defaults from a chosen `m`, or [`Default`] for `m = 16`.
#[derive(Debug, Clone, Copy)]
pub struct HnswParams {
    /// Max neighbours per node above layer 0.
    pub m: usize,
    /// Max neighbours per node at layer 0 (typically `2 * m`).
    pub m_max0: usize,
    /// Candidate pool size during insert. Locked at build time.
    pub ef_construction: usize,
    /// Candidate pool size during search. Tunable per query.
    pub ef_search: usize,
    /// Level-distribution scaling, `1 / ln(m)`.
    pub m_l: f64,
}

impl HnswParams {
    /// Build params with a chosen `m`, deriving defaults for the rest.
    ///
    /// # Panics
    /// Panics if `m <= 1` (would yield a chain, or `ln(m)` undefined).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn new(m: usize) -> Self {
        assert!(m > 1, "HNSW M must be > 1");
        Self {
            m,
            m_max0: m * 2,
            ef_construction: 200,
            ef_search: 50,
            m_l: 1.0 / (m as f64).ln(),
        }
    }

    /// Degree cap at a given layer: `m_max0` at layer 0, `m` above.
    #[must_use]
    pub const fn m_for_layer(&self, layer: usize) -> usize {
        if layer == 0 { self.m_max0 } else { self.m }
    }
}

impl Default for HnswParams {
    fn default() -> Self {
        Self::new(16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults() {
        let p = HnswParams::default();
        assert_eq!(p.m, 16);
        assert_eq!(p.m_max0, 32);
        assert_eq!(p.ef_construction, 200);
        assert_eq!(p.ef_search, 50);
        assert!((p.m_l - 1.0 / 16.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn new_derives_m_max0() {
        assert_eq!(HnswParams::new(32).m_max0, 64);
    }

    #[test]
    fn m_for_layer_zero_uses_m_max0() {
        assert_eq!(HnswParams::new(16).m_for_layer(0), 32);
    }

    #[test]
    fn m_for_layer_above_zero_uses_m() {
        let p = HnswParams::new(16);
        assert_eq!(p.m_for_layer(1), 16);
        assert_eq!(p.m_for_layer(100), 16);
    }

    #[test]
    #[should_panic(expected = "M must be > 1")]
    fn panics_on_m_one() {
        let _ = HnswParams::new(1);
    }
}

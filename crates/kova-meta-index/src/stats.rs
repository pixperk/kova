//! Column statistics for selectivity estimation on unindexed predicates.
//!
//! The planner's `ShardEstimator` consults the [`IndexCatalog`] for
//! atoms backed by a real index (`bitmap.len()` is exact and O(1)),
//! and falls back to a full metadata scan for atoms it can't
//! answer. The fallback is O(N) every time the planner runs, which
//! is fine for tiny shards and painful for big ones.
//!
//! This module is the third option : per-field summary statistics
//! kept up to date at checkpoint time, persisted alongside the
//! catalog, and consulted by the estimator for unindexed atoms.
//! For the planner's "how selective is this predicate ?" question,
//! a histogram lookup runs in microseconds where the scan path
//! takes milliseconds.
//!
//! ## What lives here
//!
//! - [`ColumnStats`] : the per-field summary. Row count, null
//!   count, distinct count, plus a kind-specific payload that
//!   matches the column's value shape (numeric, string, bool,
//!   array-of-strings).
//! - [`StatsCatalog`] : a `HashMap<field, ColumnStats>` with a
//!   single `selectivity(field, atom)` entry point that
//!   dispatches to the right per-field math.
//! - [`HistogramBucket`] : the equi-depth bucket type for numeric
//!   columns.
//!
//! ## What does NOT live here
//!
//! - Stats collection (building a `ColumnStats` from a stream of
//!   `(VectorId, Value)`) lands in a follow-up slice.
//! - Persistence (encode/decode/load) lands with the collection
//!   slice so the on-disk format is decided once.
//! - Integration with `ShardEstimator` lands last, after both
//!   collection and persistence are in.
//!
//! This slice is pure types + math + tests. Easy to verify in
//! isolation, easy to swap out if the estimation model changes.
//!
//! [`IndexCatalog`]: crate::IndexCatalog

// Selectivity math is ratios of counts. `u64` / `i64` → `f64`
// loses precision past 2^53, but our counts (rows, occurrences,
// histogram bucket counts) live well below that and the precision
// loss is irrelevant to a selectivity in [0, 1].
#![allow(clippy::cast_precision_loss)]

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use kova_core::{Metadata, Value};
use serde::{Deserialize, Serialize};

use crate::error::KovaMetaIndexError;
use crate::{CmpOp, IndexAtom};

/// Magic header on every stats file. 8 bytes, ASCII for `KOVASTA1`.
const STATS_MAGIC: &[u8; 8] = b"KOVASTA1";

/// Bumped when the on-disk stats layout changes incompatibly.
const STATS_FORMAT_VERSION: u32 = 1;

/// Fixed header bytes : magic + version.
const STATS_HEADER_LEN: usize = STATS_MAGIC.len() + std::mem::size_of::<u32>();

/// Default number of equi-depth histogram buckets for numeric
/// columns. Tuned for "small enough to bench in microseconds, large
/// enough to estimate range selectivities within a few percent."
pub const DEFAULT_HISTOGRAM_BUCKETS: usize = 20;

/// Default top-K size for string columns and array element
/// frequency tables. The most common K values are captured exactly ;
/// the tail uses uniform-over-remaining-distinct estimation.
pub const DEFAULT_TOP_K: usize = 16;

/// Per-shard catalog of column statistics. Keyed by field name,
/// just like [`crate::IndexCatalog`] ; both catalogs are read
/// side-by-side by the planner's estimator.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct StatsCatalog {
    fields: HashMap<String, ColumnStats>,
}

impl StatsCatalog {
    /// Construct an empty stats catalog.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace (or insert) the stats for `field`.
    pub fn put(&mut self, field: &str, stats: ColumnStats) {
        self.fields.insert(field.to_string(), stats);
    }

    /// Remove the stats for `field` if present.
    pub fn remove(&mut self, field: &str) {
        self.fields.remove(field);
    }

    /// Borrow the stats for `field`, or `None` if no stats were
    /// collected for it.
    #[must_use]
    pub fn get(&self, field: &str) -> Option<&ColumnStats> {
        self.fields.get(field)
    }

    /// Estimate the selectivity of `atom` against `field`. Returns
    /// a value in `[0.0, 1.0]` on a hit, or `None` when either
    /// `field` has no stats or the atom shape doesn't fit the
    /// column's kind. Callers (the estimator) treat `None` as
    /// "stats can't help, fall back to scan."
    #[must_use]
    pub fn selectivity(&self, field: &str, atom: &IndexAtom) -> Option<f64> {
        self.fields.get(field).and_then(|s| s.selectivity(atom))
    }

    /// Iterate the names of all fields with stats.
    pub fn fields(&self) -> impl Iterator<Item = &str> {
        self.fields.keys().map(String::as_str)
    }

    /// How many fields have stats.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// True if no fields have stats.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Encode the catalog into a self-describing byte buffer with
    /// the same shape `IndexCatalog` uses :
    ///
    /// ```text
    /// +----------+----------+-------------------------------+
    /// | magic[8] | ver[u32] | bincode( StatsCatalog )       |
    /// +----------+----------+-------------------------------+
    /// ```
    ///
    /// The storage layer wraps this in an atomic write (tmp +
    /// fsync + rename + dirsync) so observers see either the
    /// whole new file or the previous one, never a partial.
    ///
    /// # Errors
    /// Returns [`KovaMetaIndexError::Decode`] (the bincode error
    /// variant covers both directions) if the payload can't be
    /// serialised, which is effectively never for this shape.
    pub fn encode(&self) -> Result<Vec<u8>, KovaMetaIndexError> {
        let payload = bincode::serialize(self)?;
        let mut buf = Vec::with_capacity(STATS_HEADER_LEN + payload.len());
        buf.extend_from_slice(STATS_MAGIC);
        buf.extend_from_slice(&STATS_FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&payload);
        Ok(buf)
    }

    /// Decode a catalog from a byte buffer produced by
    /// [`Self::encode`]. Validates the magic header and version
    /// before handing the rest to bincode.
    ///
    /// # Errors
    /// - [`KovaMetaIndexError::Truncated`] if `bytes` is shorter
    ///   than the fixed header.
    /// - [`KovaMetaIndexError::BadMagic`] if the magic bytes don't
    ///   match.
    /// - [`KovaMetaIndexError::UnsupportedVersion`] if the version
    ///   field doesn't match this build's expected version.
    /// - [`KovaMetaIndexError::Decode`] if bincode rejects the
    ///   payload.
    pub fn decode(bytes: &[u8]) -> Result<Self, KovaMetaIndexError> {
        if bytes.len() < STATS_HEADER_LEN {
            return Err(KovaMetaIndexError::Truncated {
                bytes: bytes.len(),
                min: STATS_HEADER_LEN,
            });
        }
        if &bytes[..STATS_MAGIC.len()] != STATS_MAGIC {
            return Err(KovaMetaIndexError::BadMagic);
        }
        let ver_bytes: [u8; 4] = bytes[STATS_MAGIC.len()..STATS_HEADER_LEN]
            .try_into()
            .expect("4-byte slice");
        let version = u32::from_le_bytes(ver_bytes);
        if version != STATS_FORMAT_VERSION {
            return Err(KovaMetaIndexError::UnsupportedVersion {
                expected: STATS_FORMAT_VERSION,
                got: version,
            });
        }
        let cat: StatsCatalog = bincode::deserialize(&bytes[STATS_HEADER_LEN..])?;
        Ok(cat)
    }

    /// Read + decode the catalog at `path`. Returns `Ok(None)` if
    /// the file doesn't exist (fresh shard with no persisted stats).
    ///
    /// # Errors
    /// All variants of [`KovaMetaIndexError`] that
    /// [`Self::decode`] can produce, plus
    /// [`KovaMetaIndexError::Io`] for read failures.
    pub fn load(path: &Path) -> Result<Option<Self>, KovaMetaIndexError> {
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(path)?;
        Ok(Some(Self::decode(&bytes)?))
    }
}

/// Summary statistics for one column.
///
/// `row_count + null_count = total live rows in the shard` ;
/// `distinct_count` is the number of unique values observed in
/// `row_count` rows. The kind-specific payload carries the data
/// shape needed to estimate selectivity for that column's atoms.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColumnStats {
    /// Rows that observed this field with a value.
    pub row_count: u64,
    /// Rows where this field was absent.
    pub null_count: u64,
    /// Distinct values across the observed rows. Counted exactly
    /// today (via `HashSet`) ; an approximate counter like
    /// `HyperLogLog` is the natural swap if memory bites at scale.
    pub distinct_count: u64,
    /// Kind-specific payload.
    pub kind: ColumnStatsKind,
}

/// The data-shape-specific portion of `ColumnStats`. Each variant
/// carries exactly the summary needed to estimate atoms on that
/// kind of column.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ColumnStatsKind {
    /// Numeric column (`I64` or `F64` values). Range queries
    /// estimate via the equi-depth histogram ; equality estimates
    /// fall back to `1 / distinct_count` since per-bucket NDV
    /// isn't tracked.
    Numeric {
        /// Smallest observed value (cast to f64).
        min: f64,
        /// Largest observed value (cast to f64).
        max: f64,
        /// Equi-depth histogram, sorted by `lo` ascending.
        histogram: Vec<HistogramBucket>,
    },
    /// String column. The top-K most frequent values are captured
    /// exactly ; values outside the top-K are estimated uniformly
    /// over the remaining distinct values.
    String {
        /// Top-K values and their observed counts.
        top_k: Vec<(String, u64)>,
    },
    /// Boolean column. Trivial to summarise.
    Bool {
        /// Rows observed with value `true`.
        true_count: u64,
        /// Rows observed with value `false`.
        false_count: u64,
    },
    /// Array column (treated as a bag of element values for
    /// `ArrayContains` queries). The top-K most frequent ELEMENT
    /// values are captured ; average array length helps estimate
    /// the tail.
    Array {
        /// Top-K element values and their observed counts (each
        /// occurrence in any row's array counts).
        element_top_k: Vec<(String, u64)>,
        /// Average array length across `row_count` rows.
        avg_array_len: f64,
    },
    /// Column where values had inconsistent kinds across rows
    /// (e.g., sometimes string, sometimes int). No useful stats
    /// shape ; selectivity returns `None` for everything except
    /// `IsNotNull`.
    Mixed,
}

/// One bucket in an equi-depth histogram. Boundaries are inclusive
/// at `lo` and exclusive at `hi`, except for the final bucket
/// which is inclusive at both ends so `max` falls inside.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HistogramBucket {
    /// Lower bound (inclusive).
    pub lo: f64,
    /// Upper bound (exclusive, except for the last bucket).
    pub hi: f64,
    /// Number of rows whose value lies in `[lo, hi)`.
    pub count: u64,
}

impl ColumnStats {
    /// Estimate the selectivity of `atom` against this column.
    ///
    /// Returns `None` when the atom shape can't be answered from
    /// the kind on this column (e.g., `Cmp(Lt, ...)` on a String
    /// column, or `ArrayContains` on a Numeric column). Returns
    /// `Some(f)` with `f` clamped to `[0.0, 1.0]` otherwise.
    ///
    /// `Mixed` columns return `None` for every atom except
    /// `IsNotNull` (which only needs `row_count` and `null_count`).
    #[must_use]
    pub fn selectivity(&self, atom: &IndexAtom) -> Option<f64> {
        // IsNotNull doesn't care about kind — handle it first so
        // empty / Mixed stats still answer it.
        if matches!(atom, IndexAtom::IsNotNull) {
            return Some(self.selectivity_not_null());
        }
        if self.row_count == 0 {
            // No observed values. Every value-shaped atom is 0%
            // selective by definition.
            return Some(0.0);
        }
        let s = match atom {
            IndexAtom::Eq(v) => self.selectivity_eq(v)?,
            IndexAtom::Cmp(op, v) => self.selectivity_cmp(*op, v)?,
            IndexAtom::In(vs) => self.selectivity_in(vs)?,
            IndexAtom::Between(lo, hi) => self.selectivity_between(lo, hi)?,
            IndexAtom::ArrayContains(v) => self.selectivity_array_contains(v)?,
            IndexAtom::IsNotNull => unreachable!("handled above"),
        };
        Some(s.clamp(0.0, 1.0))
    }

    /// P(field is present) = `row_count / total_rows`.
    fn selectivity_not_null(&self) -> f64 {
        let total = self.row_count + self.null_count;
        if total == 0 {
            0.0
        } else {
            self.row_count as f64 / total as f64
        }
    }

    fn selectivity_eq(&self, v: &Value) -> Option<f64> {
        match (&self.kind, v) {
            (ColumnStatsKind::Numeric { min, max, .. }, _) => {
                let n = value_to_f64(v)?;
                if n < *min || n > *max {
                    return Some(0.0);
                }
                if self.distinct_count == 0 {
                    return Some(0.0);
                }
                // Uniform-over-distinct approximation : we don't
                // track per-bucket NDV, so probability of equality
                // is roughly `1 / NDV` when v is in range.
                Some(1.0 / self.distinct_count as f64)
            }
            (ColumnStatsKind::String { top_k }, Value::String(s)) => {
                if let Some((_, count)) = top_k.iter().find(|(k, _)| k == s) {
                    return Some(*count as f64 / self.row_count as f64);
                }
                // Tail value : uniform over (distinct_count - K)
                // remaining distinct values.
                let top_k_sum: u64 = top_k.iter().map(|(_, c)| *c).sum();
                let tail_rows = self.row_count.saturating_sub(top_k_sum);
                let tail_distinct = self.distinct_count.saturating_sub(top_k.len() as u64);
                if tail_distinct == 0 || tail_rows == 0 {
                    return Some(0.0);
                }
                let per_value = tail_rows as f64 / tail_distinct as f64;
                Some(per_value / self.row_count as f64)
            }
            (
                ColumnStatsKind::Bool {
                    true_count,
                    false_count,
                },
                Value::Bool(b),
            ) => {
                let total = (*true_count + *false_count) as f64;
                if total == 0.0 {
                    return Some(0.0);
                }
                let c = if *b { *true_count } else { *false_count };
                Some(c as f64 / total)
            }
            _ => None,
        }
    }

    fn selectivity_cmp(&self, op: CmpOp, v: &Value) -> Option<f64> {
        // Ne only depends on Eq, so handle it first : that lets
        // String / Bool kinds answer Ne even though strict
        // less-than isn't defined on them.
        if matches!(op, CmpOp::Ne) {
            let eq = self.selectivity_eq(v)?;
            return Some(1.0 - eq);
        }
        // Lt / Le / Gt / Ge all derive from the strict less-than
        // primitive plus equality at the boundary.
        let lt = self.selectivity_lt_strict(v)?;
        // Eq may not exist for the kind (e.g., Numeric Eq returns
        // some answer, Mixed returns None) ; treat None as 0 for
        // the cmp composition so we don't drop the whole estimate.
        let eq = self.selectivity_eq(v).unwrap_or(0.0);
        let s = match op {
            CmpOp::Lt => lt,
            CmpOp::Le => lt + eq,
            CmpOp::Gt => 1.0 - lt - eq,
            CmpOp::Ge => 1.0 - lt,
            CmpOp::Ne => unreachable!("handled above"),
        };
        Some(s)
    }

    /// `selectivity(field < v)`. Strict less-than.
    fn selectivity_lt_strict(&self, v: &Value) -> Option<f64> {
        match &self.kind {
            ColumnStatsKind::Numeric {
                min,
                max,
                histogram,
            } => {
                let n = value_to_f64(v)?;
                if n <= *min {
                    return Some(0.0);
                }
                if n > *max {
                    return Some(1.0);
                }
                let mut count = 0.0_f64;
                for b in histogram {
                    if b.hi <= n {
                        count += b.count as f64;
                    } else if b.lo >= n {
                        break;
                    } else {
                        // Boundary bucket : linearly interpolate
                        // assuming uniform density within the
                        // bucket. The textbook approximation.
                        let width = (b.hi - b.lo).max(f64::EPSILON);
                        let frac = (n - b.lo) / width;
                        count += b.count as f64 * frac;
                        break;
                    }
                }
                Some(count / self.row_count as f64)
            }
            ColumnStatsKind::Bool {
                true_count,
                false_count,
            } => {
                let total = (*true_count + *false_count) as f64;
                if total == 0.0 {
                    return Some(0.0);
                }
                match v {
                    Value::Bool(true) => {
                        // false < true : selectivity is P(false)
                        Some(*false_count as f64 / total)
                    }
                    Value::Bool(false) => Some(0.0),
                    _ => None,
                }
            }
            // String comparisons (other than equality / Ne) aren't
            // supported by the catalog either ; return None so the
            // estimator falls back to scan.
            ColumnStatsKind::String { .. }
            | ColumnStatsKind::Array { .. }
            | ColumnStatsKind::Mixed => None,
        }
    }

    fn selectivity_between(&self, lo: &Value, hi: &Value) -> Option<f64> {
        let lo_n = value_to_f64(lo)?;
        let hi_n = value_to_f64(hi)?;
        if lo_n > hi_n {
            return Some(0.0);
        }
        // Between is inclusive both sides :
        //   P(lo <= field <= hi) = P(field <= hi) - P(field < lo)
        let le_hi = {
            let lt = self.selectivity_lt_strict(hi)?;
            let eq = self.selectivity_eq(hi).unwrap_or(0.0);
            lt + eq
        };
        let lt_lo = self.selectivity_lt_strict(lo)?;
        Some(le_hi - lt_lo)
    }

    fn selectivity_in(&self, vs: &[Value]) -> Option<f64> {
        if vs.is_empty() {
            return Some(0.0);
        }
        let mut total = 0.0_f64;
        for v in vs {
            // If ANY value in the set is wrong-kind, the IN can't
            // be estimated. Bail.
            total += self.selectivity_eq(v)?;
        }
        // Cap at 1.0 : duplicates in the set could push the sum
        // above 1, and selectivities live in [0, 1].
        Some(total.min(1.0))
    }

    fn selectivity_array_contains(&self, v: &Value) -> Option<f64> {
        let (
            ColumnStatsKind::Array {
                element_top_k,
                avg_array_len,
            },
            Value::String(s),
        ) = (&self.kind, v)
        else {
            return None;
        };
        if let Some((_, count)) = element_top_k.iter().find(|(k, _)| k == s) {
            return Some(*count as f64 / self.row_count as f64);
        }
        // Tail estimate. Each row contains roughly `avg_array_len`
        // elements ; the K most common cover `top_k_sum`
        // occurrences ; the rest is split across `distinct_count - K`
        // distinct values.
        let top_k_sum: u64 = element_top_k.iter().map(|(_, c)| *c).sum();
        let total_occurrences = (*avg_array_len * self.row_count as f64).max(top_k_sum as f64);
        let tail_occurrences = total_occurrences - top_k_sum as f64;
        let tail_distinct = self
            .distinct_count
            .saturating_sub(element_top_k.len() as u64);
        if tail_distinct == 0 || tail_occurrences <= 0.0 {
            return Some(0.0);
        }
        let per_value = tail_occurrences / tail_distinct as f64;
        // Probability that a given row contains the tail value :
        // roughly `per_value / row_count`. Linear approximation
        // for low-occurrence tail values, which is when this branch
        // fires anyway.
        Some((per_value / self.row_count as f64).min(1.0))
    }
}

/// Convert a numeric [`Value`] to `f64` for histogram math.
/// Returns `None` for non-numeric variants (caller treats that as
/// "stats can't help with this atom").
fn value_to_f64(v: &Value) -> Option<f64> {
    match v {
        // i64 → f64 loses precision past 2^53 but realistic data
        // (years, scores, counts) doesn't approach that.
        Value::I64(n) => Some(*n as f64),
        Value::F64(f) => Some(*f),
        _ => None,
    }
}

// =========================================================================
// Builder : turn a stream of `Metadata` observations into a `StatsCatalog`.
// =========================================================================

/// Incremental [`StatsCatalog`] builder. Observe one row's metadata
/// at a time ; the builder routes each `(field, value)` to a
/// per-field sub-builder that tracks the kind-specific summary.
///
/// On [`Self::finish`], each sub-builder produces a
/// [`ColumnStats`]. Fields that observed values of inconsistent
/// kinds across rows collapse to [`ColumnStatsKind::Mixed`] and
/// only retain `row_count` (enough to answer `IsNotNull`).
///
/// Used at checkpoint time : after vacuum, walk every live row's
/// metadata once, finish, persist via [`StatsCatalog::encode`].
pub struct StatsBuilder {
    fields: HashMap<String, ColumnBuilder>,
    total_rows: u64,
    histogram_buckets: usize,
    top_k: usize,
}

impl Default for StatsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl StatsBuilder {
    /// Construct a builder with the default histogram bucket count
    /// ([`DEFAULT_HISTOGRAM_BUCKETS`]) and top-K size
    /// ([`DEFAULT_TOP_K`]).
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(DEFAULT_HISTOGRAM_BUCKETS, DEFAULT_TOP_K)
    }

    /// Construct a builder with explicit histogram + top-K limits.
    #[must_use]
    pub fn with_config(histogram_buckets: usize, top_k: usize) -> Self {
        Self {
            fields: HashMap::new(),
            total_rows: 0,
            histogram_buckets,
            top_k,
        }
    }

    /// Observe one row's metadata. Increments `total_rows`, and
    /// for every `(field, value)` in the bag, advances that
    /// field's sub-builder by one value.
    ///
    /// Fields missing from the bag don't get touched. Their
    /// `null_count` is reconstructed at finish time as
    /// `total_rows - row_count`.
    pub fn observe(&mut self, metadata: &Metadata) {
        self.total_rows += 1;
        for (field, value) in metadata {
            let cb = self
                .fields
                .entry(field.clone())
                .or_insert(ColumnBuilder::Empty);
            cb.observe(value);
        }
    }

    /// Force-track `field` even if no row has observed it. Useful
    /// when the caller knows the field exists in the schema and
    /// wants `null_count = total_rows` to surface explicitly.
    pub fn track_field(&mut self, field: &str) {
        self.fields
            .entry(field.to_string())
            .or_insert(ColumnBuilder::Empty);
    }

    /// Finalise into a [`StatsCatalog`]. Each tracked field gets
    /// its `ColumnStats` ; fields nobody touched stay out.
    #[must_use]
    pub fn finish(self) -> StatsCatalog {
        let mut catalog = StatsCatalog::new();
        let total = self.total_rows;
        let buckets = self.histogram_buckets;
        let top_k = self.top_k;
        for (field, builder) in self.fields {
            let stats = builder.finish(buckets, top_k, total);
            catalog.put(&field, stats);
        }
        catalog
    }
}

/// Per-field state machine. Starts `Empty`, transitions to a
/// kind-specific sub-builder on the first observed value, and
/// collapses to `Mixed` if a later value disagrees on kind.
enum ColumnBuilder {
    Empty,
    Numeric(NumericBuilder),
    String(StringBuilder),
    Bool(BoolBuilder),
    Array(ArrayBuilder),
    Mixed { row_count: u64 },
}

impl ColumnBuilder {
    fn observe(&mut self, value: &Value) {
        // Take ownership of the current state, advance, replace.
        // `mem::replace` is the standard way to drive a state
        // machine where the next variant depends on the current.
        let prev = std::mem::replace(self, ColumnBuilder::Empty);
        *self = prev.advance(value);
    }

    fn advance(self, value: &Value) -> Self {
        match (self, value) {
            (ColumnBuilder::Empty, Value::I64(_) | Value::F64(_)) => {
                let mut b = NumericBuilder::default();
                b.observe(value);
                ColumnBuilder::Numeric(b)
            }
            (ColumnBuilder::Empty, Value::String(_)) => {
                let mut b = StringBuilder::default();
                b.observe(value);
                ColumnBuilder::String(b)
            }
            (ColumnBuilder::Empty, Value::Bool(_)) => {
                let mut b = BoolBuilder::default();
                b.observe(value);
                ColumnBuilder::Bool(b)
            }
            (ColumnBuilder::Empty, Value::Array(_)) => {
                let mut b = ArrayBuilder::default();
                b.observe(value);
                ColumnBuilder::Array(b)
            }
            (ColumnBuilder::Empty, Value::Map(_)) => {
                // Maps have no useful stats shape.
                ColumnBuilder::Mixed { row_count: 1 }
            }
            (ColumnBuilder::Numeric(mut b), Value::I64(_) | Value::F64(_)) => {
                b.observe(value);
                ColumnBuilder::Numeric(b)
            }
            (ColumnBuilder::Numeric(b), _) => ColumnBuilder::Mixed {
                row_count: b.row_count + 1,
            },
            (ColumnBuilder::String(mut b), Value::String(_)) => {
                b.observe(value);
                ColumnBuilder::String(b)
            }
            (ColumnBuilder::String(b), _) => ColumnBuilder::Mixed {
                row_count: b.row_count + 1,
            },
            (ColumnBuilder::Bool(mut b), Value::Bool(_)) => {
                b.observe(value);
                ColumnBuilder::Bool(b)
            }
            (ColumnBuilder::Bool(b), _) => ColumnBuilder::Mixed {
                row_count: b.true_count + b.false_count + 1,
            },
            (ColumnBuilder::Array(mut b), Value::Array(_)) => {
                b.observe(value);
                ColumnBuilder::Array(b)
            }
            (ColumnBuilder::Array(b), _) => ColumnBuilder::Mixed {
                row_count: b.row_count + 1,
            },
            (ColumnBuilder::Mixed { row_count }, _) => ColumnBuilder::Mixed {
                row_count: row_count + 1,
            },
        }
    }

    fn finish(self, hist_buckets: usize, top_k: usize, total_rows: u64) -> ColumnStats {
        match self {
            ColumnBuilder::Empty => ColumnStats {
                row_count: 0,
                null_count: total_rows,
                distinct_count: 0,
                kind: ColumnStatsKind::Mixed,
            },
            ColumnBuilder::Numeric(b) => b.finish(hist_buckets, total_rows),
            ColumnBuilder::String(b) => b.finish(top_k, total_rows),
            ColumnBuilder::Bool(b) => b.finish(total_rows),
            ColumnBuilder::Array(b) => b.finish(top_k, total_rows),
            ColumnBuilder::Mixed { row_count } => ColumnStats {
                row_count,
                null_count: total_rows.saturating_sub(row_count),
                distinct_count: 0,
                kind: ColumnStatsKind::Mixed,
            },
        }
    }
}

#[derive(Default)]
struct NumericBuilder {
    values: Vec<f64>,
    distinct: HashSet<u64>, // bit-cast f64 for hashing
    row_count: u64,
}

impl NumericBuilder {
    fn observe(&mut self, v: &Value) {
        let Some(f) = value_to_f64(v) else { return };
        self.values.push(f);
        self.distinct.insert(f.to_bits());
        self.row_count += 1;
    }

    fn finish(mut self, hist_buckets: usize, total_rows: u64) -> ColumnStats {
        if self.values.is_empty() {
            return ColumnStats {
                row_count: 0,
                null_count: total_rows,
                distinct_count: 0,
                kind: ColumnStatsKind::Numeric {
                    min: 0.0,
                    max: 0.0,
                    histogram: vec![],
                },
            };
        }
        // f64 NaN comparison is partial ; in practice we never
        // observe NaN through Value::F64 because the binder doesn't
        // accept NaN literals, but the sort fallback to Equal keeps
        // us total.
        self.values
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let min = self.values[0];
        let max = *self.values.last().expect("non-empty checked");
        let histogram = build_equi_depth(&self.values, hist_buckets);
        ColumnStats {
            row_count: self.row_count,
            null_count: total_rows.saturating_sub(self.row_count),
            distinct_count: self.distinct.len() as u64,
            kind: ColumnStatsKind::Numeric {
                min,
                max,
                histogram,
            },
        }
    }
}

#[derive(Default)]
struct StringBuilder {
    freqs: HashMap<String, u64>,
    row_count: u64,
}

impl StringBuilder {
    fn observe(&mut self, v: &Value) {
        if let Value::String(s) = v {
            *self.freqs.entry(s.clone()).or_insert(0) += 1;
            self.row_count += 1;
        }
    }

    fn finish(self, top_k: usize, total_rows: u64) -> ColumnStats {
        let distinct_count = self.freqs.len() as u64;
        let mut entries: Vec<(String, u64)> = self.freqs.into_iter().collect();
        // Sort by count desc, then by key asc for determinism.
        entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        entries.truncate(top_k);
        ColumnStats {
            row_count: self.row_count,
            null_count: total_rows.saturating_sub(self.row_count),
            distinct_count,
            kind: ColumnStatsKind::String { top_k: entries },
        }
    }
}

#[derive(Default)]
struct BoolBuilder {
    true_count: u64,
    false_count: u64,
}

impl BoolBuilder {
    fn observe(&mut self, v: &Value) {
        if let Value::Bool(b) = v {
            if *b {
                self.true_count += 1;
            } else {
                self.false_count += 1;
            }
        }
    }

    fn finish(self, total_rows: u64) -> ColumnStats {
        let row_count = self.true_count + self.false_count;
        let distinct = u64::from(self.true_count > 0) + u64::from(self.false_count > 0);
        ColumnStats {
            row_count,
            null_count: total_rows.saturating_sub(row_count),
            distinct_count: distinct,
            kind: ColumnStatsKind::Bool {
                true_count: self.true_count,
                false_count: self.false_count,
            },
        }
    }
}

#[derive(Default)]
struct ArrayBuilder {
    element_freqs: HashMap<String, u64>,
    total_length: u64,
    row_count: u64,
}

impl ArrayBuilder {
    fn observe(&mut self, v: &Value) {
        if let Value::Array(elems) = v {
            self.row_count += 1;
            self.total_length += elems.len() as u64;
            for e in elems {
                if let Value::String(s) = e {
                    *self.element_freqs.entry(s.clone()).or_insert(0) += 1;
                }
                // Non-string array elements are ignored : the
                // catalog only indexes string-element arrays
                // (see InvertedIndex), so the stats follow suit.
            }
        }
    }

    fn finish(self, top_k: usize, total_rows: u64) -> ColumnStats {
        let distinct_count = self.element_freqs.len() as u64;
        let avg_array_len = if self.row_count == 0 {
            0.0
        } else {
            self.total_length as f64 / self.row_count as f64
        };
        let mut entries: Vec<(String, u64)> = self.element_freqs.into_iter().collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        entries.truncate(top_k);
        ColumnStats {
            row_count: self.row_count,
            null_count: total_rows.saturating_sub(self.row_count),
            distinct_count,
            kind: ColumnStatsKind::Array {
                element_top_k: entries,
                avg_array_len,
            },
        }
    }
}

/// Build an equi-depth histogram from a sorted slice of values.
///
/// The output has at most `n_buckets` entries. Each bucket holds
/// approximately `sorted.len() / n_buckets` values (the first
/// `len % n_buckets` buckets get one extra value to absorb the
/// remainder). Buckets are touching but not strictly contiguous :
/// bucket[i].hi is the largest value assigned to bucket i, which
/// may equal bucket[i+1].lo.
fn build_equi_depth(sorted: &[f64], n_buckets: usize) -> Vec<HistogramBucket> {
    let len = sorted.len();
    if len == 0 || n_buckets == 0 {
        return vec![];
    }
    let n = n_buckets.min(len);
    let chunk_base = len / n;
    let extra = len % n;

    let mut buckets = Vec::with_capacity(n);
    let mut start = 0;
    for i in 0..n {
        let count = chunk_base + usize::from(i < extra);
        let end = start + count;
        buckets.push(HistogramBucket {
            lo: sorted[start],
            hi: sorted[end - 1],
            count: count as u64,
        });
        start = end;
    }
    buckets
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Helpers ----

    fn s(x: &str) -> Value {
        Value::String(x.into())
    }
    fn i(n: i64) -> Value {
        Value::I64(n)
    }

    fn numeric_stats(
        row_count: u64,
        null_count: u64,
        ndv: u64,
        buckets: Vec<(f64, f64, u64)>,
    ) -> ColumnStats {
        let min = buckets.first().map_or(0.0, |(lo, _, _)| *lo);
        let max = buckets.last().map_or(0.0, |(_, hi, _)| *hi);
        ColumnStats {
            row_count,
            null_count,
            distinct_count: ndv,
            kind: ColumnStatsKind::Numeric {
                min,
                max,
                histogram: buckets
                    .into_iter()
                    .map(|(lo, hi, count)| HistogramBucket { lo, hi, count })
                    .collect(),
            },
        }
    }

    fn string_stats(
        row_count: u64,
        null_count: u64,
        ndv: u64,
        top_k: &[(&str, u64)],
    ) -> ColumnStats {
        ColumnStats {
            row_count,
            null_count,
            distinct_count: ndv,
            kind: ColumnStatsKind::String {
                top_k: top_k.iter().map(|(s, c)| ((*s).to_string(), *c)).collect(),
            },
        }
    }

    fn bool_stats(true_count: u64, false_count: u64, null_count: u64) -> ColumnStats {
        ColumnStats {
            row_count: true_count + false_count,
            null_count,
            distinct_count: u64::from(true_count > 0) + u64::from(false_count > 0),
            kind: ColumnStatsKind::Bool {
                true_count,
                false_count,
            },
        }
    }

    fn array_stats(
        row_count: u64,
        null_count: u64,
        ndv: u64,
        avg_array_len: f64,
        top_k: &[(&str, u64)],
    ) -> ColumnStats {
        ColumnStats {
            row_count,
            null_count,
            distinct_count: ndv,
            kind: ColumnStatsKind::Array {
                element_top_k: top_k.iter().map(|(s, c)| ((*s).to_string(), *c)).collect(),
                avg_array_len,
            },
        }
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    // ---- IsNotNull (kind-agnostic) ----

    #[test]
    fn is_not_null_uses_row_and_null_counts() {
        let st = numeric_stats(80, 20, 10, vec![(0.0, 100.0, 80)]);
        let s = st.selectivity(&IndexAtom::IsNotNull).unwrap();
        assert!(approx_eq(s, 0.8));
    }

    #[test]
    fn is_not_null_on_empty_stats_is_zero() {
        let st = numeric_stats(0, 0, 0, vec![]);
        assert_eq!(st.selectivity(&IndexAtom::IsNotNull), Some(0.0));
    }

    #[test]
    fn is_not_null_works_on_mixed_kind() {
        // Mixed columns answer IsNotNull from row/null counts ; the
        // payload kind doesn't matter for this atom.
        let st = ColumnStats {
            row_count: 10,
            null_count: 90,
            distinct_count: 0,
            kind: ColumnStatsKind::Mixed,
        };
        let s = st.selectivity(&IndexAtom::IsNotNull).unwrap();
        assert!(approx_eq(s, 0.1));
    }

    // ---- Eq ----

    #[test]
    fn eq_on_numeric_in_range_is_one_over_ndv() {
        let st = numeric_stats(100, 0, 10, vec![(0.0, 100.0, 100)]);
        let s = st.selectivity(&IndexAtom::Eq(i(50))).unwrap();
        assert!(approx_eq(s, 0.1));
    }

    #[test]
    fn eq_on_numeric_outside_range_is_zero() {
        let st = numeric_stats(100, 0, 10, vec![(0.0, 100.0, 100)]);
        assert_eq!(st.selectivity(&IndexAtom::Eq(i(-5))), Some(0.0));
        assert_eq!(st.selectivity(&IndexAtom::Eq(i(101))), Some(0.0));
    }

    #[test]
    fn eq_on_string_top_k_hit_uses_exact_count() {
        let st = string_stats(100, 0, 5, &[("docs", 60), ("blog", 30)]);
        let s = st.selectivity(&IndexAtom::Eq(s("docs"))).unwrap();
        assert!(approx_eq(s, 0.6));
    }

    #[test]
    fn eq_on_string_top_k_miss_uses_tail_estimate() {
        // 100 rows, 5 distinct values, top-2 covers (60+30) = 90 rows.
        // Tail : 10 rows over (5-2) = 3 distinct values = ~3.33 per value.
        // selectivity = 3.33 / 100 = 0.0333.
        let st = string_stats(100, 0, 5, &[("docs", 60), ("blog", 30)]);
        let sel = st.selectivity(&IndexAtom::Eq(s("rare"))).unwrap();
        assert!((sel - 10.0 / 3.0 / 100.0).abs() < 1e-6);
    }

    #[test]
    fn eq_on_bool_uses_direct_counts() {
        let st = bool_stats(30, 70, 0);
        let t = st.selectivity(&IndexAtom::Eq(Value::Bool(true))).unwrap();
        let f = st.selectivity(&IndexAtom::Eq(Value::Bool(false))).unwrap();
        assert!(approx_eq(t, 0.3));
        assert!(approx_eq(f, 0.7));
    }

    #[test]
    fn eq_on_string_with_non_string_value_returns_none() {
        let st = string_stats(100, 0, 5, &[("docs", 60)]);
        assert!(st.selectivity(&IndexAtom::Eq(i(5))).is_none());
    }

    // ---- Cmp : Lt / Le / Gt / Ge / Ne ----

    #[test]
    fn lt_on_numeric_within_first_bucket() {
        // One bucket [0, 100] with 100 rows. v=25 : fraction 0.25.
        let st = numeric_stats(100, 0, 10, vec![(0.0, 100.0, 100)]);
        let s = st.selectivity(&IndexAtom::Cmp(CmpOp::Lt, i(25))).unwrap();
        assert!(approx_eq(s, 0.25));
    }

    #[test]
    fn lt_on_numeric_at_min_is_zero() {
        let st = numeric_stats(100, 0, 10, vec![(0.0, 100.0, 100)]);
        assert_eq!(st.selectivity(&IndexAtom::Cmp(CmpOp::Lt, i(0))), Some(0.0));
    }

    #[test]
    fn lt_on_numeric_above_max_is_one() {
        let st = numeric_stats(100, 0, 10, vec![(0.0, 100.0, 100)]);
        let s = st.selectivity(&IndexAtom::Cmp(CmpOp::Lt, i(150))).unwrap();
        assert!(approx_eq(s, 1.0));
    }

    #[test]
    fn lt_spans_multiple_buckets() {
        // Two buckets : [0, 50) with 30 rows, [50, 100] with 70 rows.
        // v=75 : 30 + (75-50)/(100-50)*70 = 30 + 0.5*70 = 65 ; /100 = 0.65.
        let st = numeric_stats(100, 0, 10, vec![(0.0, 50.0, 30), (50.0, 100.0, 70)]);
        let s = st.selectivity(&IndexAtom::Cmp(CmpOp::Lt, i(75))).unwrap();
        assert!(approx_eq(s, 0.65));
    }

    #[test]
    fn ge_is_complement_of_lt() {
        let st = numeric_stats(100, 0, 10, vec![(0.0, 100.0, 100)]);
        let lt = st.selectivity(&IndexAtom::Cmp(CmpOp::Lt, i(25))).unwrap();
        let ge = st.selectivity(&IndexAtom::Cmp(CmpOp::Ge, i(25))).unwrap();
        assert!(approx_eq(lt + ge, 1.0));
    }

    #[test]
    fn le_includes_equality() {
        // Lt(50) = 0.5 ; Eq(50) ≈ 1/10 = 0.1 ; Le(50) = 0.6.
        let st = numeric_stats(100, 0, 10, vec![(0.0, 100.0, 100)]);
        let s = st.selectivity(&IndexAtom::Cmp(CmpOp::Le, i(50))).unwrap();
        assert!(approx_eq(s, 0.6));
    }

    #[test]
    fn ne_is_one_minus_eq() {
        let st = string_stats(100, 0, 5, &[("docs", 60)]);
        let eq = st.selectivity(&IndexAtom::Eq(s("docs"))).unwrap();
        let ne = st
            .selectivity(&IndexAtom::Cmp(CmpOp::Ne, s("docs")))
            .unwrap();
        assert!(approx_eq(eq + ne, 1.0));
    }

    // ---- Between ----

    #[test]
    fn between_inclusive_both_sides() {
        // [0, 100], 100 rows uniform. Between(20, 60) :
        //   Le(60) = 0.6 + 0.1 = 0.7 ; Lt(20) = 0.2 ; result = 0.5.
        let st = numeric_stats(100, 0, 10, vec![(0.0, 100.0, 100)]);
        let s = st.selectivity(&IndexAtom::Between(i(20), i(60))).unwrap();
        assert!(approx_eq(s, 0.5));
    }

    #[test]
    fn between_inverted_range_is_zero() {
        let st = numeric_stats(100, 0, 10, vec![(0.0, 100.0, 100)]);
        assert_eq!(st.selectivity(&IndexAtom::Between(i(60), i(20))), Some(0.0));
    }

    // ---- In ----

    #[test]
    fn in_sums_eq_selectivities() {
        let st = string_stats(100, 0, 5, &[("docs", 60), ("blog", 30)]);
        let s = st
            .selectivity(&IndexAtom::In(vec![s("docs"), s("blog")]))
            .unwrap();
        assert!(approx_eq(s, 0.9));
    }

    #[test]
    fn in_caps_at_one() {
        // Adversarial : repeated values would push sum > 1 ; we cap.
        let st = string_stats(100, 0, 5, &[("docs", 60), ("blog", 60)]);
        let s = st
            .selectivity(&IndexAtom::In(vec![s("docs"), s("blog"), s("docs")]))
            .unwrap();
        assert!(approx_eq(s, 1.0));
    }

    // ---- ArrayContains ----

    #[test]
    fn array_contains_top_k_hit_uses_count() {
        let st = array_stats(100, 0, 6, 2.0, &[("rust", 40), ("go", 20)]);
        let s = st
            .selectivity(&IndexAtom::ArrayContains(s("rust")))
            .unwrap();
        assert!(approx_eq(s, 0.4));
    }

    #[test]
    fn array_contains_tail_value_uses_avg_len_estimate() {
        // 100 rows, avg 2 elements, total occurrences = 200.
        // Top-K covers 40 + 20 = 60. Tail = 140 occurrences over
        // (6 - 2) = 4 distinct values = 35 per value.
        // Selectivity = 35 / 100 = 0.35.
        let st = array_stats(100, 0, 6, 2.0, &[("rust", 40), ("go", 20)]);
        let s = st
            .selectivity(&IndexAtom::ArrayContains(s("rare")))
            .unwrap();
        assert!(approx_eq(s, 0.35));
    }

    #[test]
    fn array_contains_on_wrong_kind_returns_none() {
        let st = numeric_stats(100, 0, 10, vec![(0.0, 100.0, 100)]);
        assert!(st.selectivity(&IndexAtom::ArrayContains(s("x"))).is_none());
    }

    // ---- Mixed kind ----

    #[test]
    fn mixed_kind_returns_none_except_for_is_not_null() {
        let st = ColumnStats {
            row_count: 100,
            null_count: 0,
            distinct_count: 50,
            kind: ColumnStatsKind::Mixed,
        };
        assert!(st.selectivity(&IndexAtom::Eq(i(5))).is_none());
        assert!(st.selectivity(&IndexAtom::Cmp(CmpOp::Lt, i(5))).is_none());
        assert!(st.selectivity(&IndexAtom::IsNotNull).is_some());
    }

    // ---- Selectivity is always in [0, 1] (sanity) ----

    #[test]
    fn selectivity_stays_clamped_for_pathological_histograms() {
        // Bucket counts deliberately sum to more than row_count : an
        // adversarial / corrupt stats payload. Selectivity must
        // still return something in [0, 1].
        let st = numeric_stats(100, 0, 10, vec![(0.0, 100.0, 1_000_000)]);
        let s = st.selectivity(&IndexAtom::Cmp(CmpOp::Lt, i(50))).unwrap();
        assert!((0.0..=1.0).contains(&s));
    }

    // ---- StatsCatalog ----

    #[test]
    fn catalog_dispatches_to_field_stats() {
        let mut cat = StatsCatalog::new();
        cat.put(
            "category",
            string_stats(100, 0, 5, &[("docs", 60), ("blog", 30)]),
        );
        cat.put(
            "year",
            numeric_stats(100, 0, 10, vec![(2020.0, 2025.0, 100)]),
        );

        let docs = cat
            .selectivity("category", &IndexAtom::Eq(s("docs")))
            .unwrap();
        let recent = cat
            .selectivity("year", &IndexAtom::Cmp(CmpOp::Ge, i(2023)))
            .unwrap();
        assert!(approx_eq(docs, 0.6));
        assert!(recent > 0.0 && recent < 1.0);
    }

    #[test]
    fn catalog_returns_none_for_unknown_field() {
        let cat = StatsCatalog::new();
        assert!(cat.selectivity("missing", &IndexAtom::Eq(s("x"))).is_none());
    }

    #[test]
    fn catalog_remove_drops_field() {
        let mut cat = StatsCatalog::new();
        cat.put("a", string_stats(10, 0, 2, &[("x", 5), ("y", 5)]));
        assert_eq!(cat.len(), 1);
        cat.remove("a");
        assert!(cat.is_empty());
    }

    // ---- Builder ----

    fn meta_from(pairs: &[(&str, Value)]) -> kova_core::Metadata {
        let mut m = kova_core::Metadata::new();
        for (k, v) in pairs {
            m.insert((*k).to_string(), v.clone());
        }
        m
    }

    #[test]
    fn builder_empty_finish_produces_empty_catalog() {
        let cat = StatsBuilder::new().finish();
        assert!(cat.is_empty());
    }

    #[test]
    fn builder_numeric_field_produces_numeric_stats() {
        let mut b = StatsBuilder::with_config(2, 16);
        for n in 0..10_i64 {
            b.observe(&meta_from(&[("year", i(2020 + n))]));
        }
        let cat = b.finish();
        let st = cat.get("year").unwrap();
        assert_eq!(st.row_count, 10);
        assert_eq!(st.null_count, 0);
        assert_eq!(st.distinct_count, 10);
        match &st.kind {
            ColumnStatsKind::Numeric {
                min,
                max,
                histogram,
            } => {
                assert!(approx_eq(*min, 2020.0));
                assert!(approx_eq(*max, 2029.0));
                // 2 buckets requested, 10 values → 5 per bucket.
                assert_eq!(histogram.len(), 2);
                assert_eq!(histogram[0].count, 5);
                assert_eq!(histogram[1].count, 5);
            }
            other => panic!("expected Numeric, got {other:?}"),
        }
    }

    #[test]
    fn builder_string_field_top_k_respects_config() {
        let mut b = StatsBuilder::with_config(20, 2);
        // 5 distinct categories, with counts [4, 3, 2, 1, 1] = 11 rows
        let pattern = ["a", "a", "a", "a", "b", "b", "b", "c", "c", "d", "e"];
        for p in pattern {
            b.observe(&meta_from(&[("cat", s(p))]));
        }
        let cat = b.finish();
        let st = cat.get("cat").unwrap();
        assert_eq!(st.row_count, 11);
        assert_eq!(st.distinct_count, 5);
        match &st.kind {
            ColumnStatsKind::String { top_k } => {
                assert_eq!(top_k.len(), 2, "top_k truncated to configured 2");
                assert_eq!(top_k[0], ("a".to_string(), 4));
                assert_eq!(top_k[1], ("b".to_string(), 3));
            }
            other => panic!("expected String, got {other:?}"),
        }
    }

    #[test]
    fn builder_bool_field_counts_directly() {
        let mut b = StatsBuilder::new();
        for _ in 0..7 {
            b.observe(&meta_from(&[("flag", Value::Bool(true))]));
        }
        for _ in 0..3 {
            b.observe(&meta_from(&[("flag", Value::Bool(false))]));
        }
        let cat = b.finish();
        let st = cat.get("flag").unwrap();
        match &st.kind {
            ColumnStatsKind::Bool {
                true_count,
                false_count,
            } => {
                assert_eq!(*true_count, 7);
                assert_eq!(*false_count, 3);
            }
            other => panic!("expected Bool, got {other:?}"),
        }
        assert_eq!(st.row_count, 10);
        assert_eq!(st.distinct_count, 2);
    }

    #[test]
    fn builder_array_field_collects_element_top_k_and_avg_len() {
        let mut b = StatsBuilder::with_config(20, 3);
        // row 0 : ["rust", "async"]      (2 elements)
        // row 1 : ["rust"]               (1)
        // row 2 : ["go", "rust", "tokio"](3)
        // row 3 : ["python"]             (1)
        // Total length = 7, row_count = 4, avg = 1.75
        // Element counts : rust=3, async=1, go=1, tokio=1, python=1
        // Top-3 : rust=3, then ties broken alphabetically -> async=1, go=1
        let rows = vec![
            Value::Array(vec![s("rust"), s("async")]),
            Value::Array(vec![s("rust")]),
            Value::Array(vec![s("go"), s("rust"), s("tokio")]),
            Value::Array(vec![s("python")]),
        ];
        for v in rows {
            b.observe(&meta_from(&[("tags", v)]));
        }
        let cat = b.finish();
        let st = cat.get("tags").unwrap();
        assert_eq!(st.row_count, 4);
        assert_eq!(st.distinct_count, 5);
        match &st.kind {
            ColumnStatsKind::Array {
                element_top_k,
                avg_array_len,
            } => {
                assert!(approx_eq(*avg_array_len, 1.75));
                assert_eq!(element_top_k.len(), 3);
                assert_eq!(element_top_k[0], ("rust".to_string(), 3));
                // The remaining two are tied at count 1 ; deterministic tiebreak by key.
                assert_eq!(element_top_k[1].0, "async");
                assert_eq!(element_top_k[2].0, "go");
            }
            other => panic!("expected Array, got {other:?}"),
        }
    }

    #[test]
    fn builder_mixed_types_collapses_to_mixed_kind() {
        let mut b = StatsBuilder::new();
        b.observe(&meta_from(&[("messy", i(1))]));
        b.observe(&meta_from(&[("messy", i(2))]));
        b.observe(&meta_from(&[("messy", s("oops"))])); // type changes
        b.observe(&meta_from(&[("messy", s("again"))]));
        let cat = b.finish();
        let st = cat.get("messy").unwrap();
        assert!(matches!(st.kind, ColumnStatsKind::Mixed));
        assert_eq!(st.row_count, 4);
        // distinct_count is dropped for Mixed.
        assert_eq!(st.distinct_count, 0);
    }

    #[test]
    fn builder_tracks_null_count_per_field() {
        // 5 rows total ; 3 of them have "year".
        let mut b = StatsBuilder::new();
        b.observe(&meta_from(&[("year", i(2020))]));
        b.observe(&meta_from(&[("year", i(2021))]));
        b.observe(&meta_from(&[("other", s("x"))]));
        b.observe(&meta_from(&[("year", i(2022))]));
        b.observe(&meta_from(&[("other", s("y"))]));
        let cat = b.finish();
        let year = cat.get("year").unwrap();
        assert_eq!(year.row_count, 3);
        assert_eq!(year.null_count, 2);
        let other = cat.get("other").unwrap();
        assert_eq!(other.row_count, 2);
        assert_eq!(other.null_count, 3);
    }

    #[test]
    fn builder_track_field_surfaces_fully_null_columns() {
        let mut b = StatsBuilder::new();
        b.track_field("missing");
        for n in 0..5 {
            b.observe(&meta_from(&[("present", i(n))]));
        }
        let cat = b.finish();
        let st = cat.get("missing").unwrap();
        assert_eq!(st.row_count, 0);
        assert_eq!(st.null_count, 5);
    }

    #[test]
    fn builder_n_smaller_than_buckets_uses_one_per_value() {
        // 3 values, 20 buckets requested → max 3 buckets.
        let mut b = StatsBuilder::with_config(20, 16);
        for n in [1_i64, 5, 10] {
            b.observe(&meta_from(&[("score", i(n))]));
        }
        let st = b.finish().get("score").cloned().unwrap();
        match st.kind {
            ColumnStatsKind::Numeric { histogram, .. } => {
                assert!(histogram.len() <= 3);
                let total: u64 = histogram.iter().map(|h| h.count).sum();
                assert_eq!(total, 3);
            }
            _ => panic!("expected Numeric"),
        }
    }

    #[test]
    fn builder_round_trip_through_selectivity() {
        // Build stats from a realistic stream, then query via the
        // catalog. This catches "builder produces stats that the
        // selectivity math interprets correctly" end-to-end.
        let mut b = StatsBuilder::new();
        for n in 0..100_i64 {
            let cat_val = if n % 2 == 0 { "docs" } else { "blog" };
            b.observe(&meta_from(&[
                ("category", s(cat_val)),
                ("year", i(2020 + n % 6)),
            ]));
        }
        let cat = b.finish();
        // ~50% of rows have category = 'docs'
        let docs = cat
            .selectivity("category", &IndexAtom::Eq(s("docs")))
            .unwrap();
        assert!(
            (docs - 0.5).abs() < 0.05,
            "expected ~0.5 selectivity, got {docs}"
        );
        // year stats : approximate range query
        let recent = cat
            .selectivity("year", &IndexAtom::Cmp(CmpOp::Ge, i(2023)))
            .unwrap();
        // 2023..=2025 should be roughly half the range.
        assert!((0.2..0.7).contains(&recent), "got {recent}");
    }

    // ---- Persistence ----

    #[test]
    fn encode_decode_round_trip_preserves_selectivity() {
        let mut b = StatsBuilder::new();
        for n in 0..20 {
            b.observe(&meta_from(&[
                ("category", s(if n % 2 == 0 { "docs" } else { "blog" })),
                ("year", i(2020 + n)),
                ("active", Value::Bool(n % 3 == 0)),
            ]));
        }
        let original = b.finish();
        let bytes = original.encode().unwrap();
        let back = StatsCatalog::decode(&bytes).unwrap();
        // Same fields tracked.
        let mut fa: Vec<&str> = original.fields().collect();
        let mut fb: Vec<&str> = back.fields().collect();
        fa.sort_unstable();
        fb.sort_unstable();
        assert_eq!(fa, fb);
        // Same selectivity for a spot-check atom on each kind.
        for (field, atom) in [
            ("category", IndexAtom::Eq(s("docs"))),
            ("year", IndexAtom::Cmp(CmpOp::Ge, i(2025))),
            ("active", IndexAtom::Eq(Value::Bool(true))),
        ] {
            let a = original.selectivity(field, &atom);
            let b = back.selectivity(field, &atom);
            assert_eq!(a, b, "selectivity diverged after round-trip on {field}");
        }
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let bytes = vec![0u8; 64];
        let err = StatsCatalog::decode(&bytes).unwrap_err();
        assert!(matches!(err, KovaMetaIndexError::BadMagic), "{err:?}");
    }

    #[test]
    fn decode_rejects_unsupported_version() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(STATS_MAGIC);
        bytes.extend_from_slice(&999u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 8]);
        let err = StatsCatalog::decode(&bytes).unwrap_err();
        assert!(
            matches!(
                err,
                KovaMetaIndexError::UnsupportedVersion {
                    expected: 1,
                    got: 999
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn decode_rejects_truncated() {
        let err = StatsCatalog::decode(&[0u8; 3]).unwrap_err();
        assert!(
            matches!(err, KovaMetaIndexError::Truncated { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn load_missing_file_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonexistent.bin");
        assert!(StatsCatalog::load(&path).unwrap().is_none());
    }

    #[test]
    fn load_existing_file_round_trip() {
        let mut b = StatsBuilder::new();
        for n in 0..5 {
            b.observe(&meta_from(&[("x", i(n))]));
        }
        let cat = b.finish();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stats.bin");
        std::fs::write(&path, cat.encode().unwrap()).unwrap();
        let loaded = StatsCatalog::load(&path).unwrap().unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(loaded.get("x").is_some());
    }
}

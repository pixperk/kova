//! BTree-backed range index.
//!
//! Backing : `BTreeMap<NormalizedKey, RoaringTreemap>`. The total
//! ordering on `NormalizedKey` lets range queries walk a contiguous
//! prefix of buckets.
//!
//! ## Atoms supported
//!
//! Everything [`crate::HashIndex`] supports, plus :
//!
//! | Atom | How |
//! |------|-----|
//! | [`IndexAtom::Cmp`] with `Lt`/`Le`/`Gt`/`Ge` | cursor at value, range walk, union of bucket bitmaps |
//! | [`IndexAtom::Between`] | two cursors, range walk, union |
//!
//! ## Float caveat
//!
//! Lexicographic ordering of [`f64::to_bits`] matches numeric `<`
//! only for positive finite floats. For mixed signs and NaN, the
//! bit-ordering diverges from numeric ordering. [`BTreeIndex`]
//! therefore rejects ranged queries on float fields via
//! [`MetaIndex::supports`]. Equality queries on float fields still
//! work because they match by bit-pattern, which is what the user
//! expects (NaN is never equal to NaN, two distinct NaN encodings
//! are distinct keys, etc.).
//!
//! ## Cost
//!
//! `Cmp` with `Lt/Le/Gt/Ge` is `O(log N)` to find the cursor at the
//! comparison value, then `O(buckets_in_range)` walk, then `O(union)`
//! over those buckets. `Between` is `O(log N)` plus a bounded range
//! walk. Total cost is dominated by `O(matches)` in the typical
//! case, not the index size.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::ops::{Bound, RangeBounds};

use kova_core::{Value, VectorId};
use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};

use crate::{CmpOp, IndexAtom, MetaIndex, NormalizedKey};

/// BTree-backed range index. See [module-level docs](self).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BTreeIndex {
    // Sorted map from normalized key to bitmap of matching row ids.
    buckets: BTreeMap<NormalizedKey, RoaringTreemap>,
    // Bitmap of all indexed row ids. Used to support IsNotNull and Ne atoms.
    all_indexed_ids: RoaringTreemap,
}

impl BTreeIndex {
    /// Construct an empty index. Bulk-build with
    /// [`MetaIndex::build`] or populate incrementally with
    /// [`MetaIndex::insert`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn bucket_for(&self, v: &Value) -> Option<&RoaringTreemap> {
        let key = NormalizedKey::from_value(v)?;
        self.buckets.get(&key)
    }

    fn range_union<R>(&self, range: R) -> RoaringTreemap
    where
        R: RangeBounds<NormalizedKey>,
    {
        let mut out = RoaringTreemap::new();
        for (_, bucket) in self.buckets.range(range) {
            out |= bucket;
        }
        out
    }

    fn range_cardinality<R>(&self, range: R) -> u64
    where
        R: RangeBounds<NormalizedKey>,
    {
        self.buckets.range(range).map(|(_, b)| b.len()).sum()
    }
}

fn is_float(v: &Value) -> bool {
    matches!(v, Value::F64(_))
}

fn cmp_bounds(op: CmpOp, key: NormalizedKey) -> (Bound<NormalizedKey>, Bound<NormalizedKey>) {
    match op {
        CmpOp::Lt => (Bound::Unbounded, Bound::Excluded(key)),
        CmpOp::Le => (Bound::Unbounded, Bound::Included(key)),
        CmpOp::Gt => (Bound::Excluded(key), Bound::Unbounded),
        CmpOp::Ge => (Bound::Included(key), Bound::Unbounded),
        // Ne is handled at the query/cardinality dispatch level, not as a range.
        CmpOp::Ne => (Bound::Unbounded, Bound::Unbounded),
    }
}

impl MetaIndex for BTreeIndex {
    fn build<I>(&mut self, rows: I)
    where
        I: IntoIterator<Item = (VectorId, Value)>,
    {
        for (id, value) in rows {
            self.insert(id, &value);
        }
    }

    fn insert(&mut self, id: VectorId, value: &Value) {
        let Some(key) = NormalizedKey::from_value(value) else {
            return;
        };
        self.buckets.entry(key).or_default().insert(id.into());
        self.all_indexed_ids.insert(id.into());
    }

    fn delete(&mut self, id: VectorId, value: &Value) {
        let Some(key) = NormalizedKey::from_value(value) else {
            return;
        };
        if let Entry::Occupied(mut entry) = self.buckets.entry(key) {
            entry.get_mut().remove(id.into());
            if entry.get().is_empty() {
                entry.remove();
            }
        }
        self.all_indexed_ids.remove(id.into());
    }

    fn update(&mut self, id: VectorId, old: &Value, new: &Value) {
        let old_key = NormalizedKey::from_value(old);
        let new_key = NormalizedKey::from_value(new);
        if old_key == new_key {
            return;
        }
        self.delete(id, old);
        self.insert(id, new);
    }

    fn query(&self, atom: &IndexAtom) -> RoaringTreemap {
        match atom {
            IndexAtom::Eq(v) => self.bucket_for(v).cloned().unwrap_or_default(),

            IndexAtom::In(vs) => {
                let mut out = RoaringTreemap::new();
                for v in vs {
                    if let Some(bucket) = self.bucket_for(v) {
                        out |= bucket;
                    }
                }
                out
            }

            IndexAtom::IsNotNull => self.all_indexed_ids.clone(),

            IndexAtom::Cmp(CmpOp::Ne, v) => {
                let mut out = self.all_indexed_ids.clone();
                if let Some(bucket) = self.bucket_for(v) {
                    out -= bucket;
                }
                out
            }

            IndexAtom::Cmp(op, v) => {
                if is_float(v) {
                    return RoaringTreemap::new();
                }
                let Some(key) = NormalizedKey::from_value(v) else {
                    return RoaringTreemap::new();
                };
                self.range_union(cmp_bounds(*op, key))
            }

            IndexAtom::Between(lo, hi) => {
                if is_float(lo) || is_float(hi) {
                    return RoaringTreemap::new();
                }
                let (Some(lo_k), Some(hi_k)) =
                    (NormalizedKey::from_value(lo), NormalizedKey::from_value(hi))
                else {
                    return RoaringTreemap::new();
                };
                if lo_k > hi_k {
                    return RoaringTreemap::new();
                }
                self.range_union((Bound::Included(lo_k), Bound::Included(hi_k)))
            }

            IndexAtom::ArrayContains(_) => RoaringTreemap::new(),
        }
    }

    fn cardinality(&self, atom: &IndexAtom) -> Option<u64> {
        match atom {
            IndexAtom::Eq(v) => Some(self.bucket_for(v).map_or(0, RoaringTreemap::len)),

            IndexAtom::In(vs) => {
                let mut total: u64 = 0;
                for v in vs {
                    total += self.bucket_for(v).map_or(0, RoaringTreemap::len);
                }
                Some(total)
            }

            IndexAtom::IsNotNull => Some(self.all_indexed_ids.len()),

            IndexAtom::Cmp(CmpOp::Ne, v) => {
                let total = self.all_indexed_ids.len();
                let bucket_len = self.bucket_for(v).map_or(0, RoaringTreemap::len);
                Some(total - bucket_len)
            }

            IndexAtom::Cmp(op, v) => {
                if is_float(v) {
                    return None;
                }
                let key = NormalizedKey::from_value(v)?;
                Some(self.range_cardinality(cmp_bounds(*op, key)))
            }

            IndexAtom::Between(lo, hi) => {
                if is_float(lo) || is_float(hi) {
                    return None;
                }
                let lo_k = NormalizedKey::from_value(lo)?;
                let hi_k = NormalizedKey::from_value(hi)?;
                if lo_k > hi_k {
                    return Some(0);
                }
                Some(self.range_cardinality((Bound::Included(lo_k), Bound::Included(hi_k))))
            }

            IndexAtom::ArrayContains(_) => None,
        }
    }

    fn len(&self) -> u64 {
        self.all_indexed_ids.len()
    }

    fn supports(&self, atom: &IndexAtom) -> bool {
        match atom {
            IndexAtom::Eq(_)
            | IndexAtom::In(_)
            | IndexAtom::IsNotNull
            | IndexAtom::Cmp(CmpOp::Ne, _) => true,

            IndexAtom::Cmp(CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge, v) => !is_float(v),

            IndexAtom::Between(lo, hi) => !is_float(lo) && !is_float(hi),

            IndexAtom::ArrayContains(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use kova_core::{Value, VectorId};

    use super::BTreeIndex;
    use crate::{CmpOp, IndexAtom, MetaIndex};

    fn i(n: i64) -> Value {
        Value::I64(n)
    }

    fn f(x: f64) -> Value {
        Value::F64(x)
    }

    fn s(x: &str) -> Value {
        Value::String(x.into())
    }

    fn id(n: u64) -> VectorId {
        VectorId::new(n)
    }

    fn build_year_index() -> BTreeIndex {
        let mut idx = BTreeIndex::new();
        // ids 0..10 paired with year 2020..2030.
        for (n, year) in (0u64..10).zip(2020i64..2030) {
            idx.insert(id(n), &i(year));
        }
        idx
    }

    #[test]
    fn eq_and_in_match_hash_semantics() {
        let idx = build_year_index();
        let eq = idx.query(&IndexAtom::Eq(i(2023)));
        assert_eq!(eq.len(), 1);
        assert!(eq.contains(3));

        let in_ = idx.query(&IndexAtom::In(vec![i(2020), i(2025), i(2028)]));
        assert_eq!(in_.len(), 3);
        assert!(in_.contains(0));
        assert!(in_.contains(5));
        assert!(in_.contains(8));
    }

    #[test]
    fn range_lt_excludes_boundary() {
        let idx = build_year_index();
        let lt = idx.query(&IndexAtom::Cmp(CmpOp::Lt, i(2023)));
        assert_eq!(lt.len(), 3);
        for n in 0..3 {
            assert!(lt.contains(n));
        }
        assert!(!lt.contains(3));
    }

    #[test]
    fn range_le_includes_boundary() {
        let idx = build_year_index();
        let le = idx.query(&IndexAtom::Cmp(CmpOp::Le, i(2023)));
        assert_eq!(le.len(), 4);
        for n in 0..=3 {
            assert!(le.contains(n));
        }
        assert!(!le.contains(4));
    }

    #[test]
    fn range_gt_excludes_boundary() {
        let idx = build_year_index();
        let gt = idx.query(&IndexAtom::Cmp(CmpOp::Gt, i(2026)));
        assert_eq!(gt.len(), 3);
        for n in 7..10 {
            assert!(gt.contains(n));
        }
        assert!(!gt.contains(6));
    }

    #[test]
    fn range_ge_includes_boundary() {
        let idx = build_year_index();
        let ge = idx.query(&IndexAtom::Cmp(CmpOp::Ge, i(2026)));
        assert_eq!(ge.len(), 4);
        for n in 6..10 {
            assert!(ge.contains(n));
        }
        assert!(!ge.contains(5));
    }

    #[test]
    fn between_is_inclusive_both_sides() {
        let idx = build_year_index();
        let bw = idx.query(&IndexAtom::Between(i(2023), i(2026)));
        assert_eq!(bw.len(), 4);
        for n in 3..=6 {
            assert!(bw.contains(n));
        }
        assert!(!bw.contains(2));
        assert!(!bw.contains(7));
    }

    #[test]
    fn between_with_inverted_range_is_empty() {
        let idx = build_year_index();
        // lo > hi : BTreeMap::range yields no buckets in this case
        // (it returns an empty iterator rather than panicking, as long
        // as we use bounded ranges built via the tuple form).
        let bw = idx.query(&IndexAtom::Between(i(2026), i(2023)));
        assert!(bw.is_empty());
    }

    #[test]
    fn range_iteration_is_sorted() {
        let idx = build_year_index();
        let bw = idx.query(&IndexAtom::Between(i(2021), i(2027)));
        let collected: Vec<u64> = bw.iter().collect();
        let mut sorted = collected.clone();
        sorted.sort_unstable();
        assert_eq!(collected, sorted);
    }

    #[test]
    fn ne_returns_complement() {
        let idx = build_year_index();
        let ne = idx.query(&IndexAtom::Cmp(CmpOp::Ne, i(2023)));
        assert_eq!(ne.len(), 9);
        assert!(!ne.contains(3));
        for n in (0..10).filter(|n| *n != 3) {
            assert!(ne.contains(n));
        }
    }

    #[test]
    fn is_not_null_excludes_non_indexable_values() {
        let mut idx = BTreeIndex::new();
        idx.insert(id(0), &i(10));
        idx.insert(id(1), &i(20));
        idx.insert(id(2), &Value::Array(vec![i(1)]));
        let live = idx.query(&IndexAtom::IsNotNull);
        assert_eq!(live.len(), 2);
        assert!(live.contains(0));
        assert!(live.contains(1));
        assert!(!live.contains(2));
    }

    #[test]
    fn float_range_is_rejected_by_supports() {
        let idx = BTreeIndex::new();
        assert!(!idx.supports(&IndexAtom::Cmp(CmpOp::Lt, f(0.5))));
        assert!(!idx.supports(&IndexAtom::Cmp(CmpOp::Gt, f(0.5))));
        assert!(!idx.supports(&IndexAtom::Between(f(0.1), f(0.9))));
        // equality on float still works
        assert!(idx.supports(&IndexAtom::Eq(f(0.5))));
        assert!(idx.supports(&IndexAtom::Cmp(CmpOp::Ne, f(0.5))));
    }

    #[test]
    fn float_range_query_returns_empty() {
        let mut idx = BTreeIndex::new();
        idx.insert(id(0), &f(0.1));
        idx.insert(id(1), &f(0.5));
        idx.insert(id(2), &f(0.9));
        // Range queries on floats refuse to give a possibly-wrong
        // answer ; the executor falls back to a metadata scan.
        let lt = idx.query(&IndexAtom::Cmp(CmpOp::Lt, f(0.5)));
        assert!(lt.is_empty());
        let bw = idx.query(&IndexAtom::Between(f(0.0), f(1.0)));
        assert!(bw.is_empty());
    }

    #[test]
    fn float_eq_still_works() {
        let mut idx = BTreeIndex::new();
        idx.insert(id(0), &f(0.1));
        idx.insert(id(1), &f(0.5));
        idx.insert(id(2), &f(0.9));
        let eq = idx.query(&IndexAtom::Eq(f(0.5)));
        assert_eq!(eq.len(), 1);
        assert!(eq.contains(1));
    }

    #[test]
    fn delete_drops_empty_buckets() {
        let mut idx = BTreeIndex::new();
        idx.insert(id(0), &i(7));
        idx.insert(id(1), &i(7));
        idx.delete(id(0), &i(7));
        idx.delete(id(1), &i(7));
        // After both removed, the bucket is gone : a range that crosses
        // its old key returns empty.
        let bw = idx.query(&IndexAtom::Between(i(5), i(9)));
        assert!(bw.is_empty());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn update_moves_id_across_range_boundary() {
        let mut idx = BTreeIndex::new();
        idx.insert(id(0), &i(5));
        // 5 is in [3, 7]
        assert!(idx.query(&IndexAtom::Between(i(3), i(7))).contains(0));
        idx.update(id(0), &i(5), &i(10));
        // After update, 10 is outside [3, 7]
        assert!(!idx.query(&IndexAtom::Between(i(3), i(7))).contains(0));
        assert!(idx.query(&IndexAtom::Between(i(8), i(12))).contains(0));
    }

    #[test]
    fn cardinality_matches_query_len_for_ranges() {
        let idx = build_year_index();
        let atoms = [
            IndexAtom::Eq(i(2023)),
            IndexAtom::In(vec![i(2020), i(2025)]),
            IndexAtom::Cmp(CmpOp::Lt, i(2025)),
            IndexAtom::Cmp(CmpOp::Le, i(2025)),
            IndexAtom::Cmp(CmpOp::Gt, i(2025)),
            IndexAtom::Cmp(CmpOp::Ge, i(2025)),
            IndexAtom::Between(i(2022), i(2027)),
            IndexAtom::IsNotNull,
            IndexAtom::Cmp(CmpOp::Ne, i(2023)),
        ];
        for atom in &atoms {
            let q_len = idx.query(atom).len();
            let c = idx.cardinality(atom).expect("supported atom");
            assert_eq!(c, q_len, "atom = {atom:?}");
        }
    }

    #[test]
    fn cardinality_returns_none_for_float_ranges() {
        let mut idx = BTreeIndex::new();
        idx.insert(id(0), &f(0.5));
        assert!(
            idx.cardinality(&IndexAtom::Cmp(CmpOp::Lt, f(0.5)))
                .is_none()
        );
        assert!(
            idx.cardinality(&IndexAtom::Between(f(0.0), f(1.0)))
                .is_none()
        );
        // But Eq still gives an exact count.
        assert_eq!(idx.cardinality(&IndexAtom::Eq(f(0.5))), Some(1));
    }

    #[test]
    fn supports_matches_spec() {
        let idx = BTreeIndex::new();

        assert!(idx.supports(&IndexAtom::Eq(i(0))));
        assert!(idx.supports(&IndexAtom::In(vec![i(0)])));
        assert!(idx.supports(&IndexAtom::IsNotNull));
        assert!(idx.supports(&IndexAtom::Cmp(CmpOp::Ne, i(0))));

        assert!(idx.supports(&IndexAtom::Cmp(CmpOp::Lt, i(0))));
        assert!(idx.supports(&IndexAtom::Cmp(CmpOp::Le, i(0))));
        assert!(idx.supports(&IndexAtom::Cmp(CmpOp::Gt, i(0))));
        assert!(idx.supports(&IndexAtom::Cmp(CmpOp::Ge, i(0))));
        assert!(idx.supports(&IndexAtom::Between(i(0), i(10))));

        assert!(!idx.supports(&IndexAtom::Cmp(CmpOp::Lt, f(0.0))));
        assert!(!idx.supports(&IndexAtom::Between(f(0.0), f(1.0))));
        assert!(!idx.supports(&IndexAtom::ArrayContains(s("x"))));
    }
}

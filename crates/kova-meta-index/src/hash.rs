//! Hash-backed equality index.
//!
//! Backing : `HashMap<NormalizedKey, RoaringTreemap>`. Each bucket
//! holds the ids of every row whose indexed field equals that key.
//!
//! ## Atoms supported
//!
//! | Atom | How |
//! |------|-----|
//! | [`IndexAtom::Eq`] | hash lookup, clone bucket bitmap |
//! | [`IndexAtom::In`] | k hash lookups, union of k bitmaps |
//! | [`IndexAtom::IsNotNull`] | clone the running `all_indexed_ids` bitmap |
//! | [`IndexAtom::Cmp`] with [`crate::CmpOp::Ne`] | `all_indexed_ids - bucket` |
//!
//! Range atoms ([`IndexAtom::Between`], [`IndexAtom::Cmp`] with
//! `Lt`/`Le`/`Gt`/`Ge`) are **not** supported : there is no
//! ordering on hash buckets. Use [`crate::BTreeIndex`] for those.
//!
//! ## Cost
//!
//! All supported atoms are `O(1)` lookup + `O(matches)` to clone the
//! bitmap. [`IndexAtom::In`] is `O(k)` lookups + `O(union)` for k
//! buckets.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use kova_core::{Value, VectorId};
use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};

use crate::{CmpOp, IndexAtom, MetaIndex, NormalizedKey};

/// Hash-backed equality index. See [module-level docs](self).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct HashIndex {
    // Map from normalized key to bitmap of matching row ids.
    buckets: HashMap<NormalizedKey, RoaringTreemap>,
    // Bitmap of all indexed row ids. Used to support IsNotNull and Ne atoms.
    all_indexed_ids: RoaringTreemap,
}

impl HashIndex {
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
}

impl MetaIndex for HashIndex {
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

            _ => RoaringTreemap::new(),
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

            _ => None,
        }
    }

    fn len(&self) -> u64 {
        self.all_indexed_ids.len()
    }

    fn supports(&self, atom: &IndexAtom) -> bool {
        matches!(
            atom,
            IndexAtom::Eq(_)
                | IndexAtom::In(_)
                | IndexAtom::IsNotNull
                | IndexAtom::Cmp(CmpOp::Ne, _)
        )
    }
}

#[cfg(test)]
mod tests {
    use kova_core::{Value, VectorId};

    use super::HashIndex;
    use crate::{CmpOp, IndexAtom, MetaIndex};

    fn s(x: &str) -> Value {
        Value::String(x.into())
    }

    fn id(n: u64) -> VectorId {
        VectorId::new(n)
    }

    #[test]
    fn eq_returns_only_matching_ids() {
        let mut idx = HashIndex::new();
        for n in 0..100 {
            let v = if n % 3 == 0 { s("docs") } else { s("blog") };
            idx.insert(id(n), &v);
        }

        let docs = idx.query(&IndexAtom::Eq(s("docs")));
        assert_eq!(docs.len(), 34);
        assert!(docs.contains(0));
        assert!(docs.contains(99));
        assert!(!docs.contains(1));
    }

    #[test]
    fn in_matches_union_of_eq() {
        let mut idx = HashIndex::new();
        for n in 0..30 {
            let v = match n % 3 {
                0 => s("a"),
                1 => s("b"),
                _ => s("c"),
            };
            idx.insert(id(n), &v);
        }

        let in_ab = idx.query(&IndexAtom::In(vec![s("a"), s("b")]));
        let union = idx.query(&IndexAtom::Eq(s("a"))) | idx.query(&IndexAtom::Eq(s("b")));
        assert_eq!(in_ab, union);
        assert_eq!(in_ab.len(), 20);
    }

    #[test]
    fn delete_removes_from_bucket_and_all_indexed_ids() {
        let mut idx = HashIndex::new();
        idx.insert(id(1), &s("a"));
        idx.insert(id(2), &s("a"));
        idx.insert(id(3), &s("b"));

        assert_eq!(idx.len(), 3);

        idx.delete(id(2), &s("a"));
        let a = idx.query(&IndexAtom::Eq(s("a")));
        assert!(a.contains(1));
        assert!(!a.contains(2));
        assert_eq!(a.len(), 1);
        assert_eq!(idx.len(), 2);

        idx.delete(id(1), &s("a"));
        assert!(idx.query(&IndexAtom::Eq(s("a"))).is_empty());
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn non_indexable_values_are_skipped() {
        let mut idx = HashIndex::new();
        idx.insert(id(1), &Value::Array(vec![s("rust")]));
        assert_eq!(idx.len(), 0);
        assert!(
            idx.query(&IndexAtom::Eq(Value::Array(vec![s("rust")])))
                .is_empty()
        );
    }

    #[test]
    fn is_not_null_returns_all_indexed_ids() {
        let mut idx = HashIndex::new();
        idx.insert(id(1), &s("a"));
        idx.insert(id(2), &s("b"));
        idx.insert(id(3), &Value::Array(vec![s("x")]));

        let live = idx.query(&IndexAtom::IsNotNull);
        assert_eq!(live.len(), 2);
        assert!(live.contains(1));
        assert!(live.contains(2));
        assert!(!live.contains(3));
    }

    #[test]
    fn ne_is_all_minus_bucket() {
        let mut idx = HashIndex::new();
        for n in 0..10 {
            let v = if n < 4 { s("a") } else { s("b") };
            idx.insert(id(n), &v);
        }

        let not_a = idx.query(&IndexAtom::Cmp(CmpOp::Ne, s("a")));
        assert_eq!(not_a.len(), 6);
        for n in 4..10 {
            assert!(not_a.contains(n));
        }
        for n in 0..4 {
            assert!(!not_a.contains(n));
        }
    }

    #[test]
    fn update_with_same_key_is_noop() {
        let mut idx = HashIndex::new();
        idx.insert(id(1), &s("a"));
        idx.update(id(1), &s("a"), &s("a"));
        let a = idx.query(&IndexAtom::Eq(s("a")));
        assert_eq!(a.len(), 1);
        assert!(a.contains(1));
    }

    #[test]
    fn update_moves_id_between_buckets() {
        let mut idx = HashIndex::new();
        idx.insert(id(1), &s("a"));
        idx.update(id(1), &s("a"), &s("b"));

        assert!(idx.query(&IndexAtom::Eq(s("a"))).is_empty());
        let b = idx.query(&IndexAtom::Eq(s("b")));
        assert_eq!(b.len(), 1);
        assert!(b.contains(1));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn cardinality_matches_query_len() {
        let mut idx = HashIndex::new();
        for n in 0..50 {
            let v = if n % 5 == 0 { s("hot") } else { s("cold") };
            idx.insert(id(n), &v);
        }

        let atoms = [
            IndexAtom::Eq(s("hot")),
            IndexAtom::Eq(s("missing")),
            IndexAtom::In(vec![s("hot"), s("cold")]),
            IndexAtom::IsNotNull,
            IndexAtom::Cmp(CmpOp::Ne, s("hot")),
        ];

        for atom in &atoms {
            let q_len = idx.query(atom).len();
            let c = idx.cardinality(atom).expect("supported atom");
            assert_eq!(c, q_len, "atom = {atom:?}");
        }
    }

    #[test]
    fn supports_matches_spec() {
        let idx = HashIndex::new();
        assert!(idx.supports(&IndexAtom::Eq(s("x"))));
        assert!(idx.supports(&IndexAtom::In(vec![s("x")])));
        assert!(idx.supports(&IndexAtom::IsNotNull));
        assert!(idx.supports(&IndexAtom::Cmp(CmpOp::Ne, s("x"))));

        assert!(!idx.supports(&IndexAtom::Cmp(CmpOp::Lt, s("x"))));
        assert!(!idx.supports(&IndexAtom::Cmp(CmpOp::Le, s("x"))));
        assert!(!idx.supports(&IndexAtom::Cmp(CmpOp::Gt, s("x"))));
        assert!(!idx.supports(&IndexAtom::Cmp(CmpOp::Ge, s("x"))));
        assert!(!idx.supports(&IndexAtom::Between(s("a"), s("z"))));
        assert!(!idx.supports(&IndexAtom::ArrayContains(s("x"))));
    }
}

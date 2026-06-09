//! Inverted index for array containment.
//!
//! Backing : `HashMap<NormalizedKey, RoaringTreemap>`, same shape as
//! [`crate::HashIndex`], but the semantics differ. For a row with
//! metadata `{"tags": ["a", "b", "c"]}`, the row's id ends up in the
//! buckets for `"a"`, `"b"`, AND `"c"`. The query `tags @> 'b'`
//! returns the `"b"` bucket directly.
//!
//! Multi-tag predicates like `tags @> 'a' AND tags @> 'b'` are
//! handled by the executor : it issues two queries against this
//! index and intersects the resulting bitmaps. The index itself
//! only answers single-element containment.
//!
//! ## Atoms supported
//!
//! | Atom | How |
//! |------|-----|
//! | [`IndexAtom::ArrayContains`] | hash lookup, clone bucket bitmap |
//! | [`IndexAtom::IsNotNull`] | clone the running `all_indexed_ids` bitmap |
//!
//! Everything else (`Eq`, `In`, `Cmp`, `Between`) is rejected by
//! [`MetaIndex::supports`] and the executor falls back to a metadata
//! scan.
//!
//! ## Cost
//!
//! `ArrayContains` is `O(1)` lookup + `O(matches)` clone. The
//! insert/delete cost is `O(array_length)` because the same id is
//! threaded into one bucket per array element.
//!
//! ## What about non-array fields ?
//!
//! Insert is a no-op when the value isn't a [`Value::Array`]. This
//! keeps the index valid even if some rows have the indexed field
//! with the wrong shape (we just don't index those rows). The
//! `query` path returns empty for them via the natural bucket
//! lookup (they were never inserted).
//!
//! An empty array `[]` is treated as "indexed but in no bucket" :
//! the row joins `all_indexed_ids` (so `IsNotNull` sees it) but no
//! `ArrayContains` query matches it. This matches how SQL and Mongo
//! treat empty arrays : not-null but contains nothing.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use kova_core::{Value, VectorId};
use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};

use crate::{IndexAtom, MetaIndex, NormalizedKey};

/// Inverted index for array containment. See [module-level docs](self).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct InvertedIndex {
    // Map from normalized element key to bitmap of rows whose array contains it.
    buckets: HashMap<NormalizedKey, RoaringTreemap>,
    // Bitmap of all rows that supplied a Value::Array (including empty arrays).
    all_indexed_ids: RoaringTreemap,
}

impl InvertedIndex {
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

    fn insert_element(&mut self, id: VectorId, element: &Value) {
        let Some(key) = NormalizedKey::from_value(element) else {
            return;
        };
        self.buckets.entry(key).or_default().insert(id.into());
    }

    fn delete_element(&mut self, id: VectorId, element: &Value) {
        let Some(key) = NormalizedKey::from_value(element) else {
            return;
        };
        if let Entry::Occupied(mut entry) = self.buckets.entry(key) {
            entry.get_mut().remove(id.into());
            if entry.get().is_empty() {
                entry.remove();
            }
        }
    }
}

impl MetaIndex for InvertedIndex {
    fn build<I>(&mut self, rows: I)
    where
        I: IntoIterator<Item = (VectorId, Value)>,
    {
        for (id, value) in rows {
            self.insert(id, &value);
        }
    }

    fn insert(&mut self, id: VectorId, value: &Value) {
        let Value::Array(elements) = value else {
            return;
        };
        self.all_indexed_ids.insert(id.into());
        for element in elements {
            self.insert_element(id, element);
        }
    }

    fn delete(&mut self, id: VectorId, value: &Value) {
        let Value::Array(elements) = value else {
            return;
        };
        self.all_indexed_ids.remove(id.into());
        for element in elements {
            self.delete_element(id, element);
        }
    }

    fn update(&mut self, id: VectorId, old: &Value, new: &Value) {
        if old == new {
            return;
        }
        self.delete(id, old);
        self.insert(id, new);
    }

    fn query(&self, atom: &IndexAtom) -> RoaringTreemap {
        match atom {
            IndexAtom::ArrayContains(v) => self.bucket_for(v).cloned().unwrap_or_default(),
            IndexAtom::IsNotNull => self.all_indexed_ids.clone(),
            _ => RoaringTreemap::new(),
        }
    }

    fn cardinality(&self, atom: &IndexAtom) -> Option<u64> {
        match atom {
            IndexAtom::ArrayContains(v) => Some(self.bucket_for(v).map_or(0, RoaringTreemap::len)),
            IndexAtom::IsNotNull => Some(self.all_indexed_ids.len()),
            _ => None,
        }
    }

    fn len(&self) -> u64 {
        self.all_indexed_ids.len()
    }

    fn supports(&self, atom: &IndexAtom) -> bool {
        matches!(atom, IndexAtom::ArrayContains(_) | IndexAtom::IsNotNull)
    }
}

#[cfg(test)]
mod tests {
    use kova_core::{Value, VectorId};

    use super::InvertedIndex;
    use crate::{CmpOp, IndexAtom, MetaIndex};

    fn s(x: &str) -> Value {
        Value::String(x.into())
    }

    fn arr(xs: &[&str]) -> Value {
        Value::Array(xs.iter().map(|x| s(x)).collect())
    }

    fn id(n: u64) -> VectorId {
        VectorId::new(n)
    }

    #[test]
    fn array_contains_returns_only_matching_ids() {
        let mut idx = InvertedIndex::new();
        idx.insert(id(0), &arr(&["rust", "async"]));
        idx.insert(id(1), &arr(&["go", "async"]));
        idx.insert(id(2), &arr(&["python"]));

        let rust = idx.query(&IndexAtom::ArrayContains(s("rust")));
        assert_eq!(rust.len(), 1);
        assert!(rust.contains(0));

        let async_ = idx.query(&IndexAtom::ArrayContains(s("async")));
        assert_eq!(async_.len(), 2);
        assert!(async_.contains(0));
        assert!(async_.contains(1));

        let go = idx.query(&IndexAtom::ArrayContains(s("go")));
        assert_eq!(go.len(), 1);
        assert!(go.contains(1));

        let missing = idx.query(&IndexAtom::ArrayContains(s("ruby")));
        assert!(missing.is_empty());
    }

    #[test]
    fn id_appears_in_every_element_bucket() {
        let mut idx = InvertedIndex::new();
        idx.insert(id(7), &arr(&["a", "b", "c", "d"]));
        for tag in ["a", "b", "c", "d"] {
            let q = idx.query(&IndexAtom::ArrayContains(s(tag)));
            assert!(q.contains(7), "id 7 missing from bucket {tag}");
        }
    }

    #[test]
    fn multi_tag_and_via_executor_intersection() {
        let mut idx = InvertedIndex::new();
        idx.insert(id(0), &arr(&["rust", "async"]));
        idx.insert(id(1), &arr(&["rust", "sync"]));
        idx.insert(id(2), &arr(&["go", "async"]));

        // The executor would AND `tags @> 'rust' AND tags @> 'async'`
        // by intersecting two bitmaps. We simulate that here.
        let rust = idx.query(&IndexAtom::ArrayContains(s("rust")));
        let async_ = idx.query(&IndexAtom::ArrayContains(s("async")));
        let both = rust & async_;
        assert_eq!(both.len(), 1);
        assert!(both.contains(0));
    }

    #[test]
    fn delete_removes_from_every_bucket() {
        let mut idx = InvertedIndex::new();
        idx.insert(id(0), &arr(&["a", "b"]));
        idx.insert(id(1), &arr(&["a"]));
        idx.delete(id(0), &arr(&["a", "b"]));

        let a = idx.query(&IndexAtom::ArrayContains(s("a")));
        assert_eq!(a.len(), 1);
        assert!(a.contains(1));
        assert!(!a.contains(0));

        // "b" only had id 0, so deleting it drops the whole bucket
        assert!(idx.query(&IndexAtom::ArrayContains(s("b"))).is_empty());
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn empty_array_is_indexed_but_in_no_bucket() {
        let mut idx = InvertedIndex::new();
        idx.insert(id(0), &Value::Array(vec![]));
        idx.insert(id(1), &arr(&["x"]));

        assert_eq!(idx.len(), 2);
        let live = idx.query(&IndexAtom::IsNotNull);
        assert!(live.contains(0));
        assert!(live.contains(1));

        // ArrayContains(anything) doesn't match the empty-array row
        assert!(!idx.query(&IndexAtom::ArrayContains(s("x"))).contains(0));
    }

    #[test]
    fn non_array_value_is_skipped() {
        let mut idx = InvertedIndex::new();
        idx.insert(id(0), &s("rust"));
        idx.insert(id(1), &Value::I64(42));
        idx.insert(id(2), &arr(&["rust"]));

        assert_eq!(idx.len(), 1);
        let rust = idx.query(&IndexAtom::ArrayContains(s("rust")));
        assert_eq!(rust.len(), 1);
        assert!(rust.contains(2));
        assert!(!rust.contains(0));
    }

    #[test]
    fn non_normalizable_element_is_skipped_siblings_kept() {
        let mut idx = InvertedIndex::new();
        // The Map element can't normalize to a NormalizedKey ;
        // string siblings still go into their buckets.
        let v = Value::Array(vec![
            s("rust"),
            Value::Map(std::collections::HashMap::new()),
            s("async"),
        ]);
        idx.insert(id(0), &v);

        assert_eq!(idx.len(), 1);
        assert!(idx.query(&IndexAtom::ArrayContains(s("rust"))).contains(0));
        assert!(idx.query(&IndexAtom::ArrayContains(s("async"))).contains(0));
    }

    #[test]
    fn update_moves_id_through_bucket_set_change() {
        let mut idx = InvertedIndex::new();
        idx.insert(id(0), &arr(&["a", "b"]));
        idx.update(id(0), &arr(&["a", "b"]), &arr(&["a", "c"]));

        // 'a' still has id 0 ; 'b' lost it ; 'c' gained it.
        assert!(idx.query(&IndexAtom::ArrayContains(s("a"))).contains(0));
        assert!(idx.query(&IndexAtom::ArrayContains(s("b"))).is_empty());
        assert!(idx.query(&IndexAtom::ArrayContains(s("c"))).contains(0));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn update_with_identical_value_is_noop() {
        let mut idx = InvertedIndex::new();
        idx.insert(id(0), &arr(&["a", "b"]));
        idx.update(id(0), &arr(&["a", "b"]), &arr(&["a", "b"]));
        assert!(idx.query(&IndexAtom::ArrayContains(s("a"))).contains(0));
        assert!(idx.query(&IndexAtom::ArrayContains(s("b"))).contains(0));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn cardinality_matches_query_len() {
        let mut idx = InvertedIndex::new();
        idx.insert(id(0), &arr(&["a", "b"]));
        idx.insert(id(1), &arr(&["a"]));
        idx.insert(id(2), &arr(&["c"]));
        idx.insert(id(3), &Value::Array(vec![]));

        let atoms = [
            IndexAtom::ArrayContains(s("a")),
            IndexAtom::ArrayContains(s("missing")),
            IndexAtom::IsNotNull,
        ];

        for atom in &atoms {
            let q_len = idx.query(atom).len();
            let c = idx.cardinality(atom).expect("supported atom");
            assert_eq!(c, q_len, "atom = {atom:?}");
        }
    }

    #[test]
    fn unsupported_atoms_return_empty_and_none() {
        let mut idx = InvertedIndex::new();
        idx.insert(id(0), &arr(&["a"]));

        let unsupported = [
            IndexAtom::Eq(s("a")),
            IndexAtom::In(vec![s("a")]),
            IndexAtom::Cmp(CmpOp::Lt, s("a")),
            IndexAtom::Cmp(CmpOp::Ne, s("a")),
            IndexAtom::Between(s("a"), s("z")),
        ];

        for atom in &unsupported {
            assert!(idx.query(atom).is_empty(), "atom = {atom:?}");
            assert!(idx.cardinality(atom).is_none(), "atom = {atom:?}");
            assert!(!idx.supports(atom), "atom = {atom:?}");
        }
    }

    #[test]
    fn supports_matches_spec() {
        let idx = InvertedIndex::new();
        assert!(idx.supports(&IndexAtom::ArrayContains(s("x"))));
        assert!(idx.supports(&IndexAtom::IsNotNull));

        assert!(!idx.supports(&IndexAtom::Eq(s("x"))));
        assert!(!idx.supports(&IndexAtom::In(vec![s("x")])));
        assert!(!idx.supports(&IndexAtom::Cmp(CmpOp::Lt, s("x"))));
        assert!(!idx.supports(&IndexAtom::Cmp(CmpOp::Le, s("x"))));
        assert!(!idx.supports(&IndexAtom::Cmp(CmpOp::Gt, s("x"))));
        assert!(!idx.supports(&IndexAtom::Cmp(CmpOp::Ge, s("x"))));
        assert!(!idx.supports(&IndexAtom::Cmp(CmpOp::Ne, s("x"))));
        assert!(!idx.supports(&IndexAtom::Between(s("a"), s("z"))));
    }

    #[test]
    fn build_populates_buckets() {
        let mut idx = InvertedIndex::new();
        let rows = vec![
            (id(0), arr(&["a", "b"])),
            (id(1), arr(&["a"])),
            (id(2), arr(&["c"])),
        ];
        idx.build(rows);
        assert_eq!(idx.len(), 3);
        assert_eq!(idx.query(&IndexAtom::ArrayContains(s("a"))).len(), 2);
        assert_eq!(idx.query(&IndexAtom::ArrayContains(s("b"))).len(), 1);
        assert_eq!(idx.query(&IndexAtom::ArrayContains(s("c"))).len(), 1);
    }
}

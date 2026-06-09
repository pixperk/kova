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
//!
//! That's the entire surface. Everything else falls back.
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

use kova_core::{Value, VectorId};
use roaring::RoaringTreemap;

use crate::{IndexAtom, MetaIndex};

/// Inverted index for array containment. See [module-level docs](self).
#[derive(Debug, Default)]
pub struct InvertedIndex {
    // implementation lands in slice 3 of M2.1
}

impl InvertedIndex {
    /// Construct an empty index. Bulk-build with
    /// [`MetaIndex::build`] or populate incrementally with
    /// [`MetaIndex::insert`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl MetaIndex for InvertedIndex {
    fn build<I>(&mut self, _rows: I)
    where
        I: IntoIterator<Item = (VectorId, Value)>,
    {
        todo!("InvertedIndex::build : slice 3 of M2.1")
    }

    fn insert(&mut self, _id: VectorId, _value: &Value) {
        todo!("InvertedIndex::insert : slice 3 of M2.1")
    }

    fn delete(&mut self, _id: VectorId, _value: &Value) {
        todo!("InvertedIndex::delete : slice 3 of M2.1")
    }

    fn update(&mut self, _id: VectorId, _old: &Value, _new: &Value) {
        todo!("InvertedIndex::update : slice 3 of M2.1")
    }

    fn query(&self, _atom: &IndexAtom) -> RoaringTreemap {
        todo!("InvertedIndex::query : slice 3 of M2.1")
    }

    fn cardinality(&self, _atom: &IndexAtom) -> Option<u64> {
        todo!("InvertedIndex::cardinality : slice 3 of M2.1")
    }

    fn len(&self) -> u64 {
        todo!("InvertedIndex::len : slice 3 of M2.1")
    }

    fn supports(&self, _atom: &IndexAtom) -> bool {
        todo!("InvertedIndex::supports : slice 3 of M2.1")
    }
}

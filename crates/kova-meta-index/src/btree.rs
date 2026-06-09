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
//! bit-ordering diverges from numeric ordering. v1 [`BTreeIndex`]
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

use kova_core::{Value, VectorId};
use roaring::RoaringTreemap;

use crate::{IndexAtom, MetaIndex};

/// BTree-backed range index. See [module-level docs](self).
#[derive(Debug, Default)]
pub struct BTreeIndex {
    // implementation lands in slice 2 of M2.1
}

impl BTreeIndex {
    /// Construct an empty index. Bulk-build with
    /// [`MetaIndex::build`] or populate incrementally with
    /// [`MetaIndex::insert`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl MetaIndex for BTreeIndex {
    fn build<I>(&mut self, _rows: I)
    where
        I: IntoIterator<Item = (VectorId, Value)>,
    {
        todo!("BTreeIndex::build : slice 2 of M2.1")
    }

    fn insert(&mut self, _id: VectorId, _value: &Value) {
        todo!("BTreeIndex::insert : slice 2 of M2.1")
    }

    fn delete(&mut self, _id: VectorId, _value: &Value) {
        todo!("BTreeIndex::delete : slice 2 of M2.1")
    }

    fn update(&mut self, _id: VectorId, _old: &Value, _new: &Value) {
        todo!("BTreeIndex::update : slice 2 of M2.1")
    }

    fn query(&self, _atom: &IndexAtom) -> RoaringTreemap {
        todo!("BTreeIndex::query : slice 2 of M2.1")
    }

    fn cardinality(&self, _atom: &IndexAtom) -> Option<u64> {
        todo!("BTreeIndex::cardinality : slice 2 of M2.1")
    }

    fn len(&self) -> u64 {
        todo!("BTreeIndex::len : slice 2 of M2.1")
    }

    fn supports(&self, _atom: &IndexAtom) -> bool {
        todo!("BTreeIndex::supports : slice 2 of M2.1")
    }
}

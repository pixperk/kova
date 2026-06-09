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

use kova_core::{Value, VectorId};
use roaring::RoaringTreemap;

use crate::{IndexAtom, MetaIndex};

/// Hash-backed equality index. See [module-level docs](self).
#[derive(Debug, Default)]
pub struct HashIndex {
    // implementation lands in slice 1 of M2.1
}

impl HashIndex {
    /// Construct an empty index. Bulk-build with
    /// [`MetaIndex::build`] or populate incrementally with
    /// [`MetaIndex::insert`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl MetaIndex for HashIndex {
    fn build<I>(&mut self, _rows: I)
    where
        I: IntoIterator<Item = (VectorId, Value)>,
    {
        todo!("HashIndex::build : slice 1 of M2.1")
    }

    fn insert(&mut self, _id: VectorId, _value: &Value) {
        todo!("HashIndex::insert : slice 1 of M2.1")
    }

    fn delete(&mut self, _id: VectorId, _value: &Value) {
        todo!("HashIndex::delete : slice 1 of M2.1")
    }

    fn update(&mut self, _id: VectorId, _old: &Value, _new: &Value) {
        todo!("HashIndex::update : slice 1 of M2.1")
    }

    fn query(&self, _atom: &IndexAtom) -> RoaringTreemap {
        todo!("HashIndex::query : slice 1 of M2.1")
    }

    fn cardinality(&self, _atom: &IndexAtom) -> Option<u64> {
        todo!("HashIndex::cardinality : slice 1 of M2.1")
    }

    fn len(&self) -> u64 {
        todo!("HashIndex::len : slice 1 of M2.1")
    }

    fn supports(&self, _atom: &IndexAtom) -> bool {
        todo!("HashIndex::supports : slice 1 of M2.1")
    }
}

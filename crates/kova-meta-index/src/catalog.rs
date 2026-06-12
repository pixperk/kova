//! Per-shard catalog of meta-indexes.
//!
//! Each shard owns one [`IndexCatalog`]. The catalog maps a field
//! name to the set of indexes built on that field (hash, btree,
//! and/or inverted). The shard's three-phase `insert`/`delete`/
//! `update` ops forward to [`IndexCatalog::on_insert`], etc., after
//! the WAL commit. Query planning calls [`IndexCatalog::lookup`] to
//! get a candidate id bitmap, or [`IndexCatalog::estimate`] for a
//! cheap cardinality without doing the lookup.
//!
//! ## Field can have multiple indexes
//!
//! A single field can carry both a [`HashIndex`] (for `Eq`/`In`)
//! and a [`BTreeIndex`] (for ranges). The catalog routes each atom
//! to the cheapest supporting index. Hash beats btree for equality
//! lookups ; btree wins for ranges (because hash can't answer
//! them).
//!
//! ## Routing priority
//!
//! For a `(field, atom)` lookup the catalog asks each available
//! index on the field in this order :
//!
//! 1. [`HashIndex`]
//! 2. [`BTreeIndex`]
//! 3. [`InvertedIndex`]
//!
//! and returns the first bitmap from an index whose
//! [`MetaIndex::supports`] returns `true`. The ordering is by
//! lookup cost, not by selectivity ; all supporting indexes return
//! the same bitmap.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use kova_core::{Metadata, Value, VectorId};
use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};

use crate::error::KovaMetaIndexError;
use crate::{BTreeIndex, HashIndex, IndexAtom, InvertedIndex, MetaIndex};

/// Magic header on every catalog file. 8 bytes, ASCII for "KOVAIDX1".
const CATALOG_MAGIC: &[u8; 8] = b"KOVAIDX1";

/// Bumped when the on-disk catalog layout changes incompatibly.
const CATALOG_FORMAT_VERSION: u32 = 1;

/// Fixed header bytes : magic + version.
const CATALOG_HEADER_LEN: usize = CATALOG_MAGIC.len() + std::mem::size_of::<u32>();

/// Which of the three index types a particular registration refers
/// to. Used by the named-index registry (see
/// [`IndexCatalog::create_named_index`]) to remember what kind of
/// index a name binds to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexKind {
    /// [`HashIndex`].
    Hash,
    /// [`BTreeIndex`].
    Btree,
    /// [`InvertedIndex`].
    Inverted,
}

/// Catalog of meta-indexes for one shard. See [module-level docs](self).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IndexCatalog {
    fields: HashMap<String, FieldIndexes>,
    /// Named-index registry, populated by DDL
    /// (`CREATE INDEX <name> ...`). Programmatic
    /// `add_*_index(field)` calls leave the underlying index
    /// anonymous : they don't add an entry here. DROP INDEX by name
    /// only resolves named entries.
    #[serde(default)]
    named_indexes: HashMap<String, (String, IndexKind)>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct FieldIndexes {
    hash: Option<HashIndex>,
    btree: Option<BTreeIndex>,
    inverted: Option<InvertedIndex>,
}

impl FieldIndexes {
    /// True if every index type slot is empty. Used by the catalog
    /// to garbage-collect the field entry after the last index on
    /// it is dropped.
    fn is_empty(&self) -> bool {
        self.hash.is_none() && self.btree.is_none() && self.inverted.is_none()
    }
}

impl FieldIndexes {
    fn on_insert(&mut self, id: VectorId, value: &Value) {
        if let Some(h) = self.hash.as_mut() {
            h.insert(id, value);
        }
        if let Some(b) = self.btree.as_mut() {
            b.insert(id, value);
        }
        if let Some(i) = self.inverted.as_mut() {
            i.insert(id, value);
        }
    }

    fn on_delete(&mut self, id: VectorId, value: &Value) {
        if let Some(h) = self.hash.as_mut() {
            h.delete(id, value);
        }
        if let Some(b) = self.btree.as_mut() {
            b.delete(id, value);
        }
        if let Some(i) = self.inverted.as_mut() {
            i.delete(id, value);
        }
    }

    fn on_update(&mut self, id: VectorId, old: &Value, new: &Value) {
        if let Some(h) = self.hash.as_mut() {
            h.update(id, old, new);
        }
        if let Some(b) = self.btree.as_mut() {
            b.update(id, old, new);
        }
        if let Some(i) = self.inverted.as_mut() {
            i.update(id, old, new);
        }
    }

    fn lookup(&self, atom: &IndexAtom) -> Option<RoaringTreemap> {
        if let Some(h) = self.hash.as_ref()
            && h.supports(atom)
        {
            return Some(h.query(atom));
        }
        if let Some(b) = self.btree.as_ref()
            && b.supports(atom)
        {
            return Some(b.query(atom));
        }
        if let Some(i) = self.inverted.as_ref()
            && i.supports(atom)
        {
            return Some(i.query(atom));
        }
        None
    }

    fn estimate(&self, atom: &IndexAtom) -> Option<u64> {
        if let Some(h) = self.hash.as_ref()
            && h.supports(atom)
        {
            return h.cardinality(atom);
        }
        if let Some(b) = self.btree.as_ref()
            && b.supports(atom)
        {
            return b.cardinality(atom);
        }
        if let Some(i) = self.inverted.as_ref()
            && i.supports(atom)
        {
            return i.cardinality(atom);
        }
        None
    }
}

impl IndexCatalog {
    /// Construct an empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new empty [`HashIndex`] on `field`. The new
    /// index is empty ; use [`Self::populate_field`] to backfill
    /// from existing data.
    ///
    /// # Errors
    /// Returns [`KovaMetaIndexError::IndexAlreadyExists`] if a
    /// hash index already exists on `field`. Each `(field, kind)`
    /// slot holds at most one index ; callers that want to rebuild
    /// must drop the existing one explicitly first.
    pub fn add_hash_index(&mut self, field: &str) -> Result<(), KovaMetaIndexError> {
        if self.has_kind_on(field, IndexKind::Hash) {
            return Err(KovaMetaIndexError::IndexAlreadyExists {
                field: field.to_string(),
                kind: IndexKind::Hash,
            });
        }
        self.fields.entry(field.to_string()).or_default().hash = Some(HashIndex::new());
        Ok(())
    }

    /// Register a new empty [`BTreeIndex`] on `field`.
    ///
    /// # Errors
    /// See [`Self::add_hash_index`].
    pub fn add_btree_index(&mut self, field: &str) -> Result<(), KovaMetaIndexError> {
        if self.has_kind_on(field, IndexKind::Btree) {
            return Err(KovaMetaIndexError::IndexAlreadyExists {
                field: field.to_string(),
                kind: IndexKind::Btree,
            });
        }
        self.fields.entry(field.to_string()).or_default().btree = Some(BTreeIndex::new());
        Ok(())
    }

    /// Register a new empty [`InvertedIndex`] on `field`.
    ///
    /// # Errors
    /// See [`Self::add_hash_index`].
    pub fn add_inverted_index(&mut self, field: &str) -> Result<(), KovaMetaIndexError> {
        if self.has_kind_on(field, IndexKind::Inverted) {
            return Err(KovaMetaIndexError::IndexAlreadyExists {
                field: field.to_string(),
                kind: IndexKind::Inverted,
            });
        }
        self.fields.entry(field.to_string()).or_default().inverted = Some(InvertedIndex::new());
        Ok(())
    }

    /// Bulk-load every index attached to `field` from a fresh row
    /// iterator. The iterator is materialised once and broadcast to
    /// each index. Existing index state on the field is **not**
    /// cleared first ; call [`Self::add_hash_index`] (etc.) to
    /// reset, then `populate_field` to fill.
    ///
    /// No-op if the field has no registered indexes.
    pub fn populate_field<I>(&mut self, field: &str, rows: I)
    where
        I: IntoIterator<Item = (VectorId, Value)>,
    {
        let Some(fi) = self.fields.get_mut(field) else {
            return;
        };
        let rows: Vec<(VectorId, Value)> = rows.into_iter().collect();
        if let Some(h) = fi.hash.as_mut() {
            for (id, v) in &rows {
                h.insert(*id, v);
            }
        }
        if let Some(b) = fi.btree.as_mut() {
            for (id, v) in &rows {
                b.insert(*id, v);
            }
        }
        if let Some(i) = fi.inverted.as_mut() {
            for (id, v) in &rows {
                i.insert(*id, v);
            }
        }
    }

    /// Targeted variant of [`Self::populate_field`] : insert each
    /// `(id, value)` row into ONLY the index of `kind` on `field`.
    /// Used by backfill when we know exactly which index we're
    /// populating (every call site does — `Shard::add_*_index`
    /// names the kind, `Shard::create_index` carries it, and the
    /// `Record::CreateIndex` replay arm has it in the record).
    ///
    /// No-op when the targeted slot is empty.
    pub fn populate_kind<I>(&mut self, field: &str, kind: IndexKind, rows: I)
    where
        I: IntoIterator<Item = (VectorId, Value)>,
    {
        let Some(fi) = self.fields.get_mut(field) else {
            return;
        };
        match kind {
            IndexKind::Hash => {
                if let Some(h) = fi.hash.as_mut() {
                    for (id, v) in rows {
                        h.insert(id, &v);
                    }
                }
            }
            IndexKind::Btree => {
                if let Some(b) = fi.btree.as_mut() {
                    for (id, v) in rows {
                        b.insert(id, &v);
                    }
                }
            }
            IndexKind::Inverted => {
                if let Some(i) = fi.inverted.as_mut() {
                    for (id, v) in rows {
                        i.insert(id, &v);
                    }
                }
            }
        }
    }

    /// Forward a row insertion to every index that watches a field
    /// present in `metadata`. Fields not in `metadata` are skipped
    /// for this row.
    pub fn on_insert(&mut self, id: VectorId, metadata: &Metadata) {
        for (field, fi) in &mut self.fields {
            if let Some(value) = metadata.get(field) {
                fi.on_insert(id, value);
            }
        }
    }

    /// Forward a row deletion to every index that watches a field
    /// present in `metadata`.
    pub fn on_delete(&mut self, id: VectorId, metadata: &Metadata) {
        for (field, fi) in &mut self.fields {
            if let Some(value) = metadata.get(field) {
                fi.on_delete(id, value);
            }
        }
    }

    /// Forward a row update to every watched field, handling the
    /// four presence cases :
    ///
    /// - present in both `old` and `new` : `update(id, old, new)`
    /// - present in `old` only : `delete(id, old)`
    /// - present in `new` only : `insert(id, new)`
    /// - absent from both : skipped
    pub fn on_update(&mut self, id: VectorId, old: &Metadata, new: &Metadata) {
        for (field, fi) in &mut self.fields {
            match (old.get(field), new.get(field)) {
                (Some(o), Some(n)) => fi.on_update(id, o, n),
                (Some(o), None) => fi.on_delete(id, o),
                (None, Some(n)) => fi.on_insert(id, n),
                (None, None) => {}
            }
        }
    }

    /// Look up matching ids for an atom against a field. Returns
    /// `None` if no index on the field can answer the atom (caller
    /// falls back to a metadata scan). Returns `Some(bitmap)` on
    /// success ; the bitmap may be empty if the atom matches no
    /// rows.
    #[must_use]
    pub fn lookup(&self, field: &str, atom: &IndexAtom) -> Option<RoaringTreemap> {
        self.fields.get(field)?.lookup(atom)
    }

    /// Estimate the number of rows an atom on a field would match,
    /// without doing the lookup. Returns `None` for the same
    /// reasons [`MetaIndex::cardinality`] does (no supporting
    /// index, or the index declines to estimate cheaply).
    #[must_use]
    pub fn estimate(&self, field: &str, atom: &IndexAtom) -> Option<u64> {
        self.fields.get(field)?.estimate(atom)
    }

    /// True if any index is registered on `field`.
    #[must_use]
    pub fn has_index_on(&self, field: &str) -> bool {
        self.fields.contains_key(field)
    }

    /// Iterate the names of all fields with at least one registered index.
    pub fn indexed_fields(&self) -> impl Iterator<Item = &str> {
        self.fields.keys().map(String::as_str)
    }

    /// True if an index of `kind` is registered on `field`.
    #[must_use]
    pub fn has_kind_on(&self, field: &str, kind: IndexKind) -> bool {
        let Some(fi) = self.fields.get(field) else {
            return false;
        };
        match kind {
            IndexKind::Hash => fi.hash.is_some(),
            IndexKind::Btree => fi.btree.is_some(),
            IndexKind::Inverted => fi.inverted.is_some(),
        }
    }

    /// Register a named index in the catalog's name registry and
    /// add the underlying index of the requested kind on `field`.
    /// Use this when DDL gives an index a name (`CREATE INDEX foo
    /// ON tbl (col)`) ; the programmatic
    /// [`Self::add_hash_index`] family stays anonymous.
    ///
    /// Strict on **both** axes :
    ///
    /// - the name registry must not already hold `name`
    /// - the `(field, kind)` slot must not already hold an index
    ///
    /// Callers that want to rebuild must
    /// [`Self::drop_named_index`] (or [`Self::drop_hash_index`]
    /// etc.) before re-adding. Idempotent-replace was the v1
    /// behaviour ; it caused silent state loss and was removed.
    ///
    /// # Errors
    /// - [`KovaMetaIndexError::IndexNameInUse`] if `name` is
    ///   already registered.
    /// - [`KovaMetaIndexError::IndexAlreadyExists`] if an index of
    ///   `kind` already exists on `field`.
    pub fn create_named_index(
        &mut self,
        name: &str,
        field: &str,
        kind: IndexKind,
    ) -> Result<(), KovaMetaIndexError> {
        if self.named_indexes.contains_key(name) {
            return Err(KovaMetaIndexError::IndexNameInUse {
                name: name.to_string(),
            });
        }
        // The strict `add_*_index` methods enforce the slot check
        // themselves, but we run it here too so the error mentions
        // both `name` and `(field, kind)` in the right order : the
        // duplicate-name case is more common at the DDL surface, so
        // it should win when both fire.
        match kind {
            IndexKind::Hash => self.add_hash_index(field)?,
            IndexKind::Btree => self.add_btree_index(field)?,
            IndexKind::Inverted => self.add_inverted_index(field)?,
        }
        self.named_indexes
            .insert(name.to_string(), (field.to_string(), kind));
        Ok(())
    }

    /// Drop a named index : unregister `name` and remove the
    /// underlying index of the kind it pointed to.
    ///
    /// # Errors
    /// Returns [`KovaMetaIndexError::UnknownIndexName`] if `name`
    /// was never registered.
    pub fn drop_named_index(&mut self, name: &str) -> Result<(), KovaMetaIndexError> {
        let Some((field, kind)) = self.named_indexes.remove(name) else {
            return Err(KovaMetaIndexError::UnknownIndexName {
                name: name.to_string(),
            });
        };
        self.drop_kind(&field, kind);
        Ok(())
    }

    /// Resolve a registered index name to `(field, kind)`. Returns
    /// `None` if the name was never registered.
    #[must_use]
    pub fn resolve_name(&self, name: &str) -> Option<(&str, IndexKind)> {
        self.named_indexes.get(name).map(|(f, k)| (f.as_str(), *k))
    }

    /// Iterate every (name, field, kind) the registry knows about.
    pub fn named_index_entries(&self) -> impl Iterator<Item = (&str, &str, IndexKind)> {
        self.named_indexes
            .iter()
            .map(|(n, (f, k))| (n.as_str(), f.as_str(), *k))
    }

    /// Remove the index of `kind` on `field`. Safe to call when no
    /// such index exists (no-op). Drops the `FieldIndexes` entry
    /// itself once every kind slot is empty so `has_index_on`
    /// stays accurate.
    fn drop_kind(&mut self, field: &str, kind: IndexKind) {
        let Some(fi) = self.fields.get_mut(field) else {
            return;
        };
        match kind {
            IndexKind::Hash => fi.hash = None,
            IndexKind::Btree => fi.btree = None,
            IndexKind::Inverted => fi.inverted = None,
        }
        if fi.is_empty() {
            self.fields.remove(field);
        }
    }

    /// Programmatic drop of the hash index on `field`. Anonymous
    /// counterpart to [`Self::drop_named_index`] for code paths
    /// that registered with [`Self::add_hash_index`] and didn't
    /// give it a name. No-op when no hash index is registered.
    pub fn drop_hash_index(&mut self, field: &str) {
        self.drop_kind(field, IndexKind::Hash);
    }

    /// Programmatic drop of the btree index on `field`. See
    /// [`Self::drop_hash_index`].
    pub fn drop_btree_index(&mut self, field: &str) {
        self.drop_kind(field, IndexKind::Btree);
    }

    /// Programmatic drop of the inverted index on `field`. See
    /// [`Self::drop_hash_index`].
    pub fn drop_inverted_index(&mut self, field: &str) {
        self.drop_kind(field, IndexKind::Inverted);
    }

    /// Encode the catalog into a self-describing byte buffer :
    ///
    /// ```text
    /// +----------+----------+-------------------------------+
    /// | magic[8] | ver[u32] | bincode( IndexCatalog )       |
    /// +----------+----------+-------------------------------+
    /// ```
    ///
    /// The header is fixed-size (12 bytes) ; the bincode payload is
    /// variable. The storage layer wraps this output in an atomic
    /// write (tmp + fsync + rename + dirsync) so observers see either
    /// the whole new file or the previous one, never a partial.
    ///
    /// # Errors
    /// Returns [`KovaMetaIndexError::Decode`] (yes, wrapping the
    /// bincode error type even on encode ; `bincode::Error` covers
    /// both directions) if the payload can't be serialised, which is
    /// effectively never for the catalog's shape.
    pub fn encode(&self) -> Result<Vec<u8>, KovaMetaIndexError> {
        let payload = bincode::serialize(self)?;
        let mut buf = Vec::with_capacity(CATALOG_HEADER_LEN + payload.len());
        buf.extend_from_slice(CATALOG_MAGIC);
        buf.extend_from_slice(&CATALOG_FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&payload);
        Ok(buf)
    }

    /// Decode a catalog from a byte buffer produced by
    /// [`Self::encode`]. Validates the magic header and version
    /// before handing the rest to bincode.
    ///
    /// # Errors
    /// - [`KovaMetaIndexError::Truncated`] if `bytes` is shorter than
    ///   the fixed header.
    /// - [`KovaMetaIndexError::BadMagic`] if the magic bytes don't
    ///   match.
    /// - [`KovaMetaIndexError::UnsupportedVersion`] if the version
    ///   field doesn't match this build's expected version.
    /// - [`KovaMetaIndexError::Decode`] if bincode rejects the
    ///   payload.
    pub fn decode(bytes: &[u8]) -> Result<Self, KovaMetaIndexError> {
        if bytes.len() < CATALOG_HEADER_LEN {
            return Err(KovaMetaIndexError::Truncated {
                bytes: bytes.len(),
                min: CATALOG_HEADER_LEN,
            });
        }
        if &bytes[..CATALOG_MAGIC.len()] != CATALOG_MAGIC {
            return Err(KovaMetaIndexError::BadMagic);
        }
        let ver_bytes: [u8; 4] = bytes[CATALOG_MAGIC.len()..CATALOG_HEADER_LEN]
            .try_into()
            .expect("4-byte slice");
        let version = u32::from_le_bytes(ver_bytes);
        if version != CATALOG_FORMAT_VERSION {
            return Err(KovaMetaIndexError::UnsupportedVersion {
                expected: CATALOG_FORMAT_VERSION,
                got: version,
            });
        }
        let catalog: IndexCatalog = bincode::deserialize(&bytes[CATALOG_HEADER_LEN..])?;
        Ok(catalog)
    }

    /// Read + decode the catalog at `path`. Returns `Ok(None)` if the
    /// file doesn't exist (fresh shard with no persisted catalog).
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use kova_core::{Metadata, Value, VectorId};

    use super::{CATALOG_MAGIC, IndexCatalog, IndexKind};
    use crate::{CmpOp, IndexAtom, KovaMetaIndexError};

    fn s(x: &str) -> Value {
        Value::String(x.into())
    }

    fn i(n: i64) -> Value {
        Value::I64(n)
    }

    fn arr(xs: &[&str]) -> Value {
        Value::Array(xs.iter().map(|x| s(x)).collect())
    }

    fn id(n: u64) -> VectorId {
        VectorId::new(n)
    }

    fn meta(pairs: &[(&str, Value)]) -> Metadata {
        let mut m = HashMap::new();
        for (k, v) in pairs {
            m.insert((*k).to_string(), v.clone());
        }
        m
    }

    #[test]
    fn hash_index_basic_round_trip() {
        let mut cat = IndexCatalog::new();
        cat.add_hash_index("category").unwrap();

        cat.on_insert(id(0), &meta(&[("category", s("docs"))]));
        cat.on_insert(id(1), &meta(&[("category", s("blog"))]));
        cat.on_insert(id(2), &meta(&[("category", s("docs"))]));

        let docs = cat
            .lookup("category", &IndexAtom::Eq(s("docs")))
            .expect("index hit");
        assert_eq!(docs.len(), 2);
        assert!(docs.contains(0));
        assert!(docs.contains(2));
    }

    #[test]
    fn missing_field_for_row_is_skipped() {
        let mut cat = IndexCatalog::new();
        cat.add_hash_index("category").unwrap();

        cat.on_insert(id(0), &meta(&[("category", s("docs"))]));
        // row 1 has no "category" field
        cat.on_insert(id(1), &meta(&[("other", s("x"))]));

        let docs = cat.lookup("category", &IndexAtom::Eq(s("docs"))).unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs.contains(0));
    }

    #[test]
    fn separate_indexes_on_different_fields_dont_cross_talk() {
        let mut cat = IndexCatalog::new();
        cat.add_hash_index("a").unwrap();
        cat.add_hash_index("b").unwrap();

        cat.on_insert(id(0), &meta(&[("a", s("x")), ("b", s("y"))]));
        cat.on_insert(id(1), &meta(&[("a", s("y")), ("b", s("x"))]));

        let a_eq_x = cat.lookup("a", &IndexAtom::Eq(s("x"))).unwrap();
        assert_eq!(a_eq_x.len(), 1);
        assert!(a_eq_x.contains(0));

        let b_eq_x = cat.lookup("b", &IndexAtom::Eq(s("x"))).unwrap();
        assert_eq!(b_eq_x.len(), 1);
        assert!(b_eq_x.contains(1));
    }

    #[test]
    fn two_indexes_on_same_field_route_by_atom() {
        let mut cat = IndexCatalog::new();
        cat.add_hash_index("year").unwrap();
        cat.add_btree_index("year").unwrap();

        for n in 0u64..10 {
            cat.on_insert(
                id(n),
                &meta(&[("year", i(2020 + i64::try_from(n).unwrap()))]),
            );
        }

        // Eq goes to hash (highest priority) ; both would work but
        // we trust the priority order.
        let eq = cat.lookup("year", &IndexAtom::Eq(i(2025))).unwrap();
        assert_eq!(eq.len(), 1);
        assert!(eq.contains(5));

        // Range goes to btree (hash doesn't support it).
        let lt = cat
            .lookup("year", &IndexAtom::Cmp(CmpOp::Lt, i(2023)))
            .unwrap();
        assert_eq!(lt.len(), 3);
        for n in 0..3 {
            assert!(lt.contains(n));
        }
    }

    #[test]
    fn inverted_index_round_trip() {
        let mut cat = IndexCatalog::new();
        cat.add_inverted_index("tags").unwrap();

        cat.on_insert(id(0), &meta(&[("tags", arr(&["rust", "async"]))]));
        cat.on_insert(id(1), &meta(&[("tags", arr(&["rust", "sync"]))]));
        cat.on_insert(id(2), &meta(&[("tags", arr(&["go"]))]));

        let rust = cat
            .lookup("tags", &IndexAtom::ArrayContains(s("rust")))
            .unwrap();
        assert_eq!(rust.len(), 2);
        assert!(rust.contains(0));
        assert!(rust.contains(1));
    }

    #[test]
    fn on_delete_removes_from_indexes() {
        let mut cat = IndexCatalog::new();
        cat.add_hash_index("category").unwrap();

        let m0 = meta(&[("category", s("docs"))]);
        cat.on_insert(id(0), &m0);
        cat.on_insert(id(1), &meta(&[("category", s("docs"))]));
        cat.on_delete(id(0), &m0);

        let docs = cat.lookup("category", &IndexAtom::Eq(s("docs"))).unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs.contains(1));
    }

    #[test]
    fn on_update_field_present_in_both_calls_update() {
        let mut cat = IndexCatalog::new();
        cat.add_hash_index("category").unwrap();

        let old_m = meta(&[("category", s("docs"))]);
        let new_m = meta(&[("category", s("blog"))]);
        cat.on_insert(id(0), &old_m);
        cat.on_update(id(0), &old_m, &new_m);

        assert!(
            cat.lookup("category", &IndexAtom::Eq(s("docs")))
                .unwrap()
                .is_empty()
        );
        let blog = cat.lookup("category", &IndexAtom::Eq(s("blog"))).unwrap();
        assert_eq!(blog.len(), 1);
        assert!(blog.contains(0));
    }

    #[test]
    fn on_update_field_dropped_calls_delete() {
        let mut cat = IndexCatalog::new();
        cat.add_hash_index("category").unwrap();

        let old_m = meta(&[("category", s("docs"))]);
        let new_m = meta(&[("other", s("x"))]);
        cat.on_insert(id(0), &old_m);
        cat.on_update(id(0), &old_m, &new_m);

        assert!(
            cat.lookup("category", &IndexAtom::Eq(s("docs")))
                .unwrap()
                .is_empty()
        );
        assert!(
            cat.lookup("category", &IndexAtom::IsNotNull)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn on_update_field_added_calls_insert() {
        let mut cat = IndexCatalog::new();
        cat.add_hash_index("category").unwrap();

        let old_m = meta(&[("other", s("x"))]);
        let new_m = meta(&[("category", s("docs"))]);
        cat.on_insert(id(0), &old_m);
        cat.on_update(id(0), &old_m, &new_m);

        let docs = cat.lookup("category", &IndexAtom::Eq(s("docs"))).unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs.contains(0));
    }

    #[test]
    fn populate_field_bulk_loads_all_indexes_on_field() {
        let mut cat = IndexCatalog::new();
        cat.add_hash_index("year").unwrap();
        cat.add_btree_index("year").unwrap();

        let rows: Vec<_> = (0u64..10)
            .map(|n| (id(n), i(2020 + i64::try_from(n).unwrap())))
            .collect();
        cat.populate_field("year", rows);

        assert_eq!(
            cat.lookup("year", &IndexAtom::Eq(i(2025))).unwrap().len(),
            1
        );
        assert_eq!(
            cat.lookup("year", &IndexAtom::Between(i(2022), i(2027)))
                .unwrap()
                .len(),
            6
        );
    }

    #[test]
    fn populate_field_on_unknown_field_is_noop() {
        let mut cat = IndexCatalog::new();
        cat.populate_field("ghost", [(id(0), s("x"))]);
        // Just shouldn't panic and shouldn't register the field.
        assert!(!cat.has_index_on("ghost"));
    }

    #[test]
    fn lookup_returns_none_for_unindexed_field() {
        let cat = IndexCatalog::new();
        assert!(cat.lookup("anything", &IndexAtom::Eq(s("x"))).is_none());
    }

    #[test]
    fn lookup_returns_none_for_unsupported_atom() {
        let mut cat = IndexCatalog::new();
        cat.add_hash_index("category").unwrap();
        cat.on_insert(id(0), &meta(&[("category", s("docs"))]));

        // Hash index doesn't support Lt ; no other index on this field ; lookup returns None.
        assert!(
            cat.lookup("category", &IndexAtom::Cmp(CmpOp::Lt, s("docs")))
                .is_none()
        );
    }

    #[test]
    fn estimate_matches_lookup_len() {
        let mut cat = IndexCatalog::new();
        cat.add_hash_index("category").unwrap();
        for n in 0u64..50 {
            let v = if n % 5 == 0 { s("hot") } else { s("cold") };
            cat.on_insert(id(n), &meta(&[("category", v)]));
        }

        let atoms = [
            IndexAtom::Eq(s("hot")),
            IndexAtom::IsNotNull,
            IndexAtom::Cmp(CmpOp::Ne, s("hot")),
        ];
        for atom in &atoms {
            let q_len = cat.lookup("category", atom).unwrap().len();
            let e = cat.estimate("category", atom).unwrap();
            assert_eq!(q_len, e, "atom = {atom:?}");
        }
    }

    #[test]
    fn add_index_is_strict_no_replace() {
        let mut cat = IndexCatalog::new();
        cat.add_hash_index("category").unwrap();
        cat.on_insert(id(0), &meta(&[("category", s("docs"))]));

        // Re-registering the same kind on the same field is now
        // rejected loudly. Callers that want to rebuild must
        // explicitly drop first.
        let err = cat.add_hash_index("category").unwrap_err();
        assert!(
            matches!(
                err,
                KovaMetaIndexError::IndexAlreadyExists { ref field, kind: IndexKind::Hash } if field == "category"
            ),
            "got {err:?}"
        );

        // Original bucket state is intact.
        let bm = cat.lookup("category", &IndexAtom::Eq(s("docs"))).unwrap();
        assert_eq!(bm.len(), 1);
        assert!(bm.contains(0));
    }

    #[test]
    fn add_index_after_drop_works() {
        let mut cat = IndexCatalog::new();
        cat.add_hash_index("category").unwrap();
        cat.drop_hash_index("category");
        // Slot's vacant again, re-adding succeeds.
        cat.add_hash_index("category").unwrap();
    }

    #[test]
    fn populate_kind_only_touches_targeted_index() {
        let mut cat = IndexCatalog::new();
        cat.add_hash_index("year").unwrap();
        cat.add_btree_index("year").unwrap();

        // Populate only the btree side.
        cat.populate_kind(
            "year",
            IndexKind::Btree,
            vec![(id(0), i(2024)), (id(1), i(2020))],
        );

        // BTree sees both rows ; hash side stays empty.
        let range = cat
            .lookup("year", &IndexAtom::Cmp(CmpOp::Ge, i(2020)))
            .unwrap();
        assert_eq!(range.len(), 2);

        // Hash bucket is empty because populate_kind didn't broadcast.
        let eq = cat.lookup("year", &IndexAtom::Eq(i(2024))).unwrap();
        assert_eq!(eq.len(), 0);
    }

    #[test]
    fn populate_kind_noop_for_unregistered_field() {
        let mut cat = IndexCatalog::new();
        // No index on "year" registered.
        cat.populate_kind("year", IndexKind::Hash, vec![(id(0), i(2024))]);
        assert!(!cat.has_index_on("year"));
    }

    #[test]
    fn indexed_fields_lists_registered_fields() {
        let mut cat = IndexCatalog::new();
        cat.add_hash_index("a").unwrap();
        cat.add_btree_index("b").unwrap();
        cat.add_inverted_index("c").unwrap();

        let mut fields: Vec<_> = cat.indexed_fields().collect();
        fields.sort_unstable();
        assert_eq!(fields, vec!["a", "b", "c"]);
    }

    #[test]
    fn encode_decode_round_trip_preserves_buckets() {
        let mut cat = IndexCatalog::new();
        cat.add_hash_index("category").unwrap();
        cat.add_btree_index("year").unwrap();
        cat.add_inverted_index("tags").unwrap();

        for n in 0u64..20 {
            cat.on_insert(
                id(n),
                &meta(&[
                    ("category", if n % 2 == 0 { s("docs") } else { s("blog") }),
                    ("year", i(2020 + i64::from(u8::try_from(n).unwrap()))),
                    ("tags", arr(&["rust", "async"])),
                ]),
            );
        }

        let bytes = cat.encode().unwrap();
        let back = IndexCatalog::decode(&bytes).unwrap();

        // Every supported atom must give the same answer pre- and
        // post- round-trip.
        let atoms = [
            ("category", IndexAtom::Eq(s("docs"))),
            ("category", IndexAtom::IsNotNull),
            ("year", IndexAtom::Cmp(CmpOp::Gt, i(2025))),
            ("year", IndexAtom::Between(i(2022), i(2027))),
            ("tags", IndexAtom::ArrayContains(s("rust"))),
        ];
        for (field, atom) in &atoms {
            assert_eq!(
                cat.lookup(field, atom).map(|b| b.len()),
                back.lookup(field, atom).map(|b| b.len()),
                "atom = {field}.{atom:?}"
            );
        }
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let bytes = vec![0u8; 64];
        let err = IndexCatalog::decode(&bytes).unwrap_err();
        assert!(matches!(err, KovaMetaIndexError::BadMagic), "{err:?}");
    }

    #[test]
    fn decode_rejects_unsupported_version() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(CATALOG_MAGIC);
        bytes.extend_from_slice(&999u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 8]);
        let err = IndexCatalog::decode(&bytes).unwrap_err();
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
        let err = IndexCatalog::decode(&[0u8; 3]).unwrap_err();
        assert!(
            matches!(err, KovaMetaIndexError::Truncated { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn load_missing_file_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonexistent.bin");
        assert!(IndexCatalog::load(&path).unwrap().is_none());
    }

    // ---- Named-index DDL surface ----

    #[test]
    fn create_named_index_then_lookup_works() {
        let mut cat = IndexCatalog::new();
        cat.create_named_index("idx_cat", "category", IndexKind::Hash)
            .unwrap();
        cat.on_insert(id(0), &meta(&[("category", s("docs"))]));

        // The named index is reachable via the same lookup that
        // anonymous registrations would use.
        let bm = cat.lookup("category", &IndexAtom::Eq(s("docs"))).unwrap();
        assert_eq!(bm.len(), 1);
        assert!(bm.contains(0));

        // The registry tracks the name.
        assert_eq!(
            cat.resolve_name("idx_cat"),
            Some(("category", IndexKind::Hash))
        );
        assert!(cat.has_kind_on("category", IndexKind::Hash));
    }

    #[test]
    fn create_named_index_rejects_duplicate_name() {
        let mut cat = IndexCatalog::new();
        cat.create_named_index("idx_a", "category", IndexKind::Hash)
            .unwrap();
        let err = cat
            .create_named_index("idx_a", "tags", IndexKind::Inverted)
            .unwrap_err();
        assert!(
            matches!(err, KovaMetaIndexError::IndexNameInUse { ref name } if name == "idx_a"),
            "got {err:?}"
        );
    }

    #[test]
    fn drop_named_index_unregisters_and_removes_underlying() {
        let mut cat = IndexCatalog::new();
        cat.create_named_index("idx_cat", "category", IndexKind::Hash)
            .unwrap();
        cat.on_insert(id(0), &meta(&[("category", s("docs"))]));
        assert!(cat.has_kind_on("category", IndexKind::Hash));

        cat.drop_named_index("idx_cat").unwrap();
        assert!(cat.resolve_name("idx_cat").is_none());
        assert!(!cat.has_kind_on("category", IndexKind::Hash));
        // Field entry is GC'd once empty so lookup also goes to None.
        assert!(cat.lookup("category", &IndexAtom::Eq(s("docs"))).is_none());
    }

    #[test]
    fn drop_named_index_unknown_name_errors() {
        let mut cat = IndexCatalog::new();
        let err = cat.drop_named_index("nope").unwrap_err();
        assert!(
            matches!(err, KovaMetaIndexError::UnknownIndexName { ref name } if name == "nope"),
            "got {err:?}"
        );
    }

    #[test]
    fn drop_named_keeps_other_kinds_on_same_field() {
        let mut cat = IndexCatalog::new();
        cat.create_named_index("idx_eq", "year", IndexKind::Hash)
            .unwrap();
        cat.create_named_index("idx_range", "year", IndexKind::Btree)
            .unwrap();
        cat.on_insert(id(0), &meta(&[("year", i(2024))]));

        cat.drop_named_index("idx_eq").unwrap();
        // Btree on year is still there.
        assert!(cat.has_kind_on("year", IndexKind::Btree));
        assert!(!cat.has_kind_on("year", IndexKind::Hash));
        // Range lookup still routes to btree.
        let bm = cat
            .lookup("year", &IndexAtom::Cmp(CmpOp::Ge, i(2020)))
            .unwrap();
        assert_eq!(bm.len(), 1);
    }

    #[test]
    fn programmatic_drop_hash_btree_inverted() {
        let mut cat = IndexCatalog::new();
        cat.add_hash_index("a").unwrap();
        cat.add_btree_index("b").unwrap();
        cat.add_inverted_index("c").unwrap();

        cat.drop_hash_index("a");
        cat.drop_btree_index("b");
        cat.drop_inverted_index("c");

        assert!(!cat.has_index_on("a"));
        assert!(!cat.has_index_on("b"));
        assert!(!cat.has_index_on("c"));
    }

    #[test]
    fn named_index_round_trips_through_encode_decode() {
        let mut cat = IndexCatalog::new();
        cat.create_named_index("idx_cat", "category", IndexKind::Hash)
            .unwrap();
        cat.on_insert(id(0), &meta(&[("category", s("docs"))]));

        let bytes = cat.encode().unwrap();
        let back = IndexCatalog::decode(&bytes).unwrap();
        assert_eq!(
            back.resolve_name("idx_cat"),
            Some(("category", IndexKind::Hash))
        );
        let bm = back.lookup("category", &IndexAtom::Eq(s("docs"))).unwrap();
        assert!(bm.contains(0));
    }

    #[test]
    fn three_indexes_on_three_fields_composition_works() {
        // Smoke test of the executor-style three-index AND :
        //   WHERE category = 'docs' AND year > 2022 AND tags @> 'rust'
        let mut cat = IndexCatalog::new();
        cat.add_hash_index("category").unwrap();
        cat.add_btree_index("year").unwrap();
        cat.add_inverted_index("tags").unwrap();

        for n in 0u64..20 {
            let category = if n % 2 == 0 { "docs" } else { "blog" };
            let year = 2020 + i64::try_from(n % 6).unwrap();
            let tags = if n % 3 == 0 {
                arr(&["rust", "async"])
            } else {
                arr(&["go"])
            };
            cat.on_insert(
                id(n),
                &meta(&[("category", s(category)), ("year", i(year)), ("tags", tags)]),
            );
        }

        let docs = cat.lookup("category", &IndexAtom::Eq(s("docs"))).unwrap();
        let recent = cat
            .lookup("year", &IndexAtom::Cmp(CmpOp::Gt, i(2022)))
            .unwrap();
        let rust = cat
            .lookup("tags", &IndexAtom::ArrayContains(s("rust")))
            .unwrap();

        let candidates = docs & recent & rust;
        // Spot-check : every survivor satisfies all three.
        for survivor in &candidates {
            assert_eq!(survivor % 2, 0, "category=docs requires even n");
            assert!((survivor % 6) > 2, "year > 2022 requires n%6 > 2");
            assert_eq!(survivor % 3, 0, "tags@>rust requires n%3 == 0");
        }
    }
}

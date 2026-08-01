//! File-backed [`MetadataStore`] : persistent attribute storage.
//!
//! Holds the full attribute map in memory and persists every mutation to a
//! single file via [`atomic_write`]. The file format is intentionally simple
//! since [`atomic_write`] already guarantees readers see either the old file
//! or the complete new file, never a torn write.
//!
//! # On-disk layout
//!
//! ```text
//! +----------+---------+------------------------------------------+
//! | magic[8] | ver[u32]| bincode( HashMap<VectorId, Metadata> )   |
//! +----------+---------+------------------------------------------+
//! ```
//!
//! - `magic` = b"KOVAMET1"   : catches pointing the store at the wrong file
//! - `ver`   = `FORMAT_VERSION` (little-endian u32) : reserved for migrations
//! - payload : bincode-encoded full map
//!
//! No per-entry framing, no CRC : [`atomic_write`] is the durability story.
//! On crash, the file is either the previous good state or the new good
//! state. Readers can't observe a half-written tail.
//!
//! # Performance shape
//!
//! Every `put`/`delete` rewrites the entire file. That is O(N) per mutation
//! and obviously bad at scale ; this impl is the *checkpoint* form. Once
//! `Shard` couples a [`Wal`](crate::Wal) with metadata, per-mutation
//! durability comes from the WAL and this file becomes a periodic snapshot
//! flushed under explicit control rather than every call.
//!
//! Adequate for the in-progress milestone where we need *some* persistent
//! metadata so `Shard` can be built and crash-tested end to end.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use kova_core::{Metadata, MetadataStore, Value, VectorId};

use crate::KovaStorageError;
use crate::atomic::atomic_write;

const MAGIC: &[u8; 8] = b"KOVAMET1";
const FORMAT_VERSION: u32 = 1;
const HEADER_LEN: usize = MAGIC.len() + std::mem::size_of::<u32>();

/// File-backed [`MetadataStore`] persisted via [`atomic_write`].
///
/// Opens or creates a single file. Every mutation rewrites the file
/// atomically. Reads serve from the in-memory copy.
#[derive(Debug)]
pub struct FileMetadataStore {
    path: PathBuf,
    entries: HashMap<VectorId, Metadata>,
}

impl FileMetadataStore {
    /// Open the store at `path`, creating an empty file if none exists.
    ///
    /// # Errors
    /// Returns [`KovaStorageError::Io`] on filesystem errors, or
    /// [`KovaStorageError::CorruptRecord`] / [`KovaStorageError::Decode`] if
    /// the existing file is not a valid `FileMetadataStore` payload.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, KovaStorageError> {
        let path = path.into();
        let entries = if path.exists() {
            load(&path)?
        } else {
            // Materialise an empty file so subsequent observers see a real
            // store rather than a missing path.
            let store = Self {
                path: path.clone(),
                entries: HashMap::new(),
            };
            store.flush_to_disk()?;
            return Ok(store);
        };
        Ok(Self { path, entries })
    }

    /// Path the store is persisted to.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Write the whole map out through [`atomic_write`].
    ///
    /// Private helper ; the trait-level [`MetadataStore::flush`] wraps it.
    fn flush_to_disk(&self) -> Result<(), KovaStorageError> {
        let payload = bincode::serialize(&self.entries).map_err(KovaStorageError::Encode)?;
        let mut buf = Vec::with_capacity(HEADER_LEN + payload.len());
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&payload);
        atomic_write(&self.path, &buf)
    }
}

fn load(path: &Path) -> Result<HashMap<VectorId, Metadata>, KovaStorageError> {
    let bytes = fs::read(path)?;
    if bytes.len() < HEADER_LEN {
        return Err(KovaStorageError::CorruptRecord {
            reason: format!(
                "metadata file too short : {} bytes, need at least {}",
                bytes.len(),
                HEADER_LEN
            ),
        });
    }
    if &bytes[..MAGIC.len()] != MAGIC {
        return Err(KovaStorageError::CorruptRecord {
            reason: "metadata file magic mismatch".into(),
        });
    }
    let ver_bytes: [u8; 4] = bytes[MAGIC.len()..HEADER_LEN]
        .try_into()
        .expect("slice of length 4");
    let version = u32::from_le_bytes(ver_bytes);
    if version != FORMAT_VERSION {
        return Err(KovaStorageError::CorruptRecord {
            reason: format!(
                "unsupported metadata format version : {version} (expected {FORMAT_VERSION})"
            ),
        });
    }
    bincode::deserialize(&bytes[HEADER_LEN..]).map_err(KovaStorageError::from)
}

impl MetadataStore for FileMetadataStore {
    type Error = KovaStorageError;

    /// In-memory only. The file is written by [`Self::flush`], which
    /// `Shard::checkpoint` calls.
    ///
    /// Durability is not lost : `Shard` appends and fsyncs a WAL record
    /// before calling this, so a crash replays the mutation on reopen.
    /// Flushing here instead cost two fsyncs and a full-file rewrite per
    /// row (~7.9 ms, independent of store size), on top of the WAL's own
    /// fsync.
    fn put(&mut self, id: VectorId, meta: Metadata) -> Result<(), Self::Error> {
        self.entries.insert(id, meta);
        Ok(())
    }

    fn get(&self, id: VectorId) -> Option<Metadata> {
        self.entries.get(&id).cloned()
    }

    /// Borrowing override, same shape as the
    /// [`kova_core::InMemoryMetadataStore`] one : reads serve from the
    /// in-memory map, so a reference costs nothing while the default
    /// impl's `get` would clone the whole bag.
    fn with_metadata<F, R>(&self, id: VectorId, f: F) -> Option<R>
    where
        F: FnOnce(&Metadata) -> R,
    {
        self.entries.get(&id).map(f)
    }

    /// In-memory only ; see [`Self::put`].
    fn delete(&mut self, id: VectorId) -> Result<(), Self::Error> {
        self.entries.remove(&id);
        Ok(())
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn scan_ids<F>(&self, mut pred: F) -> Vec<VectorId>
    where
        F: FnMut(&Metadata) -> bool,
    {
        self.entries
            .iter()
            .filter(|(_, m)| pred(m))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Single-pass override : iterate the in-memory `HashMap` once,
    /// emitting `(id, &value)` for every row that has the field.
    /// Mirrors the [`kova_core::InMemoryMetadataStore`] override ;
    /// both impls use `HashMap<VectorId, Metadata>` internally, so
    /// the body is identical.
    fn walk_field<F>(&self, field: &str, mut callback: F)
    where
        F: FnMut(VectorId, &Value),
    {
        for (id, meta) in &self.entries {
            if let Some(v) = meta.get(field) {
                callback(*id, v);
            }
        }
    }

    /// Apply many puts as a single flush.
    ///
    /// The default trait impl calls [`Self::put`] N times, which for
    /// this store means N full-file rewrites + fsyncs. This override
    /// updates the in-memory map for every item and then flushes the
    /// whole file once at the end : O(N) memory work + 1 fsync, instead
    /// of O(N) of each.
    ///
    /// # Atomicity / rollback
    ///
    /// On flush failure we roll back to the snapshot taken before the
    /// batch started, so the store stays consistent with what's on disk.
    /// For larger batches this is a real memory cost (clone of the live
    /// map), but it preserves the "if `put_many` returned Err, the
    /// in-memory state matches the on-disk state" invariant the singleton
    /// `put` already guarantees.
    /// In-memory only ; see [`Self::put`]. The rollback snapshot the
    /// eager version needed is gone with it : there is no fallible disk
    /// write left to unwind.
    fn put_many<I>(&mut self, items: I) -> Result<(), Self::Error>
    where
        I: IntoIterator<Item = (VectorId, Metadata)>,
        Self: Sized,
    {
        for (id, meta) in items {
            self.entries.insert(id, meta);
        }
        Ok(())
    }

    /// Write the map to disk. Called by `Shard::checkpoint`.
    fn flush(&mut self) -> Result<(), Self::Error> {
        self.flush_to_disk()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kova_core::Value;
    use tempfile::tempdir;

    fn id(n: u64) -> VectorId {
        VectorId::new(n)
    }

    fn sample_meta(tag: &str) -> Metadata {
        let mut m = Metadata::new();
        m.insert("tag".into(), Value::String(tag.into()));
        m.insert("score".into(), Value::F64(0.5));
        m
    }

    #[test]
    fn open_creates_empty_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.bin");
        let store = FileMetadataStore::open(&path).unwrap();
        assert!(path.exists());
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn put_then_get_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.bin");
        let mut store = FileMetadataStore::open(&path).unwrap();
        let meta = sample_meta("alpha");
        store.put(id(1), meta.clone()).unwrap();
        assert_eq!(store.get(id(1)), Some(meta));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn delete_removes_entry() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.bin");
        let mut store = FileMetadataStore::open(&path).unwrap();
        store.put(id(1), sample_meta("a")).unwrap();
        store.delete(id(1)).unwrap();
        assert_eq!(store.len(), 0);
        assert!(!store.contains(id(1)));
    }

    #[test]
    fn flush_then_reopen_recovers_entries() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.bin");
        {
            let mut store = FileMetadataStore::open(&path).unwrap();
            store.put(id(1), sample_meta("alpha")).unwrap();
            store.put(id(2), sample_meta("beta")).unwrap();
            store.delete(id(1)).unwrap();
            store.flush().unwrap();
        }
        let store = FileMetadataStore::open(&path).unwrap();
        assert_eq!(store.len(), 1);
        assert!(!store.contains(id(1)));
        assert_eq!(store.get(id(2)), Some(sample_meta("beta")));
    }

    /// Mutations are **not** durable on their own, by design.
    ///
    /// This store used to `atomic_write` the whole file on every `put`,
    /// which cost two fsyncs and O(rows) bytes per mutation (~7.9 ms
    /// regardless of size). Durability for metadata comes from the WAL :
    /// `Shard` logs and fsyncs before touching this store, so a crash
    /// replays the mutation. The file is a checkpoint artifact, written
    /// by `flush`.
    ///
    /// Dropping an unflushed store therefore loses the writes, and that
    /// is the contract, not a bug. Anyone using `FileMetadataStore`
    /// outside `Shard` has to call `flush` themselves.
    #[test]
    fn writes_are_not_durable_without_flush() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.bin");
        {
            let mut store = FileMetadataStore::open(&path).unwrap();
            store.put(id(1), sample_meta("alpha")).unwrap();
            // no flush
        }
        let store = FileMetadataStore::open(&path).unwrap();
        assert_eq!(
            store.len(),
            0,
            "unflushed writes should not survive ; durability is the WAL's job"
        );
    }

    #[test]
    fn put_overwrites_persisted_entry() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.bin");
        {
            let mut store = FileMetadataStore::open(&path).unwrap();
            store.put(id(7), sample_meta("first")).unwrap();
            store.put(id(7), sample_meta("second")).unwrap();
            store.flush().unwrap();
        }
        let store = FileMetadataStore::open(&path).unwrap();
        assert_eq!(store.get(id(7)), Some(sample_meta("second")));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn rejects_wrong_magic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.bin");
        // Write a file with a valid-looking length but wrong magic.
        let mut bytes = vec![0u8; HEADER_LEN + 4];
        bytes[..8].copy_from_slice(b"NOTKOVA!");
        std::fs::write(&path, &bytes).unwrap();
        let err = FileMetadataStore::open(&path).unwrap_err();
        match err {
            KovaStorageError::CorruptRecord { reason } => {
                assert!(reason.contains("magic"), "reason was: {reason}");
            }
            other => panic!("expected CorruptRecord, got {other:?}"),
        }
    }

    #[test]
    fn rejects_unsupported_version() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.bin");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&999u32.to_le_bytes());
        // Empty bincode-encoded map would still fail version check first.
        std::fs::write(&path, &bytes).unwrap();
        let err = FileMetadataStore::open(&path).unwrap_err();
        match err {
            KovaStorageError::CorruptRecord { reason } => {
                assert!(reason.contains("version"), "reason was: {reason}");
            }
            other => panic!("expected CorruptRecord, got {other:?}"),
        }
    }

    #[test]
    fn rejects_truncated_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.bin");
        std::fs::write(&path, b"KOV").unwrap();
        let err = FileMetadataStore::open(&path).unwrap_err();
        match err {
            KovaStorageError::CorruptRecord { reason } => {
                assert!(reason.contains("too short"), "reason was: {reason}");
            }
            other => panic!("expected CorruptRecord, got {other:?}"),
        }
    }

    #[test]
    fn path_accessor_returns_open_path() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.bin");
        let store = FileMetadataStore::open(&path).unwrap();
        assert_eq!(store.path(), path.as_path());
    }
}

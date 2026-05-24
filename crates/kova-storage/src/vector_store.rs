//! mmap-backed [`VectorStore`] : flat fixed-stride file with self-describing
//! slots. On reopen, walks slots to rebuild the in-memory id-to-slot map.
//!
//! ## File layout
//!
//! ```text
//!   offset 0
//!   +---------------------------------------------+
//!   |  Header (32 bytes)                          |
//!   |    magic       "KOVAVST1"          8 bytes  |
//!   |    dim         u64 LE              8 bytes  |
//!   |    next_slot   u64 LE              8 bytes  |
//!   |    reserved    u64 zero            8 bytes  |
//!   +---------------------------------------------+
//!   offset 32
//!   +---------------------------------------------+
//!   |  Slot 0      (stride = 16 + dim * 4 bytes)  |
//!   |    id        u64 LE                8 bytes  |
//!   |    present   u8 (0=empty/1=set)    1 byte   |
//!   |    padding   zero                  7 bytes  |
//!   |    vector    dim * f32            dim*4 B   |
//!   +---------------------------------------------+
//!   offset 32 + stride
//!   +---------------------------------------------+
//!   |  Slot 1      same layout                    |
//!   +---------------------------------------------+
//!   ...
//!   +---------------------------------------------+
//!   |  Slot (next_slot - 1)                       |
//!   +---------------------------------------------+
//!   |  unused capacity (zero-filled by set_len)   |
//!   +---------------------------------------------+
//! ```
//!
//! `id_to_slot` is in-memory only and rebuilt on open by walking
//! slots `0..next_slot`. Slots are self-describing : no sidecar
//! index file needed.

// File offsets and slot indices are u64 by design (matches mmap semantics
// and on-disk format). Casting to usize is safe on 64-bit targets which
// are what we ship; 32-bit users would hit the MAX_MAPPABLE_BYTES guard.
#![allow(clippy::cast_possible_truncation)]

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use kova_core::{Vector, VectorId, VectorStore};
use memmap2::MmapMut;

use crate::KovaStorageError;

//constants for the file format and growth strategy.

/// File header magic. Bumping this number invalidates old files.
const MAGIC: &[u8; 8] = b"KOVAVST1";

/// Bytes occupied by the file header (magic + dim + `next_slot` + reserved).
const HEADER_SIZE: u64 = 32;

/// Bytes occupied by the slot header (id + present + padding).
const SLOT_HEADER_SIZE: u64 = 16;

/// Initial file capacity. Doubles on grow.
const INITIAL_CAPACITY_BYTES: u64 = 1024 * 1024;

/// Hard cap so a corrupt header can't ask us to mmap absurd sizes.
const MAX_MAPPABLE_BYTES: u64 = 1024 * 1024 * 1024 * 1024; // 1 TB

/// Safely mmap a file we own.
///
/// Wraps the `unsafe` call to [`MmapMut::map_mut`] behind preconditions
/// that turn would-be UB into typed errors :
///
/// - the file must be non-empty (mmap of zero bytes is invalid on Linux)
/// - the file must be smaller than [`MAX_MAPPABLE_BYTES`]
/// - the caller must guarantee no other process or thread mutates the file
///   while the returned mapping is alive (we own the file in this crate)
fn map_file(file: &File) -> Result<MmapMut, KovaStorageError> {
    let len = file.metadata()?.len();
    if len == 0 {
        return Err(KovaStorageError::CorruptRecord {
            reason: "cannot mmap a zero-byte file (set_len first)".into(),
        });
    }
    if len > MAX_MAPPABLE_BYTES {
        return Err(KovaStorageError::CorruptRecord {
            reason: format!("file size {len} exceeds max mappable {MAX_MAPPABLE_BYTES}"),
        });
    }
    // SAFETY: caller guarantees no concurrent mutation of `file`.
    // The wrapping function documents the contract; every call site
    // in this module relies on it.
    unsafe { MmapMut::map_mut(file) }.map_err(KovaStorageError::Io)
}

/// mmap-backed [`VectorStore`].
#[derive(Debug)]
pub struct MmapVectorStore {
    file: File,
    mmap: MmapMut,
    dim: usize,
    /// Bytes per slot : `SLOT_HEADER_SIZE + dim * 4`.
    stride: u64,
    /// Next free slot index.
    next_slot: u64,
    /// In-memory map from id to slot index. Rebuilt on open.
    id_to_slot: HashMap<VectorId, u64>,
    path: PathBuf,
    capacity_bytes: u64,
}

impl MmapVectorStore {
    /// Open or create an mmap-backed store at `path` for vectors of `dim`.
    ///
    /// If the file exists, validates the header magic and stored dim,
    /// then walks slots to rebuild the in-memory id-to-slot map.
    /// If the file doesn't exist (or is empty), creates a fresh one with
    /// [`INITIAL_CAPACITY_BYTES`] of capacity.
    pub fn open(path: impl AsRef<Path>, dim: usize) -> Result<Self, KovaStorageError> {
        let path = path.as_ref().to_path_buf();
        let stride = SLOT_HEADER_SIZE + (dim as u64) * 4;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        let existing_size = file.metadata()?.len();

        if existing_size == 0 {
            // ---- fresh file branch ----
            file.set_len(INITIAL_CAPACITY_BYTES)?;
            let mut mmap = map_file(&file)?;

            // Write the header.
            mmap[0..8].copy_from_slice(MAGIC);
            mmap[8..16].copy_from_slice(&(dim as u64).to_le_bytes());
            mmap[16..24].copy_from_slice(&0u64.to_le_bytes()); // next_slot
            mmap[24..32].copy_from_slice(&0u64.to_le_bytes()); // reserved

            return Ok(Self {
                file,
                mmap,
                dim,
                stride,
                next_slot: 0,
                id_to_slot: HashMap::new(),
                path,
                capacity_bytes: INITIAL_CAPACITY_BYTES,
            });
        }

        // ---- existing file branch : recover ----
        let mmap = map_file(&file)?;

        // Validate header magic.
        if &mmap[0..8] != MAGIC {
            return Err(KovaStorageError::CorruptRecord {
                reason: format!("invalid magic in {}", path.display()),
            });
        }

        // Read stored dim; bail if it doesn't match what caller asked for.
        let stored_dim = u64::from_le_bytes(mmap[8..16].try_into().expect("8 bytes")) as usize;
        if stored_dim != dim {
            return Err(KovaStorageError::CorruptRecord {
                reason: format!(
                    "dim mismatch in {}: file has {stored_dim}, caller asked {dim}",
                    path.display()
                ),
            });
        }

        let next_slot = u64::from_le_bytes(mmap[16..24].try_into().expect("8 bytes"));

        // Walk slots 0..next_slot and rebuild id_to_slot.
        let mut id_to_slot = HashMap::with_capacity(next_slot as usize);
        for slot in 0..next_slot {
            let off = (HEADER_SIZE + slot * stride) as usize;
            let id = u64::from_le_bytes(mmap[off..off + 8].try_into().expect("8 bytes"));
            let present = mmap[off + 8];
            if present == 1 {
                id_to_slot.insert(VectorId::new(id), slot);
            }
        }

        Ok(Self {
            file,
            mmap,
            dim,
            stride,
            next_slot,
            id_to_slot,
            path,
            capacity_bytes: existing_size,
        })
    }

    /// Path to the underlying file on disk.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Grow the file and remap. Caller must ensure no references to the old mapping are alive.
    /// We never truncate or shrink the file, only grow, to avoid the risk of `SIGBUS` from accessing truncated regions.
    fn grow_to(&mut self, new_capacity: u64) -> Result<(), KovaStorageError> {
        assert!(
            new_capacity > self.capacity_bytes,
            "grow_to must increase capacity, never shrink (would risk SIGBUS)"
        );
        self.mmap.flush()?; // push dirty pages to disk
        self.file.set_len(new_capacity)?; // extend the file
        self.mmap = map_file(&self.file)?; // re-mmap the larger region
        self.capacity_bytes = new_capacity;
        Ok(())
    }

    /// Update the `next_slot` field in the header to match the in-memory `next_slot`.
    fn write_next_slot_to_header(&mut self) {
        self.mmap[16..24].copy_from_slice(&self.next_slot.to_le_bytes());
    }
}

impl VectorStore for MmapVectorStore {
    type Error = KovaStorageError;

    /// Insert or overwrite the vector for `id`.
    ///
    /// ```text
    ///   put(id, vector)
    ///       |
    ///       v
    ///   vector.dim == self.dim ?  -- no --> Err(CorruptRecord)
    ///       |
    ///       v
    ///   id in id_to_slot ?
    ///       |-- yes --> reuse existing slot (overwrite in place)
    ///       |-- no  --> slot = next_slot         (allocate fresh)
    ///       v
    ///   needed = HEADER_SIZE + (slot+1) * stride
    ///       |
    ///       v
    ///   needed > capacity ?  -- yes --> grow_to(capacity * 2), repeat
    ///       |
    ///       v
    ///   write into mmap at HEADER_SIZE + slot * stride :
    ///       [ id 8B ][ present=1 1B ][ pad 7B ][ vector dim*4 B ]
    ///       |
    ///       v
    ///   if new : next_slot += 1, write header, id_to_slot[id] = slot
    /// ```
    fn put(&mut self, id: VectorId, vector: Vector) -> Result<(), Self::Error> {
        // Validate dim.
        if vector.dim() != self.dim {
            return Err(KovaStorageError::CorruptRecord {
                reason: format!(
                    "vector dim {} does not match store dim {}",
                    vector.dim(),
                    self.dim
                ),
            });
        }

        // Decide which slot to use : overwrite if id exists, else allocate.
        let (slot, is_new) = match self.id_to_slot.get(&id) {
            Some(&existing_slot) => (existing_slot, false),
            None => (self.next_slot, true),
        };

        // If this would overflow capacity, grow first.
        let needed = HEADER_SIZE + (slot + 1) * self.stride;
        while needed > self.capacity_bytes {
            let new_cap = self.capacity_bytes.saturating_mul(2);
            self.grow_to(new_cap)?;
        }

        // Write the slot.
        let off = (HEADER_SIZE + slot * self.stride) as usize;
        self.mmap[off..off + 8].copy_from_slice(&id.get().to_le_bytes());
        self.mmap[off + 8] = 1; // present
        for b in &mut self.mmap[off + 9..off + 16] {
            *b = 0; // padding
        }
        let data_off = off + SLOT_HEADER_SIZE as usize;
        let vec_bytes: &[u8] = bytemuck::cast_slice(vector.as_slice());
        self.mmap[data_off..data_off + self.dim * 4].copy_from_slice(vec_bytes);

        // Update bookkeeping.
        if is_new {
            self.next_slot += 1;
            self.write_next_slot_to_header();
            self.id_to_slot.insert(id, slot);
        }

        Ok(())
    }

    /// Fetch the vector for `id`, if present.
    ///
    /// ```text
    ///   get(id)
    ///       |
    ///       v
    ///   slot = id_to_slot[id] ?  -- None --> return None
    ///       |
    ///       v
    ///   off = HEADER_SIZE + slot * stride
    ///   data_off = off + SLOT_HEADER_SIZE   (skip id + present + pad)
    ///       |
    ///       v
    ///   bytes = &mmap[data_off .. data_off + dim*4]
    ///       |
    ///       v
    ///   floats: &[f32] = bytemuck::cast_slice(bytes)   (zero-copy reinterpret)
    ///       |
    ///       v
    ///   Vector::try_new(floats.to_vec())   (one clone : trait returns owned)
    /// ```
    fn get(&self, id: VectorId) -> Option<Vector> {
        let slot = *self.id_to_slot.get(&id)?;
        let off = (HEADER_SIZE + slot * self.stride) as usize;
        let data_off = off + SLOT_HEADER_SIZE as usize;
        let data_bytes = &self.mmap[data_off..data_off + self.dim * 4];
        let floats: &[f32] = bytemuck::cast_slice(data_bytes);
        Vector::try_new(floats.to_vec()).ok()
    }

    fn len(&self) -> usize {
        self.id_to_slot.len()
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn v(data: Vec<f32>) -> Vector {
        Vector::try_new(data).unwrap()
    }

    fn id(n: u64) -> VectorId {
        VectorId::new(n)
    }

    // ---------- map_file helper ----------

    #[test]
    fn map_file_errors_on_empty_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.bin");
        let file = File::create(&path).unwrap();
        let err = map_file(&file).unwrap_err();
        assert!(matches!(err, KovaStorageError::CorruptRecord { .. }));
    }

    #[test]
    fn map_file_returns_mmap_for_non_empty_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonempty.bin");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .unwrap();
        file.set_len(1024).unwrap();
        let mmap = map_file(&file).unwrap();
        assert_eq!(mmap.len(), 1024);
    }

    // ---------- MmapVectorStore ----------

    #[test]
    fn new_store_is_empty() {
        let dir = tempdir().unwrap();
        let store = MmapVectorStore::open(dir.path().join("vs.dat"), 3).unwrap();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn put_then_get_roundtrip() {
        let dir = tempdir().unwrap();
        let mut store = MmapVectorStore::open(dir.path().join("vs.dat"), 3).unwrap();
        let original = v(vec![1.0, 2.0, 3.0]);
        store.put(id(42), original.clone()).unwrap();
        assert_eq!(store.get(id(42)), Some(original));
    }

    #[test]
    fn put_multiple_each_retrievable() {
        let dir = tempdir().unwrap();
        let mut store = MmapVectorStore::open(dir.path().join("vs.dat"), 2).unwrap();
        for n in 0..5_u64 {
            store.put(id(n), v(vec![n as f32, (n * 2) as f32])).unwrap();
        }
        for n in 0..5_u64 {
            assert_eq!(store.get(id(n)), Some(v(vec![n as f32, (n * 2) as f32])));
        }
        assert_eq!(store.len(), 5);
    }

    #[test]
    fn put_dim_mismatch_errors() {
        let dir = tempdir().unwrap();
        let mut store = MmapVectorStore::open(dir.path().join("vs.dat"), 3).unwrap();
        let err = store.put(id(1), v(vec![1.0, 2.0])).unwrap_err();
        assert!(matches!(err, KovaStorageError::CorruptRecord { .. }));
    }

    #[test]
    fn overwrite_existing_id() {
        let dir = tempdir().unwrap();
        let mut store = MmapVectorStore::open(dir.path().join("vs.dat"), 1).unwrap();
        store.put(id(7), v(vec![1.0])).unwrap();
        store.put(id(7), v(vec![2.0])).unwrap();
        assert_eq!(store.get(id(7)), Some(v(vec![2.0])));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn reopen_recovers_entries() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vs.dat");
        {
            let mut store = MmapVectorStore::open(&path, 2).unwrap();
            store.put(id(1), v(vec![1.0, 2.0])).unwrap();
            store.put(id(2), v(vec![3.0, 4.0])).unwrap();
            store.put(id(3), v(vec![5.0, 6.0])).unwrap();
        }
        let store = MmapVectorStore::open(&path, 2).unwrap();
        assert_eq!(store.len(), 3);
        assert_eq!(store.get(id(1)), Some(v(vec![1.0, 2.0])));
        assert_eq!(store.get(id(2)), Some(v(vec![3.0, 4.0])));
        assert_eq!(store.get(id(3)), Some(v(vec![5.0, 6.0])));
    }

    #[test]
    fn reopen_dim_mismatch_errors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vs.dat");
        {
            let mut store = MmapVectorStore::open(&path, 3).unwrap();
            store.put(id(1), v(vec![1.0, 2.0, 3.0])).unwrap();
        }
        let err = MmapVectorStore::open(&path, 5).unwrap_err();
        assert!(matches!(err, KovaStorageError::CorruptRecord { .. }));
    }

    #[test]
    fn reopen_magic_mismatch_errors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vs.dat");
        let mut f = File::create(&path).unwrap();
        f.write_all(&[0u8; 64]).unwrap();
        drop(f);
        let err = MmapVectorStore::open(&path, 3).unwrap_err();
        assert!(matches!(err, KovaStorageError::CorruptRecord { .. }));
    }

    #[test]
    fn put_grows_file_when_needed() {
        // With dim=4, stride = 16 + 16 = 32 bytes. 1 MB / 32 = ~32k slots.
        // Insert 40k to force at least one grow.
        let dir = tempdir().unwrap();
        let mut store = MmapVectorStore::open(dir.path().join("vs.dat"), 4).unwrap();
        for n in 0..40_000_u64 {
            store.put(id(n), v(vec![n as f32, 0.0, 0.0, 0.0])).unwrap();
        }
        assert_eq!(store.len(), 40_000);
        assert_eq!(store.get(id(0)), Some(v(vec![0.0, 0.0, 0.0, 0.0])));
        assert_eq!(
            store.get(id(39_999)),
            Some(v(vec![39_999.0, 0.0, 0.0, 0.0]))
        );
    }
}

//! `Manifest` : the single durable commit point of a checkpoint.
//!
//! The manifest records two things : the `checkpoint_lsn` (WAL records
//! with LSN > this are still in the live WAL ; records `<= this` are
//! baked into the snapshot), and the `snapshot_id` naming the live
//! `graph.{snapshot_id}.snapshot` file.
//!
//! The atomic write of the manifest is what makes "the new checkpoint
//! is now live" true. Everything before the manifest commits is
//! preparation that can be discarded on crash ; everything after is
//! cleanup that can be retried.
//!
//! # On-disk layout
//!
//! ```text
//!   +----------+----------+-------------------------------+
//!   | magic[8] | ver[u32] | bincode( Manifest )           |
//!   +----------+----------+-------------------------------+
//!
//!   magic = b"KOVAMAN1"   : catches "you handed me the wrong file"
//!   ver   = FORMAT_VERSION (little-endian u32)
//!   payload : bincode-encoded Manifest
//! ```
//!
//! Stored via [`atomic_write`] : tmp + fsync + rename + dirsync, so a
//! crash at any point leaves the old manifest (or no manifest) intact.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::KovaStorageError;
use crate::atomic::atomic_write;

const MAGIC: &[u8; 8] = b"KOVAMAN1";
const FORMAT_VERSION: u32 = 1;
const HEADER_LEN: usize = MAGIC.len() + std::mem::size_of::<u32>();

/// Checkpoint manifest. Names which snapshot is live and up to which
/// LSN the snapshot covers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// Format version. Currently `1`. Bumping it implies a migration.
    pub version: u32,
    /// WAL records with LSN > this are still in `wal/` ; everything
    /// `<= this` is baked into the snapshot.
    pub checkpoint_lsn: u64,
    /// Matches the suffix on the live `graph.{snapshot_id}.snapshot`
    /// file. Monotonically increases across checkpoints ; old snapshots
    /// stay on disk until the next [`crate::Shard::open`] cleans them up.
    pub snapshot_id: u64,
}

impl Manifest {
    /// Load the manifest at `path`, returning `Ok(None)` if the file
    /// doesn't exist (fresh shard, no checkpoint yet).
    ///
    /// # Errors
    /// - [`KovaStorageError::Io`] for filesystem failures.
    /// - [`KovaStorageError::CorruptRecord`] if the magic or version
    ///   don't match, or the file is shorter than the header.
    /// - [`KovaStorageError::Decode`] if bincode rejects the payload.
    pub fn load(path: &Path) -> Result<Option<Self>, KovaStorageError> {
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(path)?;
        if bytes.len() < HEADER_LEN {
            return Err(KovaStorageError::CorruptRecord {
                reason: format!(
                    "manifest file too short : {} bytes, need at least {HEADER_LEN}",
                    bytes.len(),
                ),
            });
        }
        if &bytes[..MAGIC.len()] != MAGIC {
            return Err(KovaStorageError::CorruptRecord {
                reason: "manifest magic mismatch".into(),
            });
        }
        let ver_bytes: [u8; 4] = bytes[MAGIC.len()..HEADER_LEN]
            .try_into()
            .expect("slice of length 4");
        let version = u32::from_le_bytes(ver_bytes);
        if version != FORMAT_VERSION {
            return Err(KovaStorageError::CorruptRecord {
                reason: format!(
                    "unsupported manifest format version : {version} (expected {FORMAT_VERSION})"
                ),
            });
        }
        let manifest: Manifest = bincode::deserialize(&bytes[HEADER_LEN..])?;
        Ok(Some(manifest))
    }

    /// Atomically write the manifest to `path`.
    ///
    /// After this returns Ok, the manifest is durable and a reopen
    /// will see the new contents. This is the single commit point of
    /// a checkpoint operation.
    ///
    /// # Errors
    /// [`KovaStorageError`] for any underlying I/O failure or bincode
    /// encoding failure.
    pub fn store(&self, path: &Path) -> Result<(), KovaStorageError> {
        let payload = bincode::serialize(self).map_err(KovaStorageError::Encode)?;
        let mut buf = Vec::with_capacity(HEADER_LEN + payload.len());
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&payload);
        atomic_write(path, &buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn sample() -> Manifest {
        Manifest {
            version: 1,
            checkpoint_lsn: 42,
            snapshot_id: 7,
        }
    }

    #[test]
    fn load_missing_returns_none() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("manifest");
        assert_eq!(Manifest::load(&path).unwrap(), None);
    }

    #[test]
    fn store_then_load_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("manifest");
        let m = sample();
        m.store(&path).unwrap();
        let loaded = Manifest::load(&path).unwrap().expect("manifest present");
        assert_eq!(loaded, m);
    }

    #[test]
    fn store_overwrites_previous() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("manifest");
        sample().store(&path).unwrap();
        let m2 = Manifest {
            version: 1,
            checkpoint_lsn: 100,
            snapshot_id: 8,
        };
        m2.store(&path).unwrap();
        assert_eq!(Manifest::load(&path).unwrap().unwrap(), m2);
    }

    #[test]
    fn rejects_wrong_magic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("manifest");
        let mut bytes = vec![0u8; HEADER_LEN + 4];
        bytes[..8].copy_from_slice(b"NOTKOVA!");
        fs::write(&path, &bytes).unwrap();
        match Manifest::load(&path).unwrap_err() {
            KovaStorageError::CorruptRecord { reason } => {
                assert!(reason.contains("magic"), "got: {reason}");
            }
            other => panic!("expected CorruptRecord, got {other:?}"),
        }
    }

    #[test]
    fn rejects_unsupported_version() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("manifest");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&999u32.to_le_bytes());
        fs::write(&path, &bytes).unwrap();
        match Manifest::load(&path).unwrap_err() {
            KovaStorageError::CorruptRecord { reason } => {
                assert!(reason.contains("version"), "got: {reason}");
            }
            other => panic!("expected CorruptRecord, got {other:?}"),
        }
    }

    #[test]
    fn rejects_truncated_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("manifest");
        fs::write(&path, b"KOV").unwrap();
        match Manifest::load(&path).unwrap_err() {
            KovaStorageError::CorruptRecord { reason } => {
                assert!(reason.contains("too short"), "got: {reason}");
            }
            other => panic!("expected CorruptRecord, got {other:?}"),
        }
    }
}

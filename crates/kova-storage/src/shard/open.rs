//! `Shard::open` and `Shard::open_seeded` : the production file-backed
//! combo (`FileWal + MmapVectorStore + FileMetadataStore`).
//!
//! On open, consults the manifest to decide whether to load a snapshot
//! and which WAL prefix to skip during replay. Also runs a one-shot
//! cleanup of orphan `graph.{N}.snapshot` files left over from
//! checkpoints that committed but didn't get to delete their predecessor.

use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use kova_core::Distance;
use kova_index::{HnswIndex, HnswParams};

use crate::{FileMetadataStore, FileWal, Lsn, Manifest, MmapVectorStore};

use super::{DEFAULT_SEED, Shard, ShardError};

impl<D: Distance> Shard<D, MmapVectorStore, FileMetadataStore, FileWal> {
    /// Open (or create) a file-backed shard rooted at `dir`. Uses the
    /// default RNG seed.
    ///
    /// On first open, creates the directory and empty backing files. On
    /// subsequent opens, the stores recover their persisted state and the
    /// WAL is replayed to rebuild the in-memory index.
    ///
    /// If a `manifest` is present, the corresponding
    /// `graph.{snapshot_id}.snapshot` is loaded directly into the
    /// in-memory index and WAL replay starts at `checkpoint_lsn + 1`,
    /// skipping the prefix already baked into the snapshot. If no
    /// manifest is present (fresh shard or pre-checkpoint state),
    /// replay walks the WAL from `Lsn::ZERO`.
    ///
    /// Any orphan `graph.{N}.snapshot` files (i.e. ones the manifest
    /// doesn't reference) are deleted as a one-shot cleanup at open
    /// time. They're harmless if left behind, but the next checkpoint
    /// would orphan another, so cleaning up on open keeps the directory
    /// tidy.
    ///
    /// `dim` pins the vector dimension. If the underlying `vectors.mmap`
    /// already exists with a different dim, [`MmapVectorStore::open`]
    /// surfaces the mismatch as [`crate::KovaStorageError::CorruptRecord`].
    ///
    /// # Errors
    /// Any [`crate::KovaStorageError`] from the backing primitives bubbles
    /// up as [`ShardError::Backend`].
    pub fn open(
        dir: impl AsRef<Path>,
        dim: usize,
        metric: D,
        params: HnswParams,
    ) -> Result<Self, ShardError> {
        Self::open_seeded(dir, dim, metric, params, DEFAULT_SEED)
    }

    /// Like [`Self::open`] but with an explicit RNG seed.
    ///
    /// # Errors
    /// See [`Self::open`].
    pub fn open_seeded(
        dir: impl AsRef<Path>,
        dim: usize,
        metric: D,
        params: HnswParams,
        seed: u64,
    ) -> Result<Self, ShardError> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).map_err(ShardError::backend)?;

        // ---- Load primitives (always) ----
        let wal = FileWal::open(dir.join("wal")).map_err(ShardError::backend)?;
        let vectors =
            MmapVectorStore::open(dir.join("vectors.mmap"), dim).map_err(ShardError::backend)?;
        let metadata =
            FileMetadataStore::open(dir.join("metadata.bin")).map_err(ShardError::backend)?;

        // ---- Manifest-aware index construction ----
        //
        // If a manifest exists, load the snapshot it names and replay
        // only WAL records past the captured LSN. Otherwise build a
        // fresh empty index and replay everything from LSN 0.
        let manifest = Manifest::load(&dir.join("manifest")).map_err(ShardError::backend)?;
        let (index, snapshot_id, checkpoint_lsn, replay_from) = if let Some(m) = manifest {
            let snapshot_path = dir.join(format!("graph.{}.snapshot", m.snapshot_id));
            let file = File::open(&snapshot_path).map_err(ShardError::backend)?;
            let mut reader = BufReader::new(file);
            let idx = HnswIndex::read_snapshot(metric, params, seed, vectors, &mut reader)?;
            let cp_lsn = Lsn::new(m.checkpoint_lsn);
            (idx, m.snapshot_id, cp_lsn, Lsn::new(m.checkpoint_lsn + 1))
        } else {
            let idx = HnswIndex::seeded_with_store(metric, params, seed, vectors);
            (idx, 0, Lsn::ZERO, Lsn::ZERO)
        };

        // ---- One-shot orphan cleanup ----
        // Best-effort ; failures are non-fatal.
        cleanup_orphan_snapshots(dir, snapshot_id);

        Self::from_parts_with_checkpoint_state(
            index,
            metadata,
            wal,
            Some(dir.to_path_buf()),
            snapshot_id,
            checkpoint_lsn,
            replay_from,
        )
    }
}

/// Scan `dir` for `graph.{N}.snapshot` files where `N != live_snapshot_id`
/// and delete them. Best-effort : I/O failures are swallowed silently
/// because orphans are harmless and the next checkpoint will produce
/// another one to clean up next open.
pub(super) fn cleanup_orphan_snapshots(dir: &Path, live_snapshot_id: u64) {
    let Ok(read_dir) = fs::read_dir(dir) else {
        return;
    };
    for entry in read_dir.flatten() {
        let path: PathBuf = entry.path();
        let Some(stem) = path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(parse_snapshot_id)
        else {
            continue;
        };
        if stem != live_snapshot_id {
            let _ = fs::remove_file(&path);
        }
    }
}

/// Parse `graph.{N}.snapshot` into `Some(N)`. Returns `None` for any
/// other filename.
fn parse_snapshot_id(name: &str) -> Option<u64> {
    let rest = name.strip_prefix("graph.")?;
    let id_str = rest.strip_suffix(".snapshot")?;
    id_str.parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use kova_core::{L2, Metadata, Value, Vector, VectorId};
    use kova_index::HnswParams;
    use tempfile::tempdir;

    use super::super::{Shard, ShardError};
    use super::parse_snapshot_id;

    fn v(data: Vec<f32>) -> Vector {
        Vector::try_new(data).unwrap()
    }
    fn id(n: u64) -> VectorId {
        VectorId::new(n)
    }
    fn tag_meta(tag: &str) -> Metadata {
        let mut m = Metadata::new();
        m.insert("tag".into(), Value::String(tag.into()));
        m
    }

    /// First `open` on an empty directory creates the three backing files
    /// and the shard reports zero size.
    #[test]
    fn open_creates_empty_shard_and_files() {
        let dir = tempdir().unwrap();
        let shard = Shard::open(dir.path(), 3, L2, HnswParams::default()).unwrap();

        assert!(shard.is_empty());
        assert_eq!(shard.len(), 0);

        assert!(
            dir.path().join("wal").exists(),
            "wal/ directory should exist"
        );
        assert!(
            dir.path().join("vectors.mmap").exists(),
            "vectors.mmap should exist"
        );
        assert!(
            dir.path().join("metadata.bin").exists(),
            "metadata.bin should exist"
        );
        // No manifest yet : nothing has been checkpointed.
        assert!(!dir.path().join("manifest").exists());
    }

    /// Insert + search through the file-backed combo.
    #[test]
    fn insert_then_search_file_backed() {
        let dir = tempdir().unwrap();
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();

        shard
            .insert(id(1), v(vec![1.0, 0.0]), tag_meta("alpha"))
            .unwrap();
        shard
            .insert(id(2), v(vec![0.0, 1.0]), tag_meta("beta"))
            .unwrap();

        let hits = shard.search(&v(vec![1.0, 0.05]), 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, id(1));
        assert_eq!(
            hits[0].metadata.get("tag"),
            Some(&Value::String("alpha".into()))
        );
    }

    /// Drop + reopen recovers all inserts via WAL replay (no checkpoint
    /// yet, so the full log replays from LSN 0).
    #[test]
    fn reopen_recovers_inserted_vectors() {
        let dir = tempdir().unwrap();

        {
            let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
            shard
                .insert(id(1), v(vec![1.0, 0.0]), tag_meta("alpha"))
                .unwrap();
            shard
                .insert(id(2), v(vec![0.0, 1.0]), tag_meta("beta"))
                .unwrap();
            shard
                .insert(id(3), v(vec![0.5, 0.5]), tag_meta("gamma"))
                .unwrap();
        }

        let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        assert_eq!(shard.len(), 3);
        assert!(shard.contains(id(1)));
        assert!(shard.contains(id(2)));
        assert!(shard.contains(id(3)));

        let hits = shard.search(&v(vec![1.0, 0.0]), 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, id(1));
        assert_eq!(
            hits[0].metadata.get("tag"),
            Some(&Value::String("alpha".into()))
        );
    }

    /// Reopening with a different `dim` errors via `ShardError::Backend`.
    #[test]
    fn reopen_with_wrong_dim_errors() {
        let dir = tempdir().unwrap();

        {
            let mut shard = Shard::open(dir.path(), 3, L2, HnswParams::default()).unwrap();
            shard
                .insert(id(1), v(vec![1.0, 2.0, 3.0]), Metadata::new())
                .unwrap();
        }

        let Err(err) = Shard::open(dir.path(), 4, L2, HnswParams::default()) else {
            panic!("expected error on dim mismatch");
        };
        match err {
            ShardError::Backend(ref source) => {
                let msg = format!("{source}");
                assert!(msg.contains("dim"), "expected dim error, got: {msg}");
            }
            other => panic!("expected Backend error, got {other:?}"),
        }
    }

    /// Opening, dropping, and opening again N times leaves identical state.
    #[test]
    fn replay_is_idempotent_across_multiple_reopens() {
        let dir = tempdir().unwrap();

        {
            let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
            shard
                .insert(id(1), v(vec![1.0, 0.0]), tag_meta("a"))
                .unwrap();
            shard
                .insert(id(2), v(vec![0.0, 1.0]), tag_meta("b"))
                .unwrap();
        }

        for round in 0..3 {
            let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
            assert_eq!(shard.len(), 2, "round {round}: wrong len");
            assert!(shard.contains(id(1)), "round {round}: missing id 1");
            assert!(shard.contains(id(2)), "round {round}: missing id 2");
            let hits = shard.search(&v(vec![1.0, 0.0]), 1).unwrap();
            assert_eq!(hits[0].id, id(1), "round {round}: wrong nearest");
        }
    }

    // ---------- snapshot filename parsing ----------

    #[test]
    fn parse_snapshot_id_handles_valid_and_invalid_names() {
        assert_eq!(parse_snapshot_id("graph.0.snapshot"), Some(0));
        assert_eq!(parse_snapshot_id("graph.42.snapshot"), Some(42));
        assert_eq!(parse_snapshot_id("graph.9999.snapshot"), Some(9999));
        assert_eq!(parse_snapshot_id("graph.snapshot"), None);
        assert_eq!(parse_snapshot_id("graph.abc.snapshot"), None);
        assert_eq!(parse_snapshot_id("graph.0.snapshot.tmp"), None);
        assert_eq!(parse_snapshot_id("vectors.mmap"), None);
        assert_eq!(parse_snapshot_id("manifest"), None);
        assert_eq!(parse_snapshot_id("graph.-1.snapshot"), None);
    }
}

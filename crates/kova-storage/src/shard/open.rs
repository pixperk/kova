//! `Shard::open` and `Shard::open_seeded` : the production file-backed
//! combo (`FileWal + MmapVectorStore + FileMetadataStore`).
//!
//! The generic `from_parts` is what the in-memory composition path uses ;
//! this file is specifically about wiring the directory layout to those
//! primitives and running WAL replay to bring the in-memory index in
//! sync with what's on disk.

use std::path::Path;

use kova_core::Distance;
use kova_index::HnswParams;

use crate::{FileMetadataStore, FileWal, MmapVectorStore};

use super::{DEFAULT_SEED, Shard, ShardError};

impl<D: Distance> Shard<D, MmapVectorStore, FileMetadataStore, FileWal> {
    /// Open (or create) a file-backed shard rooted at `dir`. Uses the
    /// default RNG seed.
    ///
    /// On first open, creates the directory and empty backing files. On
    /// subsequent opens, the stores recover their persisted state and the
    /// WAL is replayed to rebuild the in-memory index.
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

        let wal = FileWal::open(dir.join("wal")).map_err(ShardError::backend)?;
        let vectors =
            MmapVectorStore::open(dir.join("vectors.mmap"), dim).map_err(ShardError::backend)?;
        let metadata =
            FileMetadataStore::open(dir.join("metadata.bin")).map_err(ShardError::backend)?;

        Self::from_parts_seeded(metric, params, seed, vectors, metadata, wal)
    }
}

#[cfg(test)]
mod tests {
    use kova_core::{L2, Metadata, Value, Vector, VectorId};
    use kova_index::HnswParams;
    use tempfile::tempdir;

    use super::super::{Shard, ShardError};

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
    }

    /// Insert + search through the file-backed combo. Confirms
    /// `Shard::open` wires up the right primitives end-to-end.
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

    /// Drop a shard, open it again on the same directory, inserts are
    /// still there with their metadata intact. Exercises every recovery
    /// path : mmap slot walk, metadata bincode load, WAL segment enum,
    /// replay into a fresh HNSW index.
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

    /// Reopening with a `dim` that doesn't match the existing
    /// `vectors.mmap` header surfaces as a `ShardError::Backend` whose
    /// source message mentions "dim".
    #[test]
    fn reopen_with_wrong_dim_errors() {
        let dir = tempdir().unwrap();

        {
            let mut shard = Shard::open(dir.path(), 3, L2, HnswParams::default()).unwrap();
            shard
                .insert(id(1), v(vec![1.0, 2.0, 3.0]), Metadata::new())
                .unwrap();
        }

        // Can't use `.unwrap_err()` : the Ok variant (`Shard`) isn't Debug
        // (HnswIndex isn't Debug, deliberately).
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

    /// Opening, dropping, and opening again N times leaves the shard in
    /// the same state every time. The invariant the crash test relies on.
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
}

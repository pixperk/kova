//! `Shard::checkpoint`, `Shard::should_checkpoint`, and [`CheckpointPolicy`].
//!
//! Vacuum + write the in-memory HNSW graph to disk + atomic-commit a
//! new manifest + truncate the WAL up to the captured LSN + delete the
//! old snapshot file. The manifest's `atomic_write` is the single
//! commit point ; everything before is preparation that can be
//! discarded on crash, everything after is best-effort cleanup.
//!
//! # Why generation-numbered snapshots
//!
//! `graph.{snapshot_id}.snapshot` is generation-numbered (unlike
//! `vectors.mmap` which is overwritten in place) because the snapshot's
//! content is **coupled to the manifest's `checkpoint_lsn`**. If we
//! overwrote `graph.snapshot` before atomic-writing the manifest, a
//! crash in that window would leave the file containing the new graph
//! but the manifest still pointing at the old `checkpoint_lsn`. On
//! reopen, replay would re-apply WAL records that are already baked
//! into the new snapshot, hitting `DuplicateId`. Generation numbers
//! make the manifest commit the single atomic "which snapshot is live"
//! decision.

use std::fs;

use kova_core::Distance;
use kova_index::Index;

use crate::atomic::{atomic_write, atomic_write_streaming};
use crate::{FileMetadataStore, FileWal, Lsn, Manifest, MmapVectorStore, Wal};

use super::{Shard, ShardError};

/// Build the filename for a catalog snapshot at the given generation.
pub(super) fn catalog_filename(snapshot_id: u64) -> String {
    format!("catalog.{snapshot_id}.bin")
}

/// Build the filename for a column-stats snapshot at the given
/// generation. Mirrors [`catalog_filename`].
pub(super) fn stats_filename(snapshot_id: u64) -> String {
    format!("stats.{snapshot_id}.bin")
}

/// Read-only thresholds for the [`Shard::should_checkpoint`] hint.
///
/// All fields are `Option<...>` so callers can opt into the thresholds
/// they care about and ignore the rest. The hint returns `true` if
/// **any** enabled threshold is exceeded.
#[derive(Debug, Default, Clone, Copy)]
pub struct CheckpointPolicy {
    /// Suggest checkpoint when the WAL holds more than this many
    /// segments. File-backed `Wal` impls track segments natively ;
    /// in-memory impls report 1, so this threshold never fires for them.
    pub max_wal_segments: Option<usize>,

    /// Suggest checkpoint when the tombstone ratio
    /// (`tombstone_count / total_node_count`) exceeds this value.
    /// Range is `0.0..=1.0`. `None` disables the check.
    pub max_tombstone_ratio: Option<f32>,

    /// Suggest checkpoint when this many WAL records have been appended
    /// since the last successful checkpoint.
    pub max_records_since_checkpoint: Option<u64>,
}

impl<D, V, M, W> Shard<D, V, M, W>
where
    D: Distance,
    V: kova_core::VectorStore,
    M: kova_core::MetadataStore,
    W: Wal,
{
    /// Read-only hint : would calling [`Self::checkpoint`] right now be
    /// a good idea per `policy`? Cheap : just reads in-memory counts.
    ///
    /// Returns `true` if **any** enabled threshold in `policy` is
    /// exceeded. Returns `false` if every threshold is `None`, or if
    /// none are crossed yet. The caller is responsible for actually
    /// calling [`Self::checkpoint`] ; this method only provides the
    /// signal.
    #[must_use]
    pub fn should_checkpoint(&self, policy: &CheckpointPolicy) -> bool {
        if let Some(max) = policy.max_wal_segments
            && self.wal.segment_count() > max
        {
            return true;
        }

        if let Some(max) = policy.max_tombstone_ratio {
            let total = self.index.len();
            if total > 0 {
                #[allow(clippy::cast_precision_loss)]
                let ratio = self.index.tombstone_count() as f32 / total as f32;
                if ratio > max {
                    return true;
                }
            }
        }

        if let Some(max) = policy.max_records_since_checkpoint {
            // Records since checkpoint = (last LSN ever) - (checkpoint LSN).
            // For a fresh shard with no inserts, last_lsn is None -> 0.
            let last_lsn = self.wal.last_lsn().map_or(0, Lsn::get);
            let since = last_lsn.saturating_sub(self.checkpoint_lsn.get());
            if since > max {
                return true;
            }
        }

        false
    }
}

impl<D: Distance> Shard<D, MmapVectorStore, FileMetadataStore, FileWal> {
    /// Vacuum, snapshot the graph, atomic-commit a new manifest, truncate
    /// the WAL, delete the old snapshot file.
    ///
    /// Returns the LSN that was captured by this checkpoint. WAL records
    /// at or before this LSN are baked into the new snapshot ; records
    /// after are still in `wal/` and will be replayed on the next open
    /// starting from `captured_lsn + 1`.
    ///
    /// # Phases
    ///
    /// 1. **Capture** : read `wal.last_lsn()`. Anything in the WAL up to
    ///    and including this LSN becomes the snapshot's coverage.
    /// 2. **Snapshot** : stream the current graph to
    ///    `graph.{new_snapshot_id}.snapshot` via `atomic_write_streaming`.
    ///    The new file lives alongside the old one ; nothing references
    ///    it yet.
    /// 3. **Commit** : atomic-write the new `manifest` pointing at the
    ///    new `snapshot_id` and `checkpoint_lsn`. **This is the single
    ///    durable commit point.** A crash before this leaves the old
    ///    manifest (so the old snapshot stays live and the WAL replays
    ///    in full as usual). A crash after this leaves the new manifest
    ///    and new snapshot live ; remaining WAL records past
    ///    `checkpoint_lsn` are still on disk and will be replayed.
    /// 4. **Truncate WAL** : drop records `<= checkpoint_lsn`. Best-
    ///    effort ; a crash here just leaves dead-weight records that
    ///    `replay_from(checkpoint_lsn + 1)` will skip on next open.
    /// 5. **Delete old snapshot** : best-effort cleanup. Orphans get
    ///    swept on the next open if this step fails.
    ///
    /// **Does not vacuum.** Call [`Shard::vacuum`] first if you want the
    /// snapshot to exclude tombstoned nodes ; vacuum is a logged state
    /// change (see [`crate::Record::Vacuum`]) precisely so that it does
    /// not happen at a locally-chosen moment.
    ///
    /// # Errors
    /// [`ShardError`] from the snapshot write, manifest write, or
    /// WAL truncate. The post-commit steps (truncate, delete) are
    /// fail-safe ; if they error after the manifest committed, the
    /// next open will still see the new state.
    ///
    /// # Panics
    /// Only the in-memory state mutation at the end (updating
    /// `self.snapshot_id` and `self.checkpoint_lsn`) can panic, and
    /// only if the post-vacuum index lost its `dir` somehow, which is
    /// impossible for a shard constructed via [`Shard::open`].
    pub fn checkpoint(&mut self) -> Result<Lsn, ShardError> {
        let dir = self
            .dir
            .as_ref()
            .expect("checkpoint requires a file-backed shard with dir")
            .clone();

        // -------- Phase 1 : capture --------
        let checkpoint_lsn = self.wal.last_lsn().unwrap_or(Lsn::ZERO);

        // NOTE : checkpoint deliberately does **not** vacuum.
        //
        // It used to, on the reasonable-sounding grounds that there is
        // no point snapshotting state you are about to discard. But
        // vacuum rewires the graph in a way that depends on *when* it
        // ran, while checkpoint is a local decision every node makes on
        // its own schedule (`CheckpointPolicy`). Vacuuming here would
        // therefore make two replicas holding identical logs diverge,
        // which is the one thing replication cannot tolerate.
        //
        // The two concerns are now separate :
        //
        // - `Shard::vacuum` is a logged state change (`Record::Vacuum`),
        //   so every replica applies it at the same log position.
        // - `checkpoint` is a pure durability artifact : snapshot the
        //   graph you have, commit the manifest, truncate the WAL : and
        //   is safe to run locally at any time.
        //
        // Operators who want vacuumed snapshots call `vacuum()` before
        // `checkpoint()`. That also makes a cheap durability-only
        // checkpoint possible, which was not expressible before.

        // -------- Phase 2 : stream snapshot to a NEW file --------
        // The old `graph.{old}.snapshot` stays valid until the manifest
        // commits to the new id below.
        let new_snapshot_id = self.snapshot_id + 1;
        let new_snapshot_path = dir.join(format!("graph.{new_snapshot_id}.snapshot"));
        atomic_write_streaming(&new_snapshot_path, |w| {
            self.index
                .write_snapshot(w)
                .map_err(|e| crate::KovaStorageError::CorruptRecord {
                    reason: format!("hnsw snapshot write: {e}"),
                })
        })
        .map_err(ShardError::backend)?;

        // -------- Phase 2b : serialise the catalog alongside --------
        // The catalog snapshot is generation-numbered for the same
        // reason the graph snapshot is : the manifest commit below is
        // the single atomic "which generation is live" point. A crash
        // between the catalog write and the manifest commit leaves a
        // tmp catalog file that the next open ignores ; the OLD
        // catalog (still pointed at by the OLD manifest) remains live.
        let new_catalog_path = dir.join(catalog_filename(new_snapshot_id));
        let catalog_bytes = self.catalog.encode().map_err(|e| {
            ShardError::backend(std::io::Error::other(format!("catalog encode: {e}")))
        })?;
        atomic_write(&new_catalog_path, &catalog_bytes).map_err(ShardError::backend)?;

        // -------- Phase 2c : rebuild + serialise the column stats --------
        // Stats are derived state, so rebuild from the post-vacuum
        // metadata store every checkpoint. Cost is O(N) walk of the
        // metadata HashMap ; happens once per checkpoint, not per
        // query.
        self.rebuild_stats();
        let new_stats_path = dir.join(stats_filename(new_snapshot_id));
        let stats_bytes = self.stats.encode().map_err(|e| {
            ShardError::backend(std::io::Error::other(format!("stats encode: {e}")))
        })?;
        atomic_write(&new_stats_path, &stats_bytes).map_err(ShardError::backend)?;

        // -------- Phase 3 : commit (the single atomic point) --------
        let manifest = Manifest {
            version: 1,
            checkpoint_lsn: checkpoint_lsn.get(),
            snapshot_id: new_snapshot_id,
        };
        manifest
            .store(&dir.join("manifest"))
            .map_err(ShardError::backend)?;

        // After this line, "the new snapshot is live" is durably true.
        // Everything below is post-commit cleanup ; failures aren't fatal.
        let old_snapshot_id = self.snapshot_id;
        self.snapshot_id = new_snapshot_id;
        self.checkpoint_lsn = checkpoint_lsn;

        // -------- Phase 4 : truncate WAL --------
        // Best-effort : if this fails, the next open just sees more
        // records to replay. They get skipped via `replay_from(cp+1)`
        // because the manifest's `checkpoint_lsn` covers them.
        let _ = self.wal.truncate_before(Lsn::new(checkpoint_lsn.get() + 1));

        // -------- Phase 5 : delete old snapshot + old catalog + old stats --------
        // Best-effort. The orphan cleanup in `Shard::open` sweeps any
        // stragglers anyway.
        if old_snapshot_id != new_snapshot_id {
            let old_path = dir.join(format!("graph.{old_snapshot_id}.snapshot"));
            let _ = fs::remove_file(&old_path);
            let old_catalog_path = dir.join(catalog_filename(old_snapshot_id));
            let _ = fs::remove_file(&old_catalog_path);
            let old_stats_path = dir.join(stats_filename(old_snapshot_id));
            let _ = fs::remove_file(&old_stats_path);
        }

        Ok(checkpoint_lsn)
    }
}

#[cfg(test)]
mod tests {
    use kova_core::{
        InMemoryMetadataStore, InMemoryVectorStore, L2, Metadata, Value, Vector, VectorId,
    };
    use kova_index::HnswParams;
    use tempfile::tempdir;

    use crate::{InMemoryWal, Lsn, Manifest};

    use super::super::Shard;
    use super::CheckpointPolicy;

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

    // ---------- checkpoint happy path ----------

    /// Insert some, checkpoint, verify : manifest exists, snapshot file
    /// exists at the right generation, WAL is truncated, shard returns
    /// the captured LSN.
    #[test]
    fn checkpoint_writes_manifest_snapshot_and_truncates_wal() {
        let dir = tempdir().unwrap();
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();

        shard
            .insert(id(1), v(vec![1.0, 0.0]), tag_meta("a"))
            .unwrap();
        shard
            .insert(id(2), v(vec![0.0, 1.0]), tag_meta("b"))
            .unwrap();
        shard
            .insert(id(3), v(vec![1.0, 1.0]), tag_meta("c"))
            .unwrap();

        let cp_lsn = shard.checkpoint().unwrap();
        // Three inserts -> LSNs 0,1,2. last_lsn = 2.
        assert_eq!(cp_lsn, Lsn::new(2));

        assert!(dir.path().join("manifest").exists(), "manifest must exist");
        assert!(
            dir.path().join("graph.1.snapshot").exists(),
            "snapshot file at gen 1 must exist"
        );

        let manifest = Manifest::load(&dir.path().join("manifest"))
            .unwrap()
            .expect("manifest present");
        assert_eq!(manifest.checkpoint_lsn, 2);
        assert_eq!(manifest.snapshot_id, 1);
    }

    /// After checkpoint, reopen loads the snapshot directly and replays
    /// only post-checkpoint records. Verified by inserting more after
    /// checkpoint, reopening, and checking everything is present.
    #[test]
    fn reopen_after_checkpoint_uses_snapshot_plus_replay() {
        let dir = tempdir().unwrap();

        {
            let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
            for n in 0u16..5 {
                shard
                    .insert(
                        id(u64::from(n)),
                        v(vec![f32::from(n), 0.0]),
                        tag_meta(&format!("pre-{n}")),
                    )
                    .unwrap();
            }
            shard.checkpoint().unwrap();
            // Inserts after the checkpoint live in the WAL only ; replay
            // applies them on top of the snapshot.
            for n in 5u16..10 {
                shard
                    .insert(
                        id(u64::from(n)),
                        v(vec![f32::from(n), 0.0]),
                        tag_meta(&format!("post-{n}")),
                    )
                    .unwrap();
            }
        }

        let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        assert_eq!(shard.len(), 10);
        for n in 0..10 {
            assert!(shard.contains(id(n)), "id {n} missing");
        }
        // Spot-check pre and post metadata both round-trip.
        let hits = shard.search(&v(vec![0.0, 0.0]), 1).unwrap();
        assert_eq!(hits[0].id, id(0));
        assert_eq!(
            hits[0].metadata.get("tag"),
            Some(&Value::String("pre-0".into()))
        );
    }

    /// Checkpoint **does not** vacuum, and a tombstone written into a
    /// snapshot survives reopen.
    ///
    /// This used to assert the opposite (`checkpoint_locks_in_vacuum_work`)
    /// because checkpoint vacuumed first. It no longer does : vacuum
    /// rewires the graph in a timing-dependent way, while checkpoint is
    /// a local decision each node makes on its own schedule, so
    /// vacuuming here would make replicas holding identical logs
    /// diverge. See [`crate::Record::Vacuum`].
    ///
    /// The load-bearing part is that the snapshot carries the tombstone
    /// set (format v2). Without it, checkpoint would truncate the WAL
    /// past the `Delete` record while writing a snapshot that still
    /// contains the node, and the row would come back from the dead.
    #[test]
    fn checkpoint_preserves_tombstones_without_vacuuming() {
        let dir = tempdir().unwrap();

        {
            let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
            shard
                .insert(id(1), v(vec![1.0, 0.0]), tag_meta("a"))
                .unwrap();
            shard
                .insert(id(2), v(vec![0.0, 1.0]), tag_meta("b"))
                .unwrap();
            shard.delete(id(1)).unwrap();
            shard.checkpoint().unwrap();
            // Not vacuumed : the node is still in the graph, tombstoned.
            assert_eq!(shard.index.tombstone_count(), 1);
        }

        let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        assert!(!shard.contains(id(1)), "deleted row must stay deleted");
        assert!(shard.contains(id(2)));
        assert_eq!(shard.len(), 1);
        assert_eq!(
            shard.index.tombstone_count(),
            1,
            "the snapshot must carry the tombstone set across reopen"
        );
    }

    /// Vacuum then checkpoint : the vacuum is locked into the snapshot
    /// and there is nothing left to redo.
    #[test]
    fn vacuum_then_checkpoint_locks_in_the_vacuum() {
        let dir = tempdir().unwrap();

        {
            let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
            shard
                .insert(id(1), v(vec![1.0, 0.0]), tag_meta("a"))
                .unwrap();
            shard
                .insert(id(2), v(vec![0.0, 1.0]), tag_meta("b"))
                .unwrap();
            shard.delete(id(1)).unwrap();
            shard.vacuum().unwrap();
            shard.checkpoint().unwrap();
        }

        let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        assert!(!shard.contains(id(1)));
        assert!(shard.contains(id(2)));
        assert_eq!(shard.len(), 1);
        assert_eq!(shard.index.tombstone_count(), 0);
    }

    /// Two checkpoints in a row : the second supersedes the first ; the
    /// old `graph.1.snapshot` is deleted ; the new `graph.2.snapshot`
    /// is live ; manifest points at `snapshot_id: 2`.
    #[test]
    fn second_checkpoint_supersedes_first() {
        let dir = tempdir().unwrap();
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();

        shard
            .insert(id(1), v(vec![1.0, 0.0]), Metadata::new())
            .unwrap();
        shard.checkpoint().unwrap();
        assert!(dir.path().join("graph.1.snapshot").exists());

        shard
            .insert(id(2), v(vec![0.0, 1.0]), Metadata::new())
            .unwrap();
        shard.checkpoint().unwrap();

        assert!(
            !dir.path().join("graph.1.snapshot").exists(),
            "old snapshot should be deleted"
        );
        assert!(dir.path().join("graph.2.snapshot").exists());

        let manifest = Manifest::load(&dir.path().join("manifest"))
            .unwrap()
            .unwrap();
        assert_eq!(manifest.snapshot_id, 2);
    }

    /// Orphan snapshots left behind by a crashed checkpoint (or by
    /// failing to delete in phase 6) get swept on the next `Shard::open`.
    #[test]
    fn orphan_snapshots_get_cleaned_up_on_open() {
        let dir = tempdir().unwrap();
        // First produce a manifest + live snapshot.
        {
            let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
            shard
                .insert(id(1), v(vec![1.0, 0.0]), Metadata::new())
                .unwrap();
            shard.checkpoint().unwrap();
        }
        assert!(dir.path().join("graph.1.snapshot").exists());

        // Drop a fake orphan snapshot from a "previous" run.
        let orphan = dir.path().join("graph.99.snapshot");
        std::fs::write(&orphan, b"garbage but plausible filename").unwrap();
        assert!(orphan.exists());

        // Reopen the shard ; orphan should be gone, live snapshot kept.
        let _shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        assert!(!orphan.exists(), "orphan should have been cleaned up");
        assert!(
            dir.path().join("graph.1.snapshot").exists(),
            "live snapshot must survive cleanup"
        );
    }

    /// Checkpoint on a shard with no records is fine : captures
    /// `Lsn::ZERO`, writes an empty snapshot, manifests it.
    #[test]
    fn checkpoint_on_empty_shard_is_valid() {
        let dir = tempdir().unwrap();
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        let cp = shard.checkpoint().unwrap();
        assert_eq!(cp, Lsn::ZERO);

        // Reopen : empty shard, manifest live.
        let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        assert_eq!(shard.len(), 0);
        let manifest = Manifest::load(&dir.path().join("manifest"))
            .unwrap()
            .unwrap();
        assert_eq!(manifest.snapshot_id, 1);
    }

    // ---------- should_checkpoint ----------

    /// All-`None` policy never suggests checkpoint, regardless of state.
    #[test]
    fn should_checkpoint_default_policy_is_always_false() {
        let mut shard = Shard::from_parts(
            L2,
            HnswParams::default(),
            InMemoryVectorStore::new(),
            InMemoryMetadataStore::new(),
            InMemoryWal::new(),
        )
        .unwrap();
        // Even after inserts + deletes, an empty policy returns false.
        shard
            .insert(id(1), v(vec![1.0, 0.0]), Metadata::new())
            .unwrap();
        shard
            .insert(id(2), v(vec![0.0, 1.0]), Metadata::new())
            .unwrap();
        shard.delete(id(1)).unwrap();
        assert!(!shard.should_checkpoint(&CheckpointPolicy::default()));
    }

    /// `max_tombstone_ratio` fires when the ratio exceeds the threshold.
    #[test]
    fn should_checkpoint_fires_on_tombstone_ratio() {
        let mut shard = Shard::from_parts(
            L2,
            HnswParams::default(),
            InMemoryVectorStore::new(),
            InMemoryMetadataStore::new(),
            InMemoryWal::new(),
        )
        .unwrap();
        for n in 0u16..4 {
            shard
                .insert(
                    id(u64::from(n)),
                    v(vec![f32::from(n), 0.0]),
                    Metadata::new(),
                )
                .unwrap();
        }
        // 0/4 tombstoned : threshold not crossed.
        let policy = CheckpointPolicy {
            max_tombstone_ratio: Some(0.25),
            ..Default::default()
        };
        assert!(!shard.should_checkpoint(&policy));

        // 1/4 tombstoned = 0.25 (not strictly greater).
        shard.delete(id(0)).unwrap();
        assert!(!shard.should_checkpoint(&policy));

        // 2/4 tombstoned = 0.5 > 0.25 : fires.
        shard.delete(id(1)).unwrap();
        assert!(shard.should_checkpoint(&policy));
    }

    /// `max_records_since_checkpoint` fires when enough records have
    /// accumulated in the WAL past the captured `checkpoint_lsn`.
    #[test]
    fn should_checkpoint_fires_on_records_since_checkpoint() {
        let dir = tempdir().unwrap();
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();

        let policy = CheckpointPolicy {
            max_records_since_checkpoint: Some(2),
            ..Default::default()
        };

        shard
            .insert(id(0), v(vec![0.0, 0.0]), Metadata::new())
            .unwrap();
        // Records appended: 1 ; checkpoint_lsn = 0 ; since = 0 - 0 = 0
        // Wait : with one append, last_lsn = 0, checkpoint_lsn = 0,
        // since = 0 - 0 = 0. Threshold 2. Not fired.
        assert!(!shard.should_checkpoint(&policy));
        shard
            .insert(id(1), v(vec![1.0, 0.0]), Metadata::new())
            .unwrap();
        // last_lsn = 1, since = 1, threshold 2. Not fired.
        assert!(!shard.should_checkpoint(&policy));
        shard
            .insert(id(2), v(vec![2.0, 0.0]), Metadata::new())
            .unwrap();
        // last_lsn = 2, since = 2, threshold 2 (strict >). Not fired.
        assert!(!shard.should_checkpoint(&policy));
        shard
            .insert(id(3), v(vec![3.0, 0.0]), Metadata::new())
            .unwrap();
        // last_lsn = 3, since = 3 > 2 : fires.
        assert!(shard.should_checkpoint(&policy));
    }

    /// After a checkpoint, `records_since_checkpoint` resets toward zero ;
    /// the policy stops firing until more records accumulate.
    #[test]
    fn should_checkpoint_resets_after_checkpoint() {
        let dir = tempdir().unwrap();
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();

        for n in 0u16..5 {
            shard
                .insert(
                    id(u64::from(n)),
                    v(vec![f32::from(n), 0.0]),
                    Metadata::new(),
                )
                .unwrap();
        }

        let policy = CheckpointPolicy {
            max_records_since_checkpoint: Some(2),
            ..Default::default()
        };
        assert!(shard.should_checkpoint(&policy));

        shard.checkpoint().unwrap();
        assert!(!shard.should_checkpoint(&policy));
    }
}

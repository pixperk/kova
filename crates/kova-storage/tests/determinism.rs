//! Deterministic apply : the gate for state-machine replication.
//!
//! Raft (and any log-shipping replication) rests on one property :
//! **applying the same records in the same order to the same starting
//! state must produce the same result on every replica.** If it does
//! not, two replicas answer the same query differently and no amount
//! of consensus fixes it.
//!
//! Kova's apply path has one obvious source of nondeterminism : HNSW
//! assigns each node a random top layer, and it is seeded
//! (`StdRng::seed_from_u64`), advanced once per insert. So determinism
//! should hold *given identical insert order*, which is exactly what a
//! replicated log provides.
//!
//! "Should" is not "does". These tests check it.

use kova_core::{L2, Metadata, Value, Vector, VectorId};
use kova_index::HnswParams;
use kova_storage::{Manifest, Shard};

const DIM: usize = 8;
const N: usize = 500;
const SEED: u64 = 0xD37E_2711_D37E_2711;

/// Inline xorshift64*. Deliberately not `rand` : this test is *about*
/// reproducibility, so its own inputs should not depend on an external
/// generator staying stable across versions. Also keeps `rand` out of
/// `kova-storage`'s dependency set.
struct Xorshift(u64);

impl Xorshift {
    fn new(seed: u64) -> Self {
        // Zero is a fixed point of xorshift ; nudge it.
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`, from the top 24 bits (an f32 mantissa).
    #[allow(clippy::cast_precision_loss)]
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    fn vector(&mut self) -> Vector {
        let data: Vec<f32> = (0..DIM).map(|_| self.next_f32()).collect();
        Vector::try_new(data).expect("non-empty vector")
    }
}

/// A deterministic row set. Same every call : this is the "log" the
/// replicas replay.
fn rows(n: usize) -> Vec<(VectorId, Vector, Metadata)> {
    let mut rng = Xorshift::new(SEED);
    (0..n)
        .map(|i| {
            let v = rng.vector();
            let mut m = Metadata::new();
            m.insert(
                "bucket".into(),
                Value::I64(i64::try_from(i % 7).expect("small")),
            );
            (VectorId::new(i as u64), v, m)
        })
        .collect()
}

fn queries(n: usize) -> Vec<Vector> {
    let mut rng = Xorshift::new(SEED ^ 0x0F1E_2D3C_4B5A_6978);
    (0..n).map(|_| rng.vector()).collect()
}

type FileShard = Shard<
    L2,
    kova_storage::MmapVectorStore,
    kova_storage::FileMetadataStore,
    kova_storage::FileWal,
>;

fn build(dir: &std::path::Path, data: &[(VectorId, Vector, Metadata)]) -> FileShard {
    let mut shard = Shard::open(dir, DIM, L2, HnswParams::default()).expect("open");
    shard.insert_many(data.to_vec()).expect("insert_many");
    shard
}

/// The **strongest** fingerprint : the bytes of the graph snapshot a
/// checkpoint writes. This is precisely the artifact Phase 5 ships to a
/// bootstrapping replica, so if two shards produce identical bytes they
/// are identical in every way that can matter downstream.
///
/// Consumes the shard : checkpointing needs `&mut`, and there is no
/// reason to keep it afterwards.
fn snapshot_bytes(mut shard: FileShard, dir: &std::path::Path) -> Vec<u8> {
    shard.checkpoint().expect("checkpoint");
    // Ask the manifest which generation is live rather than assuming
    // `graph.1.snapshot` : a shard that has checkpointed before is on a
    // later generation. The generation number lives in the filename,
    // not in the payload, so snapshots from different generations are
    // still directly comparable.
    let manifest = Manifest::load(&dir.join("manifest"))
        .expect("read manifest")
        .expect("checkpoint must have written a manifest");
    std::fs::read(dir.join(format!("graph.{}.snapshot", manifest.snapshot_id)))
        .expect("read snapshot")
}

/// The property that actually matters to a user : two replicas must
/// answer the same question the same way.
fn search_fingerprint(shard: &FileShard, qs: &[Vector], k: usize) -> Vec<Vec<(u64, u32)>> {
    qs.iter()
        .map(|q| {
            shard
                .search(q, k)
                .expect("search")
                .into_iter()
                // Distances are f32 and computed identically ; compare
                // their bit patterns so a NaN or a one-ulp difference
                // cannot slip through as "equal".
                .map(|h| (h.id.get(), h.distance.to_bits()))
                .collect()
        })
        .collect()
}

#[test]
fn two_shards_built_from_the_same_sequence_agree() {
    let data = rows(N);
    let qs = queries(20);

    let dir_a = tempfile::tempdir().expect("tempdir");
    let dir_b = tempfile::tempdir().expect("tempdir");
    let a = build(dir_a.path(), &data);
    let b = build(dir_b.path(), &data);

    assert_eq!(a.len(), b.len(), "row counts diverged");

    assert_eq!(
        search_fingerprint(&a, &qs, 10),
        search_fingerprint(&b, &qs, 10),
        "two identically-built shards answered the same query differently : \
         the apply path is not deterministic and log-shipping replication \
         cannot work"
    );

    assert_eq!(
        snapshot_bytes(a, dir_a.path()),
        snapshot_bytes(b, dir_b.path()),
        "graph snapshots differ byte-for-byte between two identically-built \
         shards. Semantic equality may still hold (check whether the search \
         assertion above passed) but replicas could not be compared by \
         checksum, and snapshot transfer would not be verifiable."
    );
}

/// The Raft-shaped case : one replica builds by applying records
/// directly, another builds by replaying the log from disk. Both must
/// land in the same place.
#[test]
fn replay_from_the_wal_reproduces_direct_application() {
    let data = rows(N);
    let qs = queries(20);

    let dir_direct = tempfile::tempdir().expect("tempdir");
    let direct = build(dir_direct.path(), &data);
    let direct_search = search_fingerprint(&direct, &qs, 10);
    let direct_snapshot = snapshot_bytes(direct, dir_direct.path());

    // Second shard : build, drop (leaving only the WAL), reopen. The
    // reopen path replays every record rather than applying them live.
    let dir_replayed = tempfile::tempdir().expect("tempdir");
    {
        let _built = build(dir_replayed.path(), &data);
    }
    let replayed =
        Shard::open(dir_replayed.path(), DIM, L2, HnswParams::default()).expect("reopen");

    assert_eq!(
        direct_search,
        search_fingerprint(&replayed, &qs, 10),
        "WAL replay produced a graph that answers queries differently than \
         live application of the same records"
    );
    assert_eq!(
        direct_snapshot,
        snapshot_bytes(replayed, dir_replayed.path()),
        "WAL replay produced a byte-different graph"
    );
}

/// Determinism must survive the full mutation surface, not just
/// inserts : a real log carries deletes and metadata updates too.
#[test]
fn mixed_mutations_are_deterministic() {
    let data = rows(N);
    let qs = queries(20);

    let apply = |dir: &std::path::Path| -> FileShard {
        let mut shard = build(dir, &data);
        // Delete a deterministic scattered subset.
        let doomed: Vec<VectorId> = (0..N)
            .filter(|i| (i * 37) % 100 < 15)
            .map(|i| VectorId::new(i as u64))
            .collect();
        shard.delete_many(doomed).expect("delete_many");
        // Update another subset's metadata.
        let updates: Vec<(VectorId, Metadata)> = (0..N)
            .filter(|i| (i * 53) % 100 < 10 && (i * 37) % 100 >= 15)
            .map(|i| {
                let mut m = Metadata::new();
                m.insert("bucket".into(), Value::I64(99));
                (VectorId::new(i as u64), m)
            })
            .collect();
        shard.update_metadata(updates).expect("update_metadata");
        shard
    };

    let dir_a = tempfile::tempdir().expect("tempdir");
    let dir_b = tempfile::tempdir().expect("tempdir");
    let a = apply(dir_a.path());
    let b = apply(dir_b.path());

    assert_eq!(a.len(), b.len());
    assert_eq!(
        search_fingerprint(&a, &qs, 10),
        search_fingerprint(&b, &qs, 10)
    );
    assert_eq!(
        snapshot_bytes(a, dir_a.path()),
        snapshot_bytes(b, dir_b.path()),
        "insert + delete + update sequence is not byte-deterministic"
    );
}

/// **Vacuum is not safe to run locally on a replica.**
///
/// Vacuum rewires the neighbour lists of every survivor that pointed at
/// a tombstoned node. The result depends on *which* nodes were
/// tombstoned when it ran, so two replicas that vacuum at different
/// log positions produce different graphs, and answer the same query
/// differently, without either being "wrong".
///
/// This test pins that divergence deliberately. It is the argument for
/// making vacuum a replicated log record (`Record::Vacuum`) rather than
/// a local maintenance decision : the leader picks the log position,
/// every replica vacuums at exactly that point.
///
/// If this test ever starts *failing* : i.e. vacuum timing stops
/// mattering : the constraint can be relaxed.
#[test]
fn vacuum_timing_changes_the_graph() {
    let data = rows(N);
    let qs = queries(20);

    let half: Vec<VectorId> = (0..N / 2).map(|i| VectorId::new(i as u64)).collect();
    let rest: Vec<VectorId> = (N / 2..N)
        .filter(|i| i % 3 == 0)
        .map(|i| VectorId::new(i as u64))
        .collect();

    // Replica 1 : delete both batches, then vacuum once at the end.
    let dir_late = tempfile::tempdir().expect("tempdir");
    let mut late = build(dir_late.path(), &data);
    late.delete_many(half.clone()).expect("delete");
    late.delete_many(rest.clone()).expect("delete");
    late.vacuum().expect("vacuum");

    // Replica 2 : same deletes, but vacuum in between.
    let dir_early = tempfile::tempdir().expect("tempdir");
    let mut early = build(dir_early.path(), &data);
    early.delete_many(half).expect("delete");
    early.vacuum().expect("vacuum");
    early.delete_many(rest).expect("delete");
    early.vacuum().expect("vacuum");

    // Same live rows either way : vacuum is semantically a no-op on
    // membership.
    assert_eq!(
        late.len(),
        early.len(),
        "vacuum changed which rows are live : that would be a real bug"
    );

    // But the graphs differ, so the answers can differ.
    let late_hits = search_fingerprint(&late, &qs, 10);
    let early_hits = search_fingerprint(&early, &qs, 10);
    assert_ne!(
        late_hits, early_hits,
        "vacuum timing no longer affects search results : if this is genuinely \
         true, Record::Vacuum may not be needed and this test should be revisited"
    );
}

/// **The payoff for splitting vacuum out of checkpoint.**
///
/// `checkpoint()` is a local decision : every node runs it on its own
/// `CheckpointPolicy` schedule. It used to vacuum first, and vacuum is
/// timing-dependent (see [`vacuum_timing_changes_the_graph`]), so two
/// replicas holding *identical logs* would diverge purely because they
/// happened to checkpoint at different moments. No amount of consensus
/// fixes that: the logs agree and the graphs still differ.
///
/// Vacuum is now a logged record and checkpoint is a pure durability
/// artifact, so checkpoint timing is free. This test is that claim,
/// checked rather than asserted.
#[test]
fn checkpoint_timing_does_not_affect_the_graph() {
    let data = rows(N);
    let qs = queries(20);
    let doomed: Vec<VectorId> = (0..N)
        .filter(|i| (i * 37) % 100 < 20)
        .map(|i| VectorId::new(i as u64))
        .collect();

    // Replica 1 : checkpoint early, before the deletes.
    let dir_early = tempfile::tempdir().expect("tempdir");
    let mut early = build(dir_early.path(), &data);
    early.checkpoint().expect("checkpoint");
    early.delete_many(doomed.clone()).expect("delete");
    early.vacuum().expect("vacuum");

    // Replica 2 : identical mutations, no checkpoint until the end.
    let dir_late = tempfile::tempdir().expect("tempdir");
    let mut late = build(dir_late.path(), &data);
    late.delete_many(doomed).expect("delete");
    late.vacuum().expect("vacuum");

    assert_eq!(early.len(), late.len());
    assert_eq!(
        search_fingerprint(&early, &qs, 10),
        search_fingerprint(&late, &qs, 10),
        "checkpoint timing changed the graph : checkpoint is a local \
         decision, so this would make replicas holding identical logs \
         diverge"
    );
    // `snapshot_bytes` checkpoints again ; for `early` that is its
    // second checkpoint and for `late` its first, which is exactly the
    // asymmetry being tested : hence the manifest lookup in the helper.
    assert_eq!(
        snapshot_bytes(early, dir_early.path()),
        snapshot_bytes(late, dir_late.path()),
        "checkpoint timing changed the snapshot bytes"
    );
}

/// Repeated delete/vacuum/checkpoint cycles must keep working.
///
/// This sequence used to fail outright. `checkpoint()` vacuumed
/// internally, and a second vacuum tripped over the dangling edges the
/// first one left behind (`vacuum: affected id N missing from nodes`),
/// because vacuum's pass 1 assumed HNSW edges are symmetric. A failed
/// checkpoint also means the WAL never truncates, so the log grows
/// without bound.
#[test]
fn repeated_delete_and_checkpoint_cycles_succeed() {
    let data = rows(N);
    let dir = tempfile::tempdir().expect("tempdir");
    let mut shard = build(dir.path(), &data);

    let mut alive = N;
    for round in 0..4u64 {
        let doomed: Vec<VectorId> = (0..N)
            .filter(|i| (*i as u64) % 4 == round)
            .map(|i| VectorId::new(i as u64))
            .collect();
        alive -= doomed.len();
        shard.delete_many(doomed).expect("delete");
        shard.vacuum().expect("vacuum");
        shard
            .checkpoint()
            .unwrap_or_else(|e| panic!("checkpoint failed in round {round}: {e:?}"));
        assert_eq!(shard.len(), alive, "row count wrong after round {round}");
    }

    // And it all survives a reopen.
    drop(shard);
    let reopened = Shard::open(dir.path(), DIM, L2, HnswParams::default()).expect("reopen");
    assert_eq!(reopened.len(), alive);
}

//! Crash recovery torture test.
//!
//! Spawns the [`crash_writer`](../bin/crash_writer.rs) child, lets it
//! insert vectors for a random delay, then `SIGKILL`s it. After the
//! kill, reopens the shard on the same directory and asserts that every
//! id the child printed `ACKED` for is durably present.
//!
//! Two flavours :
//!
//! - **`*_inserts_only`** : child only inserts. Exercises crash windows
//!   in `wal.append + wal.sync + index.insert + metadata.put`.
//! - **`*_with_checkpoints`** : child interleaves [`Shard::checkpoint`]
//!   calls every K inserts. Exercises crash windows during checkpoint
//!   (vacuum, snapshot write, manifest commit, WAL truncate, old
//!   snapshot delete) on top of the insert windows.
//!
//! This is the milestone test for the storage layer : everything else
//! is interesting only if THIS passes.
//!
//! # What "durable" means here
//!
//! The child prints `ACKED <id>` after [`Shard::insert`] returns Ok,
//! which means [`Wal::sync`] has fsynced the record. Under SIGKILL on
//! Linux, the kernel preserves the file system state up to the last
//! fsync. So every ACKED id must survive into the reopened shard.
//!
//! Similarly, the child prints `CHECKPOINTED <lsn>` after
//! [`Shard::checkpoint`] returns Ok, which means the manifest is durably
//! committed. If the parent sees any CHECKPOINTED line, the manifest
//! must exist on reopen and the post-checkpoint replay path is what
//! brings the in-memory index back.
//!
//! `vectors.mmap` is also OK across SIGKILL : the kernel keeps the page
//! cache, so even un-fsynced mmap writes are still visible on reopen.
//! The full power-loss scenario (where the page cache itself is lost) is
//! NOT covered by this test ; that needs a VM-level snapshot/restore or
//! explicit `msync` calls that we don't yet make. Documented limitation.
//!
//! # Tuning
//!
//! - [`SMOKE_ITERATIONS`] runs by default (fast ; ~5-10s).
//! - [`TORTURE_ITERATIONS`] is `#[ignore]`'d ; run with
//!   `cargo test -p kova-storage --test crash_recovery -- --ignored`.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use kova_core::{L2, VectorId};
use kova_index::HnswParams;
use kova_storage::{Manifest, Shard};
use tempfile::tempdir;

const VECTORS_PER_RUN: u64 = 200;
const DIM: usize = 8;
/// Maximum delay between spawn and SIGKILL. Tuned so the kill lands
/// somewhere mid-run for typical hardware ; tweak if local SSD is much
/// faster or slower.
const KILL_DELAY_RANGE_MS: u64 = 1500;

/// Torture-only knobs : larger runs + tighter checkpoint cadence + wider
/// kill-delay distribution to hit more crash windows.
const TORTURE_VECTORS_PER_RUN: u64 = 400;
const TORTURE_CHECKPOINT_EVERY: u64 = 25; // ~16 checkpoints per torture run
const TORTURE_KILL_DELAY_RANGE_MS: u64 = 3000;

/// Per-iteration summary returned to the parent for logging.
struct IterationStats {
    n_acked: usize,
    n_checkpointed: usize,
    shard_len: usize,
}

/// Single iteration of the test : spawn child, kill after `kill_delay_ms`,
/// drain stdout, reopen shard, assert every ACKED id is present (and
/// every CHECKPOINTED line corresponds to a durably-committed manifest).
///
/// `checkpoint_every` of 0 means the child runs inserts-only ; non-zero
/// triggers `Shard::checkpoint` every K successful inserts.
fn run_one_iteration(
    iter: usize,
    kill_delay_ms: u64,
    checkpoint_every: u64,
    vectors_per_run: u64,
) -> IterationStats {
    let writer_path: PathBuf = env!("CARGO_BIN_EXE_crash_writer").into();
    let dir = tempdir().expect("tempdir");
    let dir_str = dir.path().to_string_lossy().to_string();

    let mut child = Command::new(&writer_path)
        .args([
            &dir_str,
            &vectors_per_run.to_string(),
            &DIM.to_string(),
            &checkpoint_every.to_string(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn crash_writer");

    thread::sleep(Duration::from_millis(kill_delay_ms));

    // `Child::kill` sends SIGKILL on Unix. Errors on an already-dead
    // child are fine : that just means the run finished before the kill.
    let _ = child.kill();
    let _ = child.wait();

    // Drain whatever the child managed to flush before dying. The kernel
    // preserves pipe contents past SIGKILL, so we read everything that
    // made it to the pipe buffer.
    let mut stdout_buf = String::new();
    if let Some(mut s) = child.stdout.take() {
        let _ = s.read_to_string(&mut stdout_buf);
    }

    let mut acked: Vec<u64> = Vec::new();
    let mut checkpointed_lsns: Vec<u64> = Vec::new();
    for line in stdout_buf.lines() {
        if let Some(rest) = line.strip_prefix("ACKED ")
            && let Ok(id) = rest.parse::<u64>()
        {
            acked.push(id);
        } else if let Some(rest) = line.strip_prefix("CHECKPOINTED ")
            && let Ok(lsn) = rest.parse::<u64>()
        {
            checkpointed_lsns.push(lsn);
        }
    }

    // If the child saw at least one successful checkpoint, the manifest
    // MUST be present on disk : `Shard::checkpoint` only returns Ok
    // after the manifest atomic-write completes.
    if !checkpointed_lsns.is_empty() {
        let manifest_path = dir.path().join("manifest");
        assert!(
            manifest_path.exists(),
            "iter {iter} (kill_delay={kill_delay_ms}ms, checkpoint_every={checkpoint_every}): \
             child reported {} CHECKPOINTED lines but manifest is missing",
            checkpointed_lsns.len(),
        );
        // Manifest must also parse (no torn write : atomic_write
        // guarantees old-or-new, never partial).
        let m = Manifest::load(&manifest_path)
            .expect("manifest read")
            .expect("manifest present");
        // `manifest.snapshot_id` must be either `n_checkpointed`
        // (parent saw every ACK) or `n_checkpointed + 1` (a checkpoint
        // committed durably AND incremented snapshot_id, but the kill
        // landed in the small window between `checkpoint()` returning
        // Ok and the child's `writeln!("CHECKPOINTED ...") + flush`
        // delivering to the pipe). The +1 case is a real and expected
        // window : ACK is best-effort observability ; the manifest
        // atomic-write is the durable commit point.
        let n_ckpt = checkpointed_lsns.len() as u64;
        assert!(
            m.snapshot_id == n_ckpt || m.snapshot_id == n_ckpt + 1,
            "iter {iter}: manifest snapshot_id={} expected {n_ckpt} or {} \
             (kill_delay={kill_delay_ms}ms)",
            m.snapshot_id,
            n_ckpt + 1,
        );
    }

    // Reopen the shard on the same directory and verify durability.
    let shard = Shard::open(dir.path(), DIM, L2, HnswParams::default()).expect("reopen");

    for &id in &acked {
        assert!(
            shard.contains(VectorId::new(id)),
            "iter {iter} (kill_delay={kill_delay_ms}ms, checkpoint_every={checkpoint_every}): \
             acked id {id} missing after reopen ; \
             {} ACKs total, {} checkpoints, shard.len()={}",
            acked.len(),
            checkpointed_lsns.len(),
            shard.len()
        );
    }

    IterationStats {
        n_acked: acked.len(),
        n_checkpointed: checkpointed_lsns.len(),
        shard_len: shard.len(),
    }
}

/// Deterministic kill delay derived from the iteration index. Avoids a
/// `rand` dependency while still spreading delays across the full range
/// so different iterations hit different points in the child's run.
fn kill_delay_for(iter: usize) -> u64 {
    1 + ((iter as u64).wrapping_mul(173) % KILL_DELAY_RANGE_MS)
}

/// Three-bucket kill-delay schedule for the torture variants : short
/// (early-run), medium (mid-run), long (late-run). Each bucket uses a
/// different multiplier so iterations within a bucket still hit varied
/// offsets. The point is to land kills at every plausible point in the
/// child's lifecycle, including the windows around checkpoint phases :
/// vacuum, snapshot write, manifest commit, WAL truncate.
fn torture_kill_delay_for(iter: usize) -> u64 {
    let bucket = iter % 3;
    let i = iter as u64;
    match bucket {
        // 1-500 ms : hits the first 1-4 checkpoints + early inserts.
        0 => 1 + (i.wrapping_mul(73) % 500),
        // 200-1700 ms : mid-run, hits middle checkpoints.
        1 => 200 + (i.wrapping_mul(173) % 1500),
        // 800-3000 ms : late-run, hits final checkpoints + cleanup.
        _ => 800 + (i.wrapping_mul(257) % (TORTURE_KILL_DELAY_RANGE_MS - 800)),
    }
}

// -----------------------------------------------------------------------------
// Smoke variants : ~5-10s each, run by default in `cargo test`.
// -----------------------------------------------------------------------------

const SMOKE_ITERATIONS: usize = 5;

/// Insert-only smoke : original test, kept untouched for regression coverage.
#[test]
fn crash_recovery_smoke_inserts_only() {
    for iter in 0..SMOKE_ITERATIONS {
        let delay = kill_delay_for(iter);
        let stats = run_one_iteration(iter, delay, 0, VECTORS_PER_RUN);
        eprintln!(
            "smoke inserts-only iter {iter}: kill_delay={delay}ms acks={} len={}",
            stats.n_acked, stats.shard_len
        );
    }
}

/// Checkpoint-interleaved smoke : child checkpoints every 40 inserts
/// (so up to ~5 checkpoints per 200-insert run). Exercises crash windows
/// in `Shard::checkpoint` itself, on top of the insert windows.
#[test]
fn crash_recovery_smoke_with_checkpoints() {
    for iter in 0..SMOKE_ITERATIONS {
        let delay = kill_delay_for(iter);
        let stats = run_one_iteration(iter, delay, 40, VECTORS_PER_RUN);
        eprintln!(
            "smoke checkpointed iter {iter}: kill_delay={delay}ms acks={} ckpts={} len={}",
            stats.n_acked, stats.n_checkpointed, stats.shard_len
        );
    }
}

// -----------------------------------------------------------------------------
// Torture variants : 100 iterations each, takes a few minutes. Run with
//   cargo test -p kova-storage --test crash_recovery -- --ignored --nocapture
// -----------------------------------------------------------------------------

const TORTURE_ITERATIONS: usize = 150;

#[test]
#[ignore = "expensive: spawns ~150 child processes, several minutes ; run with --ignored"]
fn crash_recovery_torture_inserts_only() {
    let mut total_acks = 0;
    for iter in 0..TORTURE_ITERATIONS {
        let delay = torture_kill_delay_for(iter);
        let stats = run_one_iteration(iter, delay, 0, TORTURE_VECTORS_PER_RUN);
        total_acks += stats.n_acked;
        eprintln!(
            "torture inserts-only iter {iter:3}: kill_delay={delay}ms acks={} len={}",
            stats.n_acked, stats.shard_len
        );
    }
    eprintln!(
        "torture inserts-only complete : {TORTURE_ITERATIONS} iterations, {total_acks} acks total",
    );
}

#[test]
#[ignore = "expensive: ~150 child processes interleaving checkpoints ; ~5-10min ; run with --ignored"]
fn crash_recovery_torture_with_checkpoints() {
    let mut total_acks = 0;
    let mut total_checkpoints = 0;
    for iter in 0..TORTURE_ITERATIONS {
        let delay = torture_kill_delay_for(iter);
        let stats = run_one_iteration(
            iter,
            delay,
            TORTURE_CHECKPOINT_EVERY,
            TORTURE_VECTORS_PER_RUN,
        );
        total_acks += stats.n_acked;
        total_checkpoints += stats.n_checkpointed;
        eprintln!(
            "torture checkpointed iter {iter:3}: kill_delay={delay}ms acks={} ckpts={} len={}",
            stats.n_acked, stats.n_checkpointed, stats.shard_len
        );
    }
    eprintln!(
        "torture checkpointed complete : {TORTURE_ITERATIONS} iterations, \
         {total_acks} acks, {total_checkpoints} checkpoints",
    );
}

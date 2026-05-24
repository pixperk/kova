//! Crash recovery torture test.
//!
//! Spawns the [`crash_writer`](../bin/crash_writer.rs) child, lets it
//! insert vectors for a random delay, then `SIGKILL`s it. After the
//! kill, reopens the shard on the same directory and asserts that every
//! id the child printed `ACKED` for is durably present.
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
use kova_storage::Shard;
use tempfile::tempdir;

const VECTORS_PER_RUN: u64 = 200;
const DIM: usize = 8;
/// Maximum delay between spawn and SIGKILL. Tuned so the kill lands
/// somewhere mid-run for typical hardware ; tweak if local SSD is much
/// faster or slower.
const KILL_DELAY_RANGE_MS: u64 = 1500;

/// Single iteration of the test : spawn child, kill after `kill_delay_ms`,
/// drain stdout, reopen shard, assert every ACKED id is present.
///
/// Returns `(n_acked, shard_len)` for the parent to log.
fn run_one_iteration(iter: usize, kill_delay_ms: u64) -> (usize, usize) {
    let writer_path: PathBuf = env!("CARGO_BIN_EXE_crash_writer").into();
    let dir = tempdir().expect("tempdir");
    let dir_str = dir.path().to_string_lossy().to_string();

    let mut child = Command::new(&writer_path)
        .args([&dir_str, &VECTORS_PER_RUN.to_string(), &DIM.to_string()])
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

    let acked: Vec<u64> = stdout_buf
        .lines()
        .filter_map(|line| line.strip_prefix("ACKED "))
        .filter_map(|n| n.parse().ok())
        .collect();

    // Reopen the shard on the same directory and verify durability.
    let shard = Shard::open(dir.path(), DIM, L2, HnswParams::default()).expect("reopen");

    for &id in &acked {
        assert!(
            shard.contains(VectorId::new(id)),
            "iter {iter} (kill_delay={kill_delay_ms}ms): \
             acked id {id} missing after reopen ; \
             {} ACKs total, shard.len()={}",
            acked.len(),
            shard.len()
        );
    }

    (acked.len(), shard.len())
}

/// Deterministic kill delay derived from the iteration index. Avoids a
/// `rand` dependency while still spreading delays across the full range
/// so different iterations hit different points in the child's run.
fn kill_delay_for(iter: usize) -> u64 {
    1 + ((iter as u64).wrapping_mul(173) % KILL_DELAY_RANGE_MS)
}

// -----------------------------------------------------------------------------
// Smoke version : ~5-10s, runs by default in `cargo test`.
// Catches regressions in the kill/reopen plumbing without taking forever.
// -----------------------------------------------------------------------------

const SMOKE_ITERATIONS: usize = 5;

#[test]
fn crash_recovery_smoke() {
    for iter in 0..SMOKE_ITERATIONS {
        let delay = kill_delay_for(iter);
        let (acks, len) = run_one_iteration(iter, delay);
        eprintln!("smoke iter {iter}: kill_delay={delay}ms acks={acks} len={len}");
    }
}

// -----------------------------------------------------------------------------
// Torture version : 100 iterations, takes a few minutes. Run with
//   cargo test -p kova-storage --test crash_recovery -- --ignored --nocapture
// -----------------------------------------------------------------------------

const TORTURE_ITERATIONS: usize = 100;

#[test]
#[ignore = "expensive: spawns ~100 child processes, ~5min total ; run with --ignored"]
fn crash_recovery_torture() {
    let mut total_acks = 0;
    for iter in 0..TORTURE_ITERATIONS {
        let delay = kill_delay_for(iter);
        let (acks, len) = run_one_iteration(iter, delay);
        total_acks += acks;
        eprintln!("torture iter {iter:3}: kill_delay={delay}ms acks={acks} len={len}");
    }
    eprintln!("torture complete : {TORTURE_ITERATIONS} iterations, {total_acks} acks total");
}

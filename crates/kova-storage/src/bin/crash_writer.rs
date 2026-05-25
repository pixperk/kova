//! Child process for the crash recovery test.
//!
//! Reads `(dir, n, dim, checkpoint_every)` from argv, opens a [`Shard`],
//! inserts `n` deterministic vectors, and prints `ACKED <id>\n` (flushed)
//! after each successful [`Shard::insert`]. If `checkpoint_every > 0`,
//! also calls [`Shard::checkpoint`] every `checkpoint_every` successful
//! inserts and prints `CHECKPOINTED <lsn>\n` after each successful
//! checkpoint.
//!
//! Designed to be SIGKILL'd at random offsets by the parent test in
//! `tests/crash_recovery.rs`.
//!
//! # Invariants
//!
//! - Every line `ACKED <id>` printed to stdout means the corresponding
//!   insert returned `Ok(())`, which means [`Wal::sync`] completed before
//!   the line was printed.
//! - Every line `CHECKPOINTED <lsn>` means
//!   [`Shard::checkpoint`] returned `Ok(lsn)` for that LSN, which means
//!   the manifest was atomic-written and is durably on disk.
//!
//! The parent test asserts that all `ACK`ed ids are durably present
//! after reopening the shard on the same directory, regardless of when
//! the kill landed (including mid-checkpoint windows).

use std::env;
use std::io::Write;
use std::path::PathBuf;
use std::process;

use kova_core::{L2, Metadata, Vector, VectorId};
use kova_index::HnswParams;
use kova_storage::Shard;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: crash_writer <dir> <n> <dim> [checkpoint_every]");
        process::exit(2);
    }
    let dir = PathBuf::from(&args[1]);
    let n: u64 = args[2].parse().expect("n must be u64");
    let dim: usize = args[3].parse().expect("dim must be usize");
    // 0 = no checkpoints ; matches the original crash_writer behaviour
    // so existing tests that pass only 3 args keep working unchanged.
    let checkpoint_every: u64 = args
        .get(4)
        .map_or(0, |s| s.parse().expect("checkpoint_every must be u64"));

    let mut shard = Shard::open(&dir, dim, L2, HnswParams::default()).expect("Shard::open");

    // Lock stdout once : we want every line flushed before the next op
    // so the ACK ordering matches the WAL ordering.
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();

    for i in 0..n {
        let vec = make_vector(i, dim);
        shard
            .insert(VectorId::new(i), vec, Metadata::new())
            .expect("Shard::insert");

        // After insert returns Ok, wal.sync() has completed : the record
        // is durable. Flush the ACK so the parent sees it even if we get
        // SIGKILL'd immediately after.
        writeln!(handle, "ACKED {i}").expect("stdout write");
        handle.flush().expect("stdout flush");

        // Periodic checkpoint. Triggered AFTER an insert (so the ACK
        // for that insert lands first), based on the 1-indexed count of
        // completed inserts.
        if checkpoint_every > 0 && (i + 1) % checkpoint_every == 0 {
            let lsn = shard.checkpoint().expect("Shard::checkpoint");
            writeln!(handle, "CHECKPOINTED {}", lsn.get()).expect("stdout write");
            handle.flush().expect("stdout flush");
        }
    }
}

/// Deterministic vector from a seed. Same `(seed, dim)` always produces
/// the same vector so the parent could (optionally) verify byte-for-byte
/// roundtrip ; we only check presence today, but the determinism makes
/// future value-equality checks free.
fn make_vector(seed: u64, dim: usize) -> Vector {
    let data: Vec<f32> = (0..dim)
        .map(|j| {
            // `raw` is bounded to 0..1000 - fits in u16, and u16 -> f32 is
            // exact (16-bit value, f32 mantissa is 23 bits).
            let raw = (seed.wrapping_mul(31).wrapping_add(j as u64) % 1000) as u16;
            f32::from(raw) / 1000.0
        })
        .collect();
    Vector::try_new(data).expect("non-empty, non-NaN vector")
}

// Benchmark harness : cast-heavy by nature.
#![allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]

//! Measure `FileMetadataStore`'s write amplification.
//!
//! `put` used to serialise the whole map and rewrite the whole file
//! through `atomic_write` on every call. The interesting part of the
//! measurement is that per-put latency came out **flat** at ~7.9 ms
//! rather than growing with the store : the two fsyncs inside
//! `atomic_write` (tmp file, then parent directory) dominate
//! serialisation entirely, so it was a hard ~125 writes/sec ceiling at
//! any size. The O(rows) part showed up as bytes instead : 852 MB
//! pushed to disk to store 436 KB.
//!
//! Writes are now in-memory and the file is written by
//! `MetadataStore::flush` at checkpoint, so this should report ~1 us
//! per put and no bytes written. Durability is unaffected : `Shard`
//! fsyncs a WAL record before touching the store.
//!
//! Run with:
//!   `cargo run --release -p kova-storage --example metadata_amplification`

use std::time::Instant;

use kova_core::{Metadata, MetadataStore, Value, VectorId};
use kova_storage::FileMetadataStore;

fn bag(i: usize) -> Metadata {
    let mut m = Metadata::new();
    m.insert("category".into(), Value::String(format!("cat-{}", i % 8)));
    m.insert("year".into(), Value::I64(2000 + (i % 25) as i64));
    m.insert("note".into(), Value::String(format!("row number {i}")));
    m
}

fn main() {
    println!("Singleton `put` into a growing store\n");
    println!("     rows      total       per put     file size   bytes written");
    let mut prev_per_put = 0.0f64;
    for &n in &[500usize, 1_000, 2_000, 4_000] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metadata.bin");
        let mut store = FileMetadataStore::open(&path).unwrap();

        let t0 = Instant::now();
        for i in 0..n {
            store.put(VectorId::new(i as u64), bag(i)).unwrap();
        }
        let elapsed = t0.elapsed();

        let final_size = std::fs::metadata(&path).unwrap().len();
        // Each put rewrites the file at its then-current size; summing
        // gives roughly n/2 * final_size of bytes pushed to disk.
        let written = (final_size as f64) * (n as f64) / 2.0;
        let per_put = elapsed.as_secs_f64() * 1e6 / n as f64;
        let growth = if prev_per_put > 0.0 {
            format!("   ({:.1}x per-put vs previous)", per_put / prev_per_put)
        } else {
            String::new()
        };
        prev_per_put = per_put;

        println!(
            "  {n:7}   {:7.0} ms   {per_put:7.0} us   {:7.0} KB   {:8.1} MB{growth}",
            elapsed.as_secs_f64() * 1e3,
            final_size as f64 / 1024.0,
            written / 1_048_576.0,
        );
    }

    println!("\nOne single-row update against an already-loaded store\n");
    for &n in &[1_000usize, 10_000] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metadata.bin");
        let mut store = FileMetadataStore::open(&path).unwrap();
        let batch: Vec<_> = (0..n).map(|i| (VectorId::new(i as u64), bag(i))).collect();
        store.put_many(batch).unwrap();
        let size = std::fs::metadata(&path).unwrap().len();

        let t0 = Instant::now();
        store.put(VectorId::new(0), bag(999_999)).unwrap();
        let one = t0.elapsed();
        println!(
            "  {n:6} rows : updating ONE row took {:6.0} us and rewrote {:5.0} KB",
            one.as_secs_f64() * 1e6,
            size as f64 / 1024.0
        );
    }
}

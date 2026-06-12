//! Persistence tests for the column stats catalog.
//!
//! Mirrors the catalog persistence test surface : verifies that
//! stats refresh at checkpoint, survive a close/reopen, get
//! orphan-cleaned on reopen, and reflect post-checkpoint mutations
//! after a fresh checkpoint.

use kova_core::{L2, Metadata, Value, Vector, VectorId};
use kova_index::HnswParams;
use kova_meta_index::IndexAtom;
use kova_storage::Shard;
use tempfile::tempdir;

fn v(data: Vec<f32>) -> Vector {
    Vector::try_new(data).unwrap()
}

fn id(n: u64) -> VectorId {
    VectorId::new(n)
}

fn s(x: &str) -> Value {
    Value::String(x.into())
}

fn i(n: i64) -> Value {
    Value::I64(n)
}

fn meta(pairs: &[(&str, Value)]) -> Metadata {
    let mut m = Metadata::new();
    for (k, val) in pairs {
        m.insert((*k).to_string(), val.clone());
    }
    m
}

#[test]
fn stats_empty_before_checkpoint() {
    let dir = tempdir().unwrap();
    let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
    for n in 0u64..10 {
        shard
            .insert(
                id(n),
                v(vec![f32::from(u16::try_from(n).unwrap()), 0.0]),
                meta(&[("category", s("docs"))]),
            )
            .unwrap();
    }
    // Stats only refresh at checkpoint, so they're empty here.
    assert!(shard.stats().is_empty());
}

#[test]
fn stats_populated_at_checkpoint() {
    let dir = tempdir().unwrap();
    let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
    for n in 0u64..10 {
        let cat = if n % 2 == 0 { "docs" } else { "blog" };
        shard
            .insert(
                id(n),
                v(vec![f32::from(u16::try_from(n).unwrap()), 0.0]),
                meta(&[("category", s(cat))]),
            )
            .unwrap();
    }
    shard.checkpoint().unwrap();
    // After checkpoint, the in-memory stats reflect the rows.
    let sel = shard
        .stats()
        .selectivity("category", &IndexAtom::Eq(s("docs")))
        .unwrap();
    // 5 of 10 rows are docs : ~0.5 selectivity.
    assert!((sel - 0.5).abs() < 0.05, "got {sel}");
}

#[test]
fn stats_survive_close_reopen_after_checkpoint() {
    let dir = tempdir().unwrap();

    {
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        for n in 0u64..10 {
            shard
                .insert(
                    id(n),
                    v(vec![f32::from(u16::try_from(n).unwrap()), 0.0]),
                    meta(&[("year", i(2020 + i64::try_from(n).unwrap()))]),
                )
                .unwrap();
        }
        shard.checkpoint().unwrap();
    }

    // Reopen : stats catalog hydrates from disk.
    let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
    let st = shard.stats().get("year").expect("stats persisted");
    assert_eq!(st.row_count, 10);
    assert_eq!(st.null_count, 0);
}

#[test]
fn stats_dropped_without_checkpoint_dont_persist() {
    let dir = tempdir().unwrap();

    {
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        for n in 0u64..5 {
            shard
                .insert(
                    id(n),
                    v(vec![f32::from(u16::try_from(n).unwrap()), 0.0]),
                    meta(&[("year", i(2020))]),
                )
                .unwrap();
        }
        // NO checkpoint here.
    }

    let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
    // Without a checkpoint, no stats were ever persisted. The
    // reopened shard has empty stats.
    assert!(shard.stats().is_empty());
}

#[test]
fn stats_refresh_on_subsequent_checkpoint() {
    let dir = tempdir().unwrap();
    let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();

    // Round 1 : 4 docs.
    for n in 0u64..4 {
        shard
            .insert(
                id(n),
                v(vec![f32::from(u16::try_from(n).unwrap()), 0.0]),
                meta(&[("category", s("docs"))]),
            )
            .unwrap();
    }
    shard.checkpoint().unwrap();
    let st1 = shard.stats().get("category").unwrap().clone();
    assert_eq!(st1.row_count, 4);

    // Round 2 : add 6 blogs. Until next checkpoint, stats are stale.
    for n in 4u64..10 {
        shard
            .insert(
                id(n),
                v(vec![f32::from(u16::try_from(n).unwrap()), 0.0]),
                meta(&[("category", s("blog"))]),
            )
            .unwrap();
    }
    // Pre-checkpoint : still seeing only round 1's rows.
    assert_eq!(shard.stats().get("category").unwrap().row_count, 4);

    shard.checkpoint().unwrap();
    // Post-checkpoint : refreshed to 10 rows.
    assert_eq!(shard.stats().get("category").unwrap().row_count, 10);
}

#[test]
fn orphan_stats_files_swept_on_open() {
    let dir = tempdir().unwrap();

    {
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        shard
            .insert(id(0), v(vec![1.0, 0.0]), meta(&[("x", i(1))]))
            .unwrap();
        shard.checkpoint().unwrap();
    }

    // Plant a fake orphan stats file from a "previous" generation.
    let orphan = dir.path().join("stats.99.bin");
    std::fs::write(&orphan, b"garbage but plausible filename").unwrap();
    assert!(orphan.exists());

    let _shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
    assert!(!orphan.exists(), "orphan stats file should be cleaned up");
    // The live stats file (generation 1) survives.
    assert!(dir.path().join("stats.1.bin").exists());
}

#[test]
fn stats_tracks_multiple_fields_with_different_kinds() {
    let dir = tempdir().unwrap();
    let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
    for n in 0u64..10 {
        shard
            .insert(
                id(n),
                v(vec![f32::from(u16::try_from(n).unwrap()), 0.0]),
                meta(&[
                    ("category", s(if n % 2 == 0 { "docs" } else { "blog" })),
                    ("year", i(2020 + i64::try_from(n).unwrap())),
                    ("active", Value::Bool(n % 3 == 0)),
                ]),
            )
            .unwrap();
    }
    shard.checkpoint().unwrap();

    let stats = shard.stats();
    assert!(stats.get("category").is_some());
    assert!(stats.get("year").is_some());
    assert!(stats.get("active").is_some());

    // Spot-check one selectivity to make sure the round-trip
    // through the on-disk format preserves the math.
    let docs = stats
        .selectivity("category", &IndexAtom::Eq(s("docs")))
        .unwrap();
    assert!((docs - 0.5).abs() < 0.05);
}

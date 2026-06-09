//! Persistence tests for the secondary-index catalog.
//!
//! Verifies that indexes registered on a file-backed shard survive
//! a close/reopen cycle when a checkpoint runs in between (the
//! durability contract documented on `Shard::add_*_index`). Also
//! checks the post-checkpoint catch-up : records appended after
//! checkpoint are replayed into the loaded catalog on the next open.

use kova_core::{L2, Metadata, Value, Vector, VectorId};
use kova_index::HnswParams;
use kova_meta_index::{CmpOp, IndexAtom};
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

fn arr(xs: &[&str]) -> Value {
    Value::Array(xs.iter().map(|x| s(x)).collect())
}

fn meta(pairs: &[(&str, Value)]) -> Metadata {
    let mut m = Metadata::new();
    for (k, val) in pairs {
        m.insert((*k).to_string(), val.clone());
    }
    m
}

#[test]
fn catalog_survives_close_after_checkpoint() {
    let dir = tempdir().unwrap();

    {
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        shard.add_hash_index("category");

        for n in 0u64..10 {
            let category = if n % 2 == 0 { "docs" } else { "blog" };
            shard
                .insert(
                    id(n),
                    v(vec![f32::from(u16::try_from(n).unwrap()), 0.0]),
                    meta(&[("category", s(category))]),
                )
                .unwrap();
        }
        shard.checkpoint().unwrap();
    }

    // Reopen ; catalog should be hydrated from the catalog.{N}.bin file.
    let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
    let docs = shard
        .catalog()
        .lookup("category", &IndexAtom::Eq(s("docs")))
        .expect("catalog was restored from disk");
    assert_eq!(docs.len(), 5);
    for n in [0, 2, 4, 6, 8] {
        assert!(docs.contains(n));
    }
}

#[test]
fn post_checkpoint_records_replay_into_loaded_catalog() {
    // Insert some, checkpoint, then insert more. On reopen, the
    // first set lives in the catalog snapshot ; the second set is
    // replayed from the WAL into the loaded catalog.
    let dir = tempdir().unwrap();

    {
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        shard.add_hash_index("category");

        for n in 0u64..5 {
            shard
                .insert(
                    id(n),
                    v(vec![f32::from(u16::try_from(n).unwrap()), 0.0]),
                    meta(&[("category", s("docs"))]),
                )
                .unwrap();
        }
        shard.checkpoint().unwrap();
        for n in 5u64..10 {
            shard
                .insert(
                    id(n),
                    v(vec![f32::from(u16::try_from(n).unwrap()), 0.0]),
                    meta(&[("category", s("docs"))]),
                )
                .unwrap();
        }
    }

    let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
    let docs = shard
        .catalog()
        .lookup("category", &IndexAtom::Eq(s("docs")))
        .unwrap();
    assert_eq!(docs.len(), 10);
    for n in 0..10 {
        assert!(docs.contains(n), "id {n} missing after replay");
    }
}

#[test]
fn indexes_without_checkpoint_dont_persist() {
    // Documented contract : indexes added after the last checkpoint
    // are transient. Reopen loads the previous (empty) catalog.
    let dir = tempdir().unwrap();

    {
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        shard
            .insert(id(0), v(vec![1.0, 0.0]), meta(&[("category", s("docs"))]))
            .unwrap();
        shard.add_hash_index("category");
        // NO checkpoint here.
    }

    let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
    assert!(
        shard
            .catalog()
            .lookup("category", &IndexAtom::Eq(s("docs")))
            .is_none(),
        "index registered without checkpoint should not persist"
    );

    // Row itself does survive : the WAL/replay path is independent.
    assert!(shard.contains(id(0)));
}

#[test]
fn all_three_index_types_persist_together() {
    let dir = tempdir().unwrap();

    {
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        shard.add_hash_index("category");
        shard.add_btree_index("year");
        shard.add_inverted_index("tags");

        for n in 0u64..6 {
            let category = if n % 2 == 0 { "docs" } else { "blog" };
            let year = 2020 + i64::try_from(n).unwrap();
            let tags = if n % 3 == 0 {
                arr(&["rust", "async"])
            } else {
                arr(&["go"])
            };
            shard
                .insert(
                    id(n),
                    v(vec![f32::from(u16::try_from(n).unwrap()), 0.0]),
                    meta(&[("category", s(category)), ("year", i(year)), ("tags", tags)]),
                )
                .unwrap();
        }
        shard.checkpoint().unwrap();
    }

    let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();

    let docs = shard
        .catalog()
        .lookup("category", &IndexAtom::Eq(s("docs")))
        .unwrap();
    assert_eq!(docs.len(), 3);

    let recent = shard
        .catalog()
        .lookup("year", &IndexAtom::Cmp(CmpOp::Gt, i(2022)))
        .unwrap();
    assert_eq!(recent.len(), 3);

    let rust = shard
        .catalog()
        .lookup("tags", &IndexAtom::ArrayContains(s("rust")))
        .unwrap();
    assert_eq!(rust.len(), 2);
}

#[test]
fn delete_after_checkpoint_replays_into_catalog() {
    let dir = tempdir().unwrap();

    {
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        shard.add_hash_index("category");
        for n in 0u64..5 {
            shard
                .insert(
                    id(n),
                    v(vec![f32::from(u16::try_from(n).unwrap()), 0.0]),
                    meta(&[("category", s("docs"))]),
                )
                .unwrap();
        }
        shard.checkpoint().unwrap();
        // Delete two rows post-checkpoint ; they live as Delete records in the WAL.
        shard.delete(id(1)).unwrap();
        shard.delete(id(3)).unwrap();
    }

    let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
    let docs = shard
        .catalog()
        .lookup("category", &IndexAtom::Eq(s("docs")))
        .unwrap();
    assert_eq!(docs.len(), 3);
    for n in [0, 2, 4] {
        assert!(docs.contains(n));
    }
    for n in [1, 3] {
        assert!(!docs.contains(n));
    }
}

#[test]
fn update_after_checkpoint_replays_into_catalog() {
    let dir = tempdir().unwrap();

    {
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        shard.add_hash_index("category");
        shard
            .insert(id(0), v(vec![1.0, 0.0]), meta(&[("category", s("docs"))]))
            .unwrap();
        shard.checkpoint().unwrap();
        shard
            .update_metadata([(id(0), meta(&[("category", s("blog"))]))])
            .unwrap();
    }

    let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
    assert!(
        shard
            .catalog()
            .lookup("category", &IndexAtom::Eq(s("docs")))
            .unwrap()
            .is_empty()
    );
    let blog = shard
        .catalog()
        .lookup("category", &IndexAtom::Eq(s("blog")))
        .unwrap();
    assert_eq!(blog.len(), 1);
    assert!(blog.contains(0));
}

#[test]
fn second_checkpoint_supersedes_first_catalog_file() {
    let dir = tempdir().unwrap();
    {
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        shard.add_hash_index("category");
        shard
            .insert(id(0), v(vec![1.0, 0.0]), meta(&[("category", s("docs"))]))
            .unwrap();
        shard.checkpoint().unwrap();
        assert!(dir.path().join("catalog.1.bin").exists());

        shard
            .insert(id(1), v(vec![0.0, 1.0]), meta(&[("category", s("blog"))]))
            .unwrap();
        shard.checkpoint().unwrap();
        assert!(dir.path().join("catalog.2.bin").exists());
        assert!(
            !dir.path().join("catalog.1.bin").exists(),
            "old catalog file should be deleted"
        );
    }

    let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
    let docs = shard
        .catalog()
        .lookup("category", &IndexAtom::Eq(s("docs")))
        .unwrap();
    assert_eq!(docs.len(), 1);
    assert!(docs.contains(0));
}

#[test]
fn orphan_catalog_files_swept_on_open() {
    let dir = tempdir().unwrap();

    // First produce a manifest + catalog at gen 1.
    {
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        shard.add_hash_index("category");
        shard
            .insert(id(0), v(vec![1.0, 0.0]), meta(&[("category", s("docs"))]))
            .unwrap();
        shard.checkpoint().unwrap();
    }
    assert!(dir.path().join("catalog.1.bin").exists());

    // Drop a fake orphan catalog from a hypothetical previous run.
    let orphan = dir.path().join("catalog.99.bin");
    std::fs::write(&orphan, b"garbage that looks like a catalog").unwrap();
    assert!(orphan.exists());

    let _shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
    assert!(!orphan.exists(), "orphan catalog should be swept on open");
    assert!(
        dir.path().join("catalog.1.bin").exists(),
        "live catalog must survive cleanup"
    );
}

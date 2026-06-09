//! End-to-end tests for the `Shard <-> IndexCatalog` wiring.
//!
//! Exercises the public surface added on top of slice 4 :
//! `add_hash_index`, `add_btree_index`, `add_inverted_index`,
//! and `catalog()` accessor, plus the synchronous index
//! maintenance through the five mutation paths
//! (`insert`, `insert_many`, `delete`, `delete_many`,
//! `update_metadata`).

use kova_core::{
    InMemoryMetadataStore, InMemoryVectorStore, L2, Metadata, Value, Vector, VectorId,
};
use kova_index::HnswParams;
use kova_meta_index::{CmpOp, IndexAtom};
use kova_storage::{InMemoryWal, Shard};

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

fn fresh_shard() -> Shard<L2, InMemoryVectorStore, InMemoryMetadataStore, InMemoryWal> {
    Shard::from_parts(
        L2,
        HnswParams::default(),
        InMemoryVectorStore::new(),
        InMemoryMetadataStore::new(),
        InMemoryWal::new(),
    )
    .unwrap()
}

#[test]
fn add_hash_index_then_insert_routes_through_catalog() {
    let mut shard = fresh_shard();
    shard.add_hash_index("category");

    shard
        .insert(id(0), v(vec![1.0, 0.0]), meta(&[("category", s("docs"))]))
        .unwrap();
    shard
        .insert(id(1), v(vec![0.0, 1.0]), meta(&[("category", s("blog"))]))
        .unwrap();
    shard
        .insert(id(2), v(vec![1.0, 1.0]), meta(&[("category", s("docs"))]))
        .unwrap();

    let docs = shard
        .catalog()
        .lookup("category", &IndexAtom::Eq(s("docs")))
        .expect("hash index handles Eq");
    assert_eq!(docs.len(), 2);
    assert!(docs.contains(0));
    assert!(docs.contains(2));
}

#[test]
fn insert_before_add_index_then_backfill() {
    // Inserts happen first ; index is added later. The backfill walks
    // the metadata store and populates the index.
    let mut shard = fresh_shard();

    shard
        .insert(id(0), v(vec![1.0, 0.0]), meta(&[("category", s("docs"))]))
        .unwrap();
    shard
        .insert(id(1), v(vec![0.0, 1.0]), meta(&[("category", s("blog"))]))
        .unwrap();

    shard.add_hash_index("category");

    let docs = shard
        .catalog()
        .lookup("category", &IndexAtom::Eq(s("docs")))
        .unwrap();
    assert_eq!(docs.len(), 1);
    assert!(docs.contains(0));
}

#[test]
fn insert_many_populates_all_indexes() {
    let mut shard = fresh_shard();
    shard.add_hash_index("category");
    shard.add_btree_index("year");
    shard.add_inverted_index("tags");

    let batch: Vec<_> = (0u64..6)
        .map(|n| {
            let category = if n % 2 == 0 { "docs" } else { "blog" };
            let year = 2020 + i64::try_from(n).unwrap();
            let tags = if n % 3 == 0 {
                arr(&["rust", "async"])
            } else {
                arr(&["go"])
            };
            (
                id(n),
                v(vec![f32::from(u16::try_from(n).unwrap()), 0.0]),
                meta(&[("category", s(category)), ("year", i(year)), ("tags", tags)]),
            )
        })
        .collect();

    shard.insert_many(batch).unwrap();

    let docs = shard
        .catalog()
        .lookup("category", &IndexAtom::Eq(s("docs")))
        .unwrap();
    assert_eq!(docs.len(), 3);
    for n in [0, 2, 4] {
        assert!(docs.contains(n));
    }

    let recent = shard
        .catalog()
        .lookup("year", &IndexAtom::Cmp(CmpOp::Gt, i(2022)))
        .unwrap();
    assert_eq!(recent.len(), 3);
    for n in [3, 4, 5] {
        assert!(recent.contains(n));
    }

    let rust = shard
        .catalog()
        .lookup("tags", &IndexAtom::ArrayContains(s("rust")))
        .unwrap();
    assert_eq!(rust.len(), 2);
    for n in [0, 3] {
        assert!(rust.contains(n));
    }
}

#[test]
fn delete_removes_id_from_catalog() {
    let mut shard = fresh_shard();
    shard.add_hash_index("category");

    shard
        .insert(id(0), v(vec![1.0, 0.0]), meta(&[("category", s("docs"))]))
        .unwrap();
    shard
        .insert(id(1), v(vec![0.0, 1.0]), meta(&[("category", s("docs"))]))
        .unwrap();

    shard.delete(id(0)).unwrap();

    let docs = shard
        .catalog()
        .lookup("category", &IndexAtom::Eq(s("docs")))
        .unwrap();
    assert_eq!(docs.len(), 1);
    assert!(docs.contains(1));
    assert!(!docs.contains(0));
}

#[test]
fn delete_many_removes_all_from_catalog() {
    let mut shard = fresh_shard();
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

    shard.delete_many([id(1), id(3)]).unwrap();

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
fn update_metadata_moves_id_between_buckets() {
    let mut shard = fresh_shard();
    shard.add_hash_index("category");

    shard
        .insert(id(0), v(vec![1.0, 0.0]), meta(&[("category", s("docs"))]))
        .unwrap();

    shard
        .update_metadata([(id(0), meta(&[("category", s("blog"))]))])
        .unwrap();

    let docs = shard
        .catalog()
        .lookup("category", &IndexAtom::Eq(s("docs")))
        .unwrap();
    let blog = shard
        .catalog()
        .lookup("category", &IndexAtom::Eq(s("blog")))
        .unwrap();
    assert!(docs.is_empty());
    assert_eq!(blog.len(), 1);
    assert!(blog.contains(0));
}

#[test]
fn update_dropping_indexed_field_removes_from_index() {
    // The new bag doesn't have the indexed field at all : the row
    // disappears from the bucket but survives in the shard.
    let mut shard = fresh_shard();
    shard.add_hash_index("category");

    shard
        .insert(id(0), v(vec![1.0, 0.0]), meta(&[("category", s("docs"))]))
        .unwrap();
    shard
        .update_metadata([(id(0), meta(&[("other", s("x"))]))])
        .unwrap();

    let docs = shard
        .catalog()
        .lookup("category", &IndexAtom::Eq(s("docs")))
        .unwrap();
    assert!(docs.is_empty());
    let live = shard
        .catalog()
        .lookup("category", &IndexAtom::IsNotNull)
        .unwrap();
    assert!(live.is_empty());

    // Row still exists in the shard ; only the catalog forgot it.
    assert!(shard.contains(id(0)));
}

#[test]
fn update_adding_indexed_field_inserts_into_index() {
    let mut shard = fresh_shard();
    shard.add_hash_index("category");

    shard
        .insert(id(0), v(vec![1.0, 0.0]), meta(&[("other", s("x"))]))
        .unwrap();
    shard
        .update_metadata([(id(0), meta(&[("category", s("docs"))]))])
        .unwrap();

    let docs = shard
        .catalog()
        .lookup("category", &IndexAtom::Eq(s("docs")))
        .unwrap();
    assert_eq!(docs.len(), 1);
    assert!(docs.contains(0));
}

#[test]
fn multi_field_composition_works_end_to_end() {
    // The whole point of having three index types : combine them
    // through bitmap operations to filter by AND across fields.
    //
    // Data design (chosen so the AND is non-empty) :
    // - category = "docs" iff n < 10
    // - year = 2024 iff n % 4 == 0, else 2020
    // - tags @> 'rust' iff n < 5
    //
    // WHERE category = 'docs' AND year > 2022 AND tags @> 'rust'
    //   = (n < 10) AND (n % 4 == 0) AND (n < 5)
    //   = n in {0, 4}
    let mut shard = fresh_shard();
    shard.add_hash_index("category");
    shard.add_btree_index("year");
    shard.add_inverted_index("tags");

    let batch: Vec<_> = (0u64..20)
        .map(|n| {
            let category = if n < 10 { "docs" } else { "blog" };
            let year = if n % 4 == 0 { 2024 } else { 2020 };
            let tags = if n < 5 {
                arr(&["rust", "async"])
            } else {
                arr(&["go"])
            };
            (
                id(n),
                v(vec![f32::from(u16::try_from(n).unwrap()), 0.0]),
                meta(&[("category", s(category)), ("year", i(year)), ("tags", tags)]),
            )
        })
        .collect();
    shard.insert_many(batch).unwrap();

    let docs = shard
        .catalog()
        .lookup("category", &IndexAtom::Eq(s("docs")))
        .unwrap();
    let recent = shard
        .catalog()
        .lookup("year", &IndexAtom::Cmp(CmpOp::Gt, i(2022)))
        .unwrap();
    let rust = shard
        .catalog()
        .lookup("tags", &IndexAtom::ArrayContains(s("rust")))
        .unwrap();
    let candidates = docs & recent & rust;

    // The intersection is exactly {0, 4} by construction.
    assert_eq!(candidates.len(), 2);
    assert!(candidates.contains(0));
    assert!(candidates.contains(4));
}

#[test]
fn lookup_returns_none_for_unindexed_field() {
    let mut shard = fresh_shard();
    shard
        .insert(id(0), v(vec![1.0, 0.0]), meta(&[("category", s("docs"))]))
        .unwrap();
    // No index registered : catalog has nothing to say.
    assert!(
        shard
            .catalog()
            .lookup("category", &IndexAtom::Eq(s("docs")))
            .is_none()
    );
}

#[test]
fn estimate_matches_lookup_len() {
    let mut shard = fresh_shard();
    shard.add_hash_index("category");
    for n in 0u64..40 {
        let v_ = if n % 5 == 0 { s("hot") } else { s("cold") };
        shard
            .insert(
                id(n),
                v(vec![f32::from(u16::try_from(n).unwrap()), 0.0]),
                meta(&[("category", v_)]),
            )
            .unwrap();
    }

    for atom in [
        IndexAtom::Eq(s("hot")),
        IndexAtom::IsNotNull,
        IndexAtom::Cmp(CmpOp::Ne, s("hot")),
    ] {
        let q = shard.catalog().lookup("category", &atom).unwrap();
        let e = shard.catalog().estimate("category", &atom).unwrap();
        assert_eq!(q.len(), e, "atom = {atom:?}");
    }
}

#[test]
fn missing_field_in_row_doesnt_affect_other_rows() {
    let mut shard = fresh_shard();
    shard.add_hash_index("category");

    shard
        .insert(id(0), v(vec![1.0, 0.0]), meta(&[("category", s("docs"))]))
        .unwrap();
    // row 1 has no "category" field
    shard
        .insert(id(1), v(vec![0.0, 1.0]), meta(&[("other", s("x"))]))
        .unwrap();
    shard
        .insert(id(2), v(vec![1.0, 1.0]), meta(&[("category", s("docs"))]))
        .unwrap();

    let docs = shard
        .catalog()
        .lookup("category", &IndexAtom::Eq(s("docs")))
        .unwrap();
    assert_eq!(docs.len(), 2);
    assert!(docs.contains(0));
    assert!(docs.contains(2));
    assert!(!docs.contains(1));
}

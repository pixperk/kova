//! Probabilistic-grammar query fuzzer for KQL.
//!
//! Drives the full parse / bind / plan / execute pipeline against a
//! random shard with random queries, asserting the pipeline either
//! succeeds or returns a typed [`KovaQueryError`]. Panics are
//! always failures.
//!
//! The fuzzer is deterministic given a seed. A failing iteration
//! prints the seed + query verbatim ; rerunning with the same seed
//! reproduces the bug.
//!
//! Scope (v1) :
//!
//! - Statement coverage : SELECT (kNN / scan-and-limit / COUNT /
//!   radius), DELETE (by-id, by-param, by-predicate, by-radius),
//!   UPDATE (by-id, by-predicate, by-radius, subscripted), VACUUM,
//!   CHECKPOINT.
//! - Predicate shapes : every atom kind, AND/OR/NOT combinators up
//!   to depth 3, subscripted field references on a designated
//!   `attrs` Map field.
//! - Value space : small ints, small floats, short strings, booleans,
//!   short arrays, nested Maps.
//!
//! Out of scope : INSERT-through-Engine (the shard fixture inserts
//! directly), index DDL (not in Phase 1), batch-param resolution
//! (one more layer of indirection, low marginal coverage).

use std::collections::HashMap;

use kova_core::{L2, Metadata, Value, Vector, VectorId};
use kova_index::HnswParams;
use kova_query::{Engine, KovaQueryError, ParamBindings, ParamValue};
use kova_storage::Shard;
use rand::seq::IndexedRandom;
use rand::{RngExt, SeedableRng, rngs::StdRng};

// =========================================================================
// Universes
// =========================================================================

/// Top-level metadata field names the generators draw from. Kept
/// small so generated predicates have a real chance of matching
/// generated rows.
const FIELDS: &[&str] = &["category", "score", "year", "active", "tags", "attrs"];

/// Field used as the nested-Map carrier for subscripted predicates.
/// The fixture always writes a Map at this field so subscript-side
/// fuzzing has actual data to hit.
const MAP_FIELD: &str = "attrs";

/// Subscript keys generated under [`MAP_FIELD`].
const SUB_KEYS: &[&str] = &["country", "priority", "color"];

const CATEGORIES: &[&str] = &["docs", "specs", "rfcs", "drafts", "archived"];
const COUNTRIES: &[&str] = &["IN", "US", "JP", "BR"];
const COLORS: &[&str] = &["red", "green", "blue"];

/// Vector dim used across the fixture. Small enough to keep
/// iterations cheap, large enough to exercise SIMD paths in the
/// distance metric.
const VECTOR_DIM: usize = 4;

// =========================================================================
// Random shard fixture
// =========================================================================

/// Fixture wrapping a fresh shard pre-seeded with `n` random rows.
/// Holds a snapshot of the seeded `(id, metadata)` pairs so tests
/// can validate against a reference without re-reading the shard.
struct Fixture {
    engine: Engine<L2>,
    rows: Vec<(VectorId, Metadata)>,
    _dir: tempfile::TempDir,
}

fn build_fixture(rng: &mut StdRng, n: usize) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let shard =
        Shard::open(dir.path(), VECTOR_DIM, L2, HnswParams::default()).expect("Shard::open");
    let mut engine = Engine::new(shard, "vectors");

    let mut rows = Vec::with_capacity(n);
    for i in 1..=(n as u64) {
        let id = VectorId::new(i);
        let vec = random_vector(rng);
        let meta = random_metadata(rng);
        engine
            .shard_mut()
            .insert(id, vec, meta.clone())
            .expect("seed insert");
        rows.push((id, meta));
    }
    Fixture {
        engine,
        rows,
        _dir: dir,
    }
}

fn random_vector(rng: &mut StdRng) -> Vector {
    let data: Vec<f32> = (0..VECTOR_DIM).map(|_| rng.random::<f32>()).collect();
    Vector::try_new(data).expect("non-empty vector")
}

fn random_metadata(rng: &mut StdRng) -> Metadata {
    let mut m = Metadata::new();
    // Always write `attrs` as a Map so subscripted predicates can hit.
    let mut attrs = HashMap::new();
    if rng.random_bool(0.8) {
        attrs.insert(
            "country".into(),
            Value::String((*COUNTRIES.choose(rng).expect("non-empty")).into()),
        );
    }
    if rng.random_bool(0.5) {
        attrs.insert(
            "color".into(),
            Value::String((*COLORS.choose(rng).expect("non-empty")).into()),
        );
    }
    if rng.random_bool(0.5) {
        attrs.insert("priority".into(), Value::I64(rng.random_range(1..=5)));
    }
    m.insert(MAP_FIELD.into(), Value::Map(attrs));

    // Other fields written with biased probabilities so predicates
    // on absent fields still get exercised.
    if rng.random_bool(0.9) {
        m.insert(
            "category".into(),
            Value::String((*CATEGORIES.choose(rng).expect("non-empty")).into()),
        );
    }
    if rng.random_bool(0.7) {
        m.insert("score".into(), Value::F64(rng.random_range(0.0..1.0)));
    }
    if rng.random_bool(0.8) {
        m.insert("year".into(), Value::I64(rng.random_range(2000..=2025)));
    }
    if rng.random_bool(0.6) {
        m.insert("active".into(), Value::Bool(rng.random_bool(0.5)));
    }
    if rng.random_bool(0.5) {
        let tag_count = rng.random_range(1..=3);
        let tags: Vec<Value> = (0..tag_count)
            .map(|_| Value::String((*CATEGORIES.choose(rng).expect("non-empty")).into()))
            .collect();
        m.insert("tags".into(), Value::Array(tags));
    }
    m
}

// =========================================================================
// Random query strings
// =========================================================================

/// One generated query : the SQL string plus its parameter bindings.
struct GenQuery {
    sql: String,
    params: ParamBindings,
}

fn gen_any_query(rng: &mut StdRng) -> GenQuery {
    // Statement-kind mix tuned for coverage. SELECT dominates because
    // it has the largest internal surface (plans A/B/C/radius/scan/COUNT).
    let r = rng.random_range(0..100);
    if r < 50 {
        gen_select(rng)
    } else if r < 75 {
        gen_delete(rng)
    } else if r < 95 {
        gen_update(rng)
    } else if r < 98 {
        plain("VACUUM vectors")
    } else {
        plain("CHECKPOINT")
    }
}

fn plain(sql: &str) -> GenQuery {
    GenQuery {
        sql: sql.into(),
        params: ParamBindings::empty(),
    }
}

fn gen_select(rng: &mut StdRng) -> GenQuery {
    let r = rng.random_range(0..100);
    if r < 15 {
        // COUNT(*) with optional WHERE.
        let pred = if rng.random_bool(0.7) {
            format!(" WHERE {}", gen_predicate(rng, 2))
        } else {
            String::new()
        };
        plain(&format!("SELECT COUNT(*) FROM vectors{pred}"))
    } else if r < 30 {
        // Scan-and-limit (no ORDER BY, has LIMIT, has WHERE).
        let pred = gen_predicate(rng, 2);
        let k = rng.random_range(1..=10);
        plain(&format!("SELECT id FROM vectors WHERE {pred} LIMIT {k}"))
    } else if r < 45 {
        // Radius search.
        let radius = rng.random_range(0.1..2.0);
        let pred = if rng.random_bool(0.5) {
            format!(" AND {}", gen_predicate(rng, 1))
        } else {
            String::new()
        };
        let mut params = ParamBindings::empty();
        params = params.with_positional(ParamValue::Vector(random_vector(rng)));
        GenQuery {
            sql: format!("SELECT id FROM vectors WHERE embedding <-> $1 < {radius:.4}{pred}"),
            params,
        }
    } else {
        // kNN SELECT (plan A/B/C depending on selectivity).
        let k = rng.random_range(1..=10);
        let pred = if rng.random_bool(0.7) {
            format!(" WHERE {}", gen_predicate(rng, 2))
        } else {
            String::new()
        };
        let projection = if rng.random_bool(0.5) { "id" } else { "*" };
        let mut params = ParamBindings::empty();
        params = params.with_positional(ParamValue::Vector(random_vector(rng)));
        GenQuery {
            sql: format!(
                "SELECT {projection} FROM vectors{pred} \
                 ORDER BY embedding <-> $1 LIMIT {k}"
            ),
            params,
        }
    }
}

fn gen_delete(rng: &mut StdRng) -> GenQuery {
    let r = rng.random_range(0..100);
    if r < 30 {
        // DELETE WHERE id = <literal>
        let id = rng.random_range(1..=20);
        plain(&format!("DELETE FROM vectors WHERE id = {id}"))
    } else if r < 50 {
        // DELETE WHERE id = $1
        let id = rng.random_range(1..=20);
        let params = ParamBindings::empty().with_positional(ParamValue::Id(VectorId::new(id)));
        GenQuery {
            sql: "DELETE FROM vectors WHERE id = $1".into(),
            params,
        }
    } else if r < 80 {
        // DELETE by metadata predicate.
        let pred = gen_predicate(rng, 2);
        plain(&format!("DELETE FROM vectors WHERE {pred}"))
    } else {
        // DELETE by radius.
        let radius = rng.random_range(0.1..2.0);
        let pred = if rng.random_bool(0.4) {
            format!(" AND {}", gen_predicate(rng, 1))
        } else {
            String::new()
        };
        let params = ParamBindings::empty().with_positional(ParamValue::Vector(random_vector(rng)));
        GenQuery {
            sql: format!("DELETE FROM vectors WHERE embedding <-> $1 < {radius:.4}{pred}"),
            params,
        }
    }
}

fn gen_update(rng: &mut StdRng) -> GenQuery {
    // Build one or two SET clauses ; at least one is required.
    let n_assigns = if rng.random_bool(0.7) { 1 } else { 2 };
    let assigns: Vec<String> = (0..n_assigns).map(|_| gen_assignment(rng)).collect();
    let set_clause = assigns.join(", ");

    // WHERE shape : 40% single-id literal, 20% predicate, 20% radius,
    // 20% param-id.
    let r = rng.random_range(0..100);
    if r < 40 {
        let id = rng.random_range(1..=20);
        plain(&format!("UPDATE vectors SET {set_clause} WHERE id = {id}"))
    } else if r < 60 {
        let id = rng.random_range(1..=20);
        let params = ParamBindings::empty().with_positional(ParamValue::Id(VectorId::new(id)));
        GenQuery {
            sql: format!("UPDATE vectors SET {set_clause} WHERE id = $1"),
            params,
        }
    } else if r < 80 {
        let pred = gen_predicate(rng, 2);
        plain(&format!("UPDATE vectors SET {set_clause} WHERE {pred}"))
    } else {
        let radius = rng.random_range(0.1..2.0);
        let params = ParamBindings::empty().with_positional(ParamValue::Vector(random_vector(rng)));
        GenQuery {
            sql: format!("UPDATE vectors SET {set_clause} WHERE embedding <-> $1 < {radius:.4}"),
            params,
        }
    }
}

fn gen_assignment(rng: &mut StdRng) -> String {
    // 30% chance of a subscripted assignment on `attrs`.
    if rng.random_bool(0.3) {
        let key = *SUB_KEYS.choose(rng).expect("non-empty");
        let value = gen_atom_value(rng);
        format!("{MAP_FIELD}['{key}'] = {value}")
    } else {
        let field = pick_settable_field(rng);
        let value = gen_atom_value(rng);
        format!("{field} = {value}")
    }
}

fn pick_settable_field(rng: &mut StdRng) -> &'static str {
    // Skip `attrs` (handled by the subscript branch) and `embedding`
    // (binder rejects ; we don't want to bias the test toward errors).
    let candidates = ["category", "score", "year", "active", "tag", "status"];
    candidates.choose(rng).expect("non-empty")
}

// =========================================================================
// Predicate generator
// =========================================================================

fn gen_predicate(rng: &mut StdRng, depth: usize) -> String {
    if depth == 0 {
        return gen_atom(rng);
    }
    let r = rng.random_range(0..100);
    if r < 20 {
        format!("NOT ({})", gen_predicate(rng, depth - 1))
    } else if r < 40 {
        format!(
            "({} AND {})",
            gen_predicate(rng, depth - 1),
            gen_predicate(rng, depth - 1),
        )
    } else if r < 55 {
        format!(
            "({} OR {})",
            gen_predicate(rng, depth - 1),
            gen_predicate(rng, depth - 1),
        )
    } else {
        gen_atom(rng)
    }
}

fn gen_atom(rng: &mut StdRng) -> String {
    let kind = rng.random_range(0..100);
    if kind < 35 {
        let field = gen_field_ref(rng);
        let value = gen_atom_value(rng);
        format!("{field} = {value}")
    } else if kind < 55 {
        let field = gen_field_ref(rng);
        let op = ["<", "<=", ">", ">=", "!=", "<>"][rng.random_range(0..6)];
        let value = gen_atom_value(rng);
        format!("{field} {op} {value}")
    } else if kind < 70 {
        // IN list of 2-3 literals.
        let field = gen_field_ref(rng);
        let n = rng.random_range(2..=3);
        let vals: Vec<String> = (0..n).map(|_| gen_literal(rng)).collect();
        format!("{field} IN ({})", vals.join(", "))
    } else if kind < 80 {
        let field = gen_field_ref(rng);
        let (lo, hi) = if rng.random_bool(0.5) {
            (rng.random_range(0..50), rng.random_range(50..100))
        } else {
            (rng.random_range(2000..2010), rng.random_range(2010..2025))
        };
        format!("{field} BETWEEN {lo} AND {hi}")
    } else if kind < 90 {
        let field = gen_field_ref(rng);
        let neg = if rng.random_bool(0.5) { "NOT " } else { "" };
        format!("{field} IS {neg}NULL")
    } else {
        // Array-contains uses a string literal target.
        let field = "tags";
        let tag = *CATEGORIES.choose(rng).expect("non-empty");
        format!("{field} @> '{tag}'")
    }
}

fn gen_field_ref(rng: &mut StdRng) -> String {
    // 25% chance of a subscripted reference on `attrs`.
    if rng.random_bool(0.25) {
        let key = *SUB_KEYS.choose(rng).expect("non-empty");
        format!("{MAP_FIELD}['{key}']")
    } else {
        (*FIELDS.choose(rng).expect("non-empty")).into()
    }
}

fn gen_atom_value(rng: &mut StdRng) -> String {
    // For predicate RHS and assignment values.
    if rng.random_bool(0.4) {
        gen_literal(rng)
    } else if rng.random_bool(0.5) {
        // Param reference — but only valid if some param actually
        // exists. Since the harness binds at most one slot ($1) and
        // it's already used for the query vector / id in many paths,
        // skip param-bound atoms in WHERE values for v1 fuzzing.
        // Fall through to a literal.
        gen_literal(rng)
    } else {
        gen_literal(rng)
    }
}

fn gen_literal(rng: &mut StdRng) -> String {
    let kind = rng.random_range(0..100);
    if kind < 30 {
        // String literal.
        let pool = [CATEGORIES, COUNTRIES, COLORS][rng.random_range(0..3)];
        let s = *pool.choose(rng).expect("non-empty");
        format!("'{s}'")
    } else if kind < 60 {
        // Integer literal.
        let n: i64 = rng.random_range(0..3000);
        n.to_string()
    } else if kind < 80 {
        // Float literal.
        let f: f64 = rng.random_range(0.0..10.0);
        format!("{f:.3}")
    } else if kind < 95 {
        if rng.random_bool(0.5) {
            "TRUE".into()
        } else {
            "FALSE".into()
        }
    } else {
        "NULL".into()
    }
}

// =========================================================================
// Fuzz harness
// =========================================================================

/// One fuzz iteration : generate a query, run it, assert no panic
/// and either Ok or a typed error.
fn fuzz_one(engine: &mut Engine<L2>, rng: &mut StdRng, seed: u64, iter: usize) {
    let GenQuery { sql, params } = gen_any_query(rng);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.execute_str(&sql, params)
    }));
    match result {
        Ok(
            Ok(_)
            | Err(
                KovaQueryError::Parse(_)
                | KovaQueryError::Bind(_)
                | KovaQueryError::Plan(_)
                | KovaQueryError::Execution(_)
                | KovaQueryError::Backend(_),
            ),
        ) => {}
        Err(panic_payload) => {
            let msg = panic_payload
                .downcast_ref::<&'static str>()
                .copied()
                .or_else(|| panic_payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("<non-string panic payload>");
            panic!(
                "fuzzer panic at seed={seed} iter={iter} on query :\n  {sql}\n\
                 panic message : {msg}"
            );
        }
    }
}

/// Drive `iterations` fuzzed queries against a freshly-seeded shard.
fn run_fuzz(seed: u64, shard_size: usize, iterations: usize) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut fx = build_fixture(&mut rng, shard_size);
    // Hold onto rows so the binding stays alive ; we don't read them
    // in this phase, but a later correctness fuzzer will.
    let _ = &fx.rows;
    for iter in 0..iterations {
        fuzz_one(&mut fx.engine, &mut rng, seed, iter);
    }
}

// =========================================================================
// Test entry points
// =========================================================================

/// Smoke run : a few hundred iterations on a small shard. Runs on
/// every `cargo test`.
#[test]
fn fuzz_smoke_500_iterations() {
    run_fuzz(0x00C0_FFEE, 20, 500);
}

/// Different seed : catches seed-dependent panics the first one
/// happens to miss.
#[test]
fn fuzz_smoke_alt_seed() {
    run_fuzz(0xDEAD_BEEF, 20, 500);
}

/// Tiny shard : exercises the empty-result + single-row edges that
/// the larger fixture rarely hits.
#[test]
fn fuzz_smoke_tiny_shard() {
    run_fuzz(0xFEED, 3, 300);
}

/// Long-run variant : invoke explicitly with `--ignored` when you
/// want serious coverage (~30 sec on a workstation).
#[test]
#[ignore = "slow ; run with --ignored"]
fn fuzz_long_run() {
    for s in 0u64..16 {
        run_fuzz(0xBAD_BEEF + s, 50, 2_000);
    }
}

/// Diagnostic : the fuzzer is only meaningful if it actually drives
/// queries through the executor. This test asserts at least 30 % of
/// generated queries succeed cleanly — guards against a regression
/// where every generated shape errors at the parser / binder and the
/// fuzzer turns into a silent-pass.
#[test]
fn fuzz_meaningful_success_rate() {
    let seed = 0xFACE_F11E;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut fx = build_fixture(&mut rng, 25);
    let iterations = 1_000;
    let mut ok = 0;
    let mut errs = 0;
    let mut panics = 0;
    for _ in 0..iterations {
        let GenQuery { sql, params } = gen_any_query(&mut rng);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fx.engine.execute_str(&sql, params)
        }));
        match result {
            Ok(Ok(_)) => ok += 1,
            Ok(Err(_)) => errs += 1,
            Err(_) => panics += 1,
        }
    }
    assert_eq!(panics, 0, "fuzzer panicked {panics} times");
    assert!(
        ok >= iterations * 30 / 100,
        "only {ok} of {iterations} queries succeeded ({errs} errors) ; \
         generator is producing too many error-only shapes"
    );
}

/// Smoke that also throws a CHECKPOINT in the middle so the
/// fixture's WAL replay path gets exercised at least once per run.
#[test]
fn fuzz_with_checkpoint_midway() {
    let seed = 0xCAFE_F00D;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut fx = build_fixture(&mut rng, 25);
    for iter in 0..200 {
        if iter == 100 {
            let _ = fx.engine.execute_str("CHECKPOINT", ParamBindings::empty());
        }
        fuzz_one(&mut fx.engine, &mut rng, seed, iter);
    }
}

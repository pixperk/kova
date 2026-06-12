// The fuzzer trades some clippy pedantry for readability ; the
// dispatch tables are inherently flat, the trace helpers prefer
// explicit Err patterns over let-else, etc.
#![allow(
    clippy::too_many_lines,
    clippy::manual_let_else,
    clippy::items_after_statements,
    clippy::match_wildcard_for_single_variants,
    clippy::match_wild_err_arm,
    clippy::doc_markdown,
    clippy::needless_borrow,
    clippy::needless_pass_by_value,
    clippy::elidable_lifetime_names
)]

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
use kova_query::{Engine, ExecutionResult, KovaQueryError, ParamBindings, ParamValue};
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
        // Param reference : but only valid if some param actually
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
/// generated queries succeed cleanly : guards against a regression
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

// =========================================================================
// Phase B : correctness against a reference implementation
// =========================================================================
//
// The harness above only asserts "no panic." Phase B adds a second
// implementation we run alongside the engine for deterministic
// operations (COUNT, scan-and-limit, DELETE, UPDATE). For each
// generated query we :
//
//   1. Bind to a LogicalStatement so we share the predicate shape
//      with the engine (no parser drift between the two paths).
//   2. Compute the expected result by walking our reference state
//      (`Vec<(VectorId, Metadata)>`) with a hand-rolled predicate
//      evaluator.
//   3. Run the engine on the same SQL.
//   4. For correctness-checkable shapes, assert engine == reference.
//   5. For mutations, also apply the same change to the reference
//      state so the next iteration sees a consistent world.
//
// kNN and radius queries are *deterministic enough to validate*
// but their result ordering is approximate. We leave correctness
// checking off for those and rely on the recall harnesses in the
// lib tests.

use kova_query::logical::{
    BoundExpr, BoundLiteral, FieldRef, LogicalAssignment, LogicalDelete, LogicalStatement,
    LogicalUpdate, PredAtom, PredicateExpr,
};

/// What kind of correctness check we run for a given LogicalStatement.
/// Anything not listed here is panic-only.
enum CheckKind<'a> {
    Count {
        pred: Option<&'a PredicateExpr>,
    },
    ScanAndLimit {
        pred: &'a PredicateExpr,
        limit: u64,
    },
    DeleteById(VectorId),
    DeleteByPredicate(&'a PredicateExpr),
    UpdateById {
        id: VectorId,
        assigns: &'a [LogicalAssignment],
    },
    UpdateByPredicate {
        pred: &'a PredicateExpr,
        assigns: &'a [LogicalAssignment],
    },
}

/// Inspect a [`LogicalStatement`] and decide whether we have a
/// correctness check for it. Returns `None` for shapes Phase B
/// doesn't validate (kNN, radius, INSERT, VACUUM, CHECKPOINT).
fn check_kind<'a>(stmt: &'a LogicalStatement, params: &ParamBindings) -> Option<CheckKind<'a>> {
    match stmt {
        LogicalStatement::Query(q) => {
            // COUNT(*) : projection is a solo CountStar.
            if q.projection.columns.len() == 1
                && matches!(
                    q.projection.columns[0],
                    kova_query::logical::BoundProjection::CountStar { .. }
                )
            {
                return Some(CheckKind::Count {
                    pred: q.predicate.as_ref(),
                });
            }
            // Scan-and-limit : no ordering, has LIMIT, has WHERE, no
            // distance-threshold (radius takes precedence in the planner).
            if q.ordering.is_empty()
                && let Some(limit) = q.limit
                && let Some(pred) = &q.predicate
                && !predicate_has_distance_threshold(pred)
            {
                return Some(CheckKind::ScanAndLimit { pred, limit });
            }
            None
        }
        LogicalStatement::Delete(LogicalDelete {
            single_id_hint,
            predicate,
            ..
        }) => {
            match single_id_hint {
                Some(kova_query::logical::IdHint::Literal(id)) => {
                    Some(CheckKind::DeleteById(VectorId::new(*id)))
                }
                Some(kova_query::logical::IdHint::Param(p)) => {
                    // Resolve the param to an id eagerly so the check
                    // doesn't need to look at params again.
                    let resolved = params.resolve(p).ok()?;
                    match resolved {
                        ParamValue::Id(id) => Some(CheckKind::DeleteById(*id)),
                        _ => None,
                    }
                }
                None => {
                    if predicate_has_distance_threshold(predicate) {
                        None
                    } else {
                        Some(CheckKind::DeleteByPredicate(predicate))
                    }
                }
            }
        }
        LogicalStatement::Update(LogicalUpdate {
            single_id_hint,
            predicate,
            assignments,
            ..
        }) => {
            // Subscripted assignments mutate metadata in a way that
            // depends on the engine's apply_assignments behaviour. We
            // mirror that in our reference, but only when *all* of
            // them are plain (no subscript) : keeps the ref simple.
            if assignments.iter().any(|a| a.subscript.is_some()) {
                return None;
            }
            match single_id_hint {
                Some(kova_query::logical::IdHint::Literal(id)) => Some(CheckKind::UpdateById {
                    id: VectorId::new(*id),
                    assigns: assignments,
                }),
                Some(kova_query::logical::IdHint::Param(p)) => {
                    let resolved = params.resolve(p).ok()?;
                    match resolved {
                        ParamValue::Id(id) => Some(CheckKind::UpdateById {
                            id: *id,
                            assigns: assignments,
                        }),
                        _ => None,
                    }
                }
                None => {
                    if predicate_has_distance_threshold(predicate) {
                        None
                    } else {
                        Some(CheckKind::UpdateByPredicate {
                            pred: predicate,
                            assigns: assignments,
                        })
                    }
                }
            }
        }
        _ => None,
    }
}

/// True if any node in the tree carries a `DistanceThreshold` atom.
/// Used to filter out queries that route to the radius operator,
/// since we don't validate distance-based shapes in Phase B.
fn predicate_has_distance_threshold(p: &PredicateExpr) -> bool {
    match p {
        PredicateExpr::Atom(PredAtom::DistanceThreshold { .. }) => true,
        PredicateExpr::And(cs) | PredicateExpr::Or(cs) => {
            cs.iter().any(predicate_has_distance_threshold)
        }
        PredicateExpr::Not(inner) => predicate_has_distance_threshold(inner),
        _ => false,
    }
}

/// Mirror of the engine's predicate evaluator for the test side.
/// Same semantics : null-safe, structural compare, subscript-aware.
/// Returns Err on shapes the reference doesn't support
/// (DistanceThreshold) so the harness can skip the comparison
/// gracefully.
fn ref_eval(
    p: &PredicateExpr,
    meta: &Metadata,
    params: &ParamBindings,
) -> Result<bool, &'static str> {
    match p {
        PredicateExpr::True => Ok(true),
        PredicateExpr::False => Ok(false),
        PredicateExpr::And(cs) => {
            for c in cs {
                if !ref_eval(c, meta, params)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        PredicateExpr::Or(cs) => {
            for c in cs {
                if ref_eval(c, meta, params)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        PredicateExpr::Not(inner) => Ok(!ref_eval(inner, meta, params)?),
        PredicateExpr::Atom(atom) => ref_eval_atom(atom, meta, params),
    }
}

fn ref_eval_atom(
    atom: &PredAtom,
    meta: &Metadata,
    params: &ParamBindings,
) -> Result<bool, &'static str> {
    match atom {
        PredAtom::Eq { field, value } => {
            let Some(want) = ref_resolve_value(value, params) else {
                return Err("non-literal param in ref WHERE");
            };
            Ok(ref_lookup(field, meta).is_some_and(|v| ref_values_eq(v, &want)))
        }
        PredAtom::Cmp { field, op, value } => {
            let Some(want) = ref_resolve_value(value, params) else {
                return Err("non-literal param in ref WHERE");
            };
            Ok(ref_lookup(field, meta)
                .and_then(|v| ref_values_cmp(v, &want, *op))
                .unwrap_or(false))
        }
        PredAtom::In { field, values } => {
            let Some(actual) = ref_lookup(field, meta) else {
                return Ok(false);
            };
            Ok(values
                .iter()
                .map(ref_literal_to_value)
                .any(|v| ref_values_eq(actual, &v)))
        }
        PredAtom::Between { field, lo, hi } => {
            let Some(actual) = ref_lookup(field, meta) else {
                return Ok(false);
            };
            let lo_v = ref_literal_to_value(lo);
            let hi_v = ref_literal_to_value(hi);
            let ge_lo = ref_values_cmp(actual, &lo_v, kova_query::ast::CmpOp::Ge).unwrap_or(false);
            let le_hi = ref_values_cmp(actual, &hi_v, kova_query::ast::CmpOp::Le).unwrap_or(false);
            Ok(ge_lo && le_hi)
        }
        PredAtom::IsNotNull { field } => Ok(ref_lookup(field, meta).is_some()),
        PredAtom::ArrayContains { field, value } => {
            let target = ref_literal_to_value(value);
            match ref_lookup(field, meta) {
                Some(Value::Array(arr)) => Ok(arr.iter().any(|v| ref_values_eq(v, &target))),
                _ => Ok(false),
            }
        }
        PredAtom::DistanceThreshold { .. } => Err("DistanceThreshold not handled by ref"),
    }
}

fn ref_lookup<'a>(field: &FieldRef, meta: &'a Metadata) -> Option<&'a Value> {
    let top = meta.get(&field.name)?;
    match &field.subscript {
        None => Some(top),
        Some(key) => match top {
            Value::Map(inner) => inner.get(key),
            _ => None,
        },
    }
}

fn ref_resolve_value(expr: &BoundExpr, params: &ParamBindings) -> Option<Value> {
    match expr {
        BoundExpr::Literal(l) => Some(ref_literal_to_value(l)),
        BoundExpr::Param(p) => {
            let resolved = params.resolve(p).ok()?;
            match resolved {
                ParamValue::String(s) => Some(Value::String(s.clone())),
                ParamValue::I64(n) => Some(Value::I64(*n)),
                ParamValue::F64(f) => Some(Value::F64(*f)),
                ParamValue::Bool(b) => Some(Value::Bool(*b)),
                ParamValue::Null => Some(Value::Array(Vec::new())),
                ParamValue::Metadata(m) => Some(Value::Map(m.clone())),
                _ => None,
            }
        }
    }
}

fn ref_literal_to_value(l: &BoundLiteral) -> Value {
    match l {
        BoundLiteral::String(s) => Value::String(s.clone()),
        BoundLiteral::I64(n) => Value::I64(*n),
        BoundLiteral::F64(f) => Value::F64(*f),
        BoundLiteral::Bool(b) => Value::Bool(*b),
        // The engine models NULL as an unmatchable sentinel ; mirror
        // that with an empty Array since no real Array literal lands
        // there.
        BoundLiteral::Null => Value::Array(Vec::new()),
    }
}

fn ref_values_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::String(x), Value::String(y)) => x == y,
        (Value::I64(x), Value::I64(y)) => x == y,
        (Value::F64(x), Value::F64(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Array(x), Value::Array(y)) => x == y,
        (Value::Map(x), Value::Map(y)) => x == y,
        (Value::I64(x), Value::F64(y)) | (Value::F64(y), Value::I64(x)) => {
            #[allow(clippy::cast_precision_loss)]
            let xf = *x as f64;
            xf == *y
        }
        _ => false,
    }
}

fn ref_values_cmp(a: &Value, b: &Value, op: kova_query::ast::CmpOp) -> Option<bool> {
    use std::cmp::Ordering;
    let ordering = match (a, b) {
        (Value::String(x), Value::String(y)) => x.partial_cmp(y),
        (Value::I64(x), Value::I64(y)) => x.partial_cmp(y),
        (Value::F64(x), Value::F64(y)) => x.partial_cmp(y),
        (Value::Bool(x), Value::Bool(y)) => x.partial_cmp(y),
        (Value::I64(x), Value::F64(y)) => {
            #[allow(clippy::cast_precision_loss)]
            let xf = *x as f64;
            xf.partial_cmp(y)
        }
        (Value::F64(x), Value::I64(y)) => {
            #[allow(clippy::cast_precision_loss)]
            let yf = *y as f64;
            x.partial_cmp(&yf)
        }
        _ => return None,
    }?;
    use kova_query::ast::CmpOp;
    Some(match op {
        CmpOp::Eq => ordering == Ordering::Equal,
        CmpOp::Ne => ordering != Ordering::Equal,
        CmpOp::Lt => ordering == Ordering::Less,
        CmpOp::Le => ordering != Ordering::Greater,
        CmpOp::Gt => ordering == Ordering::Greater,
        CmpOp::Ge => ordering != Ordering::Less,
    })
}

/// Drive one iteration of the correctness fuzzer. Generates a query,
/// binds it, picks a correctness check if one applies, runs the
/// engine, asserts the reference and engine agree, and (for DML)
/// syncs the reference state.
fn correctness_one(fx: &mut Fixture, rng: &mut StdRng, seed: u64, iter: usize) {
    let GenQuery { sql, params } = gen_any_query(rng);

    // Bind the SQL to a LogicalStatement before running. If binding
    // fails we just run the engine for panic-coverage and skip the
    // correctness check.
    let bound = kova_query::parse_str(&sql).and_then(kova_query::bind);
    let stmt = if let Ok(s) = bound {
        s
    } else {
        // Parser/binder rejection : engine will reject too. Still
        // run to catch any panic.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fx.engine.execute_str(&sql, params)
        }));
        return;
    };

    let Some(kind) = check_kind(&stmt, &params) else {
        // No correctness check for this shape. If it's a mutation we
        // can't mirror (e.g. radius DELETE / UPDATE, subscripted
        // assignment), skip it entirely so the engine and the
        // reference state stay in sync. Non-mutations run the
        // already-generated query for panic coverage (NOT a freshly
        // generated one : that'd mutate the engine off our radar).
        if matches!(
            stmt,
            LogicalStatement::Delete(_) | LogicalStatement::Update(_)
        ) {
            return;
        }
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fx.engine.execute_str(&sql, params)
        }))
        .unwrap_or_else(|_| panic!("engine panicked on `{sql}` at seed={seed} iter={iter}"));
        return;
    };

    // Reference-first : compute the expected result against the
    // reference state BEFORE running the engine. If the reference
    // can't handle some shape (returns Err mid-eval), skip without
    // touching the engine, so the two stay in sync.
    let expectation = match compute_expectation(&kind, &fx.rows, &params) {
        Some(exp) => exp,
        None => return, // ref can't evaluate ; skip
    };

    // Run the engine. Panic-catch turns any panic into a test failure.
    let engine_params = params.clone();
    let engine_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fx.engine.execute_str(&sql, engine_params)
    }));
    let engine_result = match engine_result {
        Ok(r) => r,
        Err(_) => panic!("engine panicked on `{sql}` at seed={seed} iter={iter}"),
    };

    match (kind, expectation) {
        (CheckKind::Count { .. }, Expectation::Count(expected)) => {
            let Ok(ExecutionResult::Rows { rows, .. }) = engine_result else {
                return;
            };
            assert_eq!(rows.len(), 1, "COUNT should produce 1 row : `{sql}`");
            let engine_count = match rows[0].values[0] {
                kova_query::executor::RowValue::Field(Value::I64(n)) => n,
                ref other => panic!(
                    "COUNT cell isn't I64 : `{sql}` value={other:?} (seed={seed} iter={iter})"
                ),
            };
            assert_eq!(
                u64::try_from(engine_count).unwrap_or(u64::MAX),
                expected,
                "COUNT mismatch on `{sql}` (seed={seed} iter={iter})"
            );
        }
        (CheckKind::ScanAndLimit { limit, .. }, Expectation::ScanIds(expected)) => {
            let Ok(ExecutionResult::Rows { rows, .. }) = engine_result else {
                return;
            };
            assert!(
                u64::try_from(rows.len()).unwrap_or(u64::MAX) <= limit,
                "scan-and-limit returned more rows than LIMIT : `{sql}` \
                 got {} expected <= {limit} (seed={seed} iter={iter})",
                rows.len(),
            );
            for row in rows {
                let Some(kova_query::executor::RowValue::Id(id)) = row.values.first() else {
                    panic!(
                        "scan-and-limit row missing Id cell : `{sql}` \
                         (seed={seed} iter={iter})"
                    );
                };
                assert!(
                    expected.contains(&id),
                    "scan-and-limit returned id {} that doesn't satisfy the predicate : \
                     `{sql}` (seed={seed} iter={iter})",
                    id.get()
                );
            }
        }
        (CheckKind::DeleteById(target), Expectation::DeleteByIdExpected(present)) => {
            match engine_result {
                Ok(ExecutionResult::Delete { deleted, .. }) => {
                    let expected = u64::from(present);
                    assert_eq!(
                        deleted, expected,
                        "delete-by-id count mismatch : `{sql}` (seed={seed} iter={iter})"
                    );
                    if deleted == 1 {
                        fx.rows.retain(|(id, _)| *id != target);
                    }
                }
                Ok(other) => panic!(
                    "DELETE returned non-Delete shape : `{sql}` result={other:?} \
                     (seed={seed} iter={iter})"
                ),
                Err(_) => { /* engine errored ; ref untouched */ }
            }
        }
        (CheckKind::DeleteByPredicate(_), Expectation::DeleteByPredicateTargets(targets)) => {
            if let Ok(ExecutionResult::Delete { deleted, .. }) = engine_result {
                let expected = u64::try_from(targets.len()).unwrap_or(u64::MAX);
                assert_eq!(
                    deleted, expected,
                    "delete-by-predicate count mismatch : `{sql}` \
                 (seed={seed} iter={iter})"
                );
                fx.rows.retain(|(id, _)| !targets.contains(id));
            } else { /* engine error : ref untouched */
            }
        }
        (CheckKind::UpdateById { id, assigns }, Expectation::UpdateByIdExpected(present)) => {
            if let Ok(ExecutionResult::Update { updated, .. }) = engine_result {
                if present {
                    assert_eq!(
                        updated, 1,
                        "update-by-id should have hit one row : `{sql}` \
                     (seed={seed} iter={iter})"
                    );
                    if let Some(i) = fx.rows.iter().position(|(rid, _)| *rid == id) {
                        let (_, bag) = &mut fx.rows[i];
                        ref_apply_assignments(bag, assigns, &params);
                    }
                } else {
                    assert_eq!(updated, 0, "update on missing id should be 0");
                }
            } else { /* engine error : ref untouched */
            }
        }
        (
            CheckKind::UpdateByPredicate { assigns, .. },
            Expectation::UpdateByPredicateTargets(targets),
        ) => {
            if let Ok(ExecutionResult::Update { updated, .. }) = engine_result {
                let expected = u64::try_from(targets.len()).unwrap_or(u64::MAX);
                assert_eq!(
                    updated, expected,
                    "update-by-predicate count mismatch : `{sql}` \
                 (seed={seed} iter={iter})"
                );
                for id in targets {
                    if let Some(i) = fx.rows.iter().position(|(rid, _)| *rid == id) {
                        let (_, bag) = &mut fx.rows[i];
                        ref_apply_assignments(bag, assigns, &params);
                    }
                }
            } else { /* engine error : ref untouched */
            }
        }
        _ => unreachable!("kind / expectation mismatch ; check_kind drift"),
    }
}

/// Reference-side expectation paired with each [`CheckKind`].
enum Expectation {
    Count(u64),
    ScanIds(std::collections::HashSet<VectorId>),
    DeleteByIdExpected(bool),
    DeleteByPredicateTargets(Vec<VectorId>),
    UpdateByIdExpected(bool),
    UpdateByPredicateTargets(Vec<VectorId>),
}

/// Walk the reference state and compute what the engine *should*
/// produce for this kind. Returns `None` when the reference evaluator
/// can't handle some atom in the predicate (rare ; the caller skips
/// the iteration without touching the engine).
fn compute_expectation(
    kind: &CheckKind<'_>,
    rows: &[(VectorId, Metadata)],
    params: &ParamBindings,
) -> Option<Expectation> {
    match kind {
        CheckKind::Count { pred } => {
            let mut n = 0_u64;
            for (_id, meta) in rows {
                let pass = match pred {
                    None => true,
                    Some(p) => ref_eval(p, meta, params).ok()?,
                };
                if pass {
                    n += 1;
                }
            }
            Some(Expectation::Count(n))
        }
        CheckKind::ScanAndLimit { pred, .. } => {
            let mut set = std::collections::HashSet::new();
            for (id, meta) in rows {
                if ref_eval(pred, meta, params).ok()? {
                    set.insert(*id);
                }
            }
            Some(Expectation::ScanIds(set))
        }
        CheckKind::DeleteById(target) => Some(Expectation::DeleteByIdExpected(
            rows.iter().any(|(id, _)| id == target),
        )),
        CheckKind::DeleteByPredicate(pred) => {
            let mut targets = Vec::new();
            for (id, meta) in rows {
                if ref_eval(pred, meta, params).ok()? {
                    targets.push(*id);
                }
            }
            Some(Expectation::DeleteByPredicateTargets(targets))
        }
        CheckKind::UpdateById { id, .. } => Some(Expectation::UpdateByIdExpected(
            rows.iter().any(|(rid, _)| rid == id),
        )),
        CheckKind::UpdateByPredicate { pred, .. } => {
            let mut targets = Vec::new();
            for (id, meta) in rows {
                if ref_eval(pred, meta, params).ok()? {
                    targets.push(*id);
                }
            }
            Some(Expectation::UpdateByPredicateTargets(targets))
        }
    }
}

/// Mirror of the engine's `apply_assignments` for the reference
/// state. Only handles non-subscripted assignments ; the caller's
/// `check_kind` filters subscripted ones out before reaching here.
fn ref_apply_assignments(
    bag: &mut Metadata,
    assigns: &[LogicalAssignment],
    params: &ParamBindings,
) {
    for a in assigns {
        if a.subscript.is_some() {
            // Shouldn't reach ; filtered upstream. Skip defensively.
            continue;
        }
        let Some(v) = ref_resolve_value(&a.value, params) else {
            continue;
        };
        bag.insert(a.field.clone(), v);
    }
}

/// Run the correctness fuzzer for `iterations` rounds. Each round
/// generates a query, runs it through the engine, and (where a
/// correctness check applies) asserts the engine's result matches
/// the reference.
fn run_correctness(seed: u64, shard_size: usize, iterations: usize) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut fx = build_fixture(&mut rng, shard_size);
    for iter in 0..iterations {
        correctness_one(&mut fx, &mut rng, seed, iter);
    }
}

#[test]
fn correctness_fuzz_500_iterations() {
    run_correctness(0xFEED_BEEF, 25, 500);
}

#[test]
fn correctness_fuzz_alt_seed() {
    run_correctness(0xBADD_F00D, 25, 500);
}

#[test]
#[ignore = "slow ; run with --ignored"]
fn correctness_fuzz_long_run() {
    for s in 0u64..8 {
        run_correctness(0xC0DE_DEAD + s, 50, 1_500);
    }
}

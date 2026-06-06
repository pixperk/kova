# KQL : the query language for Kova

KQL is the SQL-shaped query language Kova ships for hybrid vector +
metadata workloads. It is the public surface of the [`kova-query`](../crates/kova-query/)
crate. Everything user-visible goes through the same parse -> bind ->
plan -> execute pipeline, including DDL, DML, management ops, and
reads.

This document is the reference. The README has the highlights ; here
is the full architecture, the design choices, how Phase 1 stands,
and what Phase 2 changes.

If you are new to KQL, read top-to-bottom. The first few sections
are a guided tour : a 30-second taste, then a quick start you can
copy, then the queries you will actually write day to day. Once you
have a feel for the language, the middle of the doc opens up the
pipeline and the algorithms underneath. The reference material
(types, full statement coverage, result shapes) lives near the end,
and the closing sections cover how we know it works and what
Phase 2 changes.

---

## A 30-second taste

KQL lets you write one query that does kNN search and metadata
filtering together, and have the database figure out the right
strategy :

```sql
SELECT id, embedding <-> $q AS dist
FROM vectors
WHERE category = 'docs' AND year >= 2024
ORDER BY embedding <-> $q LIMIT 10
```

That one statement finds the 10 vectors closest to `$q`, restricted
to rows where `category = 'docs' AND year >= 2024`. Without KQL you
would do the ANN search and the metadata filter in two separate
calls, glue the results together in application code, and hope you
got the order right. With KQL, the planner picks one of three
execution strategies based on how selective your predicate is, and
hands you back the answer.

That is the elevator pitch. The rest of this doc unpacks it.

---

## What KQL is, and what it is not

KQL gives you a way to express "find the k nearest vectors to a query,
among rows that match a metadata predicate," without you having to
hand-write the planner. Everything from `SELECT COUNT(*)` to
`UPDATE vectors SET attrs['country'] = 'IN'` to
`DELETE FROM vectors WHERE embedding <-> $q < 0.5 AND category = 'old'`
is the same language, the same parser, the same executor.

It is **not** general-purpose SQL. There is exactly one table
(`vectors`), there are no JOINs, no subqueries, no window functions,
no transactions across statements. The constraints are deliberate :
the workloads vector databases serve do not need those, and shedding
them keeps the planner small enough to fit in your head.

---

## How to use it

The full public surface is exactly one type plus its `execute_str`
method :

```rust
use kova_core::{L2, Metadata, Vector, VectorId, Value};
use kova_index::HnswParams;
use kova_storage::Shard;
use kova_query::{Engine, ExecutionResult, ParamBindings, ParamValue, RowValue};

// 1. Open a shard, wrap it in an Engine. Engine is generic over the
//    distance metric ; the table name is the symbolic name KQL uses.
let shard = Shard::open("/tmp/my-shard", /*dim*/ 4, L2, HnswParams::default())?;
let mut engine = Engine::new(shard, "vectors");

// 2. Seed some data through the Engine. Single inserts go through
//    `INSERT INTO ... VALUES`. Batches go through the batch-param shape.
engine.execute_str(
    "INSERT INTO vectors (id, embedding, metadata) VALUES ($1, $2, $3)",
    ParamBindings::empty()
        .with_positional(ParamValue::Id(VectorId::new(1)))
        .with_positional(ParamValue::Vector(Vector::try_new(vec![1.0, 0.0, 0.0, 0.0])?))
        .with_positional(ParamValue::Metadata({
            let mut m = Metadata::new();
            m.insert("category".into(), Value::String("docs".into()));
            m
        })),
)?;

// 3. Run a kNN query. Parameters bind by position.
let query_vec = Vector::try_new(vec![1.0, 0.0, 0.0, 0.0])?;
let result = engine.execute_str(
    "SELECT id, embedding <-> $1 AS distance \
     FROM vectors \
     WHERE category = 'docs' \
     ORDER BY embedding <-> $1 LIMIT 10",
    ParamBindings::empty().with_positional(ParamValue::Vector(query_vec)),
)?;

// 4. Pattern-match on the result. Rows is the SELECT shape ; the
//    Delete / Update / Count / Insert / Vacuum / Checkpoint shapes
//    have their own variants.
let ExecutionResult::Rows { columns, rows } = result else {
    panic!("expected Rows");
};
for row in rows {
    let RowValue::Id(id) = row.values[0] else { continue; };
    let RowValue::Distance(d) = row.values[1] else { continue; };
    println!("id={} distance={:.3}", id.get(), d);
}
```

Named params work the same way :

```rust
let params = ParamBindings::empty()
    .with_named("q", ParamValue::Vector(query_vec))
    .with_named("cutoff", ParamValue::F64(0.5));

engine.execute_str(
    "SELECT id FROM vectors WHERE embedding <-> $q < $cutoff",
    params,
)?;
```

`ParamBindings` is a builder ; positional and named slots coexist in
the same call. Every public KQL example in this doc is one
`execute_str` call away from running.

---

## The queries you will actually write

If you only learn five query shapes, learn these. They cover most of
what real workloads look like.

### kNN with a metadata filter

The bread and butter. "Find me the 10 most similar `$q`, but only
among `docs` from 2024 onward."

```sql
SELECT id, embedding <-> $q AS dist
FROM vectors
WHERE category = 'docs' AND year >= 2024
ORDER BY embedding <-> $q LIMIT 10
```

The planner picks plan A, B, or C automatically depending on how
many rows match the WHERE clause. You do not need to think about it.

### Radius search

"Everything within distance 0.5, no top-k cap."

```sql
SELECT id FROM vectors
WHERE embedding <-> $q < 0.5 AND tag = 'archived'
```

Note that this has no `ORDER BY` and no `LIMIT`. The radius operator
is its own thing : it returns every match, no ranking required. The
`AND tag = 'archived'` is a post-filter inside the operator.

### Counting

"How many rows match this predicate?"

```sql
SELECT COUNT(*) FROM vectors WHERE attrs['country'] = 'IN'
```

COUNT(*) bypasses the kNN path entirely. Comes back as a single row
with a single I64 cell. Note also the subscripted predicate :
`attrs['country']` looks into a nested map field.

### Targeted DML

Delete one row by id (this gets a fast path) :

```sql
DELETE FROM vectors WHERE id = 42
```

Patch a nested attribute on a known row :

```sql
UPDATE vectors SET attrs['priority'] = 5 WHERE id = $1
```

Tombstone everything matching a predicate :

```sql
DELETE FROM vectors WHERE category = 'old' AND year < 2020
```

DML by id is O(1) (single get + single put). DML by predicate scans
metadata and batches the mutations under one WAL group-commit.

### Batch insert

For ingestion, you pass the whole batch as one param :

```sql
INSERT INTO vectors VALUES $batch
```

```rust
let batch = vec![
    (VectorId::new(1), v1, m1),
    (VectorId::new(2), v2, m2),
    /* ... */
];
engine.execute_str(
    "INSERT INTO vectors VALUES $batch",
    ParamBindings::empty().with_positional(ParamValue::Batch(batch)),
)?;
```

Batched inserts share one WAL fsync and one metadata flush across
the whole batch, so the per-row cost drops sharply at any reasonable
batch size.

---

If those five shapes feel comfortable, you already know 80% of KQL.
The rest of this doc explains how that 80% is built, why the
planner picks what it picks, and how you can be sure it works.

---

You have seen the queries. The rest of the doc is for the curious :
**how the engine actually answers them.** Skip ahead to "Type system
and reference" if you only need the API. Stay with us if you want to
understand what changed when KQL shipped, and why Phase 2 will be
fast without anyone having to rewrite their queries.

The next handful of sections walk through the pipeline as a whole,
then each layer in turn, then the algorithms inside the operators.

## The pipeline at a glance

```mermaid
flowchart LR
    SQL["KQL string"] -. parser .-> AST["AstStatement"]
    AST -. binder .-> LS["LogicalStatement"]
    LS -. plan_with_estimator .-> PP["PhysicalPlan"]
    PP -. executor .-> R["ExecutionResult"]
```

Four intermediate representations, three transformations. Each IR
has one job, each transformation is independently testable. The IRs
are :

| IR | Lives in | Purpose |
|----|----------|---------|
| `AstStatement` | [`ast.rs`](../crates/kova-query/src/ast.rs) | Syntax-faithful capture of what the user typed |
| `LogicalStatement` | [`logical.rs`](../crates/kova-query/src/logical.rs) | After field resolution, type checks, predicate normalisation |
| `PhysicalPlan` | [`physical.rs`](../crates/kova-query/src/physical.rs) | The operator tree the executor walks, with strategy choice baked in |
| `ExecutionResult` | [`executor.rs`](../crates/kova-query/src/executor.rs) | What the caller sees |

The boundaries are stable contracts. Grammar churn does not touch the
planner, planner churn does not touch the parser, and adding the
binder did not move either.

### Pipeline in code

```rust
let ast       = kova_query::parse_str(sql)?;          // parser
let logical   = kova_query::bind(ast)?;               // binder
let physical  = kova_query::plan_with_estimator(      // planner
    logical, &estimator, &params,
)?;
let result    = engine.execute(physical, &params)?;   // executor
```

`Engine::execute_str(sql, params)` wraps all four steps. Real callers
use that ; the internal pipeline is exposed for testing.

---

## Layer 1 : grammar and parser

The grammar lives in [`grammar.pest`](../crates/kova-query/src/grammar.pest)
and the parser builds the AST from a Pest parse tree. The parser is
strict about syntax and permissive about semantics. A parser cannot
type-check because it does not know what fields exist ; semantic
checks live in the binder.

Statements supported by the grammar (all 8 listed are accepted ; some
are bound but rejected at the binder for v1) :

```text
SELECT     INSERT     UPDATE     DELETE
VACUUM     CHECKPOINT
CREATE INDEX           DROP INDEX
```

Precedence inside predicates : `OR < AND < NOT < atom`. Atom kinds :

- `field = value` / `field <op> value` for the six comparison ops
- `field IN (lit, lit, ...)`
- `field BETWEEN lo AND hi`
- `field IS NULL` / `field IS NOT NULL`
- `field @> 'tag'` (array contains)
- `embedding <op> $q <cmp> radius` (distance threshold)

Where `field` is either `bare_name` or `bare_name['subscript_key']`.

Distance operators : `<->` (L2), `<=>` (cosine), `<#>` (negated inner
product, so smaller is closer).

Parameters : positional (`$1`, `$2`, ...) and named (`$query`,
`$cutoff`, ...). The binder resolves both into the same internal
slot, the executor looks them up the same way.

The grammar uses a shared `field_ref` rule across every atom kind so
subscript syntax is uniform : `WHERE attrs['key'] BETWEEN 0 AND 10`
is no different from `WHERE attrs BETWEEN 0 AND 10`.

The parser's surface is round-trip tested by the printer. Every
parse-able statement, when fed back through `print`, reparses to the
same AST. Property tests pin this.

---

## Layer 2 : binder

The binder turns an AST into a `LogicalStatement` plus a set of
loud-and-clear rejections. What it does :

1. **Field resolution.** Captures the field name plus optional
   subscript on every predicate atom, normalising the shape so
   downstream code does not branch on syntax.
2. **Predicate normalisation.** `IS NULL` becomes `NOT IsNotNull(f)`,
   `And` / `Or` get flattened across nesting, `Not` is pushed down to
   atoms.
3. **Type checks.** Embedding mutation is rejected here
   (`UPDATE SET embedding = $1` errors with a clear message ; HNSW
   nodes are not mutable). kNN ordering with `DESC` is rejected
   (kNN is ascending by definition). kNN without `LIMIT` is rejected.
4. **Single-id hint detection.** `WHERE id = 42` and `WHERE id = $1`
   both get a `single_id_hint`. The planner uses this to dispatch
   `DeleteById` / `DeleteByParamId` / `UpdateById` / `UpdateByParamId`
   without re-walking the predicate tree.
5. **v2-only DDL rejection.** `CREATE INDEX` and `DROP INDEX` parse
   but the binder refuses them ; index DDL is Phase 2 territory.

The result is a `LogicalStatement` that is guaranteed well-typed and
free of grammar-level ambiguity. Anything downstream of the binder
can assume the shape is sound.

---

## Layer 3 : planner

The planner decides **how** to execute. For DDL and management
statements the planning is one-to-one : `LogicalVacuum` becomes
`PhysicalPlan::Vacuum`, `LogicalCheckpoint` becomes
`PhysicalPlan::Checkpoint`, and so on. The interesting work is on
the read side.

### The three plans for SELECT

```mermaid
flowchart TD
    Q["SELECT id FROM vectors WHERE pred ORDER BY embedding <-> $q LIMIT k"]
    Q -. selectivity .-> D{estimator}
    D -. < 0.05 .-> B["Plan B<br/>MetadataScan + ExactDistance"]
    D -. 0.05 to 0.5 .-> C["Plan C<br/>FilteredKnnSearch"]
    D -. >= 0.5 .-> A["Plan A<br/>KnnSearch + post_filter"]
```

Three strategies, chosen by **estimated predicate selectivity**.

**Plan A : overfetched kNN + post-filter.** Run an unfiltered kNN
with `k = user_limit * 4`, then walk the candidates and drop the ones
that fail the WHERE clause. Wins when most rows match the predicate,
so the post-filter rarely drops anything and the overfetch cheaply
saturates the LIMIT.

**Plan B : metadata scan + exact distance.** Walk the metadata store
for ids matching the predicate, then compute exact distance for each.
Bypasses the ANN entirely. Wins when the predicate matches very few
rows : an O(matches × dim) exact distance loop beats running the full
ANN walk.

**Plan C : filtered ANN.** Run an HNSW search where the predicate is
consulted **during** the graph walk. Out-of-filter nodes still route
traversal but never enter the results heap. Wins in the middle :
tight enough that plan A's overfetch would starve, loose enough that
plan B's scan would be wasteful.

The bands :

| Selectivity | Plan |
|-------------|------|
| `< 0.05` | B |
| `[0.05, 0.5)` | C |
| `>= 0.5` | A |
| no predicate | A |

The estimator that picks is a trait :

```rust
trait SelectivityEstimator {
    fn estimate(&self, pred: &PredicateExpr, params: &ParamBindings)
        -> SelectivityEstimate;
}
```

Phase 1's implementation (`ShardEstimator`) runs the predicate
against every row and returns an exact count. Cheap because metadata
is in-memory. Phase 2 swaps in an estimator backed by histograms and
secondary-index cardinality, same trait, same dispatch.

### Radius operator

`WHERE embedding <-> $q < r` is a distinct shape : "give me everything
within distance r," not "give me the top k." The planner detects this
shape before the kNN check and emits a `RadiusSearch` operator instead.

```mermaid
flowchart LR
    SQL["WHERE embedding <-> $q < r"]
    SQL -. extract_radius_atom .-> EXT["query=$q, radius=r, residue=other AND-atoms"]
    EXT -.-> OP["PhysicalPlan::RadiusSearch"]
```

The HNSW side uses a doubling-`ef` walk : it runs kNN with growing
`ef` until either a returned hit lies outside the radius (proving the
ball is fully enclosed) or `ef` reaches the index size. Then filters
by radius. This is simpler than a true radius-walk and more robust
when the entry point is outside the ball.

AND-residue stays on as a `post_filter`. OR with a distance threshold
is rejected at plan time : the right answer is a Union operator that
merges radius balls, which is not in Phase 1.

### COUNT(*) bypass

`SELECT COUNT(*) FROM vectors [WHERE pred]` skips the kNN-shape
check entirely. The planner detects a solo `COUNT(*)` projection and
emits `PhysicalPlan::Count` directly. No ordering, no LIMIT, no kNN.

The executor either returns `shard.len()` (no predicate) or
`shard.count_matching(pred)` (with predicate). One row, one column.

### Scan-and-limit bypass

`SELECT id FROM vectors WHERE pred LIMIT k` (no ORDER BY) goes
through a `Projection(Limit(MetadataScan))` plan. The order of
returned rows is implementation-defined ; there is no kNN-shape
requirement for this path.

The same bypass refuses to plan an unbounded slice-scan : a query
with `LIMIT` but no `WHERE` is rejected loudly. Returning random
slices of large shards is the kind of foot-gun the language refuses
to ship.

### DML dispatch

```mermaid
flowchart TD
    D["DELETE FROM vectors WHERE ..."] -.-> H{predicate shape}
    H -. id = literal .-> ID["DeleteById"]
    H -. id = $param .-> PI["DeleteByParamId"]
    H -. distance threshold .-> R["DeleteByRadius"]
    H -. other metadata .-> P["DeleteByPredicate"]
```

`DELETE` and `UPDATE` use the same dispatch shape : a single-id hint
takes the fast path, a top-level distance threshold takes the radius
operator, anything else goes through the metadata-scan path.
`UPDATE` adds one more decision : subscripted assignments
(`SET attrs['key'] = ...`) and bare ones use the same operator but
different apply paths in the executor.

OR containing a distance threshold is rejected uniformly across
`SELECT`, `DELETE`, and `UPDATE` for the same Union-operator reason.

---

---

## Algorithms : how the operators actually work

Three of the planner's operators carry non-trivial algorithmic work :
the radius walk, the filtered ANN walk for plan C, and the
exact-distance scoring for plan B. The other operators are
straightforward dispatch ; these three are where the actual
correctness invariants live.

### Radius : doubling-`ef` until enclosure

The user asks "every id within distance r of q." HNSW does kNN, not
radius. The textbook way to bolt radius on is to add an
early-termination predicate to the layer-level walk : stop expanding
when the smallest candidate is farther than r.

That fails in practice when the entry point is outside the ball.
HNSW's entry point is whatever node landed on the highest layer at
insert time, which has nothing to do with the query. If the entry's
distance to q is already greater than r, the early-termination rule
fires immediately and returns nothing, even though hundreds of in-ball
points exist a few hops away.

Kova uses a different approach : run kNN with a doubling `ef` until
either some hit lies outside the radius (proof that the radius ball
is enclosed within the candidate set) or `ef` reaches the index size
(we have exhausted everything). Then filter by radius.

```text
fn search_radius(query, r):
    ef = max(ef_search, 16)
    loop:
        hits = search_layer(query, entry_points, ef, layer=0)
        if any hit has distance > r:           // the ball is enclosed
            return hits filter (d <= r)
        if ef >= index size:                   // we have everything
            return hits filter (d <= r)
        ef = min(ef * 2, index size)           // grow and retry
```

```mermaid
flowchart TD
    Start["search_radius(q, r)"] -.-> Init["ef = ef_search"]
    Init -.-> Run["search_layer(q, ef, layer=0)"]
    Run -.-> Outside{"any hit<br/>distance > r?"}
    Outside -. yes .-> Done["filter by r"]
    Outside -. no .-> Exhausted{"ef >=<br/>index size?"}
    Exhausted -. yes .-> Done
    Exhausted -. no .-> Grow["ef = ef * 2"]
    Grow -.-> Run
```

The cost is paid only when the ball is sparse : if the closest 16
neighbours are all inside the ball, we double once or twice before
seeing an outsider. The common case (a radius that captures a
reasonable slice of the local neighbourhood) terminates in one
iteration.

The correctness argument :

- `search_layer` returns its result set sorted ascending by distance.
- If any returned hit has distance > r, then every node in the index
  that lies inside the ball was either in the returned set or was
  closer to q than that outsider. Both are visible to us.
- If `ef` reaches index size, we have visited every reachable node ;
  filtering by r gives the exact answer.

What this loses compared to the textbook walk : a tiny bit of
efficiency when r is small enough that even `ef = 16` already exceeds
the in-ball population. The trade is full correctness across all
ball sizes against ~1.5x cost in the degenerate case. The recall
sweep validates the trade : every cell at every shape clears 0.9,
most at 1.0.

The implementation lives in [`hnsw/search.rs`](../crates/kova-index/src/hnsw/search.rs)
as `search_radius_impl`.

### Plan C : soft-filtering inside the HNSW walk

Plan C is the contribution that makes mid-selectivity workloads
viable. The naive options either overfetch (plan A, drops candidates
that fail the WHERE) or scan ahead-of-time (plan B, computes exact
distance on a curated id set). Both lose when the predicate matches
20-50% of the shard : overfetch starves the LIMIT because too many
candidates fail the WHERE, and ahead-of-time scanning produces too
many candidates for exact distance to beat the ANN.

Plan C threads the predicate **into** the HNSW graph walk. Concretely
it changes one rule in `search_layer` : only filter-passing nodes are
allowed into the results heap. Out-of-filter nodes are still
expanded, their neighbours are still added to candidates, the walk
still uses them as transit. They simply do not count toward the
top-k.

```text
fn search_layer_filtered(query, entry_points, ef, layer, filter):
    visited = {entry_points}
    candidates = min-heap of entry_points (by distance to query)
    results = max-heap of in-filter entry_points (by distance to query)

    while !candidates.empty():
        c = candidates.pop_smallest()

        // Termination only kicks in once results is full.
        // If results is short of ef, every candidate is worth
        // expanding (the next neighbour might be the first filter
        // match we see).
        if results.len() >= ef && c.distance > results.peek_worst().distance:
            break

        for n in c.neighbours[layer]:
            if n in visited: continue
            visited.add(n)
            n.distance = distance(query, n)

            // Expansion gating : only worth following if n could
            // contribute to results, OR if results isn't full yet.
            worst = results.peek_worst().distance if results.full else infinity
            if results.len() < ef || n.distance < worst:
                candidates.push(n)
                if filter(n):
                    results.push(n)
                    if results.len() > ef: results.pop_worst()
```

Two invariants make this work :

1. **Termination is gated on `results.len() >= ef`.** If the results
   heap is short of `ef`, every popped candidate matters because the
   next neighbour expansion might be the first filter match. Closing
   on `results.len() < ef && c.distance > results.peek_worst()`
   would short-circuit before any filter match shows up.
2. **Expansion still gates on distance.** Even though we keep
   visiting out-of-filter nodes for routing, we do not enqueue
   neighbours that are farther than the current worst result.
   Otherwise the candidate heap grows without bound.

```mermaid
flowchart TD
    Start["search_layer_filtered"] -.-> Init["seed visited + heaps with entry points"]
    Init -.-> Loop{"candidates<br/>empty?"}
    Loop -. yes .-> End["return results"]
    Loop -. no .-> Pop["pop smallest candidate c"]
    Pop -.-> Term{"results full<br/>AND c worse<br/>than results?"}
    Term -. yes .-> End
    Term -. no .-> Expand["expand c's neighbours"]
    Expand -.-> ForN["for each neighbour n"]
    ForN -.-> Filter{"filter(n)?"}
    Filter -. yes .-> AddResults["push n to results"]
    Filter -. no .-> Skip["skip results push"]
    AddResults -.-> AddCand["push n to candidates if<br/>n.distance < worst"]
    Skip -.-> AddCand
    AddCand -.-> Loop
```

The Phase 1 implementation passes the filter as a borrowed `&F`
where `F: Fn(VectorId) -> bool`. The `Shard` wrapper
(`search_filtered`) bridges from the engine's `FnMut(&Metadata) ->
bool` predicate to the index's `Fn(VectorId) -> bool` via
`RefCell`-wrapped interior mutability, looking up each id's metadata
inside the closure. The filter call cost dominates on tight
predicates (each filter call is a HashMap lookup + a predicate eval) ;
the recall sweep verifies this stays fast enough that plan C wins
its band.

Implementation : `search_filtered_impl` and `search_layer_filtered`
in [`hnsw/search.rs`](../crates/kova-index/src/hnsw/search.rs).

### Plan B : exact distance over a metadata-scanned id set

Plan B is the simplest of the three but worth pinning the shape :

```text
fn plan_b(predicate, query, k):
    ids   = shard.scan_metadata(predicate)        // O(N) walk, returns
                                                  // ids whose bag passes
    hits  = []
    for id in ids:
        v = shard.get_vector(id)
        d = distance(query, v)
        hits.push((id, d))
    hits.sort_ascending_by_distance()
    return hits.take(k)
```

The win is that no graph traversal happens. `scan_metadata` walks
the in-memory metadata HashMap directly, applying the predicate to
each row's bag. The result is a small candidate set ; exact distance
on it is `O(matches * dim)`. When matches is small, this beats both
the ANN and plan C's per-visit filter cost.

The closure-error-capture pattern keeps this safe : the predicate
might fail to evaluate on some row (e.g. a wrong-type param),
and a closure-`bool` interface can't propagate `Result`. The
executor captures the error into a `mut Option<KovaQueryError>`
outside the closure, has the closure return `false` after the first
error to short-circuit the remaining scan, then checks the option
after the scan returns. Same pattern is reused in `Count`,
`UpdateByPredicate`, and `DeleteByPredicate` ; it's the standard
shape any `FnMut(&Metadata) -> Result<bool>` substitute needs.

---

## Layer 4 : executor

The executor walks a `PhysicalPlan` against a `Shard` and produces an
`ExecutionResult`. One `Engine<D>` owns one `Shard` and dispatches
queries against it. The executor is the only layer that talks to
the storage primitives.

### Read path

Every read query is `Projection(... inner ...)` at the root.
`Projection` is the only operator that builds user-facing `Row`
values. Internal read operators (`KnnSearch`, `MetadataScan`,
`ExactDistance`, `Limit`, `FilteredKnnSearch`, `RadiusSearch`) flow
`Vec<InternalHit>` between themselves. The executor's `execute_read`
recursive walk is what does this.

```text
Projection
  Limit
    KnnSearch / FilteredKnnSearch / RadiusSearch
      OR
    ExactDistance
      MetadataScan
```

`InternalHit { id, distance: Option<f32>, metadata }` is the
internal currency. `KnnSearch` and `RadiusSearch` fill `distance`
from the HNSW walk ; `MetadataScan` leaves it `None` ; `ExactDistance`
computes it. The projection step turns this into typed `RowValue`
cells.

### Write path

`PhysicalPlan::InsertOne` / `InsertMany` / `DeleteById` /
`DeleteByParamId` / `DeleteByPredicate` / `DeleteByRadius` /
`UpdateById` / `UpdateByParamId` / `UpdateByPredicate` /
`UpdateByRadius` all sit at the top of `execute()`. Each delegates to
a per-arm `exec_*` helper. The DML arms share two reusable helpers :

- `apply_assignments(bag, assigns, params)` : mutate a metadata bag
  in place per an `UPDATE`'s `SET` clause. Handles both flat
  (`SET field = value`) and subscripted (`SET field['key'] = value`)
  shapes.
- `build_updates_from_ids(ids, assigns, params)` : fetch the current
  bag for each id, apply assignments to copies, return the staged
  batch ready for `Shard::update_metadata`.

The `RadiusOp` borrowed bundle struct is shared between
`exec_delete_by_radius` and `exec_update_by_radius`, keeping per-arm
helper signatures honest.

### Error model

`Engine::execute` returns one of four error kinds, and never panics
on bad input :

| Error | Source |
|-------|--------|
| `KovaQueryError::Parse` | Grammar didn't match |
| `KovaQueryError::Bind` | Semantic check failed in the binder |
| `KovaQueryError::Plan` | Shape the planner can't handle |
| `KovaQueryError::Execution` | Runtime issue : missing param, missing id, wrong type |
| `KovaQueryError::Backend` | Boxed error from the `Shard` layer (WAL, storage, index) |

The fuzzer (see Phase 1 status below) enforces this : 32k+ queries
across multiple seeds, zero panics.

### The result shape

`ExecutionResult` has one variant per operator family. Pattern-match
on it ; the variant tells you what came back :

```rust
pub enum ExecutionResult {
    Checkpoint { lsn: Lsn },
    Vacuum    { table: String, removed: usize },
    Insert    { table: String, inserted: u64 },
    Delete    { table: String, deleted: u64 },
    Update    { table: String, updated: u64 },
    Rows      { columns: Vec<String>, rows: Vec<Row> },
}
```

Reads always come back as `Rows`. The columns vector carries the
projected column names (in projection order) ; the rows vector
carries the cells. The cell type is :

```rust
pub enum RowValue {
    Id(VectorId),
    Distance(f32),
    Metadata(Metadata),
    Field(Value),
    Null,
}
```

| Cell | Produced when |
|------|---------------|
| `Id` | The projection includes `id` (or `*`, which expands to `[id, metadata]`) |
| `Distance` | The projection includes an `embedding <-> $q` expression. Set from the kNN walk (or ExactDistance in plan B), `None`-able only internally |
| `Metadata` | The projection includes `metadata` (the whole bag) |
| `Field(Value)` | The projection names a metadata field. Carries the raw `Value`, which is what's in the bag : `String` / `I64` / `F64` / `Bool` / `Array` / `Map` |
| `Null` | The projection asked for a field that isn't in this row's bag |

### Projection syntax

```sql
SELECT *                       FROM vectors ...   -- expands to [id, metadata]
SELECT id                      FROM vectors ...   -- one column, Id cells
SELECT id, metadata            FROM vectors ...   -- both
SELECT id, embedding <-> $q    FROM vectors ...   -- adds a Distance column
SELECT id, embedding <-> $q AS dist FROM vectors ...   -- aliased
SELECT category                FROM vectors ...   -- Field(Value::String) cells
SELECT category, year          FROM vectors ...   -- multiple Field columns
SELECT COUNT(*)                FROM vectors ...   -- solo : ExecutionResult::Rows with one I64 cell
SELECT COUNT(*) AS n           FROM vectors ...   -- column name follows the alias
```

The binder rejects mixing `*` with other items (`SELECT *, id` errors
cleanly), and the planner refuses `COUNT(*)` alongside any non-COUNT
column (no GROUP BY semantics in v1).

### Result examples

A `SELECT id, embedding <-> $1 AS dist FROM vectors ... LIMIT 2` :

```text
ExecutionResult::Rows {
    columns: ["id", "dist"],
    rows: [
        Row { values: [Id(VectorId(42)), Distance(0.013)] },
        Row { values: [Id(VectorId(17)), Distance(0.041)] },
    ],
}
```

A `SELECT COUNT(*) FROM vectors WHERE category = 'docs'` :

```text
ExecutionResult::Rows {
    columns: ["count"],
    rows: [
        Row { values: [Field(Value::I64(230))] },
    ],
}
```

A `DELETE FROM vectors WHERE category = 'old'` that matched three rows :

```text
ExecutionResult::Delete { table: "vectors", deleted: 3 }
```

A `CHECKPOINT` :

```text
ExecutionResult::Checkpoint { lsn: Lsn(40_312) }
```

The pattern is consistent : Rows for reads, table + count for DML,
table + side-effect-count for management, LSN for checkpoint. The
caller never has to translate "did it work" through stringly-typed
output ; the variant is the answer.

---

The pipeline and the algorithms cover how KQL answers a query. The
next several sections are **reference material** : the value types,
the parameter binding, the statement coverage matrix, and the result
shapes. These are the things you look up when you are actually
writing a query and need to remember the exact rules.

## Type system

The value universe is `kova_core::Value` :

```rust
enum Value {
    String(String),
    I64(i64),
    F64(f64),
    Bool(bool),
    Array(Vec<Value>),
    Map(HashMap<String, Value>),
}
type Metadata = HashMap<String, Value>;
```

Six variants, deliberately flat. The semantics for comparison and
equality are pinned :

- Same-type comparison delegates to the natural `PartialOrd` /
  `PartialEq` of the inner type.
- `I64` and `F64` are mutually coercible : `5 == 5.0` is true,
  `3 < 4.5` is true.
- All other cross-type comparisons evaluate to false. There is no
  silent string-to-number coercion.
- `Map` participates in equality (structural compare on the inner
  `HashMap`) but not ordering (no `<`/`>`/`BETWEEN` against a Map).

NULL has a single-shape semantic : a field that is absent from the
bag is equivalent to NULL. A literal `NULL` in the query is a
sentinel that compares unequal to every concrete value. `IS NULL` /
`IS NOT NULL` are the only predicates that test presence directly.

---

## Subscripted access

The `attrs['key']` shape works on both sides of the language :

- In `WHERE` clauses : every predicate atom (`Eq`, `Cmp`, `In`,
  `Between`, `IsNotNull`, `ArrayContains`) accepts a subscripted
  field reference. `WHERE attrs['country'] = 'IN'` looks up `attrs`
  in the row's metadata, expects a `Map`, and keys into it. Returns
  false silently if the field is missing or is not a Map ; same
  policy as a missing top-level field.

- In `SET` clauses : `SET attrs['priority'] = 5` creates an empty
  `Map` at `attrs` if absent, then writes the keyed value. If
  `attrs` is present but not a `Map`, the update errors loudly
  rather than overwrite typed user data.

Only one level of nesting is supported. `WHERE attrs['a']['b']` and
`SET attrs['a']['b'] = ...` are not in the grammar.

---

## Parameters

Two binding modes : positional (`$1`, `$2`) and named (`$query`,
`$radius`). Both resolve to the same `ParamValue` enum :

```rust
enum ParamValue {
    String(String), I64(i64), F64(f64), Bool(bool), Null,
    Id(VectorId),
    Vector(Vector),
    Metadata(Metadata),
    Batch(Vec<(VectorId, Vector, Metadata)>),
}
```

The first five are literal-shaped, accepted anywhere a literal would
be in the language. The last four are reserved for specific syntactic
slots : `Id` for `WHERE id = $1`, `Vector` for `ORDER BY embedding
<-> $1`, `Metadata` for `SET attrs = $1`, `Batch` for
`INSERT INTO vectors VALUES $1`. Using the wrong type in the wrong
slot surfaces as a clean `Execution` error.

```rust
let params = ParamBindings::empty()
    .with_positional(ParamValue::Vector(query_vec))
    .with_named("radius", ParamValue::F64(0.5));
```

---

## Statement coverage in Phase 1

Everything in this table works end to end :

| Statement | Shape | Notes |
|-----------|-------|-------|
| `SELECT id, embedding <-> $q AS distance FROM vectors WHERE pred ORDER BY embedding <-> $q LIMIT k` | kNN with predicate | Plan A/B/C dispatch |
| `SELECT id FROM vectors WHERE pred LIMIT k` | scan-and-limit | Order is implementation-defined |
| `SELECT id FROM vectors WHERE embedding <-> $q < r [AND pred]` | radius | Optional AND-residue as post-filter |
| `SELECT COUNT(*) FROM vectors [WHERE pred]` | aggregate bypass | One row, one column |
| `INSERT INTO vectors (id, embedding, metadata) VALUES ($1, $2, $3)` | single | All three are params |
| `INSERT INTO vectors VALUES $batch` | batch | Param is `ParamValue::Batch` |
| `DELETE FROM vectors WHERE id = <literal>` | fast | `Shard::delete` |
| `DELETE FROM vectors WHERE id = $1` | fast | Param-resolved |
| `DELETE FROM vectors WHERE pred` | scan | `delete_many` over the matched id set |
| `DELETE FROM vectors WHERE embedding <-> $q < r [AND pred]` | radius | Same `RadiusOp` shape as SELECT |
| `UPDATE vectors SET f = v [, ...] WHERE id = <literal>` | fast | Single id, multiple SET clauses |
| `UPDATE vectors SET f = v WHERE id = $1` | fast | Param-resolved |
| `UPDATE vectors SET f = v WHERE pred` | scan | Mass update over a predicate |
| `UPDATE vectors SET f = v WHERE embedding <-> $q < r [AND pred]` | radius | |
| `UPDATE vectors SET attrs['key'] = v WHERE ...` | subscripted | Lands the value in a nested Map |
| `UPDATE vectors SET attrs = $1 WHERE id = $2` | bag-replace | `$1` is `ParamValue::Metadata` |
| `VACUUM vectors` | management | Bridges to `Shard::vacuum` |
| `CHECKPOINT` | management | Bridges to `Shard::checkpoint` |

What is **not** in Phase 1 :

| Shape | Why deferred |
|-------|--------------|
| `CREATE INDEX`, `DROP INDEX` | No secondary indexes in Phase 1 |
| `OR` with a distance threshold | Needs a Union operator (Phase 2) |
| `WHERE attrs['a']['b'] = ...` | Multi-level subscripts ; grammar would need to grow |
| Multi-table queries, JOIN | Out of language scope (single `vectors` table) |
| Transactions across statements | Each statement is its own commit |

---

Reference covered. The next section is the case for trusting any of
this : what the test surface looks like, what numbers it produces,
and how the fuzzer that produces them actually works.

## What Phase 1 ships strong

Phase 1 has three load-bearing guarantees, each backed by mechanical
verification :

**No-panic across the language surface.** The fuzzer
([`fuzz_query.rs`](../crates/kova-query/tests/fuzz_query.rs))
generates random grammar-conformant queries with random parameters
against a randomly-seeded shard and asserts the pipeline either
succeeds or returns a typed error. 32,000+ queries across multiple
seeds, zero panics. A typed error is a contract ; a panic is a
bug.

**Correctness for the deterministic shapes.** The same harness runs
a second pass with a reference implementation built into the test
crate. For `COUNT(*)`, scan-and-limit, `DELETE` by id or predicate,
and `UPDATE` by id or predicate, the reference computes the
expected result by hand-walking the seeded rows ; the engine and
reference must agree on every iteration. 12,000+ queries, zero
divergence.

The reference-first design (compute expectation, then execute) was
itself caught by the fuzzer : an earlier version let the engine run
ahead and the reference fell behind on uncheckable shapes. The bug
was in the harness, not the engine, but the fix is the fuzzer
methodology Phase 2 inherits.

**Recall measured, not assumed.** The HNSW recall sweep
([`hnsw/search.rs`](../crates/kova-index/src/hnsw/search.rs))
parametrises across `(n, dim, k, selectivity, radius)` and asserts
every cell clears 0.9. Today's baseline :

| Shape | Configuration | Recall |
|-------|---------------|--------|
| kNN @10 | n=500, dim=4 | 1.000 |
| kNN @10 | n=500, dim=16 | 1.000 |
| kNN @10 | n=2k, dim=8 | 1.000 |
| kNN @10 | n=2k, dim=32 | 0.985 |
| kNN @10 | n=10k, dim=16 | 0.990 |
| Filtered @10 | n=500, dim=8, keep=0.5 | 1.000 |
| Filtered @10 | n=2k, dim=16, keep=0.2 | 1.000 |
| Filtered @10 | n=2k, dim=16, keep=0.5 | 1.000 |
| Filtered @10 | n=2k, dim=16, keep=0.8 | 1.000 |
| Filtered @10 | n=5k, dim=16, keep=0.3 | 1.000 |
| Radius | n=500, dim=4, r=0.30 | 1.000 |
| Radius | n=2k, dim=16, r=0.60 | 1.000 |
| Radius | n=5k, dim=16, r=0.50 | 1.000 |
| Radius | n=5k, dim=32, r=1.00 | 1.000 |

12 of 14 cells perfect ; the two below 1.0 are the high-dim kNN
ones where the curse of dimensionality is the natural enemy of
ANN. Filtered and radius recall hold at 1.0 even at tight selectivity
(20% kept) and high dim. Any future change that drops a cell below
0.9 fails the test with the specific cell named.

---

## How the fuzzer works

The fuzzer is the methodology that backs every claim above. It is
deterministic given a seed, hand-rolled (no proptest / quickcheck
dependency), and has two phases that build on each other.

### Phase A : no-panic across the language surface

Phase A generates random grammar-conformant queries and pushes them
through `Engine::execute_str`. It asserts the call either succeeds
or returns one of the typed `KovaQueryError` variants. Any panic is
a test failure with the seed + the panicking query printed verbatim
so the run can be reproduced.

The generator is split by statement kind. SELECT dominates (50% of
generated queries) because it has the most internal surface ; DELETE
and UPDATE each get a quarter ; VACUUM and CHECKPOINT split the
remaining 5%. Within each statement, the generator picks a shape
proportional to how much code path it exercises :

```mermaid
flowchart TD
    Any["gen_any_query"]
    Any -. 50% .-> Sel["gen_select"]
    Any -. 25% .-> Del["gen_delete"]
    Any -. 20% .-> Upd["gen_update"]
    Any -. 3% .-> Vac["VACUUM"]
    Any -. 2% .-> Chk["CHECKPOINT"]

    Sel -. 15% .-> SC["COUNT(*)"]
    Sel -. 15% .-> SS["scan-and-limit"]
    Sel -. 15% .-> SR["radius"]
    Sel -. 55% .-> SK["kNN"]
```

Predicates are generated recursively up to depth 3, mixing every
atom kind (`=`, `<`, `IN`, `BETWEEN`, `IS NULL`, `@>`) with
AND/OR/NOT combinators. Field references draw from a small fixed
universe of names ; 25% of field references are subscripted
(`attrs['key']`) so the Map-side surface gets exercised.

Values are drawn from realistic shapes : short strings from a
6-element pool, integers in `[0, 3000)`, floats in `[0, 10)`,
booleans, and `NULL`. The metadata bag generator writes each
candidate field with biased probability (`active` 60%, `category`
90%, etc.) so predicates land on both present and absent fields.

The shard fixture pre-seeds N random rows before the loop starts.
That gives the executor real data to walk : an empty shard would
silently fast-path every SELECT to "no results" and miss most of
the executor's interesting branches.

### Phase B : correctness against a reference implementation

Phase A only proves the engine doesn't crash. Phase B adds a second
implementation we run alongside the engine for the deterministic
shapes (COUNT, scan-and-limit, DELETE, UPDATE) and asserts they
agree on every iteration.

The reference impl is a hand-rolled predicate evaluator + assignment
applier that operates on the test's snapshot of `Vec<(VectorId,
Metadata)>`. It mirrors the engine's semantics for every atom kind,
the same subscript lookup, the same null handling, the same numeric
coercion. The two implementations are independent : if both agree
on the same input, both are probably right.

The harness ordering is **reference-first, engine-second**. This
matters and was load-bearing in a way that took finding the hard
way :

```text
for each iteration:
    sql, params = generate()
    stmt        = parse_and_bind(sql)
    kind        = classify(stmt)               // which check applies

    if kind has no check:
        if kind is mutation:
            skip iteration entirely
        else:
            run engine for panic coverage only
        continue

    expectation = compute_via_reference(kind, snapshot, params)
    if expectation unavailable:
        skip iteration entirely                // do not touch engine

    engine_result = engine.execute_str(sql, params)
    assert engine_result agrees with expectation
    if mutation: sync snapshot to match
```

The key choice is the **skip without running** branch. An earlier
version of the harness fell back to a generic "execute for panic
coverage" call on shapes it couldn't check. That call generated its
own random query and executed it through the engine, silently
mutating shard state that the snapshot never saw. After a few dozen
iterations, the snapshot held rows that no longer existed in the
engine. The next correctness check fired a "engine deleted 0,
expected 10" assertion and looked like an engine bug.

It wasn't. The harness was lying. The fix : never run a side-effecting
query through the engine unless either the reference can verify it
or it's known to be read-only. The fuzzer found a fuzzer bug, which
is its second job after finding real bugs.

### Counts

| Run | Seeds | Iterations per seed | Total queries |
|-----|-------|---------------------|---------------|
| `fuzz_smoke_*` | 4 | ~500 | 1,500+ |
| `fuzz_long_run` (ignored) | 16 | 2,000 | 32,000 |
| `correctness_fuzz_*` | 2 | 500 | 1,000 |
| `correctness_fuzz_long_run` (ignored) | 8 | 1,500 | 12,000 |

Per CI cycle (smoke variants only) about 2,500 queries flow through
the engine. The long-run variants push 44,000+ when invoked
explicitly. Across all of those, every COUNT, scan, DELETE, and
UPDATE that the reference picked up agreed with the engine.

### What the fuzzer doesn't catch

A few categories are out of scope by construction :

- **Approximate operators.** kNN and radius are deterministic given
  the same seed but their result *content* depends on HNSW's
  approximation. The fuzzer asserts no panic on those, but does
  not compare results against a brute-force reference. The recall
  sweep above is what does that.
- **Performance regressions.** No timing assertions. A slow query
  is still a passing query. Phase 2's M2.7 milestone is where
  latency baselines land.
- **Multi-query coherence.** Each iteration is independent. The
  fuzzer does not check that a sequence of queries preserves some
  invariant across them (other than the implicit "reference and
  engine state stay in sync" check).

What it does catch : every panic, every typed-error contract
violation, every disagreement with the reference on a checkable
shape. The harness is in [`tests/fuzz_query.rs`](../crates/kova-query/tests/fuzz_query.rs)
if you want to read the actual generators.

---

The case for trust covered. The last substantive section is the
case for staying : what Phase 2 changes, and how it manages to do so
without breaking anything Phase 1 already ships. If you are picking
up KQL now, this is what tomorrow looks like.

## What Phase 2 builds on top

Phase 2 is where KQL goes from "works" to "fast at scale on
predicate-heavy workloads." The shape is :

```mermaid
flowchart LR
    P1["Phase 1<br/>full-scan predicates,<br/>flat estimator,<br/>fixed plan bands"]
    P1 -. same grammar,<br/>same operators,<br/>swap impls behind traits .-> P2["Phase 2<br/>real indexes,<br/>histogram stats,<br/>cost-model planner"]
```

Every Phase 2 milestone is a trait-impl swap behind a stable
interface. No grammar change. No new operators on the read path.
Predicates already parsed today by Phase 1 will plan differently
in Phase 2 and execute faster, but the same query string keeps
working.

### M2.1 : secondary indexes + `RoaringBitmap` composition

Today's `MetadataScan` is an O(N) walk over the metadata store. M2.1
ships `HashIndex` (eq), `BTreeIndex` (range), and `InvertedIndex`
(array containment) for the metadata fields users care about, plus
`RoaringBitmap` composition so `field_a = X AND field_b > Y`
intersects the two bitmaps in microseconds instead of re-walking
the full shard.

The `MetadataStore` trait grows index hooks. `Shard::scan_metadata`
gains an index-driven fast path. Predicate evaluator stays the same.
Queries that hit indexed fields drop from O(N) to O(matches +
log(N)).

### M2.2 : index sidecar persistence

Built indexes live in their own snapshot files alongside the graph
snapshot. The manifest's commit point grows two more lines : index
generation and version. Crash safety story is identical to today's
graph snapshot, same generation-numbered files, same atomic
manifest swap.

### M2.3 : `CREATE INDEX` / `DROP INDEX` DDL

The grammar already accepts `CREATE INDEX foo ON vectors USING
HASH (category)` and `DROP INDEX foo`. The binder rejects both
today. M2.3 unrejects them : `LogicalCreateIndex` / `LogicalDropIndex`
flow through to a new `PhysicalPlan::CreateIndex` /
`PhysicalPlan::DropIndex`, executor builds or drops the sidecar.

### M2.4 : statistics + histograms

The `SelectivityEstimator` trait grows a histogram-backed impl.
Equi-depth histograms per indexed field, refreshed at checkpoint.
Estimates that were O(N) (`ShardEstimator` walks every row) become
O(log buckets). Estimates that are currently exact become
approximate but vastly cheaper.

### M2.5 : index-driven `MetadataScan` dispatch

The executor's `MetadataScan` arm starts consulting the index
catalog. Predicates with indexed atoms route through the bitmap
path ; non-indexed atoms still walk the metadata store. Mixed
predicates (`category = 'docs' AND priority > 5` when `category`
is indexed but `priority` is not) intersect the bitmap with the
scan output.

### M2.6 : cost-model planner

The hardcoded `PLAN_B_UPPER = 0.05` and `PLAN_A_LOWER = 0.5` bands
become a real cost model. The `dispatch_knn_plan` function grows
inputs for `k`, target recall, shard size, and the histogram-backed
selectivity estimate. The bands are computed from measured cost
coefficients, not constants.

The decision-grid test from Phase 1 keeps protecting the boundary
behaviour ; it just becomes parametric over the cost coefficients.

### M2.7 : benchmarks + public design notes

Once Phase 2's machinery is in place, the same recall sweep that
Phase 1 published gets a latency column. Numbers go out as part of
a public benchmarking writeup.

### What does not change in Phase 2

The KQL grammar, the AST, the LogicalStatement shape, the
PhysicalPlan operator set on the read path, the `ExecutionResult`
shape, the parameter binding model, the error taxonomy. Phase 1
queries are Phase 2 queries with no rewrite. The fuzzer Phase 1
already runs is the same fuzzer that pins Phase 2 against
regressions ; nothing in its harness assumes Phase 1's plan-choice
constants.

---

## Caveats and load-bearing constraints

A few things worth knowing before you build on KQL.

**One table.** The grammar carries the table name through
(`AstVacuum { table: "my_shard" }`, `LogicalVacuum { table }`, and so
on) precisely so that multi-table is a runtime config change later,
not a language change. But Phase 1 only validates against the
`Engine`'s single `Shard`. Multi-shard fan-out lives in the future
cluster layer.

**Param-id resolution is positional.** `WHERE id = $1` expects
`ParamValue::Id` ; `WHERE id = $1` with `ParamValue::I64(1)` will
silently produce zero matches because the metadata-side `id` field
does not exist. The right type is the discipline.

**Distance is always ascending.** `ORDER BY embedding <-> $q DESC`
is rejected at the binder. kNN by definition picks the smallest
distances, and "k farthest neighbours" is not in scope.

**Subscript depth is one.** `attrs['a']['b']` does not parse. If
you need it, flatten the schema or wait for the grammar to grow it.

**Updates are metadata-only.** `UPDATE vectors SET embedding = $1`
is rejected at the binder. Vector positions are graph-structural ;
delete and re-insert.

---

## Pointers

| What | Where |
|------|-------|
| Grammar | [`grammar.pest`](../crates/kova-query/src/grammar.pest) |
| AST | [`ast.rs`](../crates/kova-query/src/ast.rs) |
| Binder | [`binder.rs`](../crates/kova-query/src/binder.rs) |
| LogicalStatement | [`logical.rs`](../crates/kova-query/src/logical.rs) |
| Planner | [`planner.rs`](../crates/kova-query/src/planner.rs) |
| PhysicalPlan | [`physical.rs`](../crates/kova-query/src/physical.rs) |
| Executor | [`executor.rs`](../crates/kova-query/src/executor.rs) |
| Printer | [`printer.rs`](../crates/kova-query/src/printer.rs) |
| Fuzzer | [`tests/fuzz_query.rs`](../crates/kova-query/tests/fuzz_query.rs) |
| Recall sweep | [`hnsw/search.rs`](../crates/kova-index/src/hnsw/search.rs) |

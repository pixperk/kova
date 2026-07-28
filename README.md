# Kova

A distributed vector database, built from scratch in Rust.

Hand-rolled HNSW index, memory-mapped storage with write-ahead logging,
a SQL-inspired query language (KQL), and a shard-replicated cluster layer
with gRPC between nodes. Every byte, every index, every network call is ours.

> *"kova"* - the Turkish word for a bucket, a hive, a vessel that holds what
> matters. Where bees store honey, Kova stores vectors.

## Workspace

| Crate          | Status      | What it provides                                                       |
| -------------- | ----------- | ---------------------------------------------------------------------- |
| `kova-core`    | shipped     | `Vector`, `Distance` trait + `Cosine` / `L2` / `InnerProduct` (SIMD)   |
| `kova-index`   | shipped     | `Index` trait, `FlatIndex` baseline, `HnswIndex` (insert + search + two-pass vacuum + streaming snapshot) |
| `kova-storage` | shipped     | Segmented WAL, `MmapVectorStore` (free-list slot reuse, crash-mid-init recovery), `FileMetadataStore`, `atomic_write` (+ streaming variant), `Manifest` commit point, `Shard` (log-then-mutate, batched inserts, logical delete, vacuum, checkpoint + WAL truncate, generation-numbered snapshots). SIGKILL-tested across 55,697 acks and 962 checkpoints. |
| `kova-query`   | shipped     | KQL end to end : parser (Pest grammar, full DML + DDL), binder (typed `LogicalStatement` with hard semantic rejections), planner (three SELECT strategies dispatched by a measured cost model with a correctness gate, plus radius, COUNT and scan-and-limit bypasses), executor wired to `Shard` for both read and write paths, full DML (INSERT / UPDATE / DELETE including subscripted, radius, and param-bound shapes), `CREATE INDEX` / `DROP INDEX` end to end via the `IndexCatalog`, `Value::Map` for nested metadata, probabilistic query fuzzer with reference-impl correctness check, recall regression sweep across kNN / filtered / radius. See [`docs/query.md`](docs/query.md). |
| `kova-meta-index` | shipped | Secondary indexes on metadata fields : `HashIndex` (equality), `BTreeIndex` (range, with float-ordering gate), `InvertedIndex` (array containment). `RoaringTreemap`-backed bitmaps compose for AND / OR / NOT in microseconds. `IndexCatalog` orchestrates per-field bundles, routes lookups by priority, exposes exact cardinality for the planner. Catalog persists alongside the graph snapshot, generation-numbered like `graph.{N}.snapshot`. WAL records carry `old_metadata` so replay rebuilds the catalog without depending on the (eagerly-mutated) metadata store. See [`docs/meta-index.md`](docs/meta-index.md). |
| `kova-cluster` | not started | Consistent hashing, quorum replication, coordinator                    |
| `kova-server`  | not started | gRPC node binary                                                       |

802 lib + integration tests passing across the workspace (covering meta-index unit tests, catalog round-trip + persistence + post-checkpoint replay, plus the existing HNSW / WAL / shard / query suites). The full fuzz long-run pushes 12k+ correctness-checked queries and 32k+ no-panic queries per CI cycle. `cargo clippy --workspace --all-targets -- -D warnings` clean.

## Build

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo bench -p kova-core
```

## Features

**Index**

- **HNSW**, hand-rolled. Default params (`M=16`, `ef_construction=200`,
  `ef_search=50`) hit recall > 0.9 at 50k vectors without tuning.
- **Two-pass vacuum** physically removes tombstoned nodes and repairs
  the neighbour lists of every survivor that pointed at one. Bounded
  by the M/2 skip rule so well-connected nodes don't pay for re-search.
- **Streaming snapshot** (`write_snapshot`/`read_snapshot`) serializes
  the whole graph through a `Write` so checkpoints don't double-buffer.
- **`FlatIndex` baseline** shares the `Index` trait, used as ground
  truth for recall validation.

**Distance**

- **SIMD** via `wide::f32x8` for `L2`, `Cosine`, `InnerProduct`. 4x-17x
  over scalar; falls back to scalar where 8-lane f32 isn't available.

**Storage**

- **Memory-mapped vector store** with fixed-stride self-describing
  slots (`16 + dim*4` bytes). Free-list reuses freed slots so deletes
  followed by inserts don't grow the file. Crash-mid-init recovery on
  reopen.
- **Segmented WAL** with 64 MB rotation, length + CRC32 framing,
  torn-tail recovery, and O(1) truncation by deleting superseded
  segments.
- **File-backed metadata** with open-shaped attribute bags
  (`String` / `I64` / `F64` / `Bool` / `Array`), full-file snapshot on
  mutation via `atomic_write`.
- **Atomic file replacement** (tmp + fsync + rename + dirsync), both
  buffer and streaming variants.

**Shard (composition)**

- **3-phase ops** (validate, commit, apply) with pre-commit validation
  and panic-on-apply-failure for honest WAL semantics.
- **Batched inserts** with group-commit fsync. N records share one
  durability barrier and one metadata flush.
- **Logical delete + vacuum**: O(1) tombstones, search-time filter,
  vacuum physically removes them and re-enables id reuse.
- **Checkpoint + WAL truncate**: stop-the-world snapshot of the index,
  manifest commits the new generation atomically, WAL truncates past
  the checkpoint LSN. Generation-numbered snapshots make the swap
  crash-safe.
- **SIGKILL-tested**: 300 torture iterations, 55,697 acked inserts,
  962 checkpoints, zero failures. See [Crash recovery](#crash-recovery).

**Extensibility**

- `VectorStore`, `MetadataStore`, and `Wal` are traits with associated
  `Error` types, so a future S3-backed log, columnar metadata, or
  distributed store drops in at the trait level without touching `Shard`.

## Distance benchmarks

SIMD-accelerated `Distance` impls in `kova-core` (`wide::f32x8`, 8-wide).
Criterion mean, single core. The scalar baseline is shown for context.

| Metric          | dim   | Scalar   | SIMD    | Speedup |
| --------------- | ----- | -------- | ------- | ------- |
| `L2`            |   128 |  110 ns  |  24 ns  |  4.6x   |
| `L2`            |   768 |  718 ns  |  90 ns  |  8.0x   |
| `L2`            | 1,536 | 1,430 ns | 171 ns  |  8.4x   |
| `Cosine`        |   128 |  267 ns  |  34 ns  |  7.9x   |
| `Cosine`        |   768 | 1,950 ns | 119 ns  | 16.4x   |
| `Cosine`        | 1,536 | 3,870 ns | 220 ns  | **17.6x** |
| `InnerProduct`  |   128 |   95 ns  |  15 ns  |  6.4x   |
| `InnerProduct`  |   768 |  668 ns  |  75 ns  |  8.9x   |
| `InnerProduct`  | 1,536 | 1,310 ns | 151 ns  |  8.7x   |

`L2` and `InnerProduct` get the raw 8-wide SIMD benefit, scaling from
~5x at dim 128 up to ~8-9x at dim 1,536. `Cosine` is roughly double
that (peaking at ~18x) because the SIMD pass also folded `dot`, `|a|^2`,
and `|b|^2` into a single loop instead of three separate passes.

## HNSW vs Flat (dim 32, k=10, L2)

`HnswIndex` against the `FlatIndex` brute-force baseline. HNSW uses default
`HnswParams` (`M=16`, `ef_construction=200`, `ef_search=50`). Criterion mean,
single core, **SIMD distance**.

| N       | `FlatIndex.search` | `HnswIndex.search` | HNSW speedup |
| ------- | ------------------ | ------------------ | ------------ |
|   1,000 |  11 us             |  62 us             |  0.17x       |
|  10,000 | 119 us             | 122 us             |  1.0x        |
| 100,000 | 4.9 ms             | 312 us             | **~16x**     |

SIMD raises *both* lines on this table, but flat benefits more : its inner
loop is *just* distance computation, while HNSW spends most of its time on
graph traversal (HashMap lookups, heap operations). The HNSW crossover point
shifts to roughly 10k vectors with SIMD; below that, the linear scan wins.
At 100k HNSW is **~16x** ahead, and the gap keeps growing with N.

| Operation                               | Latency |
| --------------------------------------- | ------- |
| `HnswIndex.insert` into 1k-vector index | 265 us  |

Run the 100k benches yourself: `cargo bench -p kova-index --bench hnsw -- at_100k`.
The 100k build alone takes ~2-3 minutes.

## Recall validation

`HnswIndex` is correctness-tested against `FlatIndex` (ground truth) on
random uniform workloads. The parametrised sweep
[`recall_sweep_baseline`](crates/kova-index/src/hnsw/search.rs) runs three
recall variants across a grid and asserts every cell clears 0.9. Today's
baseline :

| Shape         | Configuration                  | Recall    |
| ------------- | ------------------------------ | --------- |
| kNN @10       | n=500, dim=4                   | **1.000** |
| kNN @10       | n=500, dim=16                  | **1.000** |
| kNN @10       | n=2k, dim=8                    | **1.000** |
| kNN @10       | n=2k, dim=32                   | 0.985     |
| kNN @10       | n=10k, dim=16                  | 0.990     |
| Filtered @10  | n=500, dim=8, keep=0.5         | **1.000** |
| Filtered @10  | n=2k, dim=16, keep=0.2 (tight) | **1.000** |
| Filtered @10  | n=2k, dim=16, keep=0.5         | **1.000** |
| Filtered @10  | n=2k, dim=16, keep=0.8         | **1.000** |
| Filtered @10  | n=5k, dim=16, keep=0.3         | **1.000** |
| Radius        | n=500, dim=4, r=0.30           | **1.000** |
| Radius        | n=2k, dim=16, r=0.60           | **1.000** |
| Radius        | n=5k, dim=16, r=0.50           | **1.000** |
| Radius        | n=5k, dim=32, r=1.00           | **1.000** |

12 of 14 cells perfect ; the two that drop are the high-dim kNN ones, where
the curse of dimensionality is the natural enemy of ANN. Filtered and
radius recall hold at 1.0 even at the tight-selectivity (20% kept) and
high-dim ends. Any future change that drops a cell below 0.9 fails the
test with the specific cell named, not a vague "recall regressed."

Plus a 50k-vector torture variant kept as `#[ignore]`'d :

```sh
cargo test -p kova-index --release -- --ignored recall_at_10_vs_flat_on_50k_dim32
```

All numbers use SIMD distance (`wide::f32x8`). The `wide` crate falls back
to scalar on platforms without 8-lane f32 SIMD, so this builds and runs
everywhere.

## Crash recovery

`Shard` is the composition layer that ties `Wal + VectorStore +
MetadataStore + HnswIndex` together under a strict log-then-mutate
discipline. The premise is that it must survive unplanned process
termination : every insert the caller saw acknowledged must be present
after reopen, every insert never acknowledged is the caller's problem.

We validate this directly with a `SIGKILL` torture test. A parent
process spawns a child binary (`crash_writer`) that opens a shard and
inserts vectors as fast as it can, printing `ACKED <id>\n` after every
insert that returned Ok. The parent waits a randomised delay, sends
`SIGKILL`, drains the child's stdout pipe, then reopens the shard on
the same directory and asserts every ACKed id is durably present.

The load-bearing invariant is the ACK ordering :

```text
   shard.insert ─── wal.sync ──>  durable on disk
                                 │
                                 ▼
                          writeln + flush ──>  ACK delivered to parent
                                 │
                       ─── kill can land here ───
```

`shard.insert` returns Ok only after `wal.sync` succeeds, so the moment
the parent reads `ACKED <id>` the record is fsynced. Pipe contents are
preserved across `SIGKILL` on Linux, so the parent sees exactly what
the child flushed. The reverse (every durable insert was ACKed) is
*not* required and does not hold - the kill can land between
`wal.sync` and the next `flush`, leaving a durable insert the parent
never knew about. The test correctly tolerates extras ; the only thing
it refuses to tolerate is a missing ACKed record.

Two flavours of torture. The **`inserts_only`** variant exercises the
insert path's crash windows (WAL append/sync, index.insert, metadata.put).
The **`with_checkpoints`** variant interleaves `Shard::checkpoint` calls
every 25 inserts, so the same SIGKILL roulette also lands inside vacuum,
snapshot write, manifest commit, WAL truncate, and old-snapshot delete
windows. The smoke versions of both run by default in `cargo test` ; the
150-iteration torture versions are `#[ignore]`'d.

| Test                                       | Iterations | Acks   | Checkpoints | Failures | Runtime (release) |
| ------------------------------------------ | ---------- | ------ | ----------- | -------- | ----------------- |
| `crash_recovery_smoke_inserts_only`        |          5 |   ~750 |           0 |        0 | ~3s               |
| `crash_recovery_smoke_with_checkpoints`    |          5 |   ~600 |         ~25 |        0 | ~3s               |
| `crash_recovery_torture_inserts_only`      |        150 | 30,038 |           0 |        0 | ~125s             |
| `crash_recovery_torture_with_checkpoints`  |        150 | 25,659 |         962 |        0 | ~127s             |

The checkpointed torture additionally asserts, on every iteration that
saw a successful checkpoint, that the on-disk `manifest`'s `snapshot_id`
matches `n_checkpointed` or `n_checkpointed + 1` (the +1 covers the
small window where the manifest commits but the child gets killed
before its post-commit `CHECKPOINTED <lsn>` print reaches the parent's
pipe : the manifest is the durable commit point ; the print is
best-effort observability).

Run them with :

```sh
cargo test -p kova-storage --test crash_recovery
cargo test -p kova-storage --release --test crash_recovery -- --ignored --nocapture
```

Kill delays are deterministic. The smoke variants use a single-bucket
schedule (`1 + (iter * 173) % 1500` ms) ; the torture variants split
into three buckets (early/mid/late) so the 150 iterations spread across
the full `1..3000` ms range, hitting every plausible point in the
child's lifecycle including post-checkpoint cleanup windows.

### Known limitation : SIGKILL is not power loss

The Linux page cache survives `SIGKILL`. Even un-fsynced mmap writes to
`vectors.mmap` remain visible on reopen because the kernel still has
the dirty pages. True power-loss recovery (where the page cache itself
disappears) needs either a VM snapshot/restore harness or explicit
`msync` on the write path. Neither is in place today. The fix tracks
the "checkpoints + log truncation" milestone, where we'll start
`msync`'ing mmap pages on critical boundaries.

## Design notes

A few architectural decisions worth calling out, both for future-me reading
this six months from now and for anyone trying to follow the code.

### Distance is a trait, not a function

`Distance` is `Send + Sync + 'static` so the metric type composes into
trait objects shared across threads. Concrete impls (`L2`, `Cosine`,
`InnerProduct`) are zero-sized unit structs. `HnswIndex` and `FlatIndex`
are both generic over `D: Distance` so the same code serves every metric
without dispatch overhead : the compiler monomorphises per metric.

The convention is `smaller = closer`. `Cosine` returns `1 - cos_similarity`
(range `[0, 2]`) so HNSW's min-heaps order correctly without per-metric
special-casing. `InnerProduct` is negated for the same reason.

### Vectors live in a `VectorStore`, not in HNSW nodes

`HnswIndex<D, V: VectorStore>` is generic over a storage backend. Nodes
hold *graph structure only* (neighbour lists per layer) ; the actual
vector bytes live in the composed `V`. Distance computations during
search/insert go through `self.vectors.get(id)`.

Why : storage strategy becomes pluggable. The same `HnswIndex` runs on
top of an `InMemoryVectorStore` (HashMap, default), the eventual
`MmapVectorStore` (file-backed, zero-copy reads), or a future
distributed store, without any HNSW changes. Vectors live in exactly one
place : no risk of drift between an in-memory copy and a persisted copy.

The trade-off : `VectorStore::get` returns owned `Vector` (clones from
the underlying storage). At realistic embedding dimensions (768-1536),
the per-clone allocation adds up. If benchmarks show it dominating, the
fix is to switch `get` to return `&[f32]` and refactor `Distance` to
accept slices instead of `&Vector` ; ~2 hours of mechanical work.
Deferred until measurements justify it.

`FlatIndex` is intentionally not refactored : as the brute-force
correctness baseline, it owns its vectors directly. The asymmetry
reflects different roles, not inconsistent design.

### Storage traits use associated error types ; `Shard` boxes them

`VectorStore`, `MetadataStore`, and `Wal` all expose an associated
`type Error : std::error::Error + Send + Sync + 'static`. Concrete impls
pick their own error universe : the file/mmap impls return
`KovaStorageError`, in-memory impls return `Infallible`, and a future
S3-backed `Wal` or distributed-log impl returns whatever shape its
backend naturally produces.

Why not pin every impl to one concrete error type :

- `KovaStorageError` is filesystem-shaped (`Io`, `CorruptRecord`,
  `Decode`, ...). It has nothing to say about S3 throttling, GCS
  service errors, or a distributed-log quorum failure. Forcing those
  into `KovaStorageError` would either grow the enum with cloud
  variants `kova-storage` has no business knowing about, or stuff
  everything into a single `Io(io::Error)` and lose the structure.
- Associated error types make `kova-storage` agnostic to the backends
  composed under it. The trait commits to a contract (`Error + Send +
  Sync + 'static`), not to a concrete enum.

`Shard<D, V, M, W>` is generic over all four primitives so the same
struct serves production (`FileWal + MmapVectorStore + FileMetadataStore`),
tests (`InMemoryWal + InMemoryVectorStore + InMemoryMetadataStore`), and
future swaps without code changes. The three backend error types
converge at the `Shard` seam via `Box<dyn Error + Send + Sync>` : not
because we want to throw away type information, but because the
composition layer genuinely cannot enumerate errors from backends it
doesn't know about. The error chain is preserved ; callers that need
the original type can `downcast` on the boxed source.

The cost is a small heap allocation per error path (cold). The win is
that adding a new backend is a trait impl, not a `kova-storage` patch.

### `MmapVectorStore` slots are self-describing, no sidecar

Each slot in the mmap file carries its own header : an 8-byte id, a
present flag, and padding to keep the vector bytes aligned. The whole
slot is `16 + dim * 4` bytes, fixed stride. The alternative would be a
*sidecar* file mapping `id -> offset`, the way most embedded KV stores
keep an index alongside the data.

The trade :

- **Cost** : ~0.5% storage overhead at 768-dim vectors. Open is O(N) :
  walk every slot to rebuild the in-memory `id_to_slot` map.
- **Win** : no two-file atomicity problem. A sidecar index and the data
  file are two separate writes ; a crash between them leaves the pair
  inconsistent and recovery has to reconcile. Self-describing slots
  can't drift from themselves : the data *is* the index.

For the sizes we care about (millions of vectors, opened rarely), the
O(N) open cost is invisible and the consistency story is much simpler.

### WAL is segmented and recoverable from day one

The write-ahead log lives in a directory of `wal-{16hex}.log` segment
files, rotated at ~64 MB. Recovery enumerates segments in LSN order,
replays each, and truncates any torn tail on the active segment.
Truncation after a future checkpoint becomes O(1) : delete superseded
segment files.

Why : a single-file WAL is technically simpler but rules out cheap
truncation and bounds nothing about replay time. Segmentation costs us
~50 LOC and unlocks the full WAL design pattern from production
databases. Same code shape will support log shipping, archive, and
multi-segment recovery without further refactoring.

### `Shard::insert` is three-phase ; apply failures after commit panic

Every insert moves through three explicit phases :

```text
1. pre-commit validation     │  duplicate id, dim mismatch
                             │  ── on Err : no state change anywhere
                             v
2. commit                    │  wal.append + wal.sync
                             │  ── after Ok : op is DURABLE
                             v
3. apply                     │  index.insert (writes through to VectorStore)
                             │  metadata.put
                             │  ── on Err : panic
```

The contract :

- `Ok(())` : committed **and** applied.
- `Err(...)` : rejected in phase 1, state unchanged.
- Phase-3 failure : process aborts. The WAL is truth ; reopen + replay
  reconciles. The caller never sees a misleading "Err" for an
  already-committed op.

The non-obvious part is the **panic**. The naive reflex is "return Err
from phase 3 too, let the caller decide." That's a trap : the operation
was already committed in phase 2. If the caller treats an Err as "didn't
happen" and retries, they reach phase 1 again with `duplicate-check :
empty`, append a *second* `Insert{id}` record, and now the WAL holds two
records for the same id. On the next reopen, replay applies the first
record fine, then hits `DuplicateId` on the second and refuses to open
the shard. One transient apply failure poisons the log permanently.

The honest answer is the Postgres model : the WAL is the commit point ;
in-memory state diverging from the WAL is a violated invariant, not a
recoverable error. Panic, let the process restart, let replay rebuild
in-memory state from the durable record.

To keep this rare, phase 1 is aggressive : it catches every failure mode
we can detect statically (duplicate id, dim mismatch, including the
first-insert case where the index hasn't pinned its own dim yet - we
fall back to the underlying `VectorStore::dim()` if the store has one).
What's left in phase 3 is genuinely exceptional : disk full, EIO,
filesystem corruption. For those, panic is the only safe call.

Tests `dim_mismatch_on_insert_does_not_poison_wal` and
`dim_mismatch_on_first_insert_is_caught_via_store_dim` enforce the
no-poison guarantee.

### Delete is logical, not structural

`Shard::delete(id)` doesn't remove the node from the HNSW graph or the
vector from `MmapVectorStore`. It marks `id` as tombstoned in an
in-memory `HashSet` and drops the entry from `MetadataStore`. The graph
node and vector bytes stay in place.

The reason is the hybrid pattern every production HNSW ships (FAISS,
hnswlib, Qdrant) : structural removal of an HNSW node requires
rewiring the neighbour lists of every node that pointed to it, which
is expensive and error-prone. Tombstoning is O(1) and preserves graph
connectivity for traversal. Cost : tombstoned vectors take up storage
+ some search-time CPU (distance is computed for them during
`search_layer`, then they're filtered before results return).

`search` filters tombstoned ids from the result list after
`search_layer`. Result count may be smaller than `k` if many candidates
in the neighbourhood are deleted ; that's what vacuum (see below) is
for : physical removal + graph repair restores both storage and recall.

Ids cannot be reused after delete until vacuum runs.
`Shard::insert(deleted_id, ...)` errors with `DuplicateId` because the
graph node is still in place. The
`reinsert_after_delete_errors_until_vacuum` test pins this contract.

### Vacuum is a two-pass graph repair, not a rebuild

`HnswIndex::vacuum_tombstones` physically removes tombstoned nodes and
patches the holes they leave in the graph. The naive approach (drop
nodes + drop their edges) breaks reachability : a survivor whose only
path to the entry point ran through a tombstoned node gets stranded.
The expensive approach (rebuild the whole graph from scratch) is
correct but wastes everything we already paid to insert.

The two-pass middle path :

```text
PASS 1 : collect (node, layer) -> {removed tombstones it pointed at}
PASS 2 : for each affected (node, layer) :
           drop the dead neighbour ids
           if remaining live neighbours >= M/2 : skip
           else : run search_layer to find fresh M neighbours
                  prune overflow with `select_neighbours_heuristic`
```

The **M/2 skip rule** is the bit that makes this cheap. HNSW only
needs *good* neighbours, not *complete* ones. A node that lost one
edge out of M still has plenty of routes to the entry point ; paying
for a re-search just to top it back up to M would dominate vacuum
time for no recall win. We only re-search nodes that fell below half
their capacity, which is rare for well-connected nodes and exactly
right for the few that mattered.

Entry-point handling is its own case. If the current entry point was
tombstoned, we pick the surviving node on the highest layer (ties
broken by id for determinism) ; if no nodes survive at the entry
layer, we drop down. `vacuum_entry_point_*` and
`vacuum_with_all_nodes_tombstoned_empties_index` pin both paths.

Cost : O(affected_nodes × M) for the prune phase, plus
O(re_searched_nodes × log N) for the re-search phase. On typical
workloads `re_searched_nodes` is a small fraction of
`affected_nodes`. `vacuum_leaves_no_dead_edges_in_live_nodes` is the
load-bearing invariant test.

### Checkpoint commits through a manifest, atomic at the manifest

`Shard::checkpoint` is the bridge between the WAL's append-only world
and the index's mutable on-disk state. Six phases :

```text
1. vacuum                 │  physically remove tombstones, repair graph
2. fsync the WAL          │  capture an LSN we can prove is durable
3. write graph snapshot   │  streaming serialize -> graph.{N+1}.snapshot
                          │  (N+1 is the new generation : doesn't touch
                          │   the old graph.{N}.snapshot the manifest
                          │   still points at)
4. atomic_write manifest  │  { version, checkpoint_lsn, snapshot_id=N+1 }
                          │  -- this is the durable commit point --
5. truncate the WAL       │  delete segments past checkpoint_lsn
6. delete graph.{N}.snapshot before this generation
```

**The manifest is the single commit point.** It's the only file that
names which snapshot is live ; everything before phase 4 is staging,
everything after is best-effort cleanup. The crash analysis :

- Kill **before phase 4** : nothing visible changed. Reopen sees the
  old manifest (or no manifest), loads `graph.{N}.snapshot` (or
  nothing), replays the full WAL. Any orphaned `graph.{N+1}.snapshot`
  is cleaned up by `cleanup_orphan_snapshots` on open.
- Kill **after phase 4** : the new generation is live. Reopen reads
  the new manifest, loads `graph.{N+1}.snapshot`, replays WAL only
  past `checkpoint_lsn`. If phase 5 or 6 didn't run, the stale WAL
  segments and old snapshot get cleaned up on the next checkpoint
  (or by `cleanup_orphan_snapshots` for the snapshot).

**Generation-numbered snapshots** (`graph.{N}.snapshot`) are what
keeps phase 3 crash-safe. Overwriting a single `graph.snapshot` file
would create a window where the file is half-written and the manifest
still points at it ; even with `atomic_write` for the file itself,
the manifest commit and the file rename can't be one operation. By
writing to a fresh generation name, the old snapshot stays valid until
the manifest commits the swap.

**`atomic_write` (and its streaming variant) is the load-bearing
primitive.** Both the manifest and each snapshot file are written to
`{path}.tmp`, fsynced, renamed onto the final path, and then the
parent directory is fsynced. POSIX guarantees the rename is atomic
and that after the dirsync the new dirent is durable. Either the old
file or the new file is observable on reopen ; never a partial write.
The streaming variant (`atomic_write_streaming`) hands the caller a
`BufWriter<File>` for the body, so large snapshots don't have to
materialise the whole payload in memory before writing.

**WAL truncation is delayed past the commit.** Phase 5 runs *after*
phase 4 returned Ok, so if it crashes mid-truncate we just have extra
WAL records that replay redundantly on reopen (idempotent : the
snapshot already has them). Crashing before phase 4 means truncation
never runs at all, which is also fine. The only ordering that would
lose data is truncating before committing the manifest, which the
code structure makes impossible.

`should_checkpoint(&CheckpointPolicy)` is a read-only hint :
`tombstone_ratio >= X` OR `records_since_checkpoint >= Y`. It returns
a bool ; the caller decides whether to actually call `checkpoint()`.
This keeps the insert path free of hidden latency : nothing in the
hot path ever blocks on snapshot I/O unless the operator explicitly
asks it to. `checkpoint_locks_in_vacuum_work` and
`orphan_snapshots_get_cleaned_up_on_open` pin the trickier paths.

### Batched inserts amortise the fsync

`Shard::insert_many(items)` groups a batch under a single WAL commit :

```text
   for each item:    wal.append(record)        (cheap, just buffered I/O)
   ─────────────────────────────────────────
   once at end:      wal.sync()                (the actual fsync)
                     metadata.put_many(...)    (one full-file flush)
                     for each: index.insert    (sequential by HNSW design)
```

Per-op cost dominates singleton `insert` because `wal.sync` is a real
disk barrier (~5-10 ms on SSD) and `FileMetadataStore::put` rewrites
the whole file with `atomic_write` (also fsynced). Group-committing
N items collapses both into a single barrier each.

Three trait additions support the batch path :

| Method                              | Default impl  | Override |
| ----------------------------------- | ------------- | -------- |
| `VectorStore::reserve(n)`           | no-op         | `MmapVectorStore` grows the file once |
| `MetadataStore::put_many(items)`    | loops `put`   | `FileMetadataStore` updates the map and flushes once at the end (with snapshot-rollback on failure) |
| `HnswIndex::reserve_store(n)`       | inherent      | pass-through to `vectors.reserve` |

All three preserve the existing trait surface (defaults cover every
current caller). `Shard::insert_many` puts `reserve_store` and dim/
duplicate validation in phase 1, so disk-full + bad input reject the
whole batch before the WAL is touched. Phase-3 apply failures panic
per the same Postgres-style rule as singleton `insert`.

Batches are atomic from the caller's view : the whole batch commits or
none of it does. There's no partial-batch state to observe.

### Metadata is not mmapped, and on purpose

Vectors live in `MmapVectorStore` because they are fixed-stride, hot, and
huge in aggregate : index search hits `vectors.get(id)` thousands of times
per query, and an `id * stride` offset calculation plus zero-copy mmap
read is the right tool.

Metadata is the opposite shape. It's variable-size (open key-value
bags), cold (read only for the final `k` candidates, not on every graph
edge), and small in aggregate. `FileMetadataStore` keeps the whole map
in memory and persists via `atomic_write` on mutation : a periodic
full-file snapshot, no mmap, no sidecar offset index, no free-list.

Forcing mmap onto variable-size data would mean building a separate
`id -> (offset, length)` sidecar plus a free-list for resize-on-update,
which is a B-tree in disguise. Different access patterns deserve
different storage strategies ; the `MetadataStore` trait is the seam so
the implementation can change without touching callers when scale
eventually demands it.

### Unsafe is encapsulated, not sprinkled

`kova-core` is `#![forbid(unsafe_code)]` : foundational types are
pure-safe Rust, no exceptions. `kova-storage` has exactly one `unsafe`
block in the whole crate, inside a private `map_file()` helper that
wraps `memmap2::MmapMut::map_mut`. The safety contract (file must not
be truncated or written to by another process while the map is live)
is documented once at the helper ; every other call site in
`MmapVectorStore` goes through the safe wrapper.

The rule : `unsafe` is a contract, and contracts are easier to audit
when there's exactly one of them. Adding a second `unsafe` block
anywhere in storage should require a comment explaining why the
existing wrapper isn't sufficient.

### `serde::Deserialize` on `Vector` is hand-rolled

`Vector::try_new` rejects NaN, ±Inf, and empty input. A blanket
`#[derive(Deserialize)]` would bypass those checks and let invalid
vectors come off disk. The hand-rolled `Deserialize` routes through
`try_new` so a CRC-valid record on disk that somehow contained NaN
cannot quietly poison the index.

The test `vector_deserialize_rejects_nan` enforces this : if anyone
"simplifies" by switching to `#[derive(Deserialize)]`, that test fails
immediately.

### `kova-query` is four IRs, not one

The query pipeline is `String -> AstStatement -> LogicalStatement ->
PhysicalPlan -> Rows`, four distinct intermediate representations
with one transformation between each. Looks like overkill for a v1
query engine ; it's not.

Each IR has exactly one job. The **AST** captures what the user
wrote, syntax-faithful and permissive enough to accept semantically-
wrong shapes (the parser doesn't know what fields exist, so it can't
type-check). The **LogicalStatement** captures what the user meant,
after field resolution, type checks, and predicate normalisation
(`IS NULL` becomes `NOT IsNotNull` so downstream code only handles
one shape ; embedding-update attempts are rejected here, not at
runtime). The **PhysicalPlan** is the operator tree the executor
walks, with the planner's strategy choice baked in. **Rows** are
what comes out.

Two upsides from the layering :

- Each transformation has its own test surface. Parser tests use
  source strings ; binder tests construct ASTs by hand and check
  semantic rejections ; planner tests construct LogicalStatements
  and assert which physical plan came out. Failures are local to
  one layer, not smeared across the whole pipeline.
- The boundaries are stable contracts. Grammar changes don't churn
  the planner ; planner cost-model changes don't churn the parser.
  Adding the binder didn't move anything else.

The "carry the table name through" rule is the load-bearing
discipline. The parser sees `VACUUM my_shard`, parses to
`AstVacuum { table: "my_shard" }`, the binder produces
`LogicalVacuum { table: "my_shard" }`, the executor matches against
its `Shard` catalog. The binder does NOT hardcode `"vectors"` and
reject anything else, even though v1 only has one table : that
would freeze a deployment decision into the language layer. Runtime
data (what tables exist, what indexes are built) belongs at the
runtime layer. Catching this early saves a multi-week refactor in
month 6.

### Three index types, not one universal index

`HashIndex` covers equality (`=`, `IN`, `IS NOT NULL`, `!=`). `BTreeIndex`
covers ranges plus everything `HashIndex` does. `InvertedIndex` covers
array containment. Three separate concrete types behind a shared
`MetaIndex` trait, instead of one universal index that handles every
atom shape.

The pull toward unification is real. One index type would mean one
backfill path, one persistence format, one set of tests. The pull is
wrong because the three shapes have **incompatible storage choices** :

- Hash buckets are random access, no ordering. Cheapest for equality
  because no comparison is needed past hash lookup.
- BTree buckets are sorted, walked via range cursor. Necessary for
  ranges but unnecessary overhead for pure equality (`HashIndex::Eq`
  is faster than `BTreeIndex::Eq`).
- Inverted index threads one row into N buckets per array element.
  Insert cost is O(array_length), unlike the others which are O(1).
  Different cost model means different design constraints.

Putting all three behind one type means picking one storage strategy
and degrading the other two. The trait split keeps each index optimal
for its atom set. The catalog routes atoms to the right index via
priority order, so the caller never has to think about which type
answers which query.

### `NormalizedKey` solves three problems at once

`Value` (from `kova-core`) is not a usable map key out of the box.
`Value::F64` contains `f64`, which is not `Eq` (NaN ≠ NaN), so
`HashMap<Value, _>` and `BTreeMap<Value, _>` both fail to derive.
`Value::Array` and `Value::Map` have no canonical single-key shape
because they're composite.

`NormalizedKey` is the bridge. Floats become `u64` via `f64::to_bits`
(total ordering, hashable bits). Strings, ints, bools pass through.
Arrays and maps return `None` from `from_value` ; the index silently
skips non-keyable values rather than erroring. That last choice keeps
batch inserts robust : one mistyped row doesn't reject the whole
batch.

The trade : `f64::to_bits` ordering matches numeric ordering only for
positive finite floats. `BTreeIndex` therefore rejects ranged queries
on floats via `supports`. Equality on floats works (bit-pattern match
is what users expect). Sortable-float encoding would lift the
restriction later if benchmarks justify it.

### `IndexCatalog::lookup` returns `Option<RoaringTreemap>`, not `Result`

The catalog has three return states for `(field, atom)` : "here's the
bitmap", "no index can answer this, fall back to scan", and "the field
isn't indexed at all." The first is `Some(bitmap)` ; the last two
collapse to `None`.

Collapsing them is deliberate. From the executor's perspective, both
mean "I have to scan myself" ; distinguishing "field unindexed" from
"atom unsupported" gives the planner no useful action. Both fall
through to the same fallback path.

Using `Result` would mean errors propagate up : "I can't answer this"
becomes an exception the caller has to handle. But "I can't answer
this" is the **normal case** for any predicate the planner hasn't
indexed yet. `None` keeps the fallback fluent ; you write
`catalog.lookup(...).unwrap_or_else(|| scan_metadata(field, atom))`
and the planner stays linear.

### WAL records carry `old_metadata` for replay coherence

`Record::Delete` and `Record::UpdateMetadata` carry the row's old
metadata bag in addition to the affected id. This is more bytes per
record than the minimum needed for the metadata store (which just
needs the id to delete, or the new bag to update).

The asymmetry the field plugs : `FileMetadataStore` persists eagerly.
By the time `Shard::open` runs WAL replay on a post-checkpoint Delete
record, the bag is already gone from disk. The catalog's replay
forwarding (`catalog.on_delete(id, &old_meta)`) needs the bag to
clear the right buckets, and the store can no longer provide it.

The honest answer is to ship the bag in the WAL record. Live `delete`
captures it pre-WAL-commit, embeds it ; replay reads it back from the
record. Both paths use the same `(id, old_meta)` pair. The cost is a
few extra hundred bytes per record (typical bag sizes), which is
small next to the vector payload an Insert already carries.

This is a WAL format change. Pre-revision WAL files won't deserialize.
For pre-release the breaking change is acceptable ; v1 will gate
with a version field on the `Record` framing for migration.

### Catalog persistence is checkpoint-gated, not write-through

The catalog file is generation-numbered alongside the graph snapshot
(`catalog.{snapshot_id}.bin` next to `graph.{snapshot_id}.snapshot`)
and written **only at `Shard::checkpoint`**. Between checkpoints, the
catalog lives in memory only.

This means indexes registered with `add_*_index` after the last
checkpoint are transient : close-without-checkpoint loses them. The
metadata store survives (data isn't lost), but `add_*_index` has to
be called again to rebuild from the live data.

The pull toward write-through ("save catalog on every `add_*_index`")
is real but creates an LSN mismatch problem. The catalog file is
named by `snapshot_id`, which only changes at checkpoint ; if you
overwrite `catalog.{N}.bin` between checkpoints, it carries state
past `checkpoint_lsn`, and reopen's WAL replay (from
`checkpoint_lsn + 1`) double-counts the records the catalog already
saw. A watermark LSN inside the catalog file would fix this but
means a third atomic commit point (catalog + graph + manifest) the
manifest swap has to coordinate.

Not worth the complexity. Same contract `vacuum()` already has :
"transient until the next checkpoint locks it in." Operators are
told to checkpoint after `add_*_index` if durability matters.

### `RoaringTreemap`, not `RoaringBitmap`

`VectorId` is a `u64` newtype, so the bitmap type that carries id sets
is `RoaringTreemap`, not `RoaringBitmap`. The two differ in addressing
range : `RoaringBitmap` handles `u32` (~4B ids), `RoaringTreemap`
handles `u64` (the full id space).

`RoaringTreemap` is implemented as `BTreeMap<u32, RoaringBitmap>` :
the upper 32 bits of each id pick an inner bitmap, the lower 32 bits
get stored inside it. Every set operation decomposes : intersection
walks both treemaps in sync, intersecting matching inner bitmaps for
each shared upper-32 chunk. The chunk-level pruning that makes
roaring fast at the `u32` level still applies at the `u64` level ;
just one extra btree layer.

The cost is one more pointer chase per operation. The win is that
the indexes never have to worry about id range exhaustion. At Kova's
scale that doesn't matter, but coding to `u64` from the start means
no migration when shards get large enough that it does.

## `kova-query` : KQL

KQL is the SQL-shaped query language Kova uses for hybrid vector +
metadata workloads. **One query expresses both an ANN search and a
metadata predicate ; the planner picks the right execution strategy
based on the predicate's selectivity, so you don't have to.** That
is the whole pitch ; the rest of this section unpacks it.

This is the README view. Full architecture, plan dispatch, type
system, algorithms, baselines, and what's still in flight live in
[`docs/query.md`](docs/query.md).

### What a real KQL query looks like

The bread-and-butter shape, kNN combined with a metadata filter :

```sql
SELECT id, embedding <-> $query AS distance, metadata
FROM vectors
WHERE category = 'docs' AND year >= 2024
ORDER BY embedding <-> $query LIMIT 10
```

That one statement returns the 10 vectors closest to `$query`,
restricted to rows where `category = 'docs' AND year >= 2024`. Before
KQL you would issue two calls (an ANN, a metadata scan) and glue them
together in application code. With KQL the engine does it.

A handful more shapes that all just work :

```sql
-- Radius search with an AND-residue post-filter
SELECT id FROM vectors
WHERE embedding <-> $q < 0.5 AND tag = 'archived'

-- COUNT with a subscripted predicate (nested attribute access)
SELECT COUNT(*) FROM vectors WHERE attrs['country'] = 'IN'

-- Patch a nested attribute, by id
UPDATE vectors SET attrs['priority'] = 5 WHERE id = $1

-- Predicate-driven delete, batched into one WAL commit
DELETE FROM vectors WHERE category = 'old' AND year < 2020
```

If those five shapes feel reasonable, you already know most of KQL.
[`docs/query.md`](docs/query.md) walks the rest.

### Why three plans, not one

The hard problem with hybrid queries is that the right strategy
depends on the workload : dim, shard size, predicate selectivity,
result limit. The planner picks between three plans automatically :

| Plan | What it does | Wins when |
|------|--------------|-----------|
| A    | Overfetched kNN (`k * 4` candidates) then post-filter. Returns whatever survives, **without retrying**. | The filter is loose. Cheapest walk when most candidates pass. |
| B    | `MetadataScan` over all rows for matching ids, exact distance per match, top-k. **Bypasses ANN entirely.** | Low selectivity, or a shard small enough that the scan is short. Also the only plan that is *exact*. |
| C    | Filtered HNSW : the predicate is consulted **during** the graph walk. Out-of-filter nodes still route traversal but never enter the result heap. | **High selectivity.** Plan A's fixed 4x overfetch is sized for a filter that rejects ; when the filter passes most rows, A does ~4x the graph work and 4x the metadata clones that C does. |

Dispatch is cost-based, not band-based. `cost_plan_a/b/c` in
[`cost.rs`](crates/kova-query/src/cost.rs) compute closed-form
latency estimates against five per-machine coefficients
(`c_hnsw_per_visit`, `c_distance_per_dim`, `c_metadata_get`,
`c_metadata_peek`, `c_filter_eval`) and the planner picks the minimum.

**Plan choice is gated on correctness before cost.** Plan A's expected
yield is `k * 4 * selectivity`, so below `selectivity = 0.25` it
*mathematically cannot* fill the LIMIT — measured, it returns **zero
rows** for `LIMIT 10` at selectivity ≤ 0.05, where B and C return every
matching row. `cost::plan_a_can_satisfy` excludes it in that regime
regardless of price ; a plan that cannot answer the query is not a fast
plan.

Validated against measured runs of each plan on an 84-cell
`(dim, n, selectivity, k)` grid, scoring against **the fastest plan that
actually returned `min(k, matching_rows)` rows** :

| Metric | Result |
|---|---|
| Dispatch accuracy | 75/84 cells |
| Mean regret | **1.055** (dispatched plan averages 1.06x slower than optimal ; 1.0 is perfect) |
| Cells where the dispatched plan under-delivered | **0/84** |
| Plan C fastest-correct | 35/84 cells, up to **3.22x**, found by the planner in 28 |

The validation harness ships behind the `internal-bench` feature at
[`examples/validate_cost_model.rs`](crates/kova-query/examples/validate_cost_model.rs)
(axes overridable via `KOVA_DIMS` / `KOVA_NS` / `KOVA_SELS` / `KOVA_KS`),
alongside a calibration runner
([`examples/calibrate_cost_coefficients.rs`](crates/kova-query/examples/calibrate_cost_coefficients.rs))
that microbenches each coefficient on the target machine, and a replay
harness ([`examples/cost_probe.rs`](crates/kova-query/examples/cost_probe.rs))
that re-scores recorded measurements against the current model in
seconds instead of re-running the hour-long sweep.

Every cost-model bug found so far has had one shape : **the formulas
scored an idealised plan rather than the one the executor runs.** The
harness exists because a model can be internally consistent and still
describe a different program than the one you shipped.

### Coverage at a glance

Everything the language accepts today, by pipeline layer :

| Layer | What lands |
|-------|------------------------|
| Grammar | All 8 statements parse (SELECT / INSERT / UPDATE / DELETE / VACUUM / CHECKPOINT / CREATE / DROP INDEX). Predicates : 6 comparison ops, `IN`, `BETWEEN`, `IS NULL`, `@>`, distance threshold, AND/OR/NOT with correct precedence. Subscripted field references (`attrs['key']`) work in both predicates and assignments. |
| Binder | Hard semantic rejections : embedding mutation, kNN without LIMIT, distance ordering with `DESC`, OR containing a distance threshold. `CREATE INDEX` / `DROP INDEX` flow through ; missing names get synthesised. |
| Planner | Three kNN strategies dispatched by predicate selectivity, plus radius operator, COUNT bypass, scan-and-limit bypass. Single-id hint detection (literal + param) fast-paths DELETE / UPDATE. |
| Executor | Full DML : single + batch INSERT, DELETE-by-id / by-param / by-predicate / by-radius, UPDATE-by-id / by-param / by-predicate / by-radius with flat or subscripted SET clauses. DDL : `Shard::create_index` / `drop_index` go through the WAL so the catalog survives reopen. VACUUM + CHECKPOINT bridge to `Shard`. |
| Types | `Value` covers `String`, `I64`, `F64`, `Bool`, `Array`, `Map` (nested bags). Numeric coercion (`5 == 5.0`) ; everything else is type-strict. |

### Why you can trust it

KQL ships with two layers of mechanical verification beyond the
unit-test suite :

| Test | Iterations | What it asserts |
|------|------------|-----------------|
| `fuzz_smoke_*` | 1,500+ per run | No panics across grammar-conformant random queries on a random shard |
| `correctness_fuzz_*` | 1,000 per run | Engine result matches a reference impl for COUNT, scan-and-limit, DELETE, UPDATE |
| `fuzz_long_run` (ignored) | 32k+ | Long-run no-panic over 16 seeds |
| `correctness_fuzz_long_run` (ignored) | 12k+ | Long-run correctness over 8 seeds |
| `plan_a_starvation` | 4 cases | A satisfiable `LIMIT k` never comes back short, through the full `execute_str` pipeline |

`plan_a_starvation` is worth a note on method : the first attempt at
that regression test was a fuzzer assertion that passed — and **kept
passing when the bug was deliberately reintroduced**, because the
fuzzer's generated `k` was too small to trigger starvation. A test that
has never been observed to fail proves nothing. The replacement was
verified failing before it was accepted as passing.

The harness is deterministic given a seed. Any failing iteration
prints the seed + query verbatim ; rerunning with the same seed
reproduces the bug. Across the full long-run sweep, every single
COUNT, scan, DELETE, and UPDATE that the reference picked up agreed
with the engine.

### The engineering decisions worth knowing

If you are reading the code, these are the calls that shape what you
see. Each one paid off enough to be worth keeping.

**Four IRs, not one.** AST -> LogicalStatement -> PhysicalPlan ->
ExecutionResult. Each IR has exactly one job, each transformation
has its own test surface. Failures stay local to one layer. The
boundaries are stable contracts so grammar changes do not churn the
planner and vice versa.

**Selectivity-driven plan dispatch, behind a trait.** The
`SelectivityEstimator` trait is what feeds the cost model. The
shipped `ShardEstimator` consults the `IndexCatalog` first
(exact cardinality in O(1) for indexed atoms), then the
`StatsCatalog` for atoms covered by column statistics
(approximate, histogram-backed), and falls back to a metadata
scan only for predicates neither layer recognises.

**Cost-based dispatch, validated against measured runs.** The
planner picks plan A / B / C by computing closed-form latency
estimates per plan against five per-machine coefficients, then
dispatching the cheapest — after excluding any plan that cannot
return a complete answer.

Six model bugs have surfaced during validation, every one the same
shape : **the formulas scored an idealised plan rather than the one
the executor runs.**

| # | The model believed | The executor actually |
|---|---|---|
| 1 | plan A retries on starvation | returns short, never retries |
| 2 | plan B's ids come from an O(1) lookup | scans every row first |
| 3 | `ef = max(2k, 50)` | `ef = ef_search.max(k)` |
| 4 | plan C's filter overhead is linear in `(1-s)` | hyperbolic in `s` — it degenerates to a full scan |
| 5 | (calibrator) its own copy of `hnsw_visits` mirrored the model | the copy went stale, skewing a derived coefficient 1.6x |
| 6 | one coefficient covers all metadata access | plans A/B *clone* a bag (250 ns), plan C *borrows* one (10 ns) |

Bugs 3 and 5 were the same mistake in two places, so `hnsw_visits` now
has exactly one definition, shared with the harnesses rather than
transcribed into them.

A seventh lived in the harness itself : it ranked plans by latency
alone, so a plan returning **zero rows** counted as the winner and a
dispatcher that correctly refused it was charged up to 6.5x regret for
doing the right thing. Ground truth is now the fastest plan that
actually returned `min(k, matching_rows)`.

A separate calibration runner microbenches each coefficient on the
target machine and prints a tuned `CostCoefficients` struct. Notably,
**calibration turned out to be second-order** : on the 84-cell grid the
shipped x86 defaults and locally-measured coefficients score the same
(75/84, regret ≈1.05). The formula fixes did the work.

The discipline is structural : the validation harness, the calibration
runner, and a replay harness that re-scores recorded measurements in
seconds all live in the repo, and anyone can re-run them.

**Single-id hint instead of duplicate paths.** `DELETE WHERE id = 42`
and `DELETE WHERE id = $1` both flow through the same `LogicalDelete
{ single_id_hint }` field, which carries either a literal id or a
parameter slot. The planner dispatches to `DeleteById` or
`DeleteByParamId` from one match. Same shape for UPDATE.

**Operator symmetry across the language.** `SELECT WHERE
embedding <-> $q < r`, `DELETE WHERE embedding <-> $q < r`, and
`UPDATE WHERE embedding <-> $q < r` all share a single `RadiusOp`
borrowed bundle struct in the executor. Adding a new radius-flavoured
operation costs one helper, not three.

**`Value::Map` is the only nested type.** Metadata bags are
`HashMap<String, Value>` and Value can recurse exactly through `Map`
(and `Array`). Lets subscripted access work on both predicates and
SET clauses without growing the grammar to multi-level paths.

**Errors are a closed taxonomy.** `KovaQueryError` has five
variants : `Parse`, `Bind`, `Plan`, `Execution`, `Backend`. The
fuzzer enforces that every code path through the pipeline returns
one of these, never panics. A panic is always a bug ; a typed error
is a contract.

**The reference fuzzer caught its own bug.** The correctness
fuzzer is reference-first : the reference impl computes the expected
result before the engine runs. An earlier version let the engine
run first and the reference fell behind on shapes it could not
evaluate, manifesting as a fake "engine deleted 0, expected 10"
mismatch. The fix was harness discipline, not engine code, but the
mode of failure is the kind of methodology the project keeps.

## `kova-meta-index` : secondary indexes

Meta-indexes are Kova's secondary-index layer for metadata
predicates. **Three index types behind one `MetaIndex` trait,
`RoaringTreemap` for composition, an `IndexCatalog` that orchestrates
per-field bundles, persistence alongside the graph snapshot.** Wires
into every `Shard::insert / delete / update` so the catalog stays in
sync without operator effort. That is the pitch ; the rest of this
section unpacks it.

This is the README view. Full architecture, atom dispatch table, the
persistence story, and the WAL coherence rules live in
[`docs/meta-index.md`](docs/meta-index.md).

### The three index types

Each index type covers a slice of the predicate space. The catalog
routes each atom to whichever index supports it cheapest.

| Index | Atoms supported | Cost |
|-------|-----------------|------|
| `HashIndex` | `Eq`, `In`, `IsNotNull`, `Cmp(Ne, _)` | O(1) lookup, O(matches) clone |
| `BTreeIndex` | Everything `HashIndex` does + `Cmp(Lt/Le/Gt/Ge, _)` and `Between` (rejected on float values) | O(log N) cursor + O(buckets in range) union |
| `InvertedIndex` | `ArrayContains`, `IsNotNull` | O(1) lookup ; insert is O(array length) |

A single field can carry multiple index types. `add_hash_index("year") +
add_btree_index("year")` makes both available ; the catalog routes
`Eq(2024)` to the hash and `Cmp(Gt, 2022)` to the btree.

### What a registration + query looks like

```rust
let mut shard = Shard::open("/tmp/my-shard", 768, L2, HnswParams::default())?;

shard.add_hash_index("category");
shard.add_btree_index("year");
shard.add_inverted_index("tags");

// ... inserts, deletes, updates : indexes update synchronously ...

// WHERE category = 'docs' AND year > 2022 AND tags @> 'rust'
let docs   = shard.catalog().lookup("category", &IndexAtom::Eq(s("docs"))).unwrap();
let recent = shard.catalog().lookup("year",     &IndexAtom::Cmp(CmpOp::Gt, i(2022))).unwrap();
let rust   = shard.catalog().lookup("tags",     &IndexAtom::ArrayContains(s("rust"))).unwrap();

let candidates = docs & recent & rust;  // roaring bitmap intersection

shard.checkpoint()?;  // catalog now durable across reopen
```

The intersection at the end runs in microseconds on a million-row
shard because roaring prunes entire 65,536-id chunks at a time when
they're absent from any input.

### Persistence at a glance

| Where things live | When they're written |
|-------------------|----------------------|
| Catalog buckets + `all_indexed_ids` (in memory) | Mutated synchronously by every `insert/delete/update` |
| `catalog.{snapshot_id}.bin` (on disk) | Written atomically at every `Shard::checkpoint`, alongside `graph.{snapshot_id}.snapshot` |
| WAL records (on disk) | Now carry `old_metadata` for `Delete` / `DeleteMany` / `UpdateMetadata` so replay can drive the catalog |
| Live catalog on reopen | Loaded from `catalog.{snapshot_id}.bin` ; WAL records from `checkpoint_lsn + 1` replay through `catalog.on_*` hooks to catch up |

The manifest is still the single durable commit point ; catalog and
graph snapshots are committed together by the manifest swap. Crash
analysis is identical to the graph snapshot story : kill before
manifest commit leaves the old generation live, kill after leaves the
new one live, orphan files get swept on next open.

### What ships strong

| Test surface | Count | What it asserts |
|--------------|-------|-----------------|
| `kova-meta-index` unit tests | 58 | Atom dispatch, supports/query symmetry, cardinality matches `lookup.len()`, persistence round-trip, magic/version/truncation rejections |
| `meta_index_integration.rs` | 12 | Catalog hooks survive every mutation path : insert / insert_many / delete / delete_many / update_metadata / backfill / multi-field composition |
| `meta_index_persistence.rs` | 8 | Close-reopen survival, post-checkpoint replay catch-up, transient indexes don't persist without checkpoint, orphan catalog sweep, all three types persist together |
| All-targets clippy `-D warnings` | clean | Including all the serde derives, the float-bit cast, the from-impls on `VectorId` |

### The engineering decisions worth knowing

If you are reading the code, these are the calls that shape what you
see. Each one paid off enough to be worth keeping.

**Three index types behind one trait.** `HashIndex`, `BTreeIndex`,
`InvertedIndex` share the `MetaIndex` surface but pick the storage
strategy that fits their atom set. Unifying them would mean degrading
one for the sake of another ; the trait split keeps each optimal.

**`NormalizedKey` is the bridge from `Value` to map keys.** `Value`
can't derive `Hash + Eq` (NaN in `F64`) or `Ord` (same reason).
`NormalizedKey` stores floats via `f64::to_bits`, excludes non-scalar
variants (`Array`, `Map`) by returning `None` from `from_value`, and
flags the float-ordering caveat for `BTreeIndex::supports` to gate.

**`IndexCatalog::lookup` returns `Option`, not `Result`.** "No index
can answer this atom" is the **normal case** for any predicate that
isn't covered yet ; the fallback is "scan." Collapsing it to `None`
keeps the planner's path linear instead of forcing it to translate
errors into fallback branches.

**WAL `Delete` and `UpdateMetadata` carry `old_metadata`.** The
metadata store persists eagerly, so by the time replay runs the bag
is gone from disk. Embedding it in the WAL record makes the record
self-sufficient : live ops capture the bag pre-commit, replay reads
it back, both paths use the same `(id, old, new)` data.

**Catalog persistence is checkpoint-gated.** Same contract as
`vacuum()` : indexes added without a subsequent `checkpoint()` are
transient. Saving on every `add_*_index` would create an LSN mismatch
between the catalog file and the WAL replay range. Operators
checkpoint after registration if durability matters.

**`RoaringTreemap`, not `RoaringBitmap`.** `VectorId` is u64.
`Treemap` is a `BTreeMap<u32, RoaringBitmap>` ; chunk-level pruning
still applies, one more pointer chase per operation. Sized for the
full id space from day one.

**Priority routing, not selectivity routing.** The catalog asks each
index on a field in fixed order : hash > btree > inverted. All
supporting indexes return the same bitmap ; picking the cheapest by
lookup cost minimises overhead. Selectivity-driven routing is for the
cost-model rework, not today.

## Longer-term scope

Beyond what `kova-query` ships today :

- **`kova-query` next steps** : meta-indexes are in (three index
  types, `RoaringTreemap` composition, `IndexCatalog`, persistence
  across reopen ; see [`docs/meta-index.md`](docs/meta-index.md)).
  Still to come : `CREATE INDEX` / `DROP INDEX` DDL unrejected at
  the binder, column statistics for unindexed predicates, the
  executor's `MetadataScan` consulting `shard.catalog().lookup`
  first, and a cost-model planner replacing the hardcoded 0.05 /
  0.5 selectivity bands. Every remaining step is a trait-impl swap
  behind a stable interface : same KQL grammar, same operators,
  same fuzzer. See [`docs/query.md`](docs/query.md) for the full
  plan.
- **`kova-cluster` + `kova-server`** : the distribution layer. Consistent
  hashing with virtual nodes, quorum replication, a coordinator that
  fans out queries and merges results across shards via gRPC. `openraft`
  for membership and leader election only ; shard logic is hand-rolled.
- **Client SDKs** in **Rust**, **Go**, and **TypeScript** so callers
  outside the project can talk to a Kova cluster idiomatically.

## Design philosophy

**The log is truth ; memory is a cache.** Every mutation is logged,
fsynced, and only then applied. If apply diverges from the log, the
process aborts and reopen rebuilds from the log. 55,697 acked inserts
and 962 checkpoints survived 300 SIGKILLs with zero data loss : the
discipline is what makes that possible, not luck.

**Pluggable seams, concrete impls.** `VectorStore`, `MetadataStore`,
and `Wal` are traits with associated error types so a future S3 log
or distributed store drops in without touching `Shard`. Inside those
seams every choice is opinionated : mmap for hot fixed-stride data,
full-file snapshots for cold variable-size data, in-memory tombstones
because the log already has the truth. Traits are where the future
lives ; impls are where the work happens.

**Built from scratch, on purpose.** Every byte format, every fsync,
every neighbour list, every checkpoint phase. An existing library
would ship the same surface in a weekend ; the point isn't to ship
fast, it's to understand the system the libraries hide.

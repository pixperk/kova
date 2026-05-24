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
| `kova-index`   | shipped     | `Index` trait, `FlatIndex` baseline, `HnswIndex` (insert + search)     |
| `kova-storage` | in progress | Segmented WAL, `MmapVectorStore`, `FileMetadataStore`, `atomic_write`, `Shard` (log-then-mutate, SIGKILL-survival tested); delete + checkpoints next |
| `kova-query`   | not started | KQL parser, planner, executor                                          |
| `kova-cluster` | not started | Consistent hashing, quorum replication, coordinator                    |
| `kova-server`  | not started | gRPC node binary                                                       |

153 tests passing across the workspace; `cargo clippy --workspace --all-targets -- -D warnings` clean.

## Build

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo bench -p kova-core
```

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

`L2` and `InnerProduct` get the raw 8-wide SIMD benefit (~5-6x). `Cosine`
gets ~12x because the SIMD pass also folded `dot`, `|a|^2`, and `|b|^2` into
a single loop (the previous scalar version did three separate passes).

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
At 100k HNSW is still **~9x** ahead, and the gap keeps growing with N.

| Operation                               | Latency |
| --------------------------------------- | ------- |
| `HnswIndex.insert` into 1k-vector index | 265 us  |

Run the 100k benches yourself: `cargo bench -p kova-index --bench hnsw -- at_100k`.
The 100k build alone takes ~2-3 minutes.

## Recall validation

`HnswIndex` is correctness-tested against `FlatIndex` (ground truth) on
random uniform workloads:

| N       | dim | Recall@10 | Notes                                                     |
| ------- | --- | --------- | --------------------------------------------------------- |
|     300 |   8 | **1.000** | default; runs in milliseconds                             |
|  10,000 |  32 | > 0.9     | default; runs in ~4s release mode                         |
|  50,000 |  32 | > 0.9     | `#[ignore]`; run with `cargo test --release -- --ignored` |

The 300-case hits the brute-force ground truth exactly; larger scales meet
the > 0.9 threshold with default `HnswParams`. No parameter tuning required
for uniform random data at these sizes.

All numbers above use SIMD distance (`wide::f32x8`). The `wide` crate falls
back to scalar on platforms without 8-lane f32 SIMD, so this builds and
runs everywhere.

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
*not* required and does not hold — the kill can land between
`wal.sync` and the next `flush`, leaving a durable insert the parent
never knew about. The test correctly tolerates extras ; the only thing
it refuses to tolerate is a missing ACKed record.

| Test                       | Iterations | Acks   | Failures | Runtime (release) |
| -------------------------- | ---------- | ------ | -------- | ----------------- |
| `crash_recovery_smoke`     |          5 |    ~750 |        0 | ~3s               |
| `crash_recovery_torture`   |        100 | 15,020 |        0 | ~113s             |

Run them with :

```sh
cargo test -p kova-storage --test crash_recovery
cargo test -p kova-storage --release --test crash_recovery -- --ignored --nocapture
```

Kill delays are deterministic — `1 + (iter * 173) % 1500` ms — so a
failure on iteration N is reproducible without a `rand` dependency. The
sequence spans from "kill before any insert lands" to "kill near the
end of the run."

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
first-insert case where the index hasn't pinned its own dim yet — we
fall back to the underlying `VectorStore::dim()` if the store has one).
What's left in phase 3 is genuinely exceptional : disk full, EIO,
filesystem corruption. For those, panic is the only safe call.

Tests `dim_mismatch_on_insert_does_not_poison_wal` and
`dim_mismatch_on_first_insert_is_caught_via_store_dim` enforce the
no-poison guarantee.

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

## Coming up : `kova-storage`

`Shard` composition + SIGKILL crash recovery have shipped (see the
[Crash recovery](#crash-recovery) section). Three milestones left to
close out the storage layer :

1. **Delete.** HNSW tombstoning (skip in `search_layer`),
   `VectorStore::remove` with a free-list of holes, `MetadataStore::delete`.
   The hybrid pattern that FAISS / hnswlib / Qdrant ship : cheap
   tombstone on the hot path, periodic vacuum on the cold path.
2. **Batched inserts.** `Shard::insert_many` with group-commit (N
   appends, 1 fsync). Plus `MmapVectorStore::reserve(n)` to skip the
   per-insert grow-remap dance, and `FileMetadataStore::flush_deferred`
   to skip the per-put rewrite. The headline win is amortised fsync —
   1000 inserts share one disk barrier instead of paying 1000.
3. **GFS-pattern checkpoints + log truncation.** Periodic snapshots of
   in-memory state, tagged with LSN ; recovery loads the checkpoint
   then replays only newer WAL records. After the checkpoint is durable,
   older WAL segments are physically deleted. Vacuum (from milestone 1)
   ties in here.

## Longer-term scope

Beyond `kova-storage` :

- **`kova-query`** : KQL, a SQL-inspired query language for hybrid
  searches that combine vector similarity with metadata filters. Pest
  grammar, planner that picks pre-filter vs post-filter based on
  selectivity, executor that walks plans against the storage layer.
- **`kova-cluster` + `kova-server`** : the distribution layer. Consistent
  hashing with virtual nodes, quorum replication, a coordinator that
  fans out queries and merges results across shards via gRPC. `openraft`
  for membership and leader election only ; shard logic is hand-rolled.
- **Client SDKs** in **Rust**, **Go**, and **TypeScript** so callers
  outside the project can talk to a Kova cluster idiomatically.

# Kova

A distributed vector database, built from scratch in Rust.

Hand-rolled HNSW index, memory-mapped storage with write-ahead logging,
a SQL-inspired query language (KQL), and a shard-replicated cluster layer
with gRPC between nodes. Every byte, every index, every network call is ours.

> *"kova"* - the Turkish word for a bucket, a hive, a vessel that holds what
> matters. Where bees store honey, Kova stores vectors.

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
| `L2`            |   128 |  110 ns  |  27 ns  |  4.1x   |
| `L2`            |   768 |  718 ns  | 120 ns  |  6.0x   |
| `L2`            | 1,536 | 1,430 ns | 263 ns  |  5.4x   |
| `Cosine`        |   128 |  267 ns  |  54 ns  |  4.9x   |
| `Cosine`        |   768 | 1,950 ns | 186 ns  | 10.5x   |
| `Cosine`        | 1,536 | 3,870 ns | 332 ns  | **11.7x** |
| `InnerProduct`  |   128 |   95 ns  |  21 ns  |  4.5x   |
| `InnerProduct`  |   768 |  668 ns  | 109 ns  |  6.1x   |
| `InnerProduct`  | 1,536 | 1,310 ns | 224 ns  |  5.8x   |

`L2` and `InnerProduct` get the raw 8-wide SIMD benefit (~5-6x). `Cosine`
gets ~12x because the SIMD pass also folded `dot`, `|a|^2`, and `|b|^2` into
a single loop (the previous scalar version did three separate passes).

## HNSW vs Flat (dim 32, k=10, L2)

`HnswIndex` against the `FlatIndex` brute-force baseline. HNSW uses default
`HnswParams` (`M=16`, `ef_construction=200`, `ef_search=50`). Criterion mean,
single core, **SIMD distance**.

| N       | `FlatIndex.search` | `HnswIndex.search` | HNSW speedup |
| ------- | ------------------ | ------------------ | ------------ |
|   1,000 |  15 us             |  79 us             |  0.19x       |
|  10,000 | 159 us             | 161 us             |  1.0x        |
| 100,000 | 3.65 ms            | 421 us             | **8.7x**     |

SIMD raises *both* lines on this table, but flat benefits more : its inner
loop is *just* distance computation, while HNSW spends most of its time on
graph traversal (HashMap lookups, heap operations). The HNSW crossover point
shifts to roughly 10k vectors with SIMD; below that, the linear scan wins.
At 100k HNSW is still **~9x** ahead, and the gap keeps growing with N.

| Operation                               | Latency |
| --------------------------------------- | ------- |
| `HnswIndex.insert` into 1k-vector index | 336 us  |

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

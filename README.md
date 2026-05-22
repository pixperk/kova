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

Scalar `Distance` impls in `kova-core`, criterion mean over ~100 samples.
SIMD optimisation should bring these down 3-5x.

| Metric          | dim 128 | dim 768 | dim 1536 |
| --------------- | ------- | ------- | -------- |
| `L2`            | 110 ns  | 718 ns  | 1.43 us  |
| `Cosine`        | 267 ns  | 1.95 us | 3.87 us  |
| `InnerProduct`  |  95 ns  | 668 ns  | 1.31 us  |

`Cosine` is ~3x `L2` because the three-pass version computes `dot`, `|a|`, and `|b|` separately. A single-pass fold is one of the planned optimisations.

## HNSW vs Flat (dim 32, k=10, L2)

`HnswIndex` against the `FlatIndex` brute-force baseline. HNSW uses default
`HnswParams` (`M=16`, `ef_construction=200`, `ef_search=50`). Criterion mean,
single core, scalar distance (no SIMD).

| N       | `FlatIndex.search` | `HnswIndex.search` | HNSW speedup |
| ------- | ------------------ | ------------------ | ------------ |
|   1,000 |  39 us             |  87 us             |  0.45x       |
|  10,000 | 329 us             | 185 us             |  1.8x        |
| 100,000 | 4.76 ms            | 378 us             | **12.6x**    |

At 1k the linear scan still beats HNSW's graph-walking constant factor. The
crossover sits around a few thousand vectors. From 10k onward HNSW's
`O(log N)` advantage is dominant: as flat grows roughly 14x going from 10k to
100k, HNSW grows only ~2x. The gap keeps widening at higher N.

| Operation                              | Latency |
| -------------------------------------- | ------- |
| `HnswIndex.insert` into 1k-vector index | 348 us  |

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

SIMD distance kernels are not yet wired; all numbers above are pure scalar.

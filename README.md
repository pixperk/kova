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
`HnswParams` (`M=16`, `ef_construction=200`, `ef_search=50`). Criterion mean
over ~100 samples, single core.

| N      | `FlatIndex.search` | `HnswIndex.search` | HNSW speedup |
| ------ | ------------------ | ------------------ | ------------ |
|  1,000 |  23 us             |  66 us             |  0.35x       |
| 10,000 | 484 us             | 156 us             |  3.1x        |

At 1k vectors the linear scan beats HNSW's graph-walking constant factor; the
crossover sits around a few thousand. At 10k, HNSW's `O(log N)` advantage
shows: latency grows ~2.4x while flat grows ~21x.

| Operation                | Latency |
| ------------------------ | ------- |
| `HnswIndex.insert` into 1k-vector index | 425 us |

Correctness is verified by a recall@10 test against `FlatIndex` on 300
random vectors (asserts > 0.9 recall). HNSW search at 10k is `O(log N)`
in distance computations vs flat's `O(N)`.

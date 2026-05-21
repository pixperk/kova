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

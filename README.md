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
```

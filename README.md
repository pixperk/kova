# Kova

A distributed vector database, built from scratch in Rust.

Hand-rolled HNSW index, memory-mapped storage with write-ahead logging,
a SQL-inspired query language (KQL), and a shard-replicated cluster layer
with gRPC between nodes. Every byte, every index, every network call is ours.

> *"kova"* — the Turkish word for a bucket, a hive, a vessel that holds what
> matters. Where bees store honey, Kova stores vectors.

## Status

Phase 0 — workspace scaffolded. See [`todo.md`](todo.md) for the roadmap and
[architecture doc](kova_architecture.pdf) for the design.

## Workspace layout

```
crates/
  kova-core/      — Vector types, distance fns        (Phase 1)
  kova-index/     — Index trait, brute force, HNSW    (Phase 2)
  kova-storage/   — WAL, mmap vector store, metadata  (Phase 3)
  kova-query/     — KQL parser, planner, executor     (Phase 4)
  kova-proto/     — gRPC protobuf definitions         (Phase 5a)
  kova-cluster/   — Consistent hashing, replication   (Phase 5b)
  kova-server/    — gRPC server, node binary          (Phase 5c)
```

Only `kova-core` exists today. The rest get created when their phase opens.

## Build

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
```

//! Durable storage for Kova.
//!
//! Hosts the write-ahead log (`wal`), the memory-mapped `VectorStore`, the
//! `MetadataStore` trait + in-memory impl, and the `Shard` type that
//! composes those with a [`kova_index::Index`]. Crash-recovery semantics
//! follow the standard log-then-mutate discipline : every mutation hits a
//! fsynced WAL record before any in-memory or on-disk state changes.

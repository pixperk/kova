//! Durable storage for Kova.
//!
//! Hosts the write-ahead log (`wal`), the memory-mapped `VectorStore`, the
//! `MetadataStore` trait + in-memory impl, and the `Shard` type that
//! composes those with a [`kova_index::Index`]. Crash-recovery semantics
//! follow the standard log-then-mutate discipline : every mutation hits a
//! fsynced WAL record before any in-memory or on-disk state changes.

mod atomic;
mod error;
mod metadata_store;
mod shard;
mod vector_store;
mod wal;
pub use atomic::atomic_write;
pub use error::KovaStorageError;
pub use metadata_store::FileMetadataStore;
pub use shard::{SearchHit, Shard, ShardError};
pub use vector_store::MmapVectorStore;
pub use wal::{FileWal, InMemoryWal, Lsn, Record, Wal};

//!wal or write-ahead log for Kova.
//!
//! The WAL is the source of truth for all mutations to
//! the index and vector store.
mod in_memory;
mod record;
pub use in_memory::InMemoryWal;
pub use record::Record;
use serde::{Deserialize, Serialize};

use crate::KovaStorageError;

/// Log sequence number, a strictly increasing identifier for WAL records.
#[repr(transparent)] // newtype wrapper around u64 for type safety and clarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Lsn(u64);

impl Lsn {
    /// The "before any record" sentinel. Used as the `from` argument to
    /// [`Wal::iter_from`] when you want to replay the entire log.
    pub const ZERO: Self = Self(0);

    /// Construct an `Lsn` from a raw `u64`.
    #[must_use]
    pub const fn new(n: u64) -> Self {
        Self(n)
    }

    /// Extract the inner `u64`.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Append-only log with crash-survival semantics.
///
/// Callers must `append` followed by `sync` to be durable : a successful
/// `sync()` means every preceding `append`'d record will survive process
/// death. `append` alone may live only in OS page cache.
pub trait Wal {
    /// Append a record. Returns the LSN it was assigned.
    /// Does not fsync : caller must explicitly [`Self::sync`] for durability.
    fn append(&mut self, record: &Record) -> Result<Lsn, KovaStorageError>;

    /// Flush any buffered writes and fsync to disk.
    fn sync(&mut self) -> Result<(), KovaStorageError>;

    /// Iterate over records with LSN `>= from`, in LSN order.
    fn iter_from(
        &self,
        from: Lsn,
    ) -> impl Iterator<Item = Result<(Lsn, Record), KovaStorageError>> + '_;

    /// Drop records with LSN `< before`. Used after a checkpoint.
    fn truncate_before(&mut self, before: Lsn) -> Result<(), KovaStorageError>;
}

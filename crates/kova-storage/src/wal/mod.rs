//!wal or write-ahead log for Kova.
//!
//! The WAL is the source of truth for all mutations to
//! the index and vector store.
mod file_wal;
mod in_memory;
mod record;
pub use file_wal::FileWal;
pub use in_memory::InMemoryWal;
pub use record::Record;
use serde::{Deserialize, Serialize};

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
///
/// The error type is an associated type so implementations targeting
/// non-filesystem backends (S3, GCS, distributed log) can surface their
/// own error universe rather than cramming everything into the
/// filesystem-shaped [`KovaStorageError`].
pub trait Wal {
    /// Error type returned by all `Wal` operations.
    ///
    /// Bounded as `Error + Send + Sync + 'static` so the generic
    /// composition layer (`kova-storage::Shard`) can box it into a
    /// `Box<dyn Error + Send + Sync>` uniformly with other backend errors.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Append a record. Returns the LSN it was assigned.
    /// Does not fsync : caller must explicitly [`Self::sync`] for durability.
    fn append(&mut self, record: &Record) -> Result<Lsn, Self::Error>;

    /// Flush any buffered writes and fsync to disk.
    fn sync(&mut self) -> Result<(), Self::Error>;

    /// Iterate over records with LSN `>= from`, in LSN order.
    fn iter_from(&self, from: Lsn)
    -> impl Iterator<Item = Result<(Lsn, Record), Self::Error>> + '_;

    /// Drop records with LSN `< before`. Used after a checkpoint.
    fn truncate_before(&mut self, before: Lsn) -> Result<(), Self::Error>;

    /// Last LSN that has been appended (durably or not). `None` if the
    /// log is empty (no records ever appended, or all were truncated).
    ///
    /// Used by checkpoint to capture the LSN up to which the snapshot
    /// covers : everything `<= last_lsn` at capture time is baked into
    /// the snapshot ; anything appended after is in the (truncated) WAL.
    fn last_lsn(&self) -> Option<Lsn>;
}

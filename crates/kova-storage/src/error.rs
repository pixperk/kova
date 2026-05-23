use thiserror::Error;

use crate::wal::Lsn;
/// `KovaStorageError` enumerates all errors that can occur in the storage layer, including I/O errors, data corruption, encoding/decoding issues, and invalid operations. Each variant provides context about the error to aid in debugging and error handling.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum KovaStorageError {
    /// I/O error from the underlying filesystem.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Record framing corrupted : length, CRC, or payload didn't validate.
    #[error("corrupt WAL record: {reason}")]
    CorruptRecord {
        /// Human-readable description of what went wrong.
        reason: String,
    },

    /// Bincode failed to decode a record's payload.
    #[error("decode error: {0}")]
    Decode(#[from] bincode::Error),

    /// Bincode failed to encode a record.
    #[error("encode error: {0}")]
    Encode(bincode::Error),

    /// Truncation target is beyond the log's end.
    #[error("invalid truncation lsn: requested {requested:?}, max {max:?}")]
    InvalidTruncationLsn {
        /// The LSN the caller asked to truncate to.
        requested: Lsn,
        /// The actual highest LSN in the log.
        max: Lsn,
    },
}

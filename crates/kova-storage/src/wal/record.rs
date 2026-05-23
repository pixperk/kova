//! WAL record framing : the on-disk frame format for [`Record`] values
//! (length + CRC + bincode payload) plus encode/decode helpers.

use kova_core::{Vector, VectorId};
use serde::{Deserialize, Serialize};

/// A single mutation applied to a shard. Persisted in the WAL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Record {
    /// Insert a new vector with the given id.
    Insert {
        /// Identifier the caller assigned.
        id: VectorId,
        /// The vector being inserted.
        vector: Vector,
    },
    /// Delete the vector with the given id.
    Delete {
        /// Identifier to remove.
        id: VectorId,
    },
}

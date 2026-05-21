//! Vector ID is the typed identifier for vectors in Kova. It is a newtype wrapper around `u64` that provides type safety and a clear distinction from other `u64` values. The `VectorId`
//! type is used throughout Kova to identify vectors in a consistent and type-safe way.

use std::fmt;

/// A typed identifier for vectors in Kova.
/// This is a newtype wrapper around `u64` that provides type safety and a clear distinction from other `u64` values.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VectorId(u64);

impl VectorId {
    /// Creates a new `VectorId` from a `u64` value.
    #[must_use]
    pub const fn new(id: u64) -> Self {
        VectorId(id)
    }

    /// Returns the underlying `u64` value of the `VectorId`.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for VectorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        //forward to inner u64's Display
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_id() {
        let id = VectorId::new(42);
        assert_eq!(id.get(), 42);
    }

    #[test]
    fn test_vector_id_equality() {
        let id1 = VectorId::new(42);
        let id2 = VectorId::new(42);
        let id3 = VectorId::new(43);
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_vector_id_display() {
        let id = VectorId::new(42);
        assert_eq!(id.to_string(), "42");
    }
}

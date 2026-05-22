//! Crate-internal helper types for scored candidates.
//!
//! [`ScoredId`] pairs a [`VectorId`] with an `f32` distance and defines a
//! total ordering on the distance via [`f32::total_cmp`]. This is what lets
//! us put scored candidates into a [`std::collections::BinaryHeap`] despite
//! `f32` not implementing `Ord`.
//!
//! Used by both [`crate::FlatIndex`] and (later) the HNSW index.

use std::cmp::Ordering;

use kova_core::VectorId;

/// A `(VectorId, distance)` pair, totally ordered by distance.
///
/// `BinaryHeap<ScoredId>` is a max-heap over the distance (top = farthest).
/// Wrap in [`std::cmp::Reverse`] to get a min-heap (top = closest), as needed
/// by HNSW's candidate frontier.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScoredId {
    pub(crate) id: VectorId,
    pub(crate) distance: f32,
}

impl PartialEq for ScoredId {
    fn eq(&self, other: &Self) -> bool {
        self.distance.total_cmp(&other.distance) == Ordering::Equal
    }
}

impl Eq for ScoredId {}

impl PartialOrd for ScoredId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.total_cmp(&other.distance)
    }
}

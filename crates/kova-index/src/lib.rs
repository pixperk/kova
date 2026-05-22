//! Vector indexes for Kova.
//!
//! Hosts the [`Index`] trait, the brute-force baseline that every other index
//! is benchmarked against, and (later) the HNSW implementation.

#![forbid(unsafe_code)]
mod error;
mod flat;
mod hnsw;
mod index;
mod scored;

pub use error::KovaIndexError;
pub use flat::FlatIndex;
pub use hnsw::{HnswIndex, HnswParams};
pub use index::Index;

//! Foundational types for Kova.
//!
//! This crate is the bedrock every other Kova crate builds on. It owns the
//! [`Vector`] type, the [`VectorId`] newtype, and the [`Distance`] trait with
//! its concrete metric implementations. Nothing here is async; everything is
//! CPU-bound.
//!
//! Phase 1 of the roadmap fills in `vector`, `id`, `distance`, and `error`.

#![forbid(unsafe_code)]

mod distance;
mod error;
mod id;
mod vector;
mod vector_store;
pub use distance::{Cosine, Distance, InnerProduct, L2};
pub use error::KovaError;
pub use id::VectorId;
pub use vector::Vector;
pub use vector_store::{InMemoryVectorStore, VectorStore};

//! `Shard` : the composition layer.
//!
//! Composes one of each persistence primitive ([`Wal`], [`VectorStore`],
//! [`MetadataStore`]) and an in-memory [`HnswIndex`] on top of the vector
//! store. Generic over all four so the same `Shard` type drives both the
//! production combo (`FileWal + MmapVectorStore + FileMetadataStore`) and
//! test fixtures (`InMemoryWal + InMemoryVectorStore + InMemoryMetadataStore`),
//! and so future backends (S3, distributed log, columnar metadata) plug in
//! without touching this file.
//!
//! The load-bearing rule is **log-then-mutate** : every `insert` appends a
//! [`Record`](crate::Record) to the WAL and `sync`s it before any in-memory
//! or on-disk state changes. On crash, the WAL is the witness ; recovery
//! replays it on reopen, idempotently re-applying every record to the
//! stores and the index.
//!
//! # Module layout
//!
//! This file (`mod.rs`) holds the cross-cutting pieces : the `Shard`
//! struct, `ShardError`, `SearchHit`, the constructors (`from_parts` /
//! `from_parts_seeded`), the read-only accessors (`contains`, `len`,
//! `is_empty`), and the private `replay` driver. Per-operation impls
//! live in sibling files :
//!
//! - [`insert`] : `insert`, `insert_many` (the 3-phase log-then-mutate)
//! - [`delete`] : `delete`, `vacuum`
//! - [`search`] : `search`
//! - [`open`] : the concrete `Shard::open` / `open_seeded` for the
//!   file-backed primitives, plus their tests
//!
//! Mirrors the per-operation layout `kova-index/src/hnsw/` uses for HNSW.
//!
//! # Layout on disk (when composed with file-backed primitives)
//!
//! ```text
//! data_dir/
//!   wal/             <- FileWal segments
//!   vectors.mmap     <- MmapVectorStore file
//!   metadata.bin     <- FileMetadataStore file
//! ```

use std::error::Error as StdError;

use kova_core::{Distance, Metadata, MetadataStore, VectorId, VectorStore};
use kova_index::{HnswIndex, HnswParams, Index, KovaIndexError};
use thiserror::Error;

use crate::{Lsn, Record, Wal};

mod delete;
mod insert;
mod open;
mod search;

/// Default RNG seed for [`Shard::from_parts`] / [`Shard::open`], mirroring
/// `HnswIndex::new`'s default. Tests that need a different seed call
/// [`Shard::from_parts_seeded`] / [`Shard::open_seeded`].
pub(super) const DEFAULT_SEED: u64 = 0xDEAD_BEEF_DEAD_BEEF;

/// Errors produced by [`Shard`] operations.
///
/// Two-level shape :
///
/// - [`Self::Index`] carries `KovaIndexError` directly (the index lives
///   inside `kova-index`, which has its own concrete error type).
/// - [`Self::Backend`] boxes any of `V::Error`, `M::Error`, or `W::Error`.
///   Generic composition cannot enumerate the three at the type level, but
///   `Box<dyn Error + Send + Sync>` preserves the full error chain so
///   callers can `downcast` if they need the original type.
///
/// Marked `#[non_exhaustive]` so future variants (`DimMismatch`,
/// `OperationRejected`, etc.) can land without a breaking change.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ShardError {
    /// Error from the index layer.
    #[error(transparent)]
    Index(#[from] KovaIndexError),

    /// Error from one of the composed backend primitives (WAL, vector
    /// store, metadata store). Boxed because the three error types differ
    /// per impl and aren't knowable at the `Shard` definition site.
    #[error("backend error: {0}")]
    Backend(#[source] Box<dyn StdError + Send + Sync + 'static>),
}

impl ShardError {
    /// Wrap any backend error into [`Self::Backend`]. Call-site shorthand :
    /// `.map_err(ShardError::backend)?` instead of `.map_err(|e| ShardError::Backend(Box::new(e)))?`.
    #[must_use]
    pub fn backend<E>(err: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::Backend(Box::new(err))
    }
}

/// A single search result returned by [`Shard::search`].
///
/// Carries the matched id, its distance to the query under the shard's
/// configured metric (smaller = closer, per the [`Distance`] convention),
/// and the metadata attached to that id at insert time. Metadata is
/// returned alongside the hit because the whole point of having a
/// metadata store is that callers can read attributes on the results path
/// without a second round-trip.
#[derive(Debug, Clone)]
pub struct SearchHit {
    /// The matched vector's id.
    pub id: VectorId,
    /// Distance under the shard's configured metric. Smaller is closer.
    pub distance: f32,
    /// Attribute bag attached to `id`. Empty if the insert carried no metadata.
    pub metadata: Metadata,
}

/// The composition layer : ties a [`Wal`], a [`VectorStore`], a
/// [`MetadataStore`], and an [`HnswIndex`] together under a single
/// log-then-mutate discipline.
///
/// Generic over all four primitives so the same struct drives production
/// (file/mmap backends) and tests (in-memory backends), and so future
/// backends (S3, distributed log) compose without code changes here.
///
/// See the module-level docs for the on-disk layout (when composed with
/// file-backed primitives) and the recovery model.
pub struct Shard<D, V, M, W>
where
    D: Distance,
    V: VectorStore,
    M: MetadataStore,
    W: Wal,
{
    pub(super) index: HnswIndex<D, V>,
    pub(super) metadata: M,
    pub(super) wal: W,
}

impl<D, V, M, W> Shard<D, V, M, W>
where
    D: Distance,
    V: VectorStore,
    M: MetadataStore,
    W: Wal,
{
    /// Compose a shard from already-constructed primitives, using the
    /// default RNG seed for the index. Runs WAL replay before returning,
    /// so the index catches up with whatever state the stores were left in.
    ///
    /// # Errors
    /// Returns [`ShardError`] if WAL iteration or any replayed mutation fails.
    pub fn from_parts(
        metric: D,
        params: HnswParams,
        vectors: V,
        metadata: M,
        wal: W,
    ) -> Result<Self, ShardError> {
        Self::from_parts_seeded(metric, params, DEFAULT_SEED, vectors, metadata, wal)
    }

    /// Like [`Self::from_parts`] but with an explicit RNG seed. Used by
    /// tests that need reproducible graph construction.
    ///
    /// # Errors
    /// See [`Self::from_parts`].
    pub fn from_parts_seeded(
        metric: D,
        params: HnswParams,
        seed: u64,
        vectors: V,
        metadata: M,
        wal: W,
    ) -> Result<Self, ShardError> {
        let index = HnswIndex::seeded_with_store(metric, params, seed, vectors);
        let mut shard = Self {
            index,
            metadata,
            wal,
        };
        shard.replay()?;
        Ok(shard)
    }

    /// Whether the shard currently holds a live entry for `id`.
    ///
    /// Returns `false` for ids that were inserted then deleted : the
    /// graph node is still in memory (until vacuum) but the id is
    /// logically gone, and that's what callers care about.
    ///
    /// O(1) ; does not touch the vector store or metadata store.
    #[must_use]
    pub fn contains(&self, id: VectorId) -> bool {
        self.index.top_layer_of(id).is_some() && !self.index.is_tombstoned(id)
    }

    /// Number of live (non-tombstoned) vectors in the shard.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len() - self.index.tombstone_count()
    }

    /// Whether the shard has any live entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Replay every WAL record (LSN order) into the index + stores.
    ///
    /// Called once during construction. The stores may already hold
    /// persisted state from a previous run ; the index is always fresh
    /// (graph structure is in-memory only). Every backend `put` is
    /// idempotent (overwrite-in-place), so re-applying a record that was
    /// already partially applied before a crash is safe.
    ///
    /// We materialise the records into a `Vec` up front because the
    /// `iter_from` borrow conflicts with the mutable borrows we need on
    /// `self.index` / `self.metadata` to apply them.
    pub(super) fn replay(&mut self) -> Result<(), ShardError> {
        let records: Vec<(Lsn, Record)> = self
            .wal
            .iter_from(Lsn::ZERO)
            .collect::<Result<_, _>>()
            .map_err(ShardError::backend)?;

        for (_lsn, record) in records {
            match record {
                Record::Insert {
                    id,
                    vector,
                    metadata,
                } => {
                    self.index.insert(id, vector)?;
                    self.metadata
                        .put(id, metadata)
                        .map_err(ShardError::backend)?;
                }
                Record::Delete { id } => {
                    // Tombstone in the index + drop from metadata. The
                    // graph node and vector bytes stay (vacuum reclaims).
                    //
                    // Ordering matters : `Shard::insert` rejects duplicate
                    // ids by graph-node presence, so the WAL never holds
                    // `Delete{id}` without a prior `Insert{id}` in the
                    // same log (the crash test inserts then dies, but
                    // the matching Insert is in the same WAL).
                    self.index.tombstone(id)?;
                    self.metadata.delete(id).map_err(ShardError::backend)?;
                }
            }
        }
        Ok(())
    }
}

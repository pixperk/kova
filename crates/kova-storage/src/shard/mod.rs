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
use std::path::PathBuf;

use kova_core::{Distance, Metadata, MetadataStore, Value, VectorId, VectorStore};
use kova_index::{HnswIndex, HnswParams, Index, KovaIndexError};
use kova_meta_index::{IndexCatalog, IndexKind, StatsBuilder, StatsCatalog};
use thiserror::Error;

use crate::{Lsn, Record, Wal};

mod checkpoint;
mod delete;
mod insert;
mod open;
mod search;
mod update;

pub use checkpoint::CheckpointPolicy;

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
    /// In-memory catalog of secondary indexes on metadata fields.
    /// Empty after open ; populated via
    /// [`Shard::add_hash_index`] / [`Shard::add_btree_index`] /
    /// [`Shard::add_inverted_index`]. Maintained synchronously in
    /// phase 3 of every mutation, after the WAL commit.
    pub(super) catalog: IndexCatalog,
    /// Per-shard column statistics. Refreshed at checkpoint time by
    /// walking the metadata store, persisted as
    /// `stats.{snapshot_id}.bin` alongside the catalog file.
    /// Consulted by the planner's `ShardEstimator` for unindexed
    /// predicates ; stays stale between checkpoints, which is fine
    /// for selectivity estimation.
    pub(super) stats: StatsCatalog,
    /// Data directory, populated by [`Shard::open`] for the file-backed
    /// combo. `None` for in-memory composition (`from_parts`) ; in that
    /// case checkpoint is a no-op (nothing to write to).
    pub(super) dir: Option<PathBuf>,
    /// Suffix on the live `graph.{snapshot_id}.snapshot` file, or `0` if
    /// no checkpoint has run yet. Each [`Shard::checkpoint`] increments
    /// this and atomic-commits the new value through the manifest.
    pub(super) snapshot_id: u64,
    /// LSN captured by the last successful checkpoint. `Lsn::ZERO` if no
    /// checkpoint has run. Used by `should_checkpoint` to estimate
    /// "records since last checkpoint" against `wal.last_lsn()`.
    pub(super) checkpoint_lsn: Lsn,
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
            catalog: IndexCatalog::new(),
            stats: StatsCatalog::new(),
            // In-memory composition has no on-disk directory and no
            // checkpoint state. The file-backed `Shard::open` populates
            // these via `Self::from_parts_with_checkpoint_state` below.
            dir: None,
            snapshot_id: 0,
            checkpoint_lsn: Lsn::ZERO,
        };
        shard.replay_from(Lsn::ZERO)?;
        Ok(shard)
    }

    /// Same as [`Self::from_parts_seeded`] but with explicit checkpoint
    /// state and a starting replay LSN. Called by the file-backed
    /// `Shard::open` after it loads a snapshot ; in-memory callers stick
    /// with `from_parts_seeded`.
    ///
    /// `catalog` is the secondary-index catalog loaded from
    /// `catalog.{snapshot_id}.bin`, or `None` if the file didn't
    /// exist (no checkpoint had been taken with indexes
    /// registered). `stats` is the analogous column-stats catalog
    /// loaded from `stats.{snapshot_id}.bin`. Either may be
    /// `None` ; replay forwards post-checkpoint records into the
    /// index catalog so it catches up with the present, but the
    /// stats catalog stays at its checkpointed values until the
    /// next checkpoint rebuilds it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_parts_with_checkpoint_state(
        index: HnswIndex<D, V>,
        metadata: M,
        wal: W,
        catalog: Option<IndexCatalog>,
        stats: Option<StatsCatalog>,
        dir: Option<PathBuf>,
        snapshot_id: u64,
        checkpoint_lsn: Lsn,
        replay_from: Lsn,
    ) -> Result<Self, ShardError> {
        let mut shard = Self {
            index,
            metadata,
            wal,
            catalog: catalog.unwrap_or_default(),
            stats: stats.unwrap_or_default(),
            dir,
            snapshot_id,
            checkpoint_lsn,
        };
        shard.replay_from(replay_from)?;
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

    /// Read-only access to the shard's column statistics. The stats
    /// reflect the metadata store as it was at the last successful
    /// checkpoint ; mutations between checkpoints are not visible
    /// here. The planner reads these for selectivity estimates on
    /// unindexed predicates.
    #[must_use]
    pub fn stats(&self) -> &StatsCatalog {
        &self.stats
    }

    /// Vector dimension pinned on the underlying HNSW index, or the
    /// store's dim if the index hasn't seen its first insert yet.
    /// Returns `None` for a brand-new in-memory shard that's never
    /// observed a vector.
    ///
    /// Used by the planner's cost model to weight distance-compute
    /// costs in plan A / B / C ranking.
    #[must_use]
    pub fn dim(&self) -> Option<usize> {
        self.index.dim().or_else(|| self.index.store_dim())
    }

    /// Walk the live metadata store and rebuild the stats catalog
    /// from scratch. Called by [`Self::checkpoint`] after vacuum so
    /// the persisted stats reflect the post-vacuum row set.
    ///
    /// O(N) walk of the metadata store. For million-row shards this
    /// takes hundreds of milliseconds ; that's a one-time cost per
    /// checkpoint, not per query.
    fn rebuild_stats(&mut self) {
        let mut builder = StatsBuilder::new();
        let ids = self.metadata.scan_ids(|_| true);
        for id in ids {
            if let Some(meta) = self.metadata.get(id) {
                builder.observe(&meta);
            }
        }
        self.stats = builder.finish();
    }

    /// Read-only access to the shard's secondary-index catalog. Use
    /// [`IndexCatalog::lookup`] or
    /// [`IndexCatalog::estimate`] to query the indexes registered via
    /// [`Self::add_hash_index`] (or its siblings).
    #[must_use]
    pub fn catalog(&self) -> &IndexCatalog {
        &self.catalog
    }

    /// Register an anonymous [`HashIndex`](kova_meta_index::HashIndex)
    /// on `field` and backfill it from the shard's current
    /// metadata. The new index is then maintained automatically by
    /// every subsequent `insert`/`delete`/`update` op.
    ///
    /// The "anonymous" qualifier matters : this index has no name
    /// in the catalog's registry, so it cannot be dropped by
    /// `DROP INDEX`. It exists solely for code paths that want to
    /// install an index without going through DDL. For named
    /// (DDL-style) indexes that survive reopen, use
    /// [`Self::create_index`].
    ///
    /// # Errors
    /// Returns [`ShardError::Backend`] wrapping
    /// [`kova_meta_index::KovaMetaIndexError::IndexAlreadyExists`]
    /// if a hash index already exists on `field`. Each
    /// `(field, kind)` slot holds at most one index.
    ///
    /// # Durability
    /// The new index is **transient** until the next successful
    /// [`Self::checkpoint`]. Anonymous adds are NOT WAL-persisted
    /// (unlike [`Self::create_index`]), so a close without a
    /// checkpoint loses them. Tests are the primary caller, and
    /// tests don't reopen ; production code paths should use
    /// [`Self::create_index`].
    pub fn add_hash_index(&mut self, field: &str) -> Result<(), ShardError> {
        self.catalog
            .add_hash_index(field)
            .map_err(ShardError::backend)?;
        self.backfill_kind(field, kova_meta_index::IndexKind::Hash);
        Ok(())
    }

    /// Anonymous [`BTreeIndex`](kova_meta_index::BTreeIndex)
    /// counterpart of [`Self::add_hash_index`].
    ///
    /// # Errors
    /// See [`Self::add_hash_index`].
    pub fn add_btree_index(&mut self, field: &str) -> Result<(), ShardError> {
        self.catalog
            .add_btree_index(field)
            .map_err(ShardError::backend)?;
        self.backfill_kind(field, kova_meta_index::IndexKind::Btree);
        Ok(())
    }

    /// Anonymous
    /// [`InvertedIndex`](kova_meta_index::InvertedIndex)
    /// counterpart of [`Self::add_hash_index`].
    ///
    /// # Errors
    /// See [`Self::add_hash_index`].
    pub fn add_inverted_index(&mut self, field: &str) -> Result<(), ShardError> {
        self.catalog
            .add_inverted_index(field)
            .map_err(ShardError::backend)?;
        self.backfill_kind(field, kova_meta_index::IndexKind::Inverted);
        Ok(())
    }

    /// DDL : register a **named** secondary index of `kind` on
    /// `field` and backfill it from current metadata. The mutation
    /// hooks ([`Self::add_hash_index`] et al.) handle live ops from
    /// here on out.
    ///
    /// Distinct from the anonymous [`Self::add_hash_index`] family
    /// in two ways :
    ///
    /// - The catalog remembers the `name` so
    ///   [`Self::drop_index`] can resolve it later.
    /// - Duplicate names are rejected loudly via
    ///   [`KovaMetaIndexError::IndexNameInUse`].
    ///
    /// # Durability
    /// Same checkpoint-gated contract as
    /// [`Self::add_hash_index`] : the index is transient until the
    /// next [`Self::checkpoint`].
    ///
    /// # Errors
    /// Returns [`ShardError::Backend`] wrapping
    /// [`KovaMetaIndexError::IndexNameInUse`] when `name` is
    /// already registered in the catalog.
    pub fn create_index(
        &mut self,
        name: &str,
        field: &str,
        kind: IndexKind,
    ) -> Result<(), ShardError> {
        // Phase 1 : pre-commit validation. Catch duplicate name and
        // slot-occupied conditions before the WAL is touched.
        if self.catalog.resolve_name(name).is_some() {
            return Err(ShardError::backend(
                kova_meta_index::KovaMetaIndexError::IndexNameInUse {
                    name: name.to_string(),
                },
            ));
        }
        if self.catalog.has_kind_on(field, kind) {
            return Err(ShardError::backend(
                kova_meta_index::KovaMetaIndexError::IndexAlreadyExists {
                    field: field.to_string(),
                    kind,
                },
            ));
        }

        // Phase 2 : commit. After sync returns Ok the DDL is durable.
        let record = Record::CreateIndex {
            name: name.to_string(),
            field: field.to_string(),
            kind,
        };
        self.wal.append(&record).map_err(ShardError::backend)?;
        self.wal.sync().map_err(ShardError::backend)?;

        // Phase 3 : apply. Phase 1 validated, so this must succeed ;
        // if the catalog disagrees the WAL has committed something
        // we can't apply, which is a broken invariant. Panic so
        // replay can reconcile.
        if let Err(e) = self.catalog.create_named_index(name, field, kind) {
            panic!(
                "Shard::create_index phase-3 apply failure on catalog.create_named_index: {e:?} \
                 (WAL has committed the record ; aborting so replay can reconcile)"
            );
        }
        self.backfill_kind(field, kind);
        Ok(())
    }

    /// DDL : drop the named secondary index. Removes both the name
    /// registration and the underlying index of the kind it
    /// pointed to.
    ///
    /// # Durability
    /// Like [`Self::create_index`] : the drop only persists when
    /// the next [`Self::checkpoint`] writes the catalog out. Before
    /// that, a reopen still sees the index.
    ///
    /// # Errors
    /// Returns [`ShardError::Backend`] wrapping
    /// [`KovaMetaIndexError::UnknownIndexName`] when `name` was
    /// never registered.
    pub fn drop_index(&mut self, name: &str) -> Result<(), ShardError> {
        // Phase 1 : pre-commit validation.
        if self.catalog.resolve_name(name).is_none() {
            return Err(ShardError::backend(
                kova_meta_index::KovaMetaIndexError::UnknownIndexName {
                    name: name.to_string(),
                },
            ));
        }

        // Phase 2 : commit.
        let record = Record::DropIndex {
            name: name.to_string(),
        };
        self.wal.append(&record).map_err(ShardError::backend)?;
        self.wal.sync().map_err(ShardError::backend)?;

        // Phase 3 : apply.
        if let Err(e) = self.catalog.drop_named_index(name) {
            panic!(
                "Shard::drop_index phase-3 apply failure on catalog.drop_named_index: {e:?} \
                 (WAL has committed the record ; aborting so replay can reconcile)"
            );
        }
        Ok(())
    }

    /// Bulk-load **one specific index** (of `kind` on `field`) from
    /// the metadata store's current contents.
    ///
    /// Single-pass walk via [`MetadataStore::walk_field`] : one
    /// iteration of the in-memory map, no per-row re-fetches.
    /// Targeted insert via [`IndexCatalog::populate_kind`] : the
    /// rows only land in the index just registered, not in
    /// pre-existing indexes on the same field.
    ///
    /// Every caller (the three `add_*_index` arms, `create_index`,
    /// and the replay-side `Record::CreateIndex` arm) knows the
    /// kind ahead of time, so the targeted version replaces the
    /// older broadcast-style `backfill_field` entirely.
    fn backfill_kind(&mut self, field: &str, kind: kova_meta_index::IndexKind) {
        let mut rows: Vec<(VectorId, Value)> = Vec::new();
        self.metadata
            .walk_field(field, |id, value| rows.push((id, value.clone())));
        self.catalog.populate_kind(field, kind, rows);
    }

    /// Replay WAL records starting from `from` (inclusive) into the
    /// index + stores.
    ///
    /// Called once during construction. Pre-snapshot callers replay
    /// from `Lsn::ZERO` ; post-checkpoint callers replay from
    /// `manifest.checkpoint_lsn + 1` since records `<=` that LSN are
    /// already baked into the loaded snapshot. The stores may already
    /// hold persisted state from a previous run ; every backend `put`
    /// is idempotent (overwrite-in-place), so re-applying a record that
    /// was already partially applied before a crash is safe.
    ///
    /// We materialise the records into a `Vec` up front because the
    /// `iter_from` borrow conflicts with the mutable borrows we need on
    /// `self.index` / `self.metadata` to apply them.
    pub(super) fn replay_from(&mut self, from: Lsn) -> Result<(), ShardError> {
        let records: Vec<(Lsn, Record)> = self
            .wal
            .iter_from(from)
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
                    // Catalog observes the row before the store
                    // consumes the bag (same ordering rule as the
                    // live `Shard::insert` path).
                    self.catalog.on_insert(id, &metadata);
                    self.metadata
                        .put(id, metadata)
                        .map_err(ShardError::backend)?;
                }
                Record::Delete { id, old_metadata } => {
                    // Tombstone in the index + drop from metadata. The
                    // graph node and vector bytes stay (vacuum reclaims).
                    //
                    // The record carries the metadata bag from delete
                    // time, so the catalog can clear the row from every
                    // bucket without depending on the (eagerly-mutated)
                    // metadata store still having it.
                    self.index.tombstone(id)?;
                    self.metadata.delete(id).map_err(ShardError::backend)?;
                    self.catalog.on_delete(id, &old_metadata);
                }
                Record::DeleteMany { items } => {
                    // Same semantics as a sequence of `Delete{id, old}`
                    // records, compacted into one frame at write time.
                    // Each id is applied independently ; partial failure
                    // on one id surfaces as a `ShardError` and aborts
                    // replay (same policy as the singleton path).
                    for (id, old_metadata) in items {
                        self.index.tombstone(id)?;
                        self.metadata.delete(id).map_err(ShardError::backend)?;
                        self.catalog.on_delete(id, &old_metadata);
                    }
                }
                Record::UpdateMetadata {
                    id,
                    old_metadata,
                    metadata,
                } => {
                    // Replace the metadata bag in full. The HNSW graph
                    // and vector store stay untouched. Idempotent : the
                    // bag is whatever the last `UpdateMetadata` record
                    // said it should be.
                    //
                    // Catalog uses the (old, new) pair carried in the
                    // record so it doesn't depend on the live store
                    // state, which has already been overwritten by
                    // prior `put`s.
                    self.catalog.on_update(id, &old_metadata, &metadata);
                    self.metadata
                        .put(id, metadata)
                        .map_err(ShardError::backend)?;
                }
                Record::CreateIndex { name, field, kind } => {
                    // Replay-time index registration. Mirrors the
                    // live path's phase-3 apply : catalog gets the
                    // new index, then `backfill_kind` walks the
                    // metadata store and populates the specific
                    // index just registered.
                    //
                    // Backfill correctness during replay relies on
                    // every preceding `Insert` / `Delete` /
                    // `UpdateMetadata` record having been applied
                    // to `self.metadata` already, which the WAL's
                    // strict ordering plus this `for` loop's
                    // ordering guarantees.
                    self.catalog
                        .create_named_index(&name, &field, kind)
                        .map_err(ShardError::backend)?;
                    self.backfill_kind(&field, kind);
                }
                Record::DropIndex { name } => {
                    self.catalog
                        .drop_named_index(&name)
                        .map_err(ShardError::backend)?;
                }
            }
        }
        Ok(())
    }
}

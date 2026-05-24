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
//! # Layout on disk (when composed with file-backed primitives)
//!
//! ```text
//! data_dir/
//!   wal/             <- FileWal segments
//!   vectors.mmap     <- MmapVectorStore file
//!   metadata.bin     <- FileMetadataStore file
//! ```

use std::error::Error as StdError;
use std::path::Path;

use kova_core::{Distance, Metadata, MetadataStore, Vector, VectorId, VectorStore};
use kova_index::{HnswIndex, HnswParams, Index, KovaIndexError};
use thiserror::Error;

use crate::{FileMetadataStore, FileWal, Lsn, MmapVectorStore, Record, Wal};

/// Default RNG seed for [`Shard::from_parts`] / [`Shard::open`], mirroring
/// `HnswIndex::new`'s default. Tests that need a different seed call
/// [`Shard::from_parts_seeded`] / [`Shard::open_seeded`].
const DEFAULT_SEED: u64 = 0xDEAD_BEEF_DEAD_BEEF;

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
    index: HnswIndex<D, V>,
    metadata: M,
    wal: W,
}

// -----------------------------------------------------------------------------
// Generic impl : works for any compatible quartet of primitives.
// -----------------------------------------------------------------------------

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

    /// Insert `(id, vector, metadata)` under the log-then-mutate discipline :
    ///
    /// 1. Reject duplicate ids upfront (cheap O(1) `HashMap` check on the index).
    /// 2. Build a [`Record::Insert`] and `append` + `sync` it to the WAL.
    ///    After `sync` returns Ok, the WAL is the durable witness.
    /// 3. Apply to the in-memory index (which also writes through to the
    ///    [`VectorStore`]), then the [`MetadataStore`].
    ///
    /// If the process dies anywhere in step 3, replay on reopen will redo
    /// the operation idempotently. If it dies in step 1 or 2, the caller's
    /// `append`/`sync` returns Err and they know the write isn't durable.
    ///
    /// # Errors
    /// - [`ShardError::Index`] with `KovaIndexError::DuplicateId` if `id`
    ///   already exists in the shard.
    /// - [`ShardError::Backend`] from the WAL `append` / `sync`, the
    ///   vector store `put` (inside `index.insert`), or the metadata store `put`.
    pub fn insert(
        &mut self,
        id: VectorId,
        vector: Vector,
        metadata: Metadata,
    ) -> Result<(), ShardError> {
        // Cheap upfront duplicate check : `HashMap` lookup, no vector clone.
        // Failing here means no WAL append, no state change ; the WAL stays
        // clean of duplicate records by construction, which keeps replay
        // simple (it can assume every Insert is for a fresh id).
        if self.index.top_layer_of(id).is_some() {
            return Err(KovaIndexError::DuplicateId { id }.into());
        }

        // 1. Build the record. Clones are unavoidable : Record needs to own
        //    the data for serialisation, and the index needs it for insertion.
        let record = Record::Insert {
            id,
            vector: vector.clone(),
            metadata: metadata.clone(),
        };

        // 2. Durability barrier.
        self.wal.append(&record).map_err(ShardError::backend)?;
        self.wal.sync().map_err(ShardError::backend)?;

        // 3. Apply. `index.insert` writes through to the underlying VectorStore.
        self.index.insert(id, vector)?;
        self.metadata
            .put(id, metadata)
            .map_err(ShardError::backend)?;

        Ok(())
    }

    /// k-nearest search. Returns hits in increasing distance order, each
    /// with its attached metadata read from the metadata store.
    ///
    /// Missing metadata (e.g. an id present in the index but absent from
    /// the metadata store, which shouldn't happen under normal operation
    /// but can after partial recovery) is filled with an empty `Metadata`
    /// rather than failing the whole query.
    ///
    /// # Errors
    /// Returns [`ShardError::Index`] if the index search fails (e.g.
    /// dimension mismatch).
    pub fn search(&self, query: &Vector, k: usize) -> Result<Vec<SearchHit>, ShardError> {
        let hits = self.index.search(query, k)?;
        let results = hits
            .into_iter()
            .map(|(id, distance)| {
                let metadata = self.metadata.get(id).unwrap_or_default();
                SearchHit {
                    id,
                    distance,
                    metadata,
                }
            })
            .collect();
        Ok(results)
    }

    /// Whether the shard currently holds an entry for `id`.
    ///
    /// O(1) check against the index's in-memory node map ; does not touch
    /// the vector store or metadata store.
    #[must_use]
    pub fn contains(&self, id: VectorId) -> bool {
        self.index.top_layer_of(id).is_some()
    }

    /// Number of vectors currently in the shard.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the shard is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.len() == 0
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
    fn replay(&mut self) -> Result<(), ShardError> {
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
                    // HNSW delete lands in the delete milestone (tombstone +
                    // free-list). For now replay only propagates the delete
                    // to the metadata store ; the index is left as-is.
                    self.metadata.delete(id).map_err(ShardError::backend)?;
                }
            }
        }
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// Concrete impl : the production file-backed combo. Wires up `FileWal +
// MmapVectorStore + FileMetadataStore` from a single data directory.
// -----------------------------------------------------------------------------

impl<D: Distance> Shard<D, MmapVectorStore, FileMetadataStore, FileWal> {
    /// Open (or create) a file-backed shard rooted at `dir`. Uses the
    /// default RNG seed.
    ///
    /// On first open, creates the directory and empty backing files. On
    /// subsequent opens, the stores recover their persisted state and the
    /// WAL is replayed to rebuild the in-memory index.
    ///
    /// `dim` pins the vector dimension. If the underlying `vectors.mmap`
    /// already exists with a different dim, [`MmapVectorStore::open`]
    /// surfaces the mismatch as [`crate::KovaStorageError::CorruptRecord`].
    ///
    /// # Errors
    /// Any [`crate::KovaStorageError`] from the backing primitives bubbles
    /// up as [`ShardError::Backend`].
    pub fn open(
        dir: impl AsRef<Path>,
        dim: usize,
        metric: D,
        params: HnswParams,
    ) -> Result<Self, ShardError> {
        Self::open_seeded(dir, dim, metric, params, DEFAULT_SEED)
    }

    /// Like [`Self::open`] but with an explicit RNG seed.
    ///
    /// # Errors
    /// See [`Self::open`].
    pub fn open_seeded(
        dir: impl AsRef<Path>,
        dim: usize,
        metric: D,
        params: HnswParams,
        seed: u64,
    ) -> Result<Self, ShardError> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).map_err(ShardError::backend)?;

        let wal = FileWal::open(dir.join("wal")).map_err(ShardError::backend)?;
        let vectors =
            MmapVectorStore::open(dir.join("vectors.mmap"), dim).map_err(ShardError::backend)?;
        let metadata =
            FileMetadataStore::open(dir.join("metadata.bin")).map_err(ShardError::backend)?;

        Self::from_parts_seeded(metric, params, seed, vectors, metadata, wal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kova_core::{InMemoryMetadataStore, InMemoryVectorStore, L2, Value};
    use kova_index::HnswParams;
    use tempfile::tempdir;

    use crate::InMemoryWal;

    fn v(data: Vec<f32>) -> Vector {
        Vector::try_new(data).unwrap()
    }

    fn id(n: u64) -> VectorId {
        VectorId::new(n)
    }

    fn tag_meta(tag: &str) -> Metadata {
        let mut m = Metadata::new();
        m.insert("tag".into(), Value::String(tag.into()));
        m
    }

    /// Smoke test : compose in-memory primitives, insert a couple of
    /// vectors, search, verify hits + metadata. Exercises every public
    /// method on the generic impl in one flow.
    #[test]
    fn insert_then_search_with_in_memory_primitives() {
        let mut shard = Shard::from_parts(
            L2,
            HnswParams::default(),
            InMemoryVectorStore::new(),
            InMemoryMetadataStore::new(),
            InMemoryWal::new(),
        )
        .unwrap();

        assert!(shard.is_empty());
        assert_eq!(shard.len(), 0);

        shard
            .insert(id(1), v(vec![1.0, 0.0]), tag_meta("alpha"))
            .unwrap();
        shard
            .insert(id(2), v(vec![0.0, 1.0]), tag_meta("beta"))
            .unwrap();

        assert_eq!(shard.len(), 2);
        assert!(shard.contains(id(1)));
        assert!(!shard.contains(id(99)));

        let hits = shard.search(&v(vec![1.0, 0.05]), 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, id(1)); // nearest to (1.0, 0.05)
        assert_eq!(
            hits[0].metadata.get("tag"),
            Some(&Value::String("alpha".into()))
        );
    }

    /// Inserting the same id twice errors with `DuplicateId` and does not
    /// append a second WAL record (the duplicate check fires before any
    /// state change).
    #[test]
    fn duplicate_insert_is_rejected() {
        let mut shard = Shard::from_parts(
            L2,
            HnswParams::default(),
            InMemoryVectorStore::new(),
            InMemoryMetadataStore::new(),
            InMemoryWal::new(),
        )
        .unwrap();

        shard.insert(id(7), v(vec![1.0]), Metadata::new()).unwrap();
        let err = shard
            .insert(id(7), v(vec![2.0]), Metadata::new())
            .unwrap_err();
        match err {
            ShardError::Index(KovaIndexError::DuplicateId { id: got }) => {
                assert_eq!(got, id(7));
            }
            other => panic!("expected DuplicateId, got {other:?}"),
        }
        assert_eq!(shard.len(), 1);
    }

    // ---------- file-backed tests : exercise the concrete `Shard::open` path ----------

    /// First `open` on an empty directory creates the three backing files
    /// and the shard reports zero size.
    #[test]
    fn open_creates_empty_shard_and_files() {
        let dir = tempdir().unwrap();
        let shard = Shard::open(dir.path(), 3, L2, HnswParams::default()).unwrap();

        assert!(shard.is_empty());
        assert_eq!(shard.len(), 0);

        assert!(
            dir.path().join("wal").exists(),
            "wal/ directory should exist"
        );
        assert!(
            dir.path().join("vectors.mmap").exists(),
            "vectors.mmap should exist"
        );
        assert!(
            dir.path().join("metadata.bin").exists(),
            "metadata.bin should exist"
        );
    }

    /// Same insert+search flow as the in-memory smoke test, but through
    /// the file-backed combo. Confirms `Shard::open` wires up the right
    /// primitives end-to-end.
    #[test]
    fn insert_then_search_file_backed() {
        let dir = tempdir().unwrap();
        let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();

        shard
            .insert(id(1), v(vec![1.0, 0.0]), tag_meta("alpha"))
            .unwrap();
        shard
            .insert(id(2), v(vec![0.0, 1.0]), tag_meta("beta"))
            .unwrap();

        let hits = shard.search(&v(vec![1.0, 0.05]), 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, id(1));
        assert_eq!(
            hits[0].metadata.get("tag"),
            Some(&Value::String("alpha".into()))
        );
    }

    /// The whole point of the persistence layer : drop a shard, open it
    /// again on the same directory, and the inserts are still there with
    /// their metadata intact.
    ///
    /// Mechanically this exercises every recovery path :
    /// - `MmapVectorStore::open` walks slots to rebuild `id_to_slot`
    /// - `FileMetadataStore::open` deserializes the bincode blob
    /// - `FileWal::open` enumerates segments
    /// - `Shard::replay` walks the WAL and re-applies every record into
    ///   the fresh in-memory HNSW index
    #[test]
    fn reopen_recovers_inserted_vectors() {
        let dir = tempdir().unwrap();

        {
            let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
            shard
                .insert(id(1), v(vec![1.0, 0.0]), tag_meta("alpha"))
                .unwrap();
            shard
                .insert(id(2), v(vec![0.0, 1.0]), tag_meta("beta"))
                .unwrap();
            shard
                .insert(id(3), v(vec![0.5, 0.5]), tag_meta("gamma"))
                .unwrap();
        } // shard dropped : releases mmap + files

        let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
        assert_eq!(shard.len(), 3);
        assert!(shard.contains(id(1)));
        assert!(shard.contains(id(2)));
        assert!(shard.contains(id(3)));

        let hits = shard.search(&v(vec![1.0, 0.0]), 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, id(1));
        assert_eq!(
            hits[0].metadata.get("tag"),
            Some(&Value::String("alpha".into()))
        );
    }

    /// Reopening with a `dim` that doesn't match the existing
    /// `vectors.mmap` header surfaces as a `ShardError::Backend` whose
    /// source mentions the dim mismatch. Caller can downcast to inspect
    /// the original `KovaStorageError` if needed ; here we just check the
    /// message contains "dim".
    #[test]
    fn reopen_with_wrong_dim_errors() {
        let dir = tempdir().unwrap();

        {
            let mut shard = Shard::open(dir.path(), 3, L2, HnswParams::default()).unwrap();
            shard
                .insert(id(1), v(vec![1.0, 2.0, 3.0]), Metadata::new())
                .unwrap();
        }

        // Can't use `.unwrap_err()` : the Ok variant (`Shard`) isn't Debug
        // (HnswIndex isn't Debug, deliberately), so we destructure manually.
        let Err(err) = Shard::open(dir.path(), 4, L2, HnswParams::default()) else {
            panic!("expected error on dim mismatch");
        };
        match err {
            ShardError::Backend(ref source) => {
                let msg = format!("{source}");
                assert!(msg.contains("dim"), "expected dim error, got: {msg}");
            }
            other => panic!("expected Backend error, got {other:?}"),
        }
    }

    /// Replay is idempotent : opening, dropping, and opening again N
    /// times leaves the shard in the same state every time. This is the
    /// invariant the crash recovery test (next milestone) will lean on.
    #[test]
    fn replay_is_idempotent_across_multiple_reopens() {
        let dir = tempdir().unwrap();

        {
            let mut shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
            shard
                .insert(id(1), v(vec![1.0, 0.0]), tag_meta("a"))
                .unwrap();
            shard
                .insert(id(2), v(vec![0.0, 1.0]), tag_meta("b"))
                .unwrap();
        }

        for round in 0..3 {
            let shard = Shard::open(dir.path(), 2, L2, HnswParams::default()).unwrap();
            assert_eq!(shard.len(), 2, "round {round}: wrong len");
            assert!(shard.contains(id(1)), "round {round}: missing id 1");
            assert!(shard.contains(id(2)), "round {round}: missing id 2");
            let hits = shard.search(&v(vec![1.0, 0.0]), 1).unwrap();
            assert_eq!(hits[0].id, id(1), "round {round}: wrong nearest");
        }
    }
}

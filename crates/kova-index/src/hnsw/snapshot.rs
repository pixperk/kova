//! HNSW graph snapshot : serialise/deserialise the graph structure
//! (nodes + `entry_point` + dim) to a `Write` / `Read`, streamed via
//! `bincode::serialize_into` so there's no intermediate full-graph
//! `Vec<u8>` buffer.
//!
//! Used by `kova-storage::Shard::checkpoint` to write `graph.snapshot`
//! and by `kova-storage::Shard::open` to restore the index from one.
//!
//! # On-disk layout
//!
//! ```text
//!   +----------+----------+-------------------------------+
//!   | magic[8] | ver[u32] | bincode( GraphSnapshot )      |
//!   +----------+----------+-------------------------------+
//!
//!   magic = b"KOVAGRA1"   : catches "you handed me the wrong file"
//!   ver   = FORMAT_VERSION (little-endian u32) ; reserved for future migrations
//!   payload : bincode-encoded GraphSnapshot
//! ```
//!
//! # What's NOT in the snapshot
//!
//! - **Vector bytes** : they live in the composed [`VectorStore`], a
//!   separate file with its own crash-safe lifecycle. The snapshot
//!   references vector ids ; the store provides the bytes.
//! - **Vector bytes** — see above.
//! - **Metric / params / RNG seed** : the caller supplies these at
//!   restore time. The graph structure is metric-agnostic ; params
//!   only affect future insertions ; seed only affects future random
//!   level assignments.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{Read, Write};

use kova_core::{Distance, VectorId, VectorStore};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::KovaIndexError;

use super::HnswIndex;
use super::node::Node;
use super::params::HnswParams;

const MAGIC: &[u8; 8] = b"KOVAGRA1";
/// Current snapshot format.
///
/// **v1 → v2 : tombstones became part of the payload.** v1 relied on an
/// invariant that no longer holds — that a snapshot is always written
/// post-vacuum, so the tombstone set is empty by construction. That was
/// true only because `Shard::checkpoint` vacuumed before snapshotting.
/// Vacuum is now a *logged* operation (replicas must vacuum at the same
/// log position, see `Record::Vacuum`) while checkpoint stayed a local
/// decision, so a snapshot can legitimately contain tombstoned nodes.
/// Omitting them resurrected deleted rows on reopen.
///
/// v1 snapshots still read : they were genuinely post-vacuum, so an
/// empty tombstone set is the correct interpretation, not a fallback.
const FORMAT_VERSION: u32 = 2;

/// The last format that omitted the tombstone set.
const FORMAT_VERSION_V1: u32 = 1;

/// Wire format for a serialised graph.
///
/// `nodes` flattens each [`Node`] to its `neighbors` field directly :
/// we don't bother serialising the `Node` wrapper because it's an
/// implementation detail. Storing `Vec<Vec<VectorId>>` keeps the format
/// stable even if `Node` grows additional fields in the future.
/// v1 payload : no tombstone set. Retained so existing snapshots keep
/// loading ; see [`FORMAT_VERSION`].
#[derive(Deserialize)]
struct GraphSnapshotV1 {
    dim: Option<usize>,
    entry_point: Option<VectorId>,
    nodes: BTreeMap<VectorId, Vec<Vec<VectorId>>>,
}

#[derive(Serialize, Deserialize)]
struct GraphSnapshot {
    /// Pinned vector dimension. `None` only if the snapshot is of an
    /// empty index (never inserted), which is degenerate but legal.
    dim: Option<usize>,
    /// Current entry point. `None` only if the graph is empty.
    entry_point: Option<VectorId>,
    /// `nodes[id][layer] = neighbours at that layer`. Top layer per
    /// node is implicit : `nodes[id].len() - 1`.
    ///
    /// **`BTreeMap`, not `HashMap`, and that is load-bearing.** bincode
    /// encodes both as a length-prefixed sequence of pairs, so the wire
    /// format is identical — but a `HashMap` is written in *hash
    /// iteration order*, which differs between instances even for
    /// identical contents. Two replicas holding the same graph would
    /// then produce snapshots that differ byte-for-byte, which rules
    /// out comparing replicas by checksum or verifying a snapshot
    /// transfer by hash. The ordered map costs one `O(n log n)` build
    /// per checkpoint and makes snapshots content-addressable.
    ///
    /// Neighbour lists are deliberately *not* sorted : their order is
    /// already deterministic (it is insertion order under a
    /// deterministic apply path), and it is semantically meaningful —
    /// `search_layer` examines neighbours in list order.
    nodes: BTreeMap<VectorId, Vec<Vec<VectorId>>>,
    /// Logically-deleted ids whose graph nodes are still present.
    ///
    /// Load-bearing : these determine which rows are live, so a snapshot
    /// that omits them silently resurrects deleted data once the WAL is
    /// truncated past the `Delete` records.
    ///
    /// `BTreeSet` for the same reason `nodes` is a `BTreeMap` — ordered
    /// output keeps snapshots byte-reproducible.
    tombstones: BTreeSet<VectorId>,
}

impl<D: Distance, V: VectorStore> HnswIndex<D, V> {
    /// Stream the graph structure into `writer`.
    ///
    /// Memory cost is `BufWriter`'s buffer (caller's choice) plus
    /// bincode's per-call working memory : independent of graph size.
    /// A 1M-vector M=16 index serialises to ~230 MB ; without
    /// streaming, that whole payload would live in a `Vec<u8>` before
    /// hitting disk.
    ///
    /// # Errors
    /// - I/O errors from `writer` surface as [`KovaIndexError::Storage`].
    /// - bincode encoding failures (effectively impossible for our
    ///   types but the API requires handling them) also surface as
    ///   [`KovaIndexError::Storage`].
    pub fn write_snapshot<W: Write>(&self, writer: &mut W) -> Result<(), KovaIndexError> {
        // Header : magic + version. Written before the bincode payload
        // so a wrong-file or wrong-version read fails fast at the
        // 12-byte mark, not after deserialising garbage.
        writer.write_all(MAGIC).map_err(|e| io_err(&e))?;
        writer
            .write_all(&FORMAT_VERSION.to_le_bytes())
            .map_err(|e| io_err(&e))?;

        // Flatten nodes into the wire format. Cloning each neighbour
        // list is cheap (`Vec<u64>`) and matches what bincode would
        // copy internally during serialisation anyway.
        let snapshot = GraphSnapshot {
            dim: self.dim,
            entry_point: self.entry_point,
            nodes: self
                .nodes
                .iter()
                .map(|(id, node)| (*id, node.neighbors.clone()))
                .collect(),
            tombstones: self.tombstones.iter().copied().collect(),
        };

        bincode::serialize_into(writer, &snapshot).map_err(|e| bincode_err(&e))?;
        Ok(())
    }

    /// Reconstruct an index from a snapshot stream.
    ///
    /// Caller provides the metric, params, seed, and a freshly-opened
    /// [`VectorStore`] : the snapshot only carries the graph structure,
    /// not the things the caller controls at runtime.
    ///
    /// Tombstones come back with the snapshot (v2 onwards). They are
    /// logical state — they decide which rows are live — so dropping
    /// them would resurrect deleted rows as soon as the WAL is truncated
    /// past the corresponding `Delete` records.
    ///
    /// # Errors
    /// - [`KovaIndexError::Storage`] if magic doesn't match (wrong file
    ///   or corrupt header).
    /// - [`KovaIndexError::Storage`] if the format version isn't
    ///   recognised (snapshot from a newer Kova, or corrupt).
    /// - [`KovaIndexError::Storage`] for I/O or bincode decode errors.
    pub fn read_snapshot<R: Read>(
        metric: D,
        params: HnswParams,
        seed: u64,
        vectors: V,
        reader: &mut R,
    ) -> Result<Self, KovaIndexError> {
        // ----- Validate magic -----
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic).map_err(|e| io_err(&e))?;
        if &magic != MAGIC {
            return Err(KovaIndexError::Storage(format!(
                "snapshot magic mismatch : got {magic:?}, expected {MAGIC:?}"
            )));
        }

        // ----- Validate version -----
        let mut version_bytes = [0u8; 4];
        reader
            .read_exact(&mut version_bytes)
            .map_err(|e| io_err(&e))?;
        let version = u32::from_le_bytes(version_bytes);
        if version != FORMAT_VERSION && version != FORMAT_VERSION_V1 {
            return Err(KovaIndexError::Storage(format!(
                "unsupported snapshot format version : {version} \
                 (this build reads {FORMAT_VERSION_V1} and {FORMAT_VERSION})"
            )));
        }

        // ----- Decode the payload -----
        //
        // v1 carried no tombstone set. Reading one as an empty set is
        // correct rather than lossy : v1 snapshots were only ever
        // written immediately after a vacuum.
        let (dim, entry_point, nodes_flat, tombstones) = if version == FORMAT_VERSION_V1 {
            let v1: GraphSnapshotV1 =
                bincode::deserialize_from(reader).map_err(|e| bincode_err(&e))?;
            (v1.dim, v1.entry_point, v1.nodes, BTreeSet::new())
        } else {
            let v2: GraphSnapshot =
                bincode::deserialize_from(reader).map_err(|e| bincode_err(&e))?;
            (v2.dim, v2.entry_point, v2.nodes, v2.tombstones)
        };

        // Rehydrate Node wrappers from the flat neighbour lists.
        let nodes: HashMap<VectorId, Node> = nodes_flat
            .into_iter()
            .map(|(id, neighbors)| (id, Node { neighbors }))
            .collect();

        Ok(Self {
            metric,
            params,
            nodes,
            vectors,
            tombstones: tombstones.into_iter().collect(),
            entry_point,
            dim,
            rng: StdRng::seed_from_u64(seed),
        })
    }
}

/// Wrap an [`std::io::Error`] into [`KovaIndexError::Storage`].
fn io_err(e: &std::io::Error) -> KovaIndexError {
    KovaIndexError::Storage(format!("snapshot io: {e}"))
}

/// Wrap a `bincode::Error` into [`KovaIndexError::Storage`].
fn bincode_err(e: &bincode::Error) -> KovaIndexError {
    KovaIndexError::Storage(format!("snapshot bincode: {e}"))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use kova_core::{InMemoryVectorStore, L2, Vector, VectorId};

    use crate::{Index, KovaIndexError};

    use super::super::{HnswIndex, HnswParams};
    use super::{FORMAT_VERSION, MAGIC};

    fn v(data: Vec<f32>) -> Vector {
        Vector::try_new(data).unwrap()
    }
    fn id(n: u64) -> VectorId {
        VectorId::new(n)
    }

    /// Round-trip on a small graph : write -> read -> compare graph
    /// structure (top layer, neighbour lists) and search behaviour.
    #[test]
    fn snapshot_roundtrip_preserves_graph_and_search() {
        // Build an index with a few inserts.
        let mut src = HnswIndex::seeded(L2, HnswParams::default(), 42);
        for n in 0u16..10 {
            src.insert(id(u64::from(n)), v(vec![f32::from(n), f32::from(n * 2)]))
                .unwrap();
        }

        // Snapshot it into a Vec<u8> via a Cursor (Vec<u8> impls Write).
        let mut buf: Vec<u8> = Vec::new();
        src.write_snapshot(&mut buf).unwrap();
        assert!(
            buf.len() > MAGIC.len() + 4,
            "snapshot must contain a payload"
        );

        // Read it back. The vectors store has to be re-populated by the
        // caller in production ; for this roundtrip test, build a fresh
        // store and copy the source's vectors into it.
        let mut vectors = InMemoryVectorStore::new();
        for n in 0u16..10 {
            let vec = src.get(id(u64::from(n))).expect("vector present");
            kova_core::VectorStore::put(&mut vectors, id(u64::from(n)), vec).unwrap();
        }
        let mut cursor = Cursor::new(buf);
        let restored =
            HnswIndex::read_snapshot(L2, HnswParams::default(), 42, vectors, &mut cursor).unwrap();

        // Graph structure matches.
        assert_eq!(restored.len(), src.len());
        assert_eq!(restored.dim(), src.dim());
        assert_eq!(restored.entry_point(), src.entry_point());
        for n in 0u16..10 {
            let id_ = id(u64::from(n));
            assert_eq!(
                restored.top_layer_of(id_),
                src.top_layer_of(id_),
                "top_layer mismatch for id {n}"
            );
        }

        // Search behaviour matches (same hits, same order).
        let q = v(vec![3.0, 6.0]);
        let src_hits = src.search(&q, 5).unwrap();
        let restored_hits = restored.search(&q, 5).unwrap();
        assert_eq!(src_hits, restored_hits, "search hits should be identical");
    }

    /// Snapshot of an empty index round-trips cleanly. Degenerate case
    /// : `dim = None`, `entry_point = None`, no nodes.
    #[test]
    fn snapshot_roundtrip_on_empty_index() {
        let src: HnswIndex<L2, InMemoryVectorStore> = HnswIndex::new(L2);

        let mut buf: Vec<u8> = Vec::new();
        src.write_snapshot(&mut buf).unwrap();

        let mut cursor = Cursor::new(buf);
        let restored = HnswIndex::read_snapshot(
            L2,
            HnswParams::default(),
            0,
            InMemoryVectorStore::new(),
            &mut cursor,
        )
        .unwrap();

        assert_eq!(restored.len(), 0);
        assert!(restored.dim().is_none());
        assert!(restored.entry_point().is_none());
    }

    /// Reading a payload with the wrong magic surfaces a clear error
    /// at the magic-check stage, not later as bincode garbage.
    #[test]
    fn read_snapshot_rejects_wrong_magic() {
        let mut bad = Vec::from(*b"NOTKOVA!");
        bad.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bad.extend_from_slice(b"some bincode garbage");
        let mut cursor = Cursor::new(bad);

        // Can't use unwrap_err : HnswIndex isn't Debug (deliberately).
        let Err(err) = HnswIndex::<L2, InMemoryVectorStore>::read_snapshot(
            L2,
            HnswParams::default(),
            0,
            InMemoryVectorStore::new(),
            &mut cursor,
        ) else {
            panic!("expected error on magic mismatch");
        };
        let KovaIndexError::Storage(msg) = err else {
            panic!("expected Storage variant, got {err:?}");
        };
        assert!(
            msg.contains("magic"),
            "expected magic-related error, got: {msg}"
        );
    }

    /// Reading a payload with the right magic but a future version
    /// surfaces as a version-mismatch error.
    #[test]
    fn read_snapshot_rejects_unsupported_version() {
        let mut bad = Vec::from(*MAGIC);
        bad.extend_from_slice(&999u32.to_le_bytes());
        bad.extend_from_slice(b"some bincode garbage");
        let mut cursor = Cursor::new(bad);

        let Err(err) = HnswIndex::<L2, InMemoryVectorStore>::read_snapshot(
            L2,
            HnswParams::default(),
            0,
            InMemoryVectorStore::new(),
            &mut cursor,
        ) else {
            panic!("expected error on version mismatch");
        };
        let KovaIndexError::Storage(msg) = err else {
            panic!("expected Storage variant, got {err:?}");
        };
        assert!(
            msg.contains("version"),
            "expected version-related error, got: {msg}"
        );
    }

    /// Truncated payload (magic present but file ends before bincode
    /// payload completes) surfaces as a bincode decode error.
    #[test]
    fn read_snapshot_rejects_truncated_payload() {
        // Only write the magic + version + the first byte of the bincode
        // payload. Deserialise will fail trying to read more.
        let mut bad = Vec::from(*MAGIC);
        bad.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bad.push(0x01); // partial payload
        let mut cursor = Cursor::new(bad);

        let res: Result<HnswIndex<L2, InMemoryVectorStore>, _> = HnswIndex::read_snapshot(
            L2,
            HnswParams::default(),
            0,
            InMemoryVectorStore::new(),
            &mut cursor,
        );
        assert!(matches!(res, Err(KovaIndexError::Storage(_))));
    }

    /// After restore, the index can accept new inserts (RNG re-seeded
    /// from caller's seed, dim stays pinned, `entry_point` preserved).
    #[test]
    fn restored_index_accepts_new_inserts() {
        let mut src = HnswIndex::seeded(L2, HnswParams::default(), 7);
        for n in 0u16..5 {
            src.insert(id(u64::from(n)), v(vec![f32::from(n), 0.0]))
                .unwrap();
        }

        let mut buf = Vec::new();
        src.write_snapshot(&mut buf).unwrap();

        let mut vectors = InMemoryVectorStore::new();
        for n in 0u16..5 {
            let vec = src.get(id(u64::from(n))).unwrap();
            kova_core::VectorStore::put(&mut vectors, id(u64::from(n)), vec).unwrap();
        }
        let mut cursor = Cursor::new(buf);
        let mut restored =
            HnswIndex::read_snapshot(L2, HnswParams::default(), 7, vectors, &mut cursor).unwrap();

        // Insert a new id ; should succeed and be searchable.
        restored.insert(id(99), v(vec![99.0, 0.0])).unwrap();
        assert_eq!(restored.len(), 6);

        let hits = restored.search(&v(vec![99.0, 0.0]), 1).unwrap();
        assert_eq!(hits[0].0, id(99));
    }
}

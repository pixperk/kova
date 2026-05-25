//! Incremental HNSW vacuum : physically remove tombstoned nodes from
//! the graph + the underlying [`VectorStore`], with proper neighbour
//! rewiring so search quality survives.
//!
//! # The two-pass algorithm
//!
//! Naive vacuum walks tombstones outermost and repairs each of their
//! neighbours per-tombstone. A node that's a neighbour of K tombstones
//! gets repaired K times (K [`search_layer`] calls).
//! [`search_layer`] is the expensive operation in vacuum, so this is
//! the bottleneck.
//!
//! Two-pass design :
//!
//! ```text
//!   PASS 1  : collect affected (node, layer) -> { tombstoned neighbours }
//!             (read-only walk of tombstones + their neighbour lists ;
//!              no graph mutations yet)
//!
//!   PASS 2  : for each affected (N, layer)
//!               drop the dead edges from N's list in one retain pass
//!               if N's count fell below repair_threshold :
//!                 search_layer for replacement candidates
//!                 select_neighbors_heuristic to pick up to m_max
//!                 add bidirectional edges + prune any overflow
//!               else : skip the expensive search (cheap path)
//!
//!   CLEANUP : remove tombstoned ids from nodes + vectors ;
//!             pick new entry_point if old was tombstoned ;
//!             clear self.tombstones
//! ```
//!
//! Two compounding wins over the naive approach :
//!
//! 1. Per-node repair, not per-edge. If N had three tombstoned
//!    neighbours, one pass covers all three. `search_layer` calls
//!    drop by a factor of (avg deletions per affected node).
//! 2. Threshold skip. Nodes whose count is still healthy (`>=
//!    repair_threshold`) don't call `search_layer` at all. Search
//!    quality survives a few missing neighbours ; it only really
//!    degrades when neighbours collapse below ~half of M.
//!
//! Never worse than naive ; substantially cheaper for clustered deletes.

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use kova_core::{Distance, VectorId, VectorStore};

use crate::KovaIndexError;

use super::HnswIndex;

/// Wraps any `V::Error` (or other `Debug` error) into
/// [`KovaIndexError::Storage`], the standard pattern for surfacing
/// foreign errors from the vector store.
fn store_err<E: Debug>(e: E) -> KovaIndexError {
    KovaIndexError::Storage(format!("{e:?}"))
}

impl<D: Distance, V: VectorStore> HnswIndex<D, V> {
    /// Physically remove all tombstoned nodes : rewire the graph so no
    /// live node points at a removed one, free their slots in the
    /// vector store, reset the tombstone set.
    ///
    /// Returns the number of nodes physically removed.
    ///
    /// The caller (e.g. `Shard::vacuum`) is responsible for snapshotting
    /// and WAL truncation around this if durability of the post-vacuum
    /// state matters. Vacuum on its own makes no on-disk commit.
    ///
    /// # Errors
    /// Returns [`KovaIndexError::Storage`] if `vectors.remove` fails
    /// for any tombstoned id. Mid-failure leaves the in-memory graph
    /// partially rewired ; the WAL is the recovery source.
    pub fn vacuum_tombstones(&mut self) -> Result<usize, KovaIndexError> {
        if self.tombstones.is_empty() {
            return Ok(0);
        }

        // Snapshot tombstone ids so we can iterate without holding an
        // immutable borrow on self.tombstones (later passes need mut
        // borrows on self for graph mutations).
        let tombstoned: Vec<VectorId> = self.tombstones.iter().copied().collect();
        let n_removed = tombstoned.len();

        // --------------------------------------------------------------
        // PASS 1 : collect affected nodes.
        //
        // For every tombstoned id T at every layer L, accumulate every
        // *live* neighbour N along with the (T)s that need to be
        // dropped from N's list at L.
        //
        //   affected[(N, L)] = { T1, T2, ... }
        //
        // Skipping tombstoned-N short-circuits work : N is about to be
        // removed anyway. If two tombstones share a live neighbour N,
        // N only gets one entry in `affected` covering both removals.
        // --------------------------------------------------------------
        let affected = self.collect_affected_at_layer(&tombstoned)?;

        // --------------------------------------------------------------
        // PASS 2 : repair each (N, layer) at most once.
        //
        // For each entry :
        //   1. drop all dead edges in a single retain
        //   2. if count >= repair_threshold, skip the search (cheap path)
        //   3. otherwise : search_layer + heuristic to fill back up,
        //      add bidirectional edges, prune overflow on the back-edges
        // --------------------------------------------------------------
        for ((n, layer), removed_neighbours) in affected {
            self.repair_node_at_layer(n, layer, &removed_neighbours, &tombstoned)?;
        }

        // --------------------------------------------------------------
        // CLEANUP : actually remove the tombstoned nodes + their
        // vector store entries. Update entry point if old was removed.
        // --------------------------------------------------------------
        for &t in &tombstoned {
            self.nodes.remove(&t);
            self.vectors.remove(t).map_err(store_err)?;
        }

        // If the entry point was tombstoned, pick a new one from the
        // surviving nodes (highest top_layer). The check is via nodes
        // membership : after the cleanup loop, tombstoned ids are gone
        // from self.nodes, so "entry point was tombstoned" is exactly
        // "entry point is no longer in nodes".
        if let Some(ep) = self.entry_point
            && !self.nodes.contains_key(&ep)
        {
            self.entry_point = self.pick_new_entry_point();
        }

        self.tombstones.clear();

        Ok(n_removed)
    }

    /// PASS 1 helper : build the `(node, layer) -> {removed tombstoned
    /// neighbours}` map.
    ///
    /// Each tombstoned id contributes to the map once per (live
    /// neighbour, layer it appears in) pair. Tombstoned neighbours of
    /// tombstoned ids are skipped : they get removed in CLEANUP without
    /// needing a repair pass.
    fn collect_affected_at_layer(
        &self,
        tombstoned: &[VectorId],
    ) -> Result<HashMap<(VectorId, usize), HashSet<VectorId>>, KovaIndexError> {
        let mut affected: HashMap<(VectorId, usize), HashSet<VectorId>> = HashMap::new();
        for &t in tombstoned {
            let t_node = self.nodes.get(&t).ok_or_else(|| {
                KovaIndexError::Storage(format!("vacuum: tombstoned id {t} missing from nodes map"))
            })?;
            for (layer, neighbours_at_layer) in t_node.neighbors.iter().enumerate() {
                for &n in neighbours_at_layer {
                    if self.tombstones.contains(&n) {
                        // n is also tombstoned ; skip ; cleanup removes both.
                        continue;
                    }
                    affected.entry((n, layer)).or_default().insert(t);
                }
            }
        }
        Ok(affected)
    }

    /// PASS 2 helper : repair node `n` at `layer` exactly once.
    ///
    /// `removed_neighbours` is the set of tombstoned ids that were N's
    /// neighbours at this layer.
    /// `tombstoned_all` is the full tombstone list, used to filter
    /// entry-point candidates when we have to seed the repair search
    /// from N's *removed* (tombstoned) neighbours' other connections.
    fn repair_node_at_layer(
        &mut self,
        n: VectorId,
        layer: usize,
        removed_neighbours: &HashSet<VectorId>,
        tombstoned_all: &[VectorId],
    ) -> Result<(), KovaIndexError> {
        // ----- Step A : drop the dead edges in one pass -----
        {
            let n_node = self.nodes.get_mut(&n).ok_or_else(|| {
                KovaIndexError::Storage(format!("vacuum: affected id {n} missing from nodes"))
            })?;
            n_node.neighbors[layer].retain(|nb| !removed_neighbours.contains(nb));
        }

        let m_max = self.params.m_for_layer(layer);
        let repair_threshold = m_max / 2;

        // ----- Step B : decide whether to do the expensive repair -----
        let current_count = {
            let n_node = self.nodes.get(&n).expect("affected node still present");
            n_node.neighbors[layer].len()
        };
        if current_count >= repair_threshold {
            // Still well-connected ; skip the search. Search quality
            // tolerates a few missing edges ; it only degrades when
            // neighbour counts collapse below the threshold.
            return Ok(());
        }

        // ----- Step C : gather entry points for the repair search -----
        //
        // Prefer N's remaining live neighbours (best signal). If N has
        // no live neighbours left, fall back to the removed tombstones'
        // *other* live neighbours : they're geographically close to N,
        // good seeds for the local search.
        let entry_points = self.gather_repair_entry_points(n, layer, removed_neighbours);
        if entry_points.is_empty() {
            // Truly stranded : no live way back into the graph for this
            // (node, layer). Skip repair ; node stays under-connected.
            // Logged via the eventual tracing layer ; pathological case.
            return Ok(());
        }

        // ----- Step D : search for replacement candidates -----
        let Some(n_vec) = self.vectors.get(n) else {
            // Defensive : n must have a vector at this point ; if not,
            // skip (no graph mutation has happened beyond Step A).
            return Ok(());
        };
        let candidates =
            self.search_layer(&n_vec, &entry_points, self.params.ef_construction, layer);
        let selected = self.select_neighbors_heuristic(&candidates, m_max);

        // ----- Step E : add bidirectional edges + prune overflow -----
        //
        // We do the additions in two phases : first collect the list of
        // candidates that pass our filters (no self-loop, no tombstoned,
        // not already a neighbour) ; then mutate. Two passes because
        // the filter step reads self.nodes immutably while the mutate
        // step needs mut borrows ; doing them interleaved confuses the
        // borrow checker for no good reason.
        let to_add: Vec<VectorId> = selected
            .iter()
            .map(|(m, _)| *m)
            .filter(|&m| m != n)
            .filter(|m| !tombstoned_all.contains(m))
            .collect();

        for m in to_add {
            // Add N -> M (skip if already there).
            let added_forward = {
                let n_node = self.nodes.get_mut(&n).expect("n still present");
                if n_node.neighbors[layer].contains(&m) {
                    false
                } else {
                    n_node.neighbors[layer].push(m);
                    true
                }
            };
            if !added_forward {
                continue;
            }

            // Add M -> N (maintain bidirectionality) ; prune if needed.
            let needs_prune = {
                let Some(m_node) = self.nodes.get_mut(&m) else {
                    // m disappeared somehow ; defensive
                    continue;
                };
                // Skip the back-edge when : m doesn't exist at this
                // layer (its top_layer < layer), or the edge is already
                // there. Both cases mean "no append, no prune."
                if layer >= m_node.neighbors.len() || m_node.neighbors[layer].contains(&n) {
                    false
                } else {
                    m_node.neighbors[layer].push(n);
                    m_node.neighbors[layer].len() > m_max
                }
            };
            if needs_prune {
                self.prune_overflow_neighbours(m, layer)?;
            }
        }

        Ok(())
    }

    /// Build the entry-point list for a repair search at `(n, layer)`.
    ///
    /// Order of preference :
    /// 1. N's own remaining (post-dead-edge-removal) neighbours
    /// 2. Other live neighbours of the tombstoned ids that were removed
    ///    from N's list (they're locally close to N)
    ///
    /// Returns empty if neither path yields anything : truly stranded.
    fn gather_repair_entry_points(
        &self,
        n: VectorId,
        layer: usize,
        removed_neighbours: &HashSet<VectorId>,
    ) -> Vec<VectorId> {
        let n_node = self.nodes.get(&n).expect("repair: n present");

        // Path 1 : N's surviving live neighbours.
        let mut eps: Vec<VectorId> = n_node.neighbors[layer]
            .iter()
            .copied()
            .filter(|nb| !self.tombstones.contains(nb))
            .collect();
        if !eps.is_empty() {
            return eps;
        }

        // Path 2 : tombstoned-T's other live neighbours at this layer.
        for &t in removed_neighbours {
            let Some(t_node) = self.nodes.get(&t) else {
                continue;
            };
            if layer >= t_node.neighbors.len() {
                continue;
            }
            for &other in &t_node.neighbors[layer] {
                if other != n && !self.tombstones.contains(&other) {
                    eps.push(other);
                }
            }
            if !eps.is_empty() {
                break;
            }
        }
        eps
    }

    /// Re-run the selection heuristic on `id`'s neighbour list at
    /// `layer` and prune it back down to `m_for_layer(layer)`.
    ///
    /// Called when adding a back-edge made `id`'s list overflow.
    /// Without pruning, neighbour lists would grow unboundedly across
    /// vacuum cycles and search would degrade.
    ///
    /// ```text
    ///   prune_overflow_neighbours(id, layer)
    ///       |
    ///       v
    ///   current = clone of nodes[id].neighbors[layer]
    ///       |
    ///       v
    ///   scored = [ (n, distance(id, n)) for n in current if vectors.get(n).is_some() ]
    ///       |
    ///       v
    ///   sort scored ascending by distance
    ///       |
    ///       v
    ///   kept = select_neighbors_heuristic(scored, m_for_layer(layer))
    ///       |
    ///       v
    ///   nodes[id].neighbors[layer] = kept
    /// ```
    fn prune_overflow_neighbours(
        &mut self,
        id: VectorId,
        layer: usize,
    ) -> Result<(), KovaIndexError> {
        let m_max = self.params.m_for_layer(layer);

        // Snapshot the current list (cheap : Vec<u64>) so we can read
        // it while we'll later mutate the node's list.
        let current: Vec<VectorId> = {
            let node = self.nodes.get(&id).ok_or_else(|| {
                KovaIndexError::Storage(format!("prune: id {id} missing from nodes"))
            })?;
            node.neighbors[layer].clone()
        };

        let Some(id_vec) = self.vectors.get(id) else {
            // Defensive : id must have a vector at this point.
            return Ok(());
        };

        // Score every current neighbour by distance to id. select_-
        // neighbors_heuristic expects ascending-sorted input.
        let mut scored: Vec<(VectorId, f32)> = current
            .iter()
            .filter_map(|&n| {
                self.vectors
                    .get(n)
                    .map(|nv| (n, self.metric.distance(&id_vec, &nv)))
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let kept = self.select_neighbors_heuristic(&scored, m_max);

        // Replace the neighbour list with the heuristic-pruned set.
        let node = self.nodes.get_mut(&id).expect("prune: id still present");
        node.neighbors[layer] = kept.into_iter().map(|(nid, _)| nid).collect();

        Ok(())
    }

    /// Pick a fresh entry point after vacuum removed the old one.
    /// Returns the live node with the highest `top_layer`, or `None`
    /// if the graph is empty.
    fn pick_new_entry_point(&self) -> Option<VectorId> {
        self.nodes
            .iter()
            .max_by_key(|(_, node)| node.top_layer())
            .map(|(id, _)| *id)
    }
}

#[cfg(test)]
mod tests {
    use kova_core::{InMemoryVectorStore, L2, Vector, VectorId};

    use crate::{Index, KovaIndexError};

    use super::super::HnswIndex;

    fn v(data: Vec<f32>) -> Vector {
        Vector::try_new(data).unwrap()
    }
    fn id(n: u64) -> VectorId {
        VectorId::new(n)
    }

    fn fresh_index() -> HnswIndex<L2, InMemoryVectorStore> {
        HnswIndex::new(L2)
    }

    // ---------- baseline ----------

    #[test]
    fn vacuum_on_empty_tombstones_is_noop() {
        let mut idx = fresh_index();
        idx.insert(id(1), v(vec![1.0, 2.0])).unwrap();
        let removed = idx.vacuum_tombstones().unwrap();
        assert_eq!(removed, 0);
        assert_eq!(idx.len(), 1);
        assert!(idx.top_layer_of(id(1)).is_some());
    }

    #[test]
    fn vacuum_on_empty_index_is_noop() {
        let mut idx = fresh_index();
        let removed = idx.vacuum_tombstones().unwrap();
        assert_eq!(removed, 0);
        assert_eq!(idx.len(), 0);
        assert!(idx.entry_point().is_none());
    }

    // ---------- removal ----------

    #[test]
    fn vacuum_removes_tombstoned_nodes() {
        let mut idx = fresh_index();
        idx.insert(id(1), v(vec![1.0, 0.0])).unwrap();
        idx.insert(id(2), v(vec![0.0, 1.0])).unwrap();
        idx.insert(id(3), v(vec![0.5, 0.5])).unwrap();
        idx.tombstone(id(2)).unwrap();

        let removed = idx.vacuum_tombstones().unwrap();
        assert_eq!(removed, 1);
        assert!(idx.top_layer_of(id(2)).is_none());
        assert!(idx.top_layer_of(id(1)).is_some());
        assert!(idx.top_layer_of(id(3)).is_some());
        assert_eq!(idx.tombstone_count(), 0);
        assert!(!idx.is_tombstoned(id(2)));
    }

    #[test]
    fn vacuum_returns_count_of_removed_nodes() {
        let mut idx = fresh_index();
        for n in 0u16..10 {
            idx.insert(id(u64::from(n)), v(vec![f32::from(n), 0.0]))
                .unwrap();
        }
        for n in 0..5 {
            idx.tombstone(id(n)).unwrap();
        }
        let removed = idx.vacuum_tombstones().unwrap();
        assert_eq!(removed, 5);
        assert_eq!(idx.len(), 5);
    }

    #[test]
    fn vacuum_with_all_nodes_tombstoned_empties_index() {
        let mut idx = fresh_index();
        for n in 0u16..5 {
            idx.insert(id(u64::from(n)), v(vec![f32::from(n), 0.0]))
                .unwrap();
        }
        for n in 0..5 {
            idx.tombstone(id(n)).unwrap();
        }
        idx.vacuum_tombstones().unwrap();
        assert_eq!(idx.len(), 0);
        assert!(idx.entry_point().is_none());
        // Dim stays pinned even after vacuum-to-empty.
        assert_eq!(idx.dim(), Some(2));
    }

    // ---------- entry point ----------

    #[test]
    fn vacuum_updates_entry_point_when_old_was_tombstoned() {
        let mut idx = fresh_index();
        idx.insert(id(1), v(vec![1.0, 0.0])).unwrap();
        idx.insert(id(2), v(vec![0.0, 1.0])).unwrap();
        idx.insert(id(3), v(vec![0.5, 0.5])).unwrap();

        let old_ep = idx.entry_point().expect("entry point set after inserts");
        idx.tombstone(old_ep).unwrap();
        idx.vacuum_tombstones().unwrap();

        let new_ep = idx.entry_point().expect("new entry point picked");
        assert_ne!(new_ep, old_ep, "entry point should have moved");
        assert!(
            idx.top_layer_of(new_ep).is_some(),
            "new entry point must be a live node"
        );
    }

    #[test]
    fn vacuum_preserves_entry_point_when_old_was_not_tombstoned() {
        let mut idx = fresh_index();
        idx.insert(id(1), v(vec![1.0, 0.0])).unwrap();
        idx.insert(id(2), v(vec![0.0, 1.0])).unwrap();
        idx.insert(id(3), v(vec![0.5, 0.5])).unwrap();

        let old_ep = idx.entry_point().unwrap();
        // Tombstone a non-entry-point id.
        let to_remove = [id(1), id(2), id(3)]
            .into_iter()
            .find(|&i| i != old_ep)
            .unwrap();
        idx.tombstone(to_remove).unwrap();
        idx.vacuum_tombstones().unwrap();

        assert_eq!(idx.entry_point(), Some(old_ep));
    }

    // ---------- search behaviour after vacuum ----------

    #[test]
    fn vacuum_search_excludes_removed_ids() {
        let mut idx = fresh_index();
        idx.insert(id(1), v(vec![1.0, 0.0])).unwrap();
        idx.insert(id(2), v(vec![0.0, 1.0])).unwrap();
        idx.insert(id(3), v(vec![1.0, 1.0])).unwrap();

        // Before vacuum : tombstoned id is filtered by `search`'s
        // existing tombstone post-filter.
        idx.tombstone(id(1)).unwrap();
        let hits = idx.search(&v(vec![1.0, 0.0]), 3).unwrap();
        let hit_ids: Vec<_> = hits.iter().map(|(i, _)| *i).collect();
        assert!(!hit_ids.contains(&id(1)));

        // After vacuum : id 1 isn't in the graph at all.
        idx.vacuum_tombstones().unwrap();
        let hits = idx.search(&v(vec![1.0, 0.0]), 3).unwrap();
        let hit_ids: Vec<_> = hits.iter().map(|(i, _)| *i).collect();
        assert!(!hit_ids.contains(&id(1)));
        assert!(hit_ids.contains(&id(2)) || hit_ids.contains(&id(3)));
    }

    // ---------- id reuse (the v1-limitation lift) ----------

    #[test]
    fn vacuum_allows_reinsert_of_previously_tombstoned_id() {
        let mut idx = fresh_index();
        idx.insert(id(1), v(vec![1.0, 0.0])).unwrap();
        idx.insert(id(2), v(vec![0.0, 1.0])).unwrap();
        idx.tombstone(id(1)).unwrap();

        // Pre-vacuum : reinsert fails because graph node is still there.
        let err = idx.insert(id(1), v(vec![9.0, 9.0])).unwrap_err();
        assert!(matches!(err, KovaIndexError::DuplicateId { .. }));

        // After vacuum : id 1 is gone from the graph, reinsert succeeds.
        idx.vacuum_tombstones().unwrap();
        idx.insert(id(1), v(vec![9.0, 9.0])).unwrap();
        assert_eq!(idx.len(), 2);
        assert!(idx.top_layer_of(id(1)).is_some());

        // The new vector for id 1 is what got inserted, not the old one.
        let hits = idx.search(&v(vec![9.0, 9.0]), 1).unwrap();
        assert_eq!(hits[0].0, id(1));
    }

    // ---------- graph integrity after vacuum ----------

    #[test]
    fn vacuum_leaves_no_dead_edges_in_live_nodes() {
        // After vacuum, no live node should have a tombstoned id in any
        // of its neighbour lists.
        let mut idx = fresh_index();
        for n in 0u16..20 {
            idx.insert(id(u64::from(n)), v(vec![f32::from(n), f32::from(n * 2)]))
                .unwrap();
        }
        for n in (0..20).step_by(3) {
            idx.tombstone(id(n)).unwrap();
        }
        let removed_ids: std::collections::HashSet<VectorId> = (0..20).step_by(3).map(id).collect();

        idx.vacuum_tombstones().unwrap();

        // Direct check : poke at the index's internals via top_layer_of
        // for every supposedly-live id, and assert search never returns
        // a removed id.
        for n in 0..20 {
            let i = id(n);
            if removed_ids.contains(&i) {
                assert!(
                    idx.top_layer_of(i).is_none(),
                    "tombstoned id {n} still in graph"
                );
            } else {
                assert!(
                    idx.top_layer_of(i).is_some(),
                    "live id {n} missing from graph"
                );
            }
        }

        // Search a query : results should NEVER include a removed id.
        let q = v(vec![5.0, 10.0]);
        let hits = idx.search(&q, 20).unwrap();
        for (hit_id, _) in &hits {
            assert!(
                !removed_ids.contains(hit_id),
                "search returned tombstoned id {hit_id}"
            );
        }
    }
}

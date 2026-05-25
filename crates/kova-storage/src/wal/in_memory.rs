use crate::{KovaStorageError, Lsn, Record, Wal};

/// In-memory [`Wal`] implementation for tests : no disk, no fsync.
///
/// All operations are infallible (`sync` is a no-op), so tests against
/// `Shard` can focus on logic without filesystem setup or crash drills.
#[derive(Debug, Default)]
pub struct InMemoryWal {
    records: Vec<(Lsn, Record)>,
    next_lsn: u64,
}

impl InMemoryWal {
    /// Create a new empty `InMemoryWal`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            next_lsn: 0,
        }
    }
}

impl Wal for InMemoryWal {
    type Error = KovaStorageError;

    fn append(&mut self, record: &Record) -> Result<Lsn, KovaStorageError> {
        let lsn = Lsn::new(self.next_lsn);
        self.records.push((lsn, record.clone()));
        self.next_lsn += 1;
        Ok(lsn)
    }

    fn sync(&mut self) -> Result<(), KovaStorageError> {
        // Nothing to flush : everything's already in memory.
        Ok(())
    }

    fn iter_from(
        &self,
        from: Lsn,
    ) -> impl Iterator<Item = Result<(Lsn, Record), KovaStorageError>> + '_ {
        self.records
            .iter()
            .filter(move |(lsn, _)| *lsn >= from)
            .map(|(lsn, record)| Ok((*lsn, record.clone())))
    }

    fn truncate_before(&mut self, before: Lsn) -> Result<(), KovaStorageError> {
        self.records.retain(|(lsn, _)| *lsn >= before);
        Ok(())
    }

    fn last_lsn(&self) -> Option<Lsn> {
        // `next_lsn` is the LSN the NEXT append will get. The last one
        // we appended (durably or not) is therefore `next_lsn - 1`,
        // unless we've never appended (next_lsn == 0).
        if self.next_lsn == 0 {
            None
        } else {
            Some(Lsn::new(self.next_lsn - 1))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kova_core::{Metadata, Vector, VectorId};

    #[allow(clippy::cast_precision_loss)]
    fn ins(n: u64) -> Record {
        Record::Insert {
            id: VectorId::new(n),
            vector: Vector::try_new(vec![n as f32]).unwrap(),
            metadata: Metadata::new(),
        }
    }

    #[test]
    fn new_is_empty() {
        let wal = InMemoryWal::new();
        assert!(wal.iter_from(Lsn::ZERO).next().is_none());
    }

    #[test]
    fn append_returns_monotonic_lsns() {
        let mut wal = InMemoryWal::new();
        let l1 = wal.append(&ins(1)).unwrap();
        let l2 = wal.append(&ins(2)).unwrap();
        let l3 = wal.append(&ins(3)).unwrap();
        assert!(l1 < l2 && l2 < l3);
    }

    #[test]
    fn iter_from_zero_returns_all() {
        let mut wal = InMemoryWal::new();
        for n in 0..5 {
            wal.append(&ins(n)).unwrap();
        }
        assert_eq!(wal.iter_from(Lsn::ZERO).count(), 5);
    }

    #[test]
    fn iter_from_filters_below() {
        let mut wal = InMemoryWal::new();
        let lsns: Vec<_> = (0..5).map(|n| wal.append(&ins(n)).unwrap()).collect();
        let from = lsns[2];
        let records: Vec<_> = wal.iter_from(from).collect::<Result<_, _>>().unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].0, lsns[2]);
    }

    #[test]
    fn truncate_before_drops_old() {
        let mut wal = InMemoryWal::new();
        let lsns: Vec<_> = (0..5).map(|n| wal.append(&ins(n)).unwrap()).collect();
        wal.truncate_before(lsns[3]).unwrap();
        assert_eq!(wal.iter_from(Lsn::ZERO).count(), 2);
    }

    #[test]
    fn sync_is_noop() {
        let mut wal = InMemoryWal::new();
        wal.append(&ins(1)).unwrap();
        assert!(wal.sync().is_ok());
    }

    #[test]
    fn last_lsn_is_none_on_empty_wal() {
        let wal = InMemoryWal::new();
        assert_eq!(wal.last_lsn(), None);
    }

    #[test]
    fn last_lsn_matches_most_recent_append() {
        let mut wal = InMemoryWal::new();
        let l1 = wal.append(&ins(1)).unwrap();
        assert_eq!(wal.last_lsn(), Some(l1));
        let l2 = wal.append(&ins(2)).unwrap();
        assert_eq!(wal.last_lsn(), Some(l2));
    }

    #[test]
    fn last_lsn_unchanged_by_truncate() {
        // truncate_before only drops records ; the next-LSN counter
        // (and therefore last_lsn) keeps advancing monotonically.
        let mut wal = InMemoryWal::new();
        for n in 0..5 {
            wal.append(&ins(n)).unwrap();
        }
        let before_truncate = wal.last_lsn();
        wal.truncate_before(Lsn::new(3)).unwrap();
        assert_eq!(wal.last_lsn(), before_truncate);
    }
}

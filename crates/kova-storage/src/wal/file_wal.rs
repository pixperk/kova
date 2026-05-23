use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::KovaStorageError;
use crate::wal::record::{decode_record, encode_record};
use crate::wal::{Lsn, Record, Wal};

/// File-backed [`Wal`] : appends frames to a single segment file with
/// `BufWriter` and `fdatasync` for durability.
///
/// Single-segment for now; segmentation lands in Day 6-7. Truncation is
/// recorded in memory (`truncated_before`) until segmentation makes
/// physical truncation cheap.
#[derive(Debug)]
pub struct FileWal {
    ///buffered write side of the file. appends go here
    writer: BufWriter<File>,
    ///path is remembered to open separate read handle for iterators
    path: PathBuf,
    ///next LSN to be assigned
    next_lsn: u64,
    ///in-memory record of truncation point. all records with LSN < this are logically truncated and should be ignored by iterators
    truncated_before: Lsn,
}

impl FileWal {
    /// Open an existing WAL file or create a new empty one.
    ///
    /// On open we walk the file from start, count records to recover
    /// `next_lsn`, and truncate any torn tail (so subsequent appends start
    /// from the last valid byte).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, KovaStorageError> {
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        // --- recovery walk ---
        file.seek(SeekFrom::Start(0))?;
        let mut next_lsn: u64 = 0;
        let mut last_valid_pos: u64 = 0;
        loop {
            match decode_record(&mut file) {
                Ok(Some(_record)) => {
                    next_lsn += 1;
                    last_valid_pos = file.stream_position()?;
                }
                // Clean EOF or torn tail : both end the walk. The set_len
                // below truncates anything past the last valid record.
                Ok(None) | Err(KovaStorageError::CorruptRecord { .. }) => break,
                Err(e) => return Err(e),
            }
        }

        // Truncate any torn tail past the last valid record.
        file.set_len(last_valid_pos)?;
        file.seek(SeekFrom::End(0))?;

        Ok(Self {
            writer: BufWriter::new(file),
            path,
            next_lsn,
            truncated_before: Lsn::ZERO,
        })
    }
}

impl Wal for FileWal {
    /// Append a record and return its assigned LSN. Caller must call `sync()` for durability.
    fn append(&mut self, record: &Record) -> Result<Lsn, KovaStorageError> {
        let lsn = Lsn::new(self.next_lsn);
        let frame = encode_record(record)?;
        self.writer.write_all(&frame)?;
        self.next_lsn += 1;
        Ok(lsn)
    }

    /// Flush buffered data and fsync to disk for durability.
    /// `fsync` vs `fdatasync`: since we only append and never modify or
    /// delete, we don't need to worry about metadata updates that `fsync`
    /// would cover. `fdatasync` (via `sync_data`) is sufficient and faster.
    fn sync(&mut self) -> Result<(), KovaStorageError> {
        self.writer.flush()?;
        self.writer.get_mut().sync_data()?;
        Ok(())
    }

    /// Drop records with LSN `< before` from iteration.
    fn truncate_before(&mut self, before: Lsn) -> Result<(), KovaStorageError> {
        // TODO : Physical truncation requires segmentation. Until then,
        // we just record the high-water mark so iter_from filters it out.
        if before > self.truncated_before {
            self.truncated_before = before;
        }
        Ok(())
    }

    fn iter_from(
        &self,
        from: Lsn,
    ) -> impl Iterator<Item = Result<(Lsn, Record), KovaStorageError>> + '_ {
        FileWalIter::open(&self.path, from.max(self.truncated_before))
    }
}

//custom iterator as iter_from returns a fresh read handle and we need state
struct FileWalIter {
    file: Option<File>,
    next_lsn: u64,
    skip_below: Lsn,
    error_to_yield: Option<KovaStorageError>,
}

impl FileWalIter {
    fn open(path: &Path, skip_below: Lsn) -> Self {
        match File::open(path) {
            Ok(file) => Self {
                file: Some(file),
                next_lsn: 0,
                skip_below,
                error_to_yield: None,
            },
            Err(e) => Self {
                file: None,
                next_lsn: 0,
                skip_below,
                error_to_yield: Some(e.into()),
            },
        }
    }
}

impl Iterator for FileWalIter {
    type Item = Result<(Lsn, Record), KovaStorageError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(e) = self.error_to_yield.take() {
            return Some(Err(e));
        }
        let file = self.file.as_mut()?;
        loop {
            let lsn = Lsn::new(self.next_lsn);
            match decode_record(file) {
                Ok(Some(record)) => {
                    self.next_lsn += 1;
                    if lsn >= self.skip_below {
                        return Some(Ok((lsn, record)));
                    }
                    // else: consumed but filtered, continue
                }
                Ok(None) => return None,
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kova_core::{Vector, VectorId};
    use tempfile::tempdir;

    #[allow(clippy::cast_precision_loss)]
    fn ins(n: u64) -> Record {
        Record::Insert {
            id: VectorId::new(n),
            vector: Vector::try_new(vec![n as f32]).unwrap(),
        }
    }

    #[test]
    fn open_creates_empty_file() {
        let dir = tempdir().unwrap();
        let wal = FileWal::open(dir.path().join("wal.log")).unwrap();
        assert!(wal.iter_from(Lsn::ZERO).next().is_none());
    }

    #[test]
    fn append_then_sync_then_iter_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.log");
        let mut wal = FileWal::open(&path).unwrap();
        wal.append(&ins(1)).unwrap();
        wal.append(&ins(2)).unwrap();
        wal.sync().unwrap();
        let records: Vec<_> = wal.iter_from(Lsn::ZERO).collect::<Result<_, _>>().unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].1, ins(1));
        assert_eq!(records[1].1, ins(2));
    }

    #[test]
    fn append_without_sync_invisible_to_iter() {
        // Demonstrates the durability contract: only synced records are visible.
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.log");
        let mut wal = FileWal::open(&path).unwrap();
        wal.append(&ins(1)).unwrap();
        // NO sync
        assert!(wal.iter_from(Lsn::ZERO).next().is_none());
    }

    #[test]
    fn reopen_recovers_all_records() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.log");
        {
            let mut wal = FileWal::open(&path).unwrap();
            for n in 0..5 {
                wal.append(&ins(n)).unwrap();
            }
            wal.sync().unwrap();
        }
        // Reopen
        let wal = FileWal::open(&path).unwrap();
        let records: Vec<_> = wal.iter_from(Lsn::ZERO).collect::<Result<_, _>>().unwrap();
        assert_eq!(records.len(), 5);
    }

    #[test]
    fn reopen_assigns_next_lsn_after_existing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.log");
        {
            let mut wal = FileWal::open(&path).unwrap();
            wal.append(&ins(0)).unwrap();
            wal.append(&ins(1)).unwrap();
            wal.sync().unwrap();
        }
        let mut wal = FileWal::open(&path).unwrap();
        let lsn = wal.append(&ins(2)).unwrap();
        assert_eq!(lsn, Lsn::new(2));
    }

    #[test]
    fn torn_tail_is_truncated_on_reopen() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.log");
        {
            let mut wal = FileWal::open(&path).unwrap();
            wal.append(&ins(0)).unwrap();
            wal.append(&ins(1)).unwrap();
            wal.sync().unwrap();
        }
        // Append 3 garbage bytes to simulate torn tail
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(&[0xff, 0xff, 0xff]).unwrap();
        file.sync_data().unwrap();
        drop(file);

        // Reopen : torn tail is detected, file truncated to last valid record
        let wal = FileWal::open(&path).unwrap();
        let records: Vec<_> = wal.iter_from(Lsn::ZERO).collect::<Result<_, _>>().unwrap();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn truncate_before_filters_iter() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.log");
        let mut wal = FileWal::open(&path).unwrap();
        for n in 0..5 {
            wal.append(&ins(n)).unwrap();
        }
        wal.sync().unwrap();
        wal.truncate_before(Lsn::new(3)).unwrap();
        let records: Vec<_> = wal.iter_from(Lsn::ZERO).collect::<Result<_, _>>().unwrap();
        assert_eq!(records.len(), 2); // LSNs 3, 4
    }
}

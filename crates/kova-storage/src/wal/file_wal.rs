//! Segmented file-backed [`Wal`].
//!
//! The WAL lives in a directory. Each segment is one file, named by its
//! starting LSN (`wal-{16-hex}.log`). The active segment is the one
//! currently being appended to; older segments are finalised, read-only.
//! Rotation happens when the active segment crosses
//! `max_segment_bytes`. Truncation physically removes finalised segments
//! whose last LSN is below the truncation point.

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::KovaStorageError;
use crate::wal::record::{decode_record, encode_record};
use crate::wal::{Lsn, Record, Wal};

/// Default cap on a single segment's size. Rotation kicks in beyond this.
const DEFAULT_MAX_SEGMENT_BYTES: u64 = 64 * 1024 * 1024;

/// File-backed segmented [`Wal`].
///
/// On disk the WAL lives in `dir/` containing one file per segment, named
/// `wal-{16-hex of start_lsn}.log`. Recovery enumerates all such files,
/// replays them in LSN order, and truncates any torn tail on the last
/// (active) segment.
#[derive(Debug)]
pub struct FileWal {
    dir: PathBuf,
    active: ActiveSegment,
    finalized: Vec<FinalizedSegment>,
    next_lsn: u64,
    truncated_before: Lsn,
    max_segment_bytes: u64,
}

/// The current write target.
#[derive(Debug)]
struct ActiveSegment {
    writer: BufWriter<File>,
    start_lsn: Lsn,
    bytes_written: u64,
}

/// Metadata for a read-only segment we've already finalised.
#[derive(Debug, Clone)]
struct FinalizedSegment {
    path: PathBuf,
    start_lsn: Lsn,
    /// Inclusive : the last LSN that lives in this segment.
    end_lsn: Lsn,
}

impl FileWal {
    /// Open the WAL rooted at `dir`. Creates the directory and a fresh
    /// initial segment if `dir` is empty or missing.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, KovaStorageError> {
        Self::open_with_segment_size(dir, DEFAULT_MAX_SEGMENT_BYTES)
    }

    /// Open with a custom max segment size. Useful in tests to force
    /// rotation without writing megabytes.
    pub fn open_with_segment_size(
        dir: impl AsRef<Path>,
        max_segment_bytes: u64,
    ) -> Result<Self, KovaStorageError> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;

        // Enumerate segment files in the directory and sort by start_lsn.
        let mut found: Vec<(u64, PathBuf)> = Vec::new();
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if let Some(start_lsn) = parse_segment_filename(name) {
                found.push((start_lsn, path));
            }
        }
        found.sort_by_key(|(lsn, _)| *lsn);

        if found.is_empty() {
            return Self::fresh_wal(dir, max_segment_bytes);
        }

        // Walk every segment except the last as finalised. The last becomes active.
        let last_idx = found.len() - 1;
        let mut finalized: Vec<FinalizedSegment> = Vec::new();

        for (start_lsn, path) in found.iter().take(last_idx) {
            let mut file = File::open(path)?;
            let (end_lsn_exclusive, _last_valid_pos) = walk_segment(&mut file, *start_lsn)?;
            if end_lsn_exclusive > *start_lsn {
                finalized.push(FinalizedSegment {
                    path: path.clone(),
                    start_lsn: Lsn::new(*start_lsn),
                    end_lsn: Lsn::new(end_lsn_exclusive - 1),
                });
            }
        }

        // Open the last segment for read+write and recover any torn tail.
        let (active_start_lsn, active_path) = &found[last_idx];
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(active_path)?;
        file.seek(SeekFrom::Start(0))?;
        let (next_lsn, last_valid_pos) = walk_segment(&mut file, *active_start_lsn)?;
        file.set_len(last_valid_pos)?;
        file.seek(SeekFrom::End(0))?;

        Ok(Self {
            dir,
            active: ActiveSegment {
                writer: BufWriter::new(file),
                start_lsn: Lsn::new(*active_start_lsn),
                bytes_written: last_valid_pos,
            },
            finalized,
            next_lsn,
            truncated_before: Lsn::ZERO,
            max_segment_bytes,
        })
    }

    /// Create a fresh WAL in an empty directory.
    fn fresh_wal(dir: PathBuf, max_segment_bytes: u64) -> Result<Self, KovaStorageError> {
        let path = segment_path(&dir, Lsn::ZERO);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        Ok(Self {
            dir,
            active: ActiveSegment {
                writer: BufWriter::new(file),
                start_lsn: Lsn::ZERO,
                bytes_written: 0,
            },
            finalized: Vec::new(),
            next_lsn: 0,
            truncated_before: Lsn::ZERO,
            max_segment_bytes,
        })
    }

    /// Finalise the active segment and start a new one at `next_lsn`.
    fn rotate(&mut self) -> Result<(), KovaStorageError> {
        // Flush + fdatasync the active segment before declaring it immutable.
        self.active.writer.flush()?;
        self.active.writer.get_mut().sync_data()?;

        let old_start = self.active.start_lsn;
        let old_path = segment_path(&self.dir, old_start);
        let end_lsn = Lsn::new(self.next_lsn.saturating_sub(1));
        self.finalized.push(FinalizedSegment {
            path: old_path,
            start_lsn: old_start,
            end_lsn,
        });

        let new_start = Lsn::new(self.next_lsn);
        let new_path = segment_path(&self.dir, new_start);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&new_path)?;
        self.active = ActiveSegment {
            writer: BufWriter::new(file),
            start_lsn: new_start,
            bytes_written: 0,
        };
        Ok(())
    }
}

impl Wal for FileWal {
    type Error = KovaStorageError;

    fn append(&mut self, record: &Record) -> Result<Lsn, KovaStorageError> {
        let frame = encode_record(record)?;
        if self.active.bytes_written + frame.len() as u64 > self.max_segment_bytes
            && self.active.bytes_written > 0
        {
            self.rotate()?;
        }
        let lsn = Lsn::new(self.next_lsn);
        self.active.writer.write_all(&frame)?;
        self.active.bytes_written += frame.len() as u64;
        self.next_lsn += 1;
        Ok(lsn)
    }

    fn sync(&mut self) -> Result<(), KovaStorageError> {
        self.active.writer.flush()?;
        self.active.writer.get_mut().sync_data()?;
        Ok(())
    }

    fn truncate_before(&mut self, before: Lsn) -> Result<(), KovaStorageError> {
        if before > self.truncated_before {
            self.truncated_before = before;
        }
        // Physically delete any finalised segment whose last LSN is below `before`.
        let mut still_alive = Vec::with_capacity(self.finalized.len());
        for seg in self.finalized.drain(..) {
            if seg.end_lsn < before {
                fs::remove_file(&seg.path)?;
            } else {
                still_alive.push(seg);
            }
        }
        self.finalized = still_alive;
        Ok(())
    }

    fn last_lsn(&self) -> Option<Lsn> {
        // `next_lsn` is the LSN the next append will get. Last-appended
        // is `next_lsn - 1` ; `None` if nothing's ever been appended.
        if self.next_lsn == 0 {
            None
        } else {
            Some(Lsn::new(self.next_lsn - 1))
        }
    }

    fn iter_from(
        &self,
        from: Lsn,
    ) -> impl Iterator<Item = Result<(Lsn, Record), KovaStorageError>> + '_ {
        let skip_below = from.max(self.truncated_before);
        let mut segments: Vec<(PathBuf, Lsn)> = self
            .finalized
            .iter()
            .filter(|s| s.end_lsn >= skip_below)
            .map(|s| (s.path.clone(), s.start_lsn))
            .collect();
        segments.push((
            segment_path(&self.dir, self.active.start_lsn),
            self.active.start_lsn,
        ));
        SegmentedWalIter::new(segments, skip_below)
    }
}

/// Multi-segment iterator: walks segments in LSN order, opening each on demand.
struct SegmentedWalIter {
    segments: Vec<(PathBuf, Lsn)>,
    current_idx: usize,
    current_file: Option<File>,
    current_lsn: u64,
    skip_below: Lsn,
    finished: bool,
}

impl SegmentedWalIter {
    fn new(segments: Vec<(PathBuf, Lsn)>, skip_below: Lsn) -> Self {
        Self {
            segments,
            current_idx: 0,
            current_file: None,
            current_lsn: 0,
            skip_below,
            finished: false,
        }
    }
}

impl Iterator for SegmentedWalIter {
    type Item = Result<(Lsn, Record), KovaStorageError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        loop {
            // Open the next segment if we don't have one.
            if self.current_file.is_none() {
                if self.current_idx >= self.segments.len() {
                    self.finished = true;
                    return None;
                }
                let (path, start_lsn) = &self.segments[self.current_idx];
                match File::open(path) {
                    Ok(f) => {
                        self.current_file = Some(f);
                        self.current_lsn = start_lsn.get();
                    }
                    Err(e) => {
                        self.finished = true;
                        return Some(Err(e.into()));
                    }
                }
            }

            let file = self.current_file.as_mut().expect("opened above");
            let lsn = Lsn::new(self.current_lsn);
            match decode_record(file) {
                Ok(Some(record)) => {
                    self.current_lsn += 1;
                    if lsn >= self.skip_below {
                        return Some(Ok((lsn, record)));
                    }
                    // filtered, continue
                }
                Ok(None) => {
                    // End of this segment, move to next.
                    self.current_file = None;
                    self.current_idx += 1;
                }
                Err(e) => {
                    // Corruption in any segment ends the walk.
                    self.finished = true;
                    return Some(Err(e));
                }
            }
        }
    }
}

// --- helpers ---

/// Compute the path of a segment given its starting LSN.
fn segment_path(dir: &Path, start_lsn: Lsn) -> PathBuf {
    dir.join(format!("wal-{:016x}.log", start_lsn.get()))
}

/// Parse a segment filename like `wal-0000000000000000.log` into its `start_lsn`.
fn parse_segment_filename(name: &str) -> Option<u64> {
    let stripped = name.strip_prefix("wal-")?.strip_suffix(".log")?;
    if stripped.len() != 16 {
        return None;
    }
    u64::from_str_radix(stripped, 16).ok()
}

/// Walk a segment file from its current cursor, counting records.
/// Returns `(next_lsn_after_segment, last_valid_byte_pos)`.
///
/// Stops on either clean EOF or a torn/CRC-corrupt frame. Real I/O errors
/// propagate.
fn walk_segment(file: &mut File, start_lsn: u64) -> Result<(u64, u64), KovaStorageError> {
    let mut lsn = start_lsn;
    let mut last_valid_pos = file.stream_position()?;
    loop {
        match decode_record(file) {
            Ok(Some(_)) => {
                lsn += 1;
                last_valid_pos = file.stream_position()?;
            }
            Ok(None) | Err(KovaStorageError::CorruptRecord { .. }) => {
                return Ok((lsn, last_valid_pos));
            }
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kova_core::{Metadata, Vector, VectorId};
    use tempfile::tempdir;

    #[allow(clippy::cast_precision_loss)]
    fn ins(n: u64) -> Record {
        Record::Insert {
            id: VectorId::new(n),
            vector: Vector::try_new(vec![n as f32]).unwrap(),
            metadata: Metadata::new(),
        }
    }

    #[test]
    fn open_creates_empty_dir() {
        let dir = tempdir().unwrap();
        let wal = FileWal::open(dir.path()).unwrap();
        assert!(wal.iter_from(Lsn::ZERO).next().is_none());
    }

    #[test]
    fn append_then_sync_then_iter_roundtrip() {
        let dir = tempdir().unwrap();
        let mut wal = FileWal::open(dir.path()).unwrap();
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
        let dir = tempdir().unwrap();
        let mut wal = FileWal::open(dir.path()).unwrap();
        wal.append(&ins(1)).unwrap();
        assert!(wal.iter_from(Lsn::ZERO).next().is_none());
    }

    #[test]
    fn reopen_recovers_all_records() {
        let dir = tempdir().unwrap();
        {
            let mut wal = FileWal::open(dir.path()).unwrap();
            for n in 0..5 {
                wal.append(&ins(n)).unwrap();
            }
            wal.sync().unwrap();
        }
        let wal = FileWal::open(dir.path()).unwrap();
        let records: Vec<_> = wal.iter_from(Lsn::ZERO).collect::<Result<_, _>>().unwrap();
        assert_eq!(records.len(), 5);
    }

    #[test]
    fn reopen_assigns_next_lsn_after_existing() {
        let dir = tempdir().unwrap();
        {
            let mut wal = FileWal::open(dir.path()).unwrap();
            wal.append(&ins(0)).unwrap();
            wal.append(&ins(1)).unwrap();
            wal.sync().unwrap();
        }
        let mut wal = FileWal::open(dir.path()).unwrap();
        let lsn = wal.append(&ins(2)).unwrap();
        assert_eq!(lsn, Lsn::new(2));
    }

    #[test]
    fn torn_tail_in_active_is_truncated_on_reopen() {
        let dir = tempdir().unwrap();
        {
            let mut wal = FileWal::open(dir.path()).unwrap();
            wal.append(&ins(0)).unwrap();
            wal.append(&ins(1)).unwrap();
            wal.sync().unwrap();
        }
        // Append 3 garbage bytes to the only segment.
        let seg_path = segment_path(dir.path(), Lsn::ZERO);
        let mut file = OpenOptions::new().append(true).open(&seg_path).unwrap();
        file.write_all(&[0xff, 0xff, 0xff]).unwrap();
        file.sync_data().unwrap();
        drop(file);

        let wal = FileWal::open(dir.path()).unwrap();
        let records: Vec<_> = wal.iter_from(Lsn::ZERO).collect::<Result<_, _>>().unwrap();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn truncate_before_filters_iter() {
        let dir = tempdir().unwrap();
        let mut wal = FileWal::open(dir.path()).unwrap();
        for n in 0..5 {
            wal.append(&ins(n)).unwrap();
        }
        wal.sync().unwrap();
        wal.truncate_before(Lsn::new(3)).unwrap();
        let records: Vec<_> = wal.iter_from(Lsn::ZERO).collect::<Result<_, _>>().unwrap();
        assert_eq!(records.len(), 2);
    }

    // ---------- segmentation tests ----------

    #[test]
    fn rotates_when_segment_full() {
        let dir = tempdir().unwrap();
        // Tiny segment cap forces rotation after a couple of records.
        let mut wal = FileWal::open_with_segment_size(dir.path(), 32).unwrap();
        for n in 0..6 {
            wal.append(&ins(n)).unwrap();
        }
        wal.sync().unwrap();
        // We should have rotated at least once.
        assert!(!wal.finalized.is_empty(), "expected rotation");
        // Total record count survives across segments.
        let records: Vec<_> = wal.iter_from(Lsn::ZERO).collect::<Result<_, _>>().unwrap();
        assert_eq!(records.len(), 6);
    }

    #[test]
    fn recovery_walks_multiple_segments_in_order() {
        let dir = tempdir().unwrap();
        {
            let mut wal = FileWal::open_with_segment_size(dir.path(), 32).unwrap();
            for n in 0..8 {
                wal.append(&ins(n)).unwrap();
            }
            wal.sync().unwrap();
        }
        let wal = FileWal::open_with_segment_size(dir.path(), 32).unwrap();
        let records: Vec<_> = wal.iter_from(Lsn::ZERO).collect::<Result<_, _>>().unwrap();
        assert_eq!(records.len(), 8);
        for (i, (lsn, _)) in records.iter().enumerate() {
            assert_eq!(*lsn, Lsn::new(i as u64));
        }
    }

    #[test]
    fn truncate_before_deletes_finalized_segments() {
        let dir = tempdir().unwrap();
        let mut wal = FileWal::open_with_segment_size(dir.path(), 32).unwrap();
        for n in 0..8 {
            wal.append(&ins(n)).unwrap();
        }
        wal.sync().unwrap();
        let segments_before = wal.finalized.len();
        assert!(segments_before > 0);

        // Truncate up to a high LSN to remove early segments.
        wal.truncate_before(Lsn::new(6)).unwrap();
        let segments_after = wal.finalized.len();
        assert!(
            segments_after < segments_before,
            "expected some finalised segments removed"
        );
        // Disk inspection: ensure the directory has fewer files.
        let file_count = fs::read_dir(dir.path()).unwrap().count();
        assert!(file_count < segments_before + 1);
    }

    // ---------- last_lsn ----------

    #[test]
    fn last_lsn_is_none_on_empty_wal() {
        let dir = tempdir().unwrap();
        let wal = FileWal::open(dir.path()).unwrap();
        assert_eq!(wal.last_lsn(), None);
    }

    #[test]
    fn last_lsn_matches_most_recent_append() {
        let dir = tempdir().unwrap();
        let mut wal = FileWal::open(dir.path()).unwrap();
        let l1 = wal.append(&ins(1)).unwrap();
        assert_eq!(wal.last_lsn(), Some(l1));
        let l2 = wal.append(&ins(2)).unwrap();
        let l3 = wal.append(&ins(3)).unwrap();
        wal.sync().unwrap();
        assert_eq!(wal.last_lsn(), Some(l3));
        assert!(l2 < l3);
    }

    #[test]
    fn last_lsn_persists_across_reopen() {
        let dir = tempdir().unwrap();
        let before_close = {
            let mut wal = FileWal::open(dir.path()).unwrap();
            for n in 0..7 {
                wal.append(&ins(n)).unwrap();
            }
            wal.sync().unwrap();
            wal.last_lsn()
        };
        let reopened = FileWal::open(dir.path()).unwrap();
        assert_eq!(reopened.last_lsn(), before_close);
    }
}

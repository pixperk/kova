//! Atomic file-write utility : tmp + fsync + rename + dirsync.
//!
//! Crash-safe replacement of file contents. Observers see either the old
//! file (or no file) or the complete new file, never a partial write.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::KovaStorageError;

/// Write `contents` to `path` atomically.
///
/// Steps :
///   1. write to `path.tmp`
///   2. `fsync` the temp file (its bytes are now durable)
///   3. rename `path.tmp` to `path`         (POSIX-atomic)
///   4. `fsync` the parent directory         (makes the rename durable)
///
/// On crash at any step, observers see a consistent state : either the
/// old file (or no file at all), or the complete new file. Never a
/// partial write.
///
/// # Errors
/// Returns [`KovaStorageError::Io`] if any underlying file operation fails.
pub fn atomic_write(path: &Path, contents: &[u8]) -> Result<(), KovaStorageError> {
    let tmp_path = tmp_path_for(path);

    // 1. Write to temp file. truncate(true) clears any stale leftover.
    {
        let mut tmp = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;
        tmp.write_all(contents)?;
        // 2. fsync the temp file's contents.
        tmp.sync_data()?;
    }

    // 3. Atomically rename into place.
    std::fs::rename(&tmp_path, path)?;

    // 4. fsync the parent directory so the rename itself is durable.
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    File::open(dir)?.sync_all()?;

    Ok(())
}

/// Same atomicity contract as [`atomic_write`], but the caller writes
/// the contents to disk via a `BufWriter<File>` rather than handing
/// over a `&[u8]`. Use this when the full payload would be large enough
/// that buffering it in memory before writing is wasteful (e.g. an
/// HNSW graph snapshot of hundreds of MB).
///
/// Steps :
///   1. open `path.tmp` for writing, wrap in a `BufWriter`
///   2. invoke `write_fn(&mut writer)` so the caller can stream bytes
///   3. flush the buffer, then `fsync` the temp file (durable bytes)
///   4. rename `path.tmp` to `path` (POSIX-atomic)
///   5. `fsync` the parent directory (durable rename)
///
/// On crash at any step, observers see a consistent state : the old
/// file (or no file), or the complete new file. The closure's error
/// is propagated as-is ; only successful closure runs produce a
/// renamed `path`.
///
/// # Errors
/// Returns [`KovaStorageError`] for I/O failures at any step, or
/// whatever error the closure produces.
pub fn atomic_write_streaming<F>(path: &Path, write_fn: F) -> Result<(), KovaStorageError>
where
    F: FnOnce(&mut BufWriter<File>) -> Result<(), KovaStorageError>,
{
    let tmp_path = tmp_path_for(path);

    // 1. Open the temp file with a buffered writer.
    let tmp_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp_path)?;
    let mut writer = BufWriter::new(tmp_file);

    // 2. Let the caller stream bytes through the writer. Bail without
    //    renaming if they error : the .tmp file is harmless garbage.
    write_fn(&mut writer)?;

    // 3. Flush BufWriter's internal buffer, recover the File, fsync it.
    writer.flush()?;
    let tmp_file = writer
        .into_inner()
        .map_err(|e| KovaStorageError::Io(e.into_error()))?;
    tmp_file.sync_data()?;

    // 4. Atomically rename into place.
    std::fs::rename(&tmp_path, path)?;

    // 5. fsync the parent directory so the rename itself is durable.
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    File::open(dir)?.sync_all()?;

    Ok(())
}

/// Build the temp filename in the same directory : `foo.bin` -> `foo.bin.tmp`.
///
/// Keeping the temp file in the same directory matters : POSIX `rename` is
/// only atomic when source and destination are on the same filesystem.
fn tmp_path_for(path: &Path) -> PathBuf {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    tmp.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn writes_full_contents() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("data.bin");
        atomic_write(&path, b"hello world").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"hello world");
    }

    #[test]
    fn overwrites_existing_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("data.bin");
        fs::write(&path, b"old contents").unwrap();
        atomic_write(&path, b"new contents").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new contents");
    }

    #[test]
    fn no_tmp_file_left_on_success() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("data.bin");
        atomic_write(&path, b"x").unwrap();
        let tmp = dir.path().join("data.bin.tmp");
        assert!(!tmp.exists(), "tmp file should be renamed away");
    }

    #[test]
    fn overwrites_stale_tmp_file() {
        // Simulates a previous run that died after step 1 : leaves a
        // .tmp file in place. The next atomic_write should overwrite it
        // cleanly, not error.
        let dir = tempdir().unwrap();
        let path = dir.path().join("data.bin");
        let tmp = dir.path().join("data.bin.tmp");
        fs::write(&tmp, b"stale leftover").unwrap();
        atomic_write(&path, b"fresh write").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"fresh write");
        assert!(!tmp.exists());
    }

    // ---------- atomic_write_streaming ----------

    #[test]
    fn streaming_writes_all_buffered_bytes() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("data.bin");
        atomic_write_streaming(&path, |w| {
            w.write_all(b"hello ").map_err(KovaStorageError::Io)?;
            w.write_all(b"world").map_err(KovaStorageError::Io)?;
            Ok(())
        })
        .unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"hello world");
    }

    #[test]
    fn streaming_overwrites_existing_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("data.bin");
        fs::write(&path, b"old contents").unwrap();
        atomic_write_streaming(&path, |w| {
            w.write_all(b"new contents").map_err(KovaStorageError::Io)
        })
        .unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new contents");
    }

    #[test]
    fn streaming_leaves_old_file_intact_on_closure_error() {
        // Closure error must NOT touch the canonical path : the old
        // file (or absence of one) is preserved. Only successful
        // closure runs produce a renamed `path`.
        let dir = tempdir().unwrap();
        let path = dir.path().join("data.bin");
        fs::write(&path, b"original").unwrap();

        let err = atomic_write_streaming(&path, |w| {
            w.write_all(b"partial ").map_err(KovaStorageError::Io)?;
            // Bail before completing.
            Err(KovaStorageError::CorruptRecord {
                reason: "test-induced failure".into(),
            })
        });
        assert!(matches!(err, Err(KovaStorageError::CorruptRecord { .. })));

        // Canonical file is untouched.
        assert_eq!(fs::read(&path).unwrap(), b"original");
    }

    #[test]
    fn streaming_handles_large_payload_without_buffering_full_size() {
        // Stream 4 MiB of bytes ; should land on disk byte-for-byte.
        // No assertion about peak memory ; this test just exercises the
        // streaming path with a non-trivial payload.
        let dir = tempdir().unwrap();
        let path = dir.path().join("big.bin");
        let chunk = vec![0xAB_u8; 1024];
        atomic_write_streaming(&path, |w| {
            for _ in 0..4096 {
                w.write_all(&chunk).map_err(KovaStorageError::Io)?;
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), 4 * 1024 * 1024);
    }

    #[test]
    fn streaming_no_tmp_file_left_on_success() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("data.bin");
        atomic_write_streaming(&path, |w| w.write_all(b"x").map_err(KovaStorageError::Io)).unwrap();
        assert!(!dir.path().join("data.bin.tmp").exists());
    }
}

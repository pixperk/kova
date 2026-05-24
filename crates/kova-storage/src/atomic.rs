//! Atomic file-write utility : tmp + fsync + rename + dirsync.
//!
//! Crash-safe replacement of file contents. Observers see either the old
//! file (or no file) or the complete new file, never a partial write.

use std::fs::{File, OpenOptions};
use std::io::Write;
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
}

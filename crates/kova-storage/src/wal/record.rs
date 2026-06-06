//! WAL record framing : the on-disk frame format for [`Record`] values
//! (length + CRC + bincode payload) plus encode/decode helpers.

use std::io::{self, Read};

use kova_core::{Metadata, Vector, VectorId};
use serde::{Deserialize, Serialize};

use crate::KovaStorageError;

/// A single mutation applied to a shard. Persisted in the WAL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Record {
    /// Insert a new vector with the given id.
    Insert {
        /// Identifier the caller assigned.
        id: VectorId,
        /// The vector being inserted.
        vector: Vector,
        /// Metadata to associate with the vector.
        metadata: Metadata,
    },
    /// Delete the vector with the given id.
    Delete {
        /// Identifier to remove.
        id: VectorId,
    },
    /// Delete a batch of ids in one record. Semantically equivalent
    /// to N `Delete { id }` records on replay ; the compact form
    /// keeps the WAL smaller and lets a batched DELETE-by-predicate
    /// land as a single frame.
    DeleteMany {
        /// Identifiers to remove. Order isn't preserved through
        /// replay (each id is applied independently).
        ids: Vec<VectorId>,
    },
    /// Replace the metadata bag attached to `id`. The vector and
    /// graph node are untouched ; only the metadata store mutates.
    /// Replay re-applies the assignment idempotently.
    UpdateMetadata {
        /// Target identifier.
        id: VectorId,
        /// New metadata bag (replaces the old one in full).
        metadata: Metadata,
    },
}

/// Result of attempting to fill a buffer from a [`Read`] source.
///
/// Distinguishes "the log ended cleanly on a boundary" (`CleanEof`) from
/// "the log was mid-record when EOF hit" (`Torn`), which `Read::read_exact`
/// conflates into a single `UnexpectedEof`.
#[derive(Debug, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum ReadOutcome {
    /// Buffer was filled completely.
    Full,
    /// Reader was at EOF and we read 0 bytes : the log ended on a frame boundary.
    CleanEof,
    /// Read some bytes but hit EOF mid-buffer : the record is torn.
    Torn {
        /// How many bytes made it into the buffer before EOF.
        bytes_read: usize,
    },
}

/// Read into `buf` until full, EOF, or error.
///
/// Returns [`ReadOutcome`] distinguishing clean EOF from a torn read.
/// Retries on `Interrupted` (POSIX `EINTR` convention).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn read_full_or_torn<R: Read>(
    reader: &mut R,
    buf: &mut [u8],
) -> io::Result<ReadOutcome> {
    let mut total = 0;
    while total < buf.len() {
        match reader.read(&mut buf[total..]) {
            Ok(0) => {
                return Ok(if total == 0 {
                    ReadOutcome::CleanEof
                } else {
                    ReadOutcome::Torn { bytes_read: total }
                });
            }
            Ok(n) => total += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(ReadOutcome::Full)
}

/// Encode a record to its on-disk frame: 4-byte LE length, 4-byte LE CRC, payload.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn encode_record(record: &Record) -> Result<Vec<u8>, KovaStorageError> {
    let payload = bincode::serialize(record).map_err(KovaStorageError::Encode)?;

    let len = u32::try_from(payload.len()).map_err(|_| KovaStorageError::CorruptRecord {
        reason: format!("record too large: {} bytes", payload.len()),
    })?;
    let crc = crc32fast::hash(&payload);

    let mut out = Vec::with_capacity(8 + payload.len());
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Decode one frame from `reader`. Returns:
/// - `Ok(None)` if the reader is at clean EOF (no more records).
/// - `Ok(Some(record))` for a valid record.
/// - `Err(CorruptRecord { .. })` for a torn or CRC-invalid frame.
///
/// Frame structure: `||length (4 bytes LE)||CRC (4 bytes LE)||bincode payload (length bytes)||`
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn decode_record<R: Read>(reader: &mut R) -> Result<Option<Record>, KovaStorageError> {
    // --- length ---
    let mut len_buf = [0u8; 4];
    match read_full_or_torn(reader, &mut len_buf)? {
        ReadOutcome::CleanEof => return Ok(None),
        ReadOutcome::Torn { bytes_read } => {
            return Err(KovaStorageError::CorruptRecord {
                reason: format!("torn length field: read {bytes_read} of 4 bytes"),
            });
        }
        ReadOutcome::Full => {}
    }
    let len = u32::from_le_bytes(len_buf) as usize;

    // --- CRC ---
    let mut crc_buf = [0u8; 4];
    match read_full_or_torn(reader, &mut crc_buf)? {
        ReadOutcome::Full => {}
        _ => {
            return Err(KovaStorageError::CorruptRecord {
                reason: "torn CRC field".into(),
            });
        }
    }
    let expected_crc = u32::from_le_bytes(crc_buf);

    // --- payload ---
    let mut payload = vec![0u8; len];
    match read_full_or_torn(reader, &mut payload)? {
        ReadOutcome::Full => {}
        _ => {
            return Err(KovaStorageError::CorruptRecord {
                reason: format!("torn payload: expected {len} bytes"),
            });
        }
    }

    // --- verify CRC ---
    let actual_crc = crc32fast::hash(&payload);
    if actual_crc != expected_crc {
        return Err(KovaStorageError::CorruptRecord {
            reason: format!("CRC mismatch: expected {expected_crc:#x}, got {actual_crc:#x}"),
        });
    }

    // --- decode payload ---
    let record: Record = bincode::deserialize(&payload)?;
    Ok(Some(record))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_insert() -> Record {
        Record::Insert {
            id: VectorId::new(42),
            vector: Vector::try_new(vec![1.0, 2.0, 3.0]).unwrap(),
            metadata: Metadata::new(),
        }
    }

    fn sample_delete() -> Record {
        Record::Delete {
            id: VectorId::new(7),
        }
    }

    // ---------- read_full_or_torn ----------

    #[test]
    fn read_full_or_torn_full() {
        let mut cur = Cursor::new(vec![1, 2, 3, 4]);
        let mut buf = [0u8; 4];
        assert_eq!(
            read_full_or_torn(&mut cur, &mut buf).unwrap(),
            ReadOutcome::Full
        );
        assert_eq!(buf, [1, 2, 3, 4]);
    }

    #[test]
    fn read_full_or_torn_clean_eof() {
        let mut cur = Cursor::new(Vec::<u8>::new());
        let mut buf = [0u8; 4];
        assert_eq!(
            read_full_or_torn(&mut cur, &mut buf).unwrap(),
            ReadOutcome::CleanEof
        );
    }

    #[test]
    fn read_full_or_torn_torn() {
        let mut cur = Cursor::new(vec![1, 2]);
        let mut buf = [0u8; 4];
        assert_eq!(
            read_full_or_torn(&mut cur, &mut buf).unwrap(),
            ReadOutcome::Torn { bytes_read: 2 }
        );
    }

    // ---------- encode/decode roundtrips ----------

    #[test]
    fn roundtrip_insert() {
        let r = sample_insert();
        let bytes = encode_record(&r).unwrap();
        let mut cur = Cursor::new(bytes);
        assert_eq!(decode_record(&mut cur).unwrap().unwrap(), r);
    }

    #[test]
    fn roundtrip_delete() {
        let r = sample_delete();
        let bytes = encode_record(&r).unwrap();
        let mut cur = Cursor::new(bytes);
        assert_eq!(decode_record(&mut cur).unwrap().unwrap(), r);
    }

    // ---------- failure modes ----------

    #[test]
    fn clean_eof_returns_none() {
        let mut cur = Cursor::new(Vec::<u8>::new());
        assert!(decode_record(&mut cur).unwrap().is_none());
    }

    #[test]
    fn torn_length_field_errors() {
        let mut cur = Cursor::new(vec![1u8, 2u8]); // 2 bytes; need 4 for length
        let err = decode_record(&mut cur).unwrap_err();
        assert!(matches!(err, KovaStorageError::CorruptRecord { .. }));
    }

    #[test]
    fn crc_mismatch_errors() {
        let mut bytes = encode_record(&sample_insert()).unwrap();
        bytes[10] ^= 0xFF; // flip a byte in the payload (past the 8-byte header)
        let mut cur = Cursor::new(bytes);
        let err = decode_record(&mut cur).unwrap_err();
        assert!(matches!(err, KovaStorageError::CorruptRecord { .. }));
    }

    #[test]
    fn multiple_records_then_torn_tail() {
        let mut bytes = Vec::new();
        bytes.extend(encode_record(&sample_insert()).unwrap());
        bytes.extend(encode_record(&sample_delete()).unwrap());
        bytes.extend(encode_record(&sample_insert()).unwrap());
        bytes.extend(&[0xff, 0xff, 0xff]); // 3 garbage bytes (torn length)

        let mut cur = Cursor::new(bytes);
        assert!(decode_record(&mut cur).unwrap().is_some());
        assert!(decode_record(&mut cur).unwrap().is_some());
        assert!(decode_record(&mut cur).unwrap().is_some());
        assert!(matches!(
            decode_record(&mut cur).unwrap_err(),
            KovaStorageError::CorruptRecord { .. }
        ));
    }
}

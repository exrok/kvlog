//! On-disk framing constants and varint helpers for the bucket log stream.
//!
//! The stream interns msgs, targets, key names, services, strings, UUIDs,
//! spans, and archetypes against the index's existing dedup state. Each
//! identity is declared once via a `*Decl` frame at first sighting and
//! referenced by a small id thereafter. A `Record` frame carries an archetype
//! id, a timestamp delta, an optional span reference, and the field values in
//! archetype field order.

/// File magic for a bucket log stream header. ASCII `KVBL`.
pub const STREAM_HEADER_MAGIC: [u8; 4] = *b"KVBL";

/// File magic for a bucket log stream footer. ASCII `KVBE`.
pub const STREAM_FOOTER_MAGIC: [u8; 4] = *b"KVBE";

/// Current stream format version.
pub const STREAM_VERSION: u8 = 1;

/// Byte length of the file header.
pub const STREAM_HEADER_LEN: usize = 16;

/// Byte length of the file footer.
pub const STREAM_FOOTER_LEN: usize = 24;

/// Body frame tags. Each frame is `[u8 tag][uvarint payload_len][payload bytes]`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameTag {
    /// `[uvarint offset][uvarint len][bytes]`. Declares a general string or
    /// bytes value at a bucket.data offset. Reused for both String and Bytes
    /// FieldKinds and for the 8-byte payloads of large I64/U64 field values.
    StringDecl = 0x01,
    /// `[uvarint msg_id][uvarint offset][uvarint len][bytes]`. Declares a
    /// message string at a sequential msg id and bucket.data offset.
    MsgDecl = 0x02,
    /// `[uvarint uuid_id][16 bytes]`. Declares a UUID at a bucket.data offset.
    UuidDecl = 0x03,
    /// `[uvarint key_id][uvarint len][utf8 name]`. Declares a dynamic KeyID
    /// for this bucket. The wire key_id is stream-local; replay re-interns
    /// the name to obtain a process-local KeyID.
    KeyNameDecl = 0x04,
    /// `[uvarint target_id][uvarint len][utf8 name]`. Declares a target id.
    /// Replay re-interns to obtain a process-local target u16.
    TargetDecl = 0x05,
    /// `[u8 service_id][uvarint len][utf8 name]`. Declares a service id.
    ServiceDecl = 0x06,
    /// `[uvarint span_id][8 bytes SpanID][u8 has_parent][optional 8 bytes
    /// parent SpanID]`. Declares a span at a sequential span id.
    SpanDecl = 0x07,
    /// Archetype declaration. See [`Self::Record`] for the field layout.
    /// `[u8 level][u8 flags: bit0=has_span][uvarint service_id_or_0]
    /// [uvarint msg_id_plus_one][uvarint target_id_plus_one]
    /// [uvarint n_fields][n × uvarint key_descriptor]`.
    /// Key descriptor: low bit `0` = static (upper bits = StaticKey ordinal),
    /// low bit `1` = dynamic (upper bits = stream-local key id).
    ArchetypeDecl = 0x08,
    /// `[uvarint archetype_id][zigzag varint timestamp_delta_ns]
    /// [optional span_payload][n × [u8 field_kind][value bytes]]`.
    Record = 0x09,
}

impl FrameTag {
    #[inline]
    pub fn from_u8(byte: u8) -> Option<FrameTag> {
        match byte {
            0x01 => Some(FrameTag::StringDecl),
            0x02 => Some(FrameTag::MsgDecl),
            0x03 => Some(FrameTag::UuidDecl),
            0x04 => Some(FrameTag::KeyNameDecl),
            0x05 => Some(FrameTag::TargetDecl),
            0x06 => Some(FrameTag::ServiceDecl),
            0x07 => Some(FrameTag::SpanDecl),
            0x08 => Some(FrameTag::ArchetypeDecl),
            0x09 => Some(FrameTag::Record),
            _ => None,
        }
    }
}

/// Span kinds inside a Record's span payload. Mirrors `kvlog::SpanInfo`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanKindTag {
    StartWithParent = 0,
    Start = 1,
    Current = 2,
    End = 3,
}

impl SpanKindTag {
    #[inline]
    pub fn from_u8(byte: u8) -> Option<SpanKindTag> {
        match byte {
            0 => Some(SpanKindTag::StartWithParent),
            1 => Some(SpanKindTag::Start),
            2 => Some(SpanKindTag::Current),
            3 => Some(SpanKindTag::End),
            _ => None,
        }
    }
}

/// Decoded file header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamHeader {
    pub version: u8,
    pub generation: u64,
}

impl StreamHeader {
    pub fn encode(self) -> [u8; STREAM_HEADER_LEN] {
        let mut bytes = [0u8; STREAM_HEADER_LEN];
        bytes[0..4].copy_from_slice(&STREAM_HEADER_MAGIC);
        bytes[4] = self.version;
        bytes[8..16].copy_from_slice(&self.generation.to_le_bytes());
        bytes
    }

    pub fn decode(bytes: &[u8; STREAM_HEADER_LEN]) -> Option<StreamHeader> {
        if bytes[0..4] != STREAM_HEADER_MAGIC {
            return None;
        }
        let version = bytes[4];
        let generation = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        Some(StreamHeader { version, generation })
    }
}

/// Decoded file footer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamFooter {
    pub total_entries: u64,
    pub close_ts_ns: u64,
}

impl StreamFooter {
    pub fn encode(self) -> [u8; STREAM_FOOTER_LEN] {
        let mut bytes = [0u8; STREAM_FOOTER_LEN];
        bytes[0..4].copy_from_slice(&STREAM_FOOTER_MAGIC);
        bytes[4..12].copy_from_slice(&self.total_entries.to_le_bytes());
        bytes[12..20].copy_from_slice(&self.close_ts_ns.to_le_bytes());
        bytes
    }

    pub fn decode(bytes: &[u8; STREAM_FOOTER_LEN]) -> Option<StreamFooter> {
        if bytes[0..4] != STREAM_FOOTER_MAGIC {
            return None;
        }
        let total_entries = u64::from_le_bytes(bytes[4..12].try_into().unwrap());
        let close_ts_ns = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        Some(StreamFooter { total_entries, close_ts_ns })
    }
}

/// Replay-side errors.
#[derive(Debug)]
pub enum ReadError {
    TruncatedHeader,
    InvalidHeaderMagic,
    UnsupportedVersion(u8),
    InvalidFooter,
    TruncatedFrame,
    InvalidFrameTag(u8),
    InvalidVarint,
    InvalidFieldKind(u8),
    InvalidSpanKind(u8),
    InvalidUtf8,
    UnknownStringId,
    UnknownArchetypeId,
    UnknownSpanId,
    UnknownKeyId,
    UnknownTargetId,
    UnknownServiceId,
    BucketCapacityExceeded,
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadError::TruncatedHeader => write!(f, "stream truncated before file header"),
            ReadError::InvalidHeaderMagic => write!(f, "stream header magic mismatch"),
            ReadError::UnsupportedVersion(v) => write!(f, "unsupported stream version: {v}"),
            ReadError::InvalidFooter => write!(f, "stream footer present but malformed"),
            ReadError::TruncatedFrame => write!(f, "frame truncated"),
            ReadError::InvalidFrameTag(t) => write!(f, "unknown frame tag: 0x{t:02x}"),
            ReadError::InvalidVarint => write!(f, "varint exceeds 10 bytes"),
            ReadError::InvalidFieldKind(k) => write!(f, "invalid field kind: 0x{k:02x}"),
            ReadError::InvalidSpanKind(k) => write!(f, "invalid span kind: 0x{k:02x}"),
            ReadError::InvalidUtf8 => write!(f, "decl name is not valid utf-8"),
            ReadError::UnknownStringId => write!(f, "string id referenced before declaration"),
            ReadError::UnknownArchetypeId => write!(f, "archetype id referenced before declaration"),
            ReadError::UnknownSpanId => write!(f, "span id referenced before declaration"),
            ReadError::UnknownKeyId => write!(f, "key id referenced before declaration"),
            ReadError::UnknownTargetId => write!(f, "target id referenced before declaration"),
            ReadError::UnknownServiceId => write!(f, "service id referenced before declaration"),
            ReadError::BucketCapacityExceeded => write!(f, "bucket capacity exceeded during replay"),
        }
    }
}

impl std::error::Error for ReadError {}

/// Append a uvarint (LEB128) onto the buffer.
#[inline]
pub fn write_uvarint(buf: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buf.push((value as u8) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

/// Read a uvarint from the slice and advance the cursor.
#[inline]
pub fn read_uvarint(cursor: &mut &[u8]) -> Result<u64, ReadError> {
    let Some((&first, rest)) = cursor.split_first() else {
        return Err(ReadError::TruncatedFrame);
    };
    if first < 0x80 {
        *cursor = rest;
        return Ok(first as u64);
    }

    let mut value: u64 = (first & 0x7F) as u64;
    let mut shift: u32 = 7;
    let bytes = rest;
    for (i, &byte) in bytes.iter().enumerate() {
        if shift >= 64 {
            return Err(ReadError::InvalidVarint);
        }
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            *cursor = &bytes[i + 1..];
            return Ok(value);
        }
        shift += 7;
    }
    Err(ReadError::TruncatedFrame)
}

/// Append a zigzag-encoded signed varint onto the buffer.
#[inline]
pub fn write_zigzag(buf: &mut Vec<u8>, value: i64) {
    let zig = ((value << 1) ^ (value >> 63)) as u64;
    write_uvarint(buf, zig);
}

/// Read a zigzag-encoded signed varint and advance the cursor.
#[inline]
pub fn read_zigzag(cursor: &mut &[u8]) -> Result<i64, ReadError> {
    let zig = read_uvarint(cursor)?;
    Ok(((zig >> 1) as i64) ^ -((zig & 1) as i64))
}

/// Read a fixed number of bytes from the cursor.
#[inline]
pub fn read_slice<'a>(cursor: &mut &'a [u8], len: usize) -> Result<&'a [u8], ReadError> {
    if cursor.len() < len {
        return Err(ReadError::TruncatedFrame);
    }
    let (head, tail) = cursor.split_at(len);
    *cursor = tail;
    Ok(head)
}

/// Read a u8 from the cursor.
#[inline]
pub fn read_u8(cursor: &mut &[u8]) -> Result<u8, ReadError> {
    let Some((&byte, rest)) = cursor.split_first() else {
        return Err(ReadError::TruncatedFrame);
    };
    *cursor = rest;
    Ok(byte)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uvarint_roundtrip() {
        let cases: &[u64] = &[0, 1, 127, 128, 255, 256, 16_383, 16_384, u32::MAX as u64, u64::MAX];
        for &value in cases {
            let mut buf = Vec::new();
            write_uvarint(&mut buf, value);
            let mut cursor: &[u8] = &buf;
            let decoded = read_uvarint(&mut cursor).unwrap();
            assert_eq!(decoded, value, "uvarint round-trip failed for {value}");
            assert!(cursor.is_empty(), "cursor not fully consumed for {value}");
        }
    }

    #[test]
    fn zigzag_roundtrip() {
        let cases: &[i64] = &[0, 1, -1, 63, -64, i32::MIN as i64, i32::MAX as i64, i64::MIN, i64::MAX];
        for &value in cases {
            let mut buf = Vec::new();
            write_zigzag(&mut buf, value);
            let mut cursor: &[u8] = &buf;
            let decoded = read_zigzag(&mut cursor).unwrap();
            assert_eq!(decoded, value, "zigzag round-trip failed for {value}");
            assert!(cursor.is_empty(), "cursor not fully consumed for {value}");
        }
    }

    #[test]
    fn truncated_uvarint() {
        let bytes: &[u8] = &[0x80, 0x80]; // continuation never terminates
        let mut cursor = bytes;
        assert!(matches!(read_uvarint(&mut cursor), Err(ReadError::TruncatedFrame)));
    }

    #[test]
    fn header_roundtrip() {
        let header = StreamHeader { version: STREAM_VERSION, generation: 0xDEAD_BEEF };
        let bytes = header.encode();
        assert_eq!(&bytes[..4], b"KVBL");
        let decoded = StreamHeader::decode(&bytes).unwrap();
        assert_eq!(decoded, header);
    }

    #[test]
    fn footer_roundtrip() {
        let footer = StreamFooter { total_entries: 100, close_ts_ns: 42 };
        let bytes = footer.encode();
        assert_eq!(&bytes[..4], b"KVBE");
        let decoded = StreamFooter::decode(&bytes).unwrap();
        assert_eq!(decoded, footer);
    }

    #[test]
    fn header_magic_mismatch() {
        let mut bytes = [0u8; STREAM_HEADER_LEN];
        bytes[0..4].copy_from_slice(b"NOPE");
        assert!(StreamHeader::decode(&bytes).is_none());
    }
}

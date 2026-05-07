//! Stream replay: decodes a bucket log stream and installs its contents
//! directly into an `Index`, bypassing the per-record hashing the normal
//! ingest path would re-do.
//!
//! Mutation primitives live on `Index` as `pub(crate) fn install_*`. The
//! replay loop here decodes frames and dispatches to those primitives.

use std::collections::HashMap;
use std::num::NonZeroU64;

use kvlog::{LogLevel, SpanID};
use smallvec::SmallVec;

use crate::field_table::KeyID;
use crate::index::archetype::ServiceId;
use crate::index::{f60::f64_to_f60, i60, Field, FieldKind, Index, InternedRange};

use super::format::{
    read_slice, read_u8, read_uvarint, read_zigzag, FrameTag, ReadError, SpanKindTag, StreamFooter, StreamHeader,
    STREAM_FOOTER_LEN, STREAM_FOOTER_MAGIC, STREAM_HEADER_LEN, STREAM_VERSION,
};

/// Per-call replay state. Maps wire-side global identities (KeyID, target u16,
/// ServiceId) to the local ids the replaying process assigns. Strings, UUIDs,
/// msgs, archetypes, and spans use the bucket's natural ids and need no
/// remapping.
struct Replayer {
    key_map: HashMap<u16, KeyID>,
    target_map: HashMap<u16, u16>,
    service_map: [Option<ServiceId>; 256],
    /// Wire offset -> string len, populated from StringDecl frames so that
    /// Record string-value references can recover the InternedRange.
    string_lens: HashMap<u32, u16>,
    last_timestamp_ns: u64,
}

impl Replayer {
    fn new() -> Self {
        Replayer {
            key_map: HashMap::new(),
            target_map: HashMap::new(),
            service_map: [None; 256],
            string_lens: HashMap::new(),
            last_timestamp_ns: 0,
        }
    }
}

/// Apply a stream slice (header + body + optional footer) to `index`.
/// Returns the number of records applied.
pub fn ingest(index: &mut Index, bytes: &[u8]) -> Result<u64, ReadError> {
    if bytes.len() < STREAM_HEADER_LEN {
        return Err(ReadError::TruncatedHeader);
    }
    let header_bytes: &[u8; STREAM_HEADER_LEN] = bytes[..STREAM_HEADER_LEN].try_into().unwrap();
    let header = StreamHeader::decode(header_bytes).ok_or(ReadError::InvalidHeaderMagic)?;
    if header.version != STREAM_VERSION {
        return Err(ReadError::UnsupportedVersion(header.version));
    }
    let after_header = &bytes[STREAM_HEADER_LEN..];

    let body = if after_header.len() >= STREAM_FOOTER_LEN
        && after_header[after_header.len() - STREAM_FOOTER_LEN..][..4] == STREAM_FOOTER_MAGIC
    {
        let footer_start = after_header.len() - STREAM_FOOTER_LEN;
        let footer_bytes: &[u8; STREAM_FOOTER_LEN] = after_header[footer_start..].try_into().unwrap();
        StreamFooter::decode(footer_bytes).ok_or(ReadError::InvalidFooter)?;
        &after_header[..footer_start]
    } else {
        after_header
    };

    let mut replayer = Replayer::new();
    let mut cursor = body;
    let mut records: u64 = 0;
    let mut field_buf: SmallVec<[Field; 24]> = SmallVec::new();

    while !cursor.is_empty() {
        let tag_byte = read_u8(&mut cursor)?;
        let tag = FrameTag::from_u8(tag_byte).ok_or(ReadError::InvalidFrameTag(tag_byte))?;
        let payload_len = read_uvarint(&mut cursor)? as usize;
        let payload = read_slice(&mut cursor, payload_len)?;
        let mut p = payload;
        match tag {
            FrameTag::StringDecl => {
                let offset = read_uvarint(&mut p)? as u32;
                let len = read_uvarint(&mut p)? as u16;
                let bytes = read_slice(&mut p, len as usize)?;
                index.install_string_decl(offset, len, bytes)?;
                replayer.string_lens.insert(offset, len);
            }
            FrameTag::MsgDecl => {
                let msg_id = read_uvarint(&mut p)? as u16;
                let offset = read_uvarint(&mut p)? as u32;
                let len = read_uvarint(&mut p)? as u16;
                let bytes = read_slice(&mut p, len as usize)?;
                index.install_msg_decl(msg_id, offset, len, bytes)?;
            }
            FrameTag::UuidDecl => {
                let offset = read_uvarint(&mut p)? as u32;
                let bytes = read_slice(&mut p, 16)?;
                let mut arr = [0u8; 16];
                arr.copy_from_slice(bytes);
                index.install_uuid_decl(offset, &arr)?;
            }
            FrameTag::KeyNameDecl => {
                let wire_key_id = read_uvarint(&mut p)? as u16;
                let len = read_uvarint(&mut p)? as usize;
                let bytes = read_slice(&mut p, len)?;
                let name = std::str::from_utf8(bytes).map_err(|_| ReadError::InvalidUtf8)?;
                let local = KeyID::intern(name);
                replayer.key_map.insert(wire_key_id, local);
            }
            FrameTag::TargetDecl => {
                let wire_target_id = read_uvarint(&mut p)? as u16;
                let len = read_uvarint(&mut p)? as usize;
                let bytes = read_slice(&mut p, len)?;
                let local = index.install_target(bytes).ok_or(ReadError::BucketCapacityExceeded)?;
                replayer.target_map.insert(wire_target_id, local);
            }
            FrameTag::ServiceDecl => {
                let wire_service_id = read_u8(&mut p)?;
                let len = read_uvarint(&mut p)? as usize;
                let bytes = read_slice(&mut p, len)?;
                let name = std::str::from_utf8(bytes).map_err(|_| ReadError::InvalidUtf8)?;
                let local = ServiceId::intern(name);
                replayer.service_map[wire_service_id as usize] = Some(local);
            }
            FrameTag::SpanDecl => {
                let span_id = read_uvarint(&mut p)? as u32;
                let span_full_bytes = read_slice(&mut p, 8)?;
                let span_full = u64::from_le_bytes(span_full_bytes.try_into().unwrap());
                let has_parent = read_u8(&mut p)?;
                let parent = if has_parent != 0 {
                    let parent_bytes = read_slice(&mut p, 8)?;
                    let parent_full = u64::from_le_bytes(parent_bytes.try_into().unwrap());
                    NonZeroU64::new(parent_full).map(SpanID::from)
                } else {
                    None
                };
                // `flags` was added in stream v2. v1 streams have no
                // trailing bytes here; treat absent flags as 0.
                let flags = if p.is_empty() { 0 } else { read_uvarint(&mut p)? as u32 };
                let span = NonZeroU64::new(span_full).map(SpanID::from).ok_or(ReadError::UnknownSpanId)?;
                index.install_span_decl(span_id, span, parent, flags)?;
            }
            FrameTag::ArchetypeDecl => {
                let level_mask = read_u8(&mut p)?;
                let flags = read_u8(&mut p)?;
                let in_span = (flags & 0b1) != 0;
                let service_u8 = read_uvarint(&mut p)? as u8;
                let service = if service_u8 == 0 {
                    None
                } else {
                    Some(replayer.service_map[service_u8 as usize].ok_or(ReadError::UnknownServiceId)?)
                };
                let msg_id_plus_one = read_uvarint(&mut p)? as u16;
                let target_id_plus_one = read_uvarint(&mut p)? as u16;
                let n_fields = read_uvarint(&mut p)? as u16;
                let mut field_keys: SmallVec<[u16; 24]> = SmallVec::new();
                for _ in 0..n_fields {
                    let descriptor = read_uvarint(&mut p)?;
                    let raw = (descriptor >> 1) as u16;
                    let local_raw = if descriptor & 1 == 0 {
                        raw
                    } else {
                        replayer.key_map.get(&raw).copied().ok_or(ReadError::UnknownKeyId)?.raw()
                    };
                    field_keys.push(local_raw);
                }
                let local_target = if target_id_plus_one == 0 {
                    0
                } else {
                    let wire = target_id_plus_one - 1;
                    *replayer.target_map.get(&wire).ok_or(ReadError::UnknownTargetId)?
                };
                let level = level_from_mask(level_mask);
                let msg_id = if msg_id_plus_one == 0 { 0 } else { msg_id_plus_one - 1 };
                index.install_archetype_decl(msg_id, local_target, level, in_span, service, &field_keys)?;
            }
            FrameTag::Record => {
                let archetype_id = read_uvarint(&mut p)? as u16;
                let delta = read_zigzag(&mut p)?;
                replayer.last_timestamp_ns = (replayer.last_timestamp_ns as i64).wrapping_add(delta) as u64;
                let timestamp = replayer.last_timestamp_ns;
                let in_span = index.archetype_in_span_for(archetype_id).ok_or(ReadError::UnknownArchetypeId)?;
                let n_fields = index.archetype_field_count_for(archetype_id).ok_or(ReadError::UnknownArchetypeId)?;
                let (span_kind, span_id) = if in_span {
                    let kind_byte = read_u8(&mut p)?;
                    let kind = SpanKindTag::from_u8(kind_byte).ok_or(ReadError::InvalidSpanKind(kind_byte))?;
                    let sid = read_uvarint(&mut p)? as u32;
                    (Some(kind), Some(sid))
                } else {
                    (None, None)
                };
                field_buf.clear();
                for _ in 0..n_fields {
                    let field = decode_field(&mut p, &replayer.string_lens)?;
                    field_buf.push(field);
                }
                index.install_record(archetype_id, timestamp, span_kind, span_id, &field_buf)?;
                records += 1;
            }
        }
    }

    index.flush_pending();
    Ok(records)
}

fn level_from_mask(mask: u8) -> LogLevel {
    match mask & 0xF {
        0b0001 => LogLevel::Debug,
        0b0010 => LogLevel::Info,
        0b0100 => LogLevel::Warn,
        0b1000 => LogLevel::Error,
        _ => LogLevel::Info,
    }
}

fn decode_field(cursor: &mut &[u8], string_lens: &HashMap<u32, u16>) -> Result<Field, ReadError> {
    let kind_byte = read_u8(cursor)?;
    let kind = decode_field_kind(kind_byte)?;
    let mask = match kind {
        FieldKind::String | FieldKind::Bytes => {
            let offset = read_uvarint(cursor)? as u32;
            let len = *string_lens.get(&offset).ok_or(ReadError::UnknownStringId)?;
            let range = InternedRange { offset, data: 0, len };
            range.field_mask()
        }
        FieldKind::I60 => {
            let v = read_zigzag(cursor)?;
            i60::from_i64(v)
        }
        FieldKind::I64 | FieldKind::U64 => read_uvarint(cursor)?,
        FieldKind::F60 | FieldKind::Seconds => {
            let buf = read_slice(cursor, 8)?;
            let v = f64::from_le_bytes(buf.try_into().unwrap());
            f64_to_f60(v)
        }
        FieldKind::UUID => read_uvarint(cursor)?,
        FieldKind::Bool => {
            let b = read_u8(cursor)?;
            (b & 1) as u64
        }
        FieldKind::None => 0,
        FieldKind::Timestamp => {
            let buf = read_slice(cursor, 8)?;
            let v = i64::from_le_bytes(buf.try_into().unwrap());
            (v as u64) & ((1u64 << 60) - 1)
        }
        FieldKind::_Reserved2
        | FieldKind::_Reserved3
        | FieldKind::_Reserved4
        | FieldKind::_Reserved5
        | FieldKind::_Reserved6 => {
            return Err(ReadError::InvalidFieldKind(kind_byte));
        }
    };
    Ok(Field::new(kind, mask))
}

fn decode_field_kind(byte: u8) -> Result<FieldKind, ReadError> {
    match byte {
        0 => Ok(FieldKind::None),
        1 => Ok(FieldKind::String),
        2 => Ok(FieldKind::Bytes),
        3 => Ok(FieldKind::I60),
        4 => Ok(FieldKind::I64),
        5 => Ok(FieldKind::U64),
        6 => Ok(FieldKind::F60),
        7 => Ok(FieldKind::Bool),
        8 => Ok(FieldKind::UUID),
        9 => Ok(FieldKind::Seconds),
        10 => Ok(FieldKind::Timestamp),
        _ => Err(ReadError::InvalidFieldKind(byte)),
    }
}

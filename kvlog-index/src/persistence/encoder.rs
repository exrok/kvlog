//! Bucket log stream encoder state.
//!
//! State is intentionally minimal: high-water marks against the index's
//! existing dedup state (`new_ranges`, `new_uuids`, `archetype_map`,
//! `msg_map`, `span_table`) and bitsets for globally-interned identities
//! (KeyID, target id, ServiceId) declared once per bucket file. The actual
//! emit logic lives on `Index` (see `Index::emit_record` in `index.rs`)
//! where it has direct access to the bucket arrays without exposing private
//! fields.

use crate::field_table::{KeyID, MIN_DYN_KEY};
use crate::index::archetype::{Archetype, ServiceId};
use crate::index::{f60::f60_to_f64, Bucket, Field, FieldKind, InternedRange};
use crate::shared_interner::SharedIntermentBuffer;
use ahash::RandomState;
use hashbrown::HashTable;
use kvlog::encoding::SpanInfo;

use super::format::{
    write_uvarint, write_zigzag, FrameTag, SpanKindTag, StreamFooter, StreamHeader, STREAM_FOOTER_LEN,
    STREAM_HEADER_LEN, STREAM_VERSION,
};

/// Encoder state attached to an `Index` when persistence is enabled.
/// All fields persist across mid-bucket buffer swaps and reset on bucket
/// rotation via [`Self::reset`].
pub struct BucketLogStreamEncoder {
    /// Number of entries in `Index::new_ranges` already emitted as
    /// `StringDecl` frames. Resets to 0 every `commit()` because
    /// `new_ranges` is drained then.
    pub(crate) new_ranges_cursor: usize,

    /// Number of entries in `Index::new_uuids` already emitted as
    /// `UuidDecl` frames. Resets to 0 every `commit()`.
    pub(crate) new_uuids_cursor: usize,

    /// Number of archetypes already declared (= last seen `archetype_map.len()`).
    pub(crate) declared_archetype_count: u16,

    /// Number of messages already declared (= last seen `msg_map.len()`).
    pub(crate) declared_msg_count: u16,

    /// Number of spans already declared (= last seen `span_table.len()`).
    pub(crate) declared_span_count: u32,

    /// Per-bucket bitset of declared KeyIDs (1 bit per `KeyID::raw()`).
    pub(crate) declared_keys: Vec<u64>,

    /// Per-bucket bitset of declared target u16 ids.
    pub(crate) declared_targets: Vec<u64>,

    /// Per-bucket bitset of declared ServiceId values (1..=255).
    pub(crate) declared_services: [u64; 4],

    /// Last record's timestamp in ns. Used for delta encoding.
    pub(crate) last_timestamp_ns: u64,
}

impl BucketLogStreamEncoder {
    pub fn new() -> Self {
        BucketLogStreamEncoder {
            new_ranges_cursor: 0,
            new_uuids_cursor: 0,
            declared_archetype_count: 0,
            declared_msg_count: 0,
            declared_span_count: 0,
            declared_keys: Vec::new(),
            declared_targets: Vec::new(),
            declared_services: [0u64; 4],
            last_timestamp_ns: 0,
        }
    }

    /// Reset all dedup state. Called on bucket rotation.
    pub fn reset(&mut self) {
        self.new_ranges_cursor = 0;
        self.new_uuids_cursor = 0;
        self.declared_archetype_count = 0;
        self.declared_msg_count = 0;
        self.declared_span_count = 0;
        self.declared_keys.clear();
        self.declared_targets.clear();
        self.declared_services = [0u64; 4];
        self.last_timestamp_ns = 0;
    }

    /// Reset cursors that follow the index's per-record staging vectors.
    /// Called after `commit()` drains `new_ranges` and `new_uuids`.
    pub fn after_commit(&mut self) {
        self.new_ranges_cursor = 0;
        self.new_uuids_cursor = 0;
    }

    pub fn key_declared(&self, key_raw: u16) -> bool {
        let word = key_raw as usize >> 6;
        let bit = key_raw as usize & 0b11_1111;
        self.declared_keys.get(word).copied().unwrap_or(0) & (1u64 << bit) != 0
    }

    pub fn mark_key_declared(&mut self, key_raw: u16) {
        let word = key_raw as usize >> 6;
        let bit = key_raw as usize & 0b11_1111;
        if self.declared_keys.len() <= word {
            self.declared_keys.resize(word + 1, 0);
        }
        self.declared_keys[word] |= 1u64 << bit;
    }

    pub fn target_declared(&self, target_raw: u16) -> bool {
        let word = target_raw as usize >> 6;
        let bit = target_raw as usize & 0b11_1111;
        self.declared_targets.get(word).copied().unwrap_or(0) & (1u64 << bit) != 0
    }

    pub fn mark_target_declared(&mut self, target_raw: u16) {
        let word = target_raw as usize >> 6;
        let bit = target_raw as usize & 0b11_1111;
        if self.declared_targets.len() <= word {
            self.declared_targets.resize(word + 1, 0);
        }
        self.declared_targets[word] |= 1u64 << bit;
    }

    pub fn service_declared(&self, service_id: u8) -> bool {
        let word = (service_id as usize) >> 6;
        let bit = (service_id as usize) & 0b11_1111;
        self.declared_services[word] & (1u64 << bit) != 0
    }

    pub fn mark_service_declared(&mut self, service_id: u8) {
        let word = (service_id as usize) >> 6;
        let bit = (service_id as usize) & 0b11_1111;
        self.declared_services[word] |= 1u64 << bit;
    }
}

/// Append the file header.
pub fn write_header(buf: &mut Vec<u8>, generation: u64) {
    let header = StreamHeader { version: STREAM_VERSION, generation };
    buf.extend_from_slice(&header.encode());
}

/// Append the file footer.
pub fn write_footer(buf: &mut Vec<u8>, close_ts_ns: u64, total_entries: u64) {
    let footer = StreamFooter { total_entries, close_ts_ns };
    buf.extend_from_slice(&footer.encode());
}

/// Emit decl frames for any newly-seen identities and a Record frame for the
/// log entry just written into the bucket. Called between
/// `Index::write_current_to_bucket` and `Index::commit`.
///
/// Reads the just-staged entries out of `new_ranges` / `new_uuids` (committed
/// later in `commit()`), the just-built archetype/span/msg out of the bucket
/// arrays, and emits the wire frames into `buf`. No hash maps are duplicated.
pub fn record(
    enc: &mut BucketLogStreamEncoder,
    buf: &mut Vec<u8>,
    bucket: &Bucket,
    new_ranges: &[(InternedRange, u64)],
    new_uuids: &[(u32, u64)],
    msg_map: &HashTable<InternedRange>,
    msg_count: usize,
    span_count: usize,
    targets: &SharedIntermentBuffer,
    log_index: u32,
    timestamp: u64,
    span_info: &SpanInfo,
) {
    let archetype_id = unsafe { bucket.archetype_id_at(log_index) };
    let archetype = unsafe { bucket.archetype_at(archetype_id) };

    // Emit MsgDecl if msg_map grew this record.
    if msg_count > enc.declared_msg_count as usize {
        let msg_id = enc.declared_msg_count;
        let msg_range = InternedRange { offset: archetype.msg_offset, data: msg_id, len: archetype.msg_len };
        let msg_bytes = unsafe { bucket.data_bytes(msg_range) };
        emit_msg_decl(buf, msg_id, msg_range.offset, msg_range.len, msg_bytes);
        enc.declared_msg_count = msg_count as u16;
    }

    // Emit StringDecl for newly-staged general strings.
    for (range, _hash) in &new_ranges[enc.new_ranges_cursor..] {
        let bytes = unsafe { bucket.data_bytes(*range) };
        emit_string_decl(buf, range.offset, range.len, bytes);
    }
    enc.new_ranges_cursor = new_ranges.len();

    // Emit UuidDecl for newly-staged UUIDs.
    for (uuid_offset, _hash) in &new_uuids[enc.new_uuids_cursor..] {
        let bytes = unsafe { bucket.uuid_at(*uuid_offset) };
        emit_uuid_decl(buf, *uuid_offset, bytes);
    }
    enc.new_uuids_cursor = new_uuids.len();

    // Emit SpanDecls for every slot newly allocated in span_data since the
    // last record. A single record can allocate more than one slot: the
    // record's own span at the lowest new slot, then any cross-bucket
    // parents replicated above it.
    //
    // INVARIANT (relied on for replay parent_index resolution):
    //
    //   The slot range [declared_span_count, span_count) for a single
    //   `record()` call always forms a single ancestor chain ordered
    //   child-at-low-slot, root-at-high-slot. This holds because
    //   `Index::install_live_span` allocates the leaf first, then iterates
    //   up the parent chain with each step appending to `spans_used` — and
    //   `record()` is called exactly once per log entry.
    //
    // We emit this batch in REVERSE slot order so replay's
    // `install_span_decl` sees parents before children within the batch.
    // Combined with the fact that any parent from an earlier batch is
    // already in `span_table` from its own prior emission, this lets
    // replay resolve `parent_index` inline without a post-pass fixup.
    let new_start = enc.declared_span_count;
    let new_end = span_count as u32;
    if new_end > new_start {
        for slot in (new_start..new_end).rev() {
            let sr = unsafe { bucket.span_range_at(slot) };
            emit_span_decl(buf, slot, sr.id.as_u64(), sr.parent.map(|p| p.as_u64()), sr.flags);
        }
        enc.declared_span_count = new_end;
    }

    // Emit ArchetypeDecl if archetype_map grew.
    if (archetype_id as usize) >= enc.declared_archetype_count as usize {
        // ArchetypeDecl references service, target, and dynamic key ids. Emit
        // those dependency declarations only when a new archetype needs them,
        // not on every record with an already-declared archetype.
        if let Some(service) = archetype.service() {
            let svc_u8 = service.as_u8();
            if !enc.service_declared(svc_u8) {
                emit_service_decl(buf, svc_u8, service.as_str());
                enc.mark_service_declared(svc_u8);
            }
        }

        for &key_raw in archetype.field_keys() {
            let raw = key_raw.raw();
            if raw >= MIN_DYN_KEY && !enc.key_declared(raw) {
                let name = unsafe { KeyID::new(raw) }.as_str();
                emit_key_name_decl(buf, raw, name);
                enc.mark_key_declared(raw);
            }
        }

        let target_id = archetype.target_id;
        if !enc.target_declared(target_id) {
            let mapper = targets.mapper();
            if let Some(text) = mapper.get(target_id) {
                emit_target_decl(buf, target_id, text);
            } else {
                emit_target_decl(buf, target_id, "");
            }
            enc.mark_target_declared(target_id);
        }

        // Recover the msg_id by hashing the msg bytes back through msg_map.
        // The msg_id is InternedRange.data — set when intern_msg ran. Cheap
        // enough: one lookup per unique archetype.
        let msg_range = InternedRange { offset: archetype.msg_offset, data: 0, len: archetype.msg_len };
        let msg_bytes = unsafe { bucket.data_bytes(msg_range) };
        let msg_id = lookup_msg_id(bucket, msg_map, msg_bytes).unwrap_or(0);
        emit_archetype_decl(buf, archetype, msg_id);
        enc.declared_archetype_count = archetype_id + 1;
    }

    // Record frame.
    emit_record_frame(buf, enc, bucket, archetype_id, archetype, log_index, timestamp, span_info);
    enc.last_timestamp_ns = timestamp;
}

fn lookup_msg_id(bucket: &Bucket, msg_map: &HashTable<InternedRange>, msg_bytes: &[u8]) -> Option<u16> {
    if msg_bytes.is_empty() {
        // intern_msg of empty bytes never happens (msg=b"" is the default and gets interned with len=0)
        // but still we need to handle the case.
        return None;
    }
    let hash = bucket.random_state.hash_one(msg_bytes);
    let interned = msg_map.find(hash, |v| unsafe { bucket.data_bytes(*v) == msg_bytes })?;
    Some(interned.data)
}

fn emit_service_decl(buf: &mut Vec<u8>, service_id: u8, name: &str) {
    let pos = begin_frame(buf, FrameTag::ServiceDecl);
    buf.push(service_id);
    write_uvarint(buf, name.len() as u64);
    buf.extend_from_slice(name.as_bytes());
    finish_frame(buf, pos);
}

fn emit_key_name_decl(buf: &mut Vec<u8>, key_id: u16, name: &str) {
    let pos = begin_frame(buf, FrameTag::KeyNameDecl);
    write_uvarint(buf, key_id as u64);
    write_uvarint(buf, name.len() as u64);
    buf.extend_from_slice(name.as_bytes());
    finish_frame(buf, pos);
}

fn emit_target_decl(buf: &mut Vec<u8>, target_id: u16, name: &str) {
    let pos = begin_frame(buf, FrameTag::TargetDecl);
    write_uvarint(buf, target_id as u64);
    write_uvarint(buf, name.len() as u64);
    buf.extend_from_slice(name.as_bytes());
    finish_frame(buf, pos);
}

fn emit_msg_decl(buf: &mut Vec<u8>, msg_id: u16, offset: u32, len: u16, bytes: &[u8]) {
    let pos = begin_frame(buf, FrameTag::MsgDecl);
    write_uvarint(buf, msg_id as u64);
    write_uvarint(buf, offset as u64);
    write_uvarint(buf, len as u64);
    buf.extend_from_slice(bytes);
    finish_frame(buf, pos);
}

fn emit_string_decl(buf: &mut Vec<u8>, offset: u32, len: u16, bytes: &[u8]) {
    let pos = begin_frame(buf, FrameTag::StringDecl);
    write_uvarint(buf, offset as u64);
    write_uvarint(buf, len as u64);
    buf.extend_from_slice(bytes);
    finish_frame(buf, pos);
}

fn emit_uuid_decl(buf: &mut Vec<u8>, uuid_offset: u32, bytes: &[u8; 16]) {
    let pos = begin_frame(buf, FrameTag::UuidDecl);
    write_uvarint(buf, uuid_offset as u64);
    buf.extend_from_slice(bytes);
    finish_frame(buf, pos);
}

fn emit_span_decl(buf: &mut Vec<u8>, span_id: u32, span_full: u64, parent: Option<u64>, flags: u32) {
    let pos = begin_frame(buf, FrameTag::SpanDecl);
    write_uvarint(buf, span_id as u64);
    buf.extend_from_slice(&span_full.to_le_bytes());
    match parent {
        Some(p) => {
            buf.push(1);
            buf.extend_from_slice(&p.to_le_bytes());
        }
        None => {
            buf.push(0);
        }
    }
    write_uvarint(buf, flags as u64);
    finish_frame(buf, pos);
}

fn emit_archetype_decl(buf: &mut Vec<u8>, archetype: &Archetype, msg_id: u16) {
    let pos = begin_frame(buf, FrameTag::ArchetypeDecl);
    // Level: 4 bits as raw mask, plus a single bit for in_span.
    buf.push((archetype.mask & 0xF) as u8);
    let flags = if archetype.in_span() { 1u8 } else { 0u8 };
    buf.push(flags);
    write_uvarint(buf, archetype.raw_service() as u64);
    // msg_id+1, target_id+1: 0 means "absent" if we ever support that. Today both are always present.
    write_uvarint(buf, msg_id as u64 + 1);
    write_uvarint(buf, archetype.target_id as u64 + 1);
    let keys = archetype.field_keys();
    write_uvarint(buf, keys.len() as u64);
    for &k in keys {
        // Static keys: low bit 0, ordinal in upper bits.
        // Dynamic keys: low bit 1, raw KeyID in upper bits.
        let raw = k.raw();
        let descriptor = if raw < MIN_DYN_KEY { (raw as u64) << 1 } else { ((raw as u64) << 1) | 1 };
        write_uvarint(buf, descriptor);
    }
    finish_frame(buf, pos);
}

fn emit_record_frame(
    buf: &mut Vec<u8>,
    enc: &BucketLogStreamEncoder,
    bucket: &Bucket,
    archetype_id: u16,
    archetype: &Archetype,
    log_index: u32,
    timestamp: u64,
    span_info: &SpanInfo,
) {
    let pos = begin_frame(buf, FrameTag::Record);
    write_uvarint(buf, archetype_id as u64);
    let delta = (timestamp as i64).wrapping_sub(enc.last_timestamp_ns as i64);
    write_zigzag(buf, delta);
    if archetype.in_span() {
        let span_idx = unsafe { bucket.span_index_at(log_index) };
        match span_info {
            SpanInfo::Start { parent: Some(_), .. } => {
                buf.push(SpanKindTag::StartWithParent as u8);
            }
            SpanInfo::Start { parent: None, .. } => {
                buf.push(SpanKindTag::Start as u8);
            }
            SpanInfo::Current { .. } => {
                buf.push(SpanKindTag::Current as u8);
            }
            SpanInfo::End { .. } => {
                buf.push(SpanKindTag::End as u8);
            }
            SpanInfo::None => {
                // Should never happen if archetype.in_span() is true.
                buf.push(SpanKindTag::Current as u8);
            }
        }
        write_uvarint(buf, span_idx as u64);
    }
    let fields = unsafe { bucket.fields_at(log_index) };
    for f in fields {
        encode_field(buf, *f);
    }
    finish_frame(buf, pos);
}

fn encode_field(buf: &mut Vec<u8>, f: Field) {
    let kind = f.kind();
    buf.push(kind as u8);
    match kind {
        FieldKind::String | FieldKind::Bytes => {
            let range = InternedRange::from_field_mask(f.value_mask());
            write_uvarint(buf, range.offset as u64);
        }
        FieldKind::I60 => {
            write_zigzag(buf, crate::index::i60::to_i64(f.value_mask()));
        }
        FieldKind::I64 | FieldKind::U64 => {
            // value_mask holds the data offset; replay reads the u64 from bucket.data.
            write_uvarint(buf, f.value_mask());
        }
        FieldKind::F60 | FieldKind::Seconds => {
            buf.extend_from_slice(&f60_to_f64(f.value_mask()).to_le_bytes());
        }
        FieldKind::UUID => {
            write_uvarint(buf, f.value_mask());
        }
        FieldKind::Bool => {
            buf.push((f.value_mask() & 1) as u8);
        }
        FieldKind::None => {}
        FieldKind::Timestamp => {
            // value_mask is i64 milliseconds clamped into 60 bits.
            buf.extend_from_slice(&(f.value_mask() as i64).to_le_bytes());
        }
        FieldKind::_Reserved2
        | FieldKind::_Reserved3
        | FieldKind::_Reserved4
        | FieldKind::_Reserved5
        | FieldKind::_Reserved6 => {
            panic!("Reserved FieldKind cannot appear in a stored field");
        }
    }
}

/// Begin a frame: write tag and reserve space for a uvarint length, returning
/// the position of the length placeholder. Pair with [`finish_frame`] after
/// the payload bytes have been written.
pub(crate) fn begin_frame(buf: &mut Vec<u8>, tag: FrameTag) -> usize {
    buf.push(tag as u8);
    let len_pos = buf.len();
    buf.push(0);
    len_pos
}

/// Finalize a frame opened by [`begin_frame`]. Patches the length placeholder
/// to the canonical uvarint encoding of the actual payload size. Most frames
/// are shorter than 128 bytes, so the one-byte reservation is already
/// canonical and no payload movement is needed.
pub(crate) fn finish_frame(buf: &mut Vec<u8>, len_pos: usize) {
    let payload_len = buf.len() - len_pos - 1;
    if payload_len < 0x80 {
        buf[len_pos] = payload_len as u8;
        return;
    }

    let mut tmp = [0u8; 10];
    let mut value = payload_len as u64;
    let mut written = 0usize;
    while value >= 0x80 {
        tmp[written] = (value as u8) | 0x80;
        value >>= 7;
        written += 1;
    }
    tmp[written] = value as u8;
    written += 1;

    let payload_start = len_pos + 1;
    let old_len = buf.len();
    let extra = written - 1;
    buf.resize(old_len + extra, 0);
    buf.copy_within(payload_start..old_len, payload_start + extra);
    buf[len_pos..len_pos + written].copy_from_slice(&tmp[..written]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_round_trip_through_begin_finish() {
        let mut buf = Vec::new();
        let pos = begin_frame(&mut buf, FrameTag::StringDecl);
        // 3 byte payload
        buf.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
        finish_frame(&mut buf, pos);
        // tag + uvarint(3)=1 + 3 payload bytes = 5 bytes total
        assert_eq!(buf.len(), 5);
        assert_eq!(buf[0], FrameTag::StringDecl as u8);
        assert_eq!(buf[1], 3); // uvarint 3
        assert_eq!(&buf[2..5], &[0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn large_frame_round_trip() {
        let mut buf = Vec::new();
        let pos = begin_frame(&mut buf, FrameTag::Record);
        let payload = vec![0x42u8; 200];
        buf.extend_from_slice(&payload);
        finish_frame(&mut buf, pos);
        // tag + uvarint(200) = 2 bytes + 200 payload = 203
        assert_eq!(buf.len(), 203);
        assert_eq!(buf[0], FrameTag::Record as u8);
        // uvarint(200) = [200|0x80, 1] = [0xC8, 0x01]
        assert_eq!(buf[1], 0xC8);
        assert_eq!(buf[2], 0x01);
        assert_eq!(&buf[3..], &payload[..]);
    }

    #[test]
    fn bitset_keys() {
        let mut enc = BucketLogStreamEncoder::new();
        assert!(!enc.key_declared(0));
        assert!(!enc.key_declared(127));
        assert!(!enc.key_declared(1000));
        enc.mark_key_declared(0);
        enc.mark_key_declared(127);
        enc.mark_key_declared(1000);
        assert!(enc.key_declared(0));
        assert!(enc.key_declared(127));
        assert!(enc.key_declared(1000));
        assert!(!enc.key_declared(64));
        assert!(!enc.key_declared(999));
        enc.reset();
        assert!(!enc.key_declared(0));
    }

    #[test]
    fn bitset_services() {
        let mut enc = BucketLogStreamEncoder::new();
        assert!(!enc.service_declared(0));
        assert!(!enc.service_declared(255));
        enc.mark_service_declared(0);
        enc.mark_service_declared(64);
        enc.mark_service_declared(255);
        assert!(enc.service_declared(0));
        assert!(enc.service_declared(64));
        assert!(enc.service_declared(255));
        assert!(!enc.service_declared(63));
    }
}

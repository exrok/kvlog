//! End-to-end tests for the bucket log stream: write entries through
//! `Index::write` with persistence enabled, replay the resulting bytes into
//! a fresh `Index`, and check the rehydrated index reports the same
//! observable state.

use kvlog::encoding::{munch_log_with_span, Encoder};
use kvlog::{Encode, LogLevel};
use std::io::{IoSlice, Write};

use crate::field_table::KeyID;
use crate::index::archetype::ServiceId;
use crate::index::Index;
use crate::persistence::encoder as stream_encoder;
use crate::persistence::{
    BucketSnapshotScratch, IndexConfig, LoadedBucket, PersistentInterners, SnapshotLoadError, ValidationMode,
};
use crate::query::QueryExpr;

fn persistent_index() -> Box<Index> {
    Box::new(Index::with_config(IndexConfig { persistence_enabled: true }))
}

fn write_into(index: &mut Index, svc: Option<ServiceId>, raw: &[u8]) {
    let mut data = raw;
    let (timestamp, level, span_info, fields) = munch_log_with_span(&mut data).expect("parse entry");
    index.write(timestamp, level, span_info, svc, fields).expect("write entry");
}

fn produce_entry_with(timestamp: u64, level: LogLevel, build: impl FnOnce(kvlog::encoding::FieldEncoder)) -> Vec<u8> {
    let mut encoder = Encoder::new();
    {
        let field_encoder = encoder.append(level, timestamp);
        build(field_encoder);
    }
    encoder.bytes().to_vec()
}

fn drain_active(index: &mut Index) -> Vec<u8> {
    // Complete the bucket so the active buffer (header + body + footer) is
    // pushed into pending_finalized.
    index.complete_bucket();
    let drained = index.take_finalized_log_buffer().expect("finalized buffer");
    assert!(drained.closed);
    drained.bytes
}

fn native_snapshot_bytes(index: &Index) -> (Vec<u8>, PersistentInterners) {
    let globals = PersistentInterners::capture_from_reader(index.reader());
    let bytes = {
        let bucket = index.reader().newest_bucket().expect("bucket");
        let mut scratch = BucketSnapshotScratch::default();
        let snapshot = bucket.snapshot_slices(&mut scratch, &globals).expect("snapshot");
        let mut bytes = Vec::with_capacity(snapshot.byte_len);
        for part in snapshot.iter() {
            bytes.extend_from_slice(part);
        }
        assert_eq!(bytes.len(), snapshot.byte_len);
        assert_eq!(u32::from_ne_bytes(bytes[16..20].try_into().unwrap()), 8);
        assert_eq!(u64::from_ne_bytes(bytes[112..120].try_into().unwrap()), 0);
        assert_eq!(u64::from_ne_bytes(bytes[120..128].try_into().unwrap()), 0);

        let io_slices: Vec<IoSlice<'_>> = snapshot.iter().map(IoSlice::new).collect();
        let mut vectored = Vec::with_capacity(snapshot.byte_len);
        let written = vectored.write_vectored(&io_slices).expect("vectored write");
        assert_eq!(written, snapshot.byte_len);
        assert_eq!(vectored, bytes);
        bytes
    };
    (bytes, globals)
}

fn borrow_aligned<'a>(bytes: &[u8], backing: &'a mut Vec<u8>, globals: &'a PersistentInterners) -> LoadedBucket<'a> {
    backing.clear();
    backing.resize(bytes.len() + 63, 0);
    let start = (64 - ((backing.as_ptr() as usize) & 63)) & 63;
    backing[start..start + bytes.len()].copy_from_slice(bytes);
    LoadedBucket::borrow(&backing[start..start + bytes.len()], globals, ValidationMode::Full).expect("borrow snapshot")
}

#[test]
fn single_record_round_trip() {
    let mut idx = persistent_index();
    let svc = ServiceId::intern("svc-single");
    let raw = produce_entry_with(1_000_000, LogLevel::Info, |mut fe| {
        "hello".encode_log_value_into(fe.key("msg"));
        "module".encode_log_value_into(fe.key("target"));
        42i64.encode_log_value_into(fe.key("count"));
    });
    write_into(&mut idx, Some(svc), &raw);
    let bytes = drain_active(&mut idx);

    let mut replay = Box::new(Index::new());
    let n = replay.ingest(&bytes).expect("ingest");
    assert_eq!(n, 1);
}

#[test]
fn multiple_records_share_msg_and_target() {
    let mut idx = persistent_index();
    let svc = ServiceId::intern("svc-shared");
    for ts in 1u64..=10 {
        let raw = produce_entry_with(ts * 1_000, LogLevel::Info, |mut fe| {
            "shared message".encode_log_value_into(fe.key("msg"));
            "module-a".encode_log_value_into(fe.key("target"));
            (ts as i64).encode_log_value_into(fe.key("count"));
        });
        write_into(&mut idx, Some(svc), &raw);
    }
    let bytes = drain_active(&mut idx);
    let mut replay = Box::new(Index::new());
    let n = replay.ingest(&bytes).expect("ingest");
    assert_eq!(n, 10);
}

#[test]
fn dynamic_keys_round_trip() {
    let mut idx = persistent_index();
    let svc = ServiceId::intern("svc-dyn");
    let raw = produce_entry_with(2_000, LogLevel::Warn, |mut fe| {
        "with dyn".encode_log_value_into(fe.key("msg"));
        "tg".encode_log_value_into(fe.key("target"));
        "value-a".encode_log_value_into(fe.key("custom_a"));
        7u64.encode_log_value_into(fe.key("custom_b"));
    });
    write_into(&mut idx, Some(svc), &raw);
    let bytes = drain_active(&mut idx);
    let mut replay = Box::new(Index::new());
    assert_eq!(replay.ingest(&bytes).expect("ingest"), 1);
}

#[test]
fn all_field_kinds_round_trip() {
    let mut idx = persistent_index();
    let svc = ServiceId::intern("svc-kinds");
    let uuid = uuid::Uuid::from_bytes([1u8; 16]);
    let raw = produce_entry_with(3_000, LogLevel::Error, |mut fe| {
        "kinds".encode_log_value_into(fe.key("msg"));
        "tg".encode_log_value_into(fe.key("target"));
        "string-val".encode_log_value_into(fe.key("s"));
        42i32.encode_log_value_into(fe.key("i32"));
        (-99i64).encode_log_value_into(fe.key("i64s"));
        7u32.encode_log_value_into(fe.key("u32"));
        i64::MAX.encode_log_value_into(fe.key("i64big"));
        u64::MAX.encode_log_value_into(fe.key("u64big"));
        1.5f32.encode_log_value_into(fe.key("f32"));
        2.75f64.encode_log_value_into(fe.key("f64"));
        uuid.encode_log_value_into(fe.key("u"));
        true.encode_log_value_into(fe.key("b1"));
        false.encode_log_value_into(fe.key("b2"));
        let none: Option<u64> = None;
        none.encode_log_value_into(fe.key("nothing"));
    });
    write_into(&mut idx, Some(svc), &raw);
    let bytes = drain_active(&mut idx);
    let mut replay = Box::new(Index::new());
    let n = replay.ingest(&bytes).expect("ingest");
    assert_eq!(n, 1);
}

#[test]
fn span_start_current_end_round_trip() {
    use kvlog::SpanID;
    use std::num::NonZeroU64;
    let mut idx = persistent_index();
    let svc = ServiceId::intern("svc-span");
    let span = SpanID::from(NonZeroU64::new(0xABCD).unwrap());
    let parent = SpanID::from(NonZeroU64::new(0x1234).unwrap());

    // Start with parent.
    let mut e = Encoder::new();
    {
        let mut fe = e.append(LogLevel::Info, 100);
        "start".encode_log_value_into(fe.key("msg"));
        "tg".encode_log_value_into(fe.key("target"));
        fe.start_span_with_parent(span, Some(parent));
    }
    let raw_start = e.bytes().to_vec();
    write_into(&mut idx, Some(svc), &raw_start);

    // Current.
    let mut e = Encoder::new();
    {
        let mut fe = e.append(LogLevel::Info, 200);
        "current".encode_log_value_into(fe.key("msg"));
        "tg".encode_log_value_into(fe.key("target"));
        fe.apply_span(span);
    }
    let raw_current = e.bytes().to_vec();
    write_into(&mut idx, Some(svc), &raw_current);

    // End.
    let mut e = Encoder::new();
    {
        let mut fe = e.append(LogLevel::Info, 300);
        "end".encode_log_value_into(fe.key("msg"));
        "tg".encode_log_value_into(fe.key("target"));
        fe.end_span(span);
    }
    let raw_end = e.bytes().to_vec();
    write_into(&mut idx, Some(svc), &raw_end);

    let bytes = drain_active(&mut idx);
    let mut replay = Box::new(Index::new());
    assert_eq!(replay.ingest(&bytes).expect("ingest"), 3);
}

#[test]
fn header_and_footer_present() {
    let mut idx = persistent_index();
    let svc = ServiceId::intern("svc-header");
    let raw = produce_entry_with(1, LogLevel::Info, |mut fe| {
        "h".encode_log_value_into(fe.key("msg"));
        "tg".encode_log_value_into(fe.key("target"));
    });
    write_into(&mut idx, Some(svc), &raw);
    let bytes = drain_active(&mut idx);
    assert_eq!(&bytes[..4], b"KVBL");
    assert!(bytes.len() >= 16 + 24, "must contain header + footer");
    assert_eq!(&bytes[bytes.len() - 24..bytes.len() - 20], b"KVBE");
}

#[test]
fn partial_buffers_concatenated() {
    let mut idx = persistent_index();
    let svc = ServiceId::intern("svc-partial");

    let raw1 = produce_entry_with(1, LogLevel::Info, |mut fe| {
        "p1".encode_log_value_into(fe.key("msg"));
        "tg".encode_log_value_into(fe.key("target"));
    });
    write_into(&mut idx, Some(svc), &raw1);

    // Mid-bucket swap.
    let drained1 = idx.swap_log_buffer(Vec::new()).unwrap();
    assert!(!drained1.closed);

    let raw2 = produce_entry_with(2, LogLevel::Info, |mut fe| {
        "p2".encode_log_value_into(fe.key("msg"));
        "tg".encode_log_value_into(fe.key("target"));
    });
    write_into(&mut idx, Some(svc), &raw2);

    let final_bytes = drain_active(&mut idx);

    let mut combined = Vec::new();
    combined.extend_from_slice(&drained1.bytes);
    combined.extend_from_slice(&final_bytes);

    let mut replay = Box::new(Index::new());
    assert_eq!(replay.ingest(&combined).expect("ingest concat"), 2);
}

#[test]
fn empty_stream_with_only_header_and_footer() {
    let mut buf = Vec::new();
    stream_encoder::write_header(&mut buf, 0);
    stream_encoder::write_footer(&mut buf, 123, 0);
    let mut replay = Box::new(Index::new());
    assert_eq!(replay.ingest(&buf).expect("ingest empty"), 0);
}

#[test]
fn ingest_rejects_truncated_header() {
    let mut replay = Box::new(Index::new());
    let err = replay.ingest(&[0u8; 4]).expect_err("must fail");
    assert!(matches!(err, crate::persistence::format::ReadError::TruncatedHeader));
}

#[test]
fn ingest_rejects_bad_magic() {
    let mut bad = vec![0u8; 16];
    bad[..4].copy_from_slice(b"WRNG");
    let mut replay = Box::new(Index::new());
    let err = replay.ingest(&bad).expect_err("must fail");
    assert!(matches!(err, crate::persistence::format::ReadError::InvalidHeaderMagic));
}

#[test]
fn native_bucket_snapshot_round_trip_and_query() {
    let mut idx = Index::new();
    let svc = ServiceId::intern("snapshot-svc");
    let uuid = uuid::Uuid::from_bytes([9u8; 16]);

    let raw1 = produce_entry_with(10_000, LogLevel::Info, |mut fe| {
        "native one".encode_log_value_into(fe.key("msg"));
        "snapshot-target".encode_log_value_into(fe.key("target"));
        "value-a".encode_log_value_into(fe.key("custom_a"));
        123i64.encode_log_value_into(fe.key("count"));
        uuid.encode_log_value_into(fe.key("uuid"));
    });
    write_into(&mut idx, Some(svc), &raw1);

    let raw2 = produce_entry_with(20_000, LogLevel::Warn, |mut fe| {
        "native two".encode_log_value_into(fe.key("msg"));
        "snapshot-target".encode_log_value_into(fe.key("target"));
        "value-b".encode_log_value_into(fe.key("custom_a"));
        false.encode_log_value_into(fe.key("flag"));
    });
    write_into(&mut idx, Some(svc), &raw2);

    let (bytes, globals) = native_snapshot_bytes(&idx);
    let mut encoded_globals = Vec::new();
    globals.encode(&mut encoded_globals);
    let decoded_globals = PersistentInterners::decode(&encoded_globals).expect("decode globals");
    assert_eq!(decoded_globals.fingerprint(), globals.fingerprint());

    let mut backing = Vec::new();
    let loaded = borrow_aligned(&bytes, &mut backing, &decoded_globals);
    let bucket = loaded.read().expect("loaded bucket");
    assert_eq!(bucket.entry_count(), 2);

    let messages: Vec<Vec<u8>> = bucket.entries().map(|entry| entry.message().to_vec()).collect();
    assert_eq!(messages, vec![b"native one".to_vec(), b"native two".to_vec()]);
    assert_eq!(loaded.target_mapper().get(bucket.entries().next().unwrap().target_id()), Some("snapshot-target"));
    {
        let maps = bucket.maps();
        let custom_key = KeyID::try_from_str("custom_a").expect("custom key");
        assert!(maps.field_text(&bucket, custom_key, b"value-a").is_some());
        let uuid_key = KeyID::try_from_str("uuid").expect("uuid key");
        assert!(maps.field_uuid(&bucket, uuid_key, uuid).is_some());
    }

    let query = QueryExpr::new("custom_a = \"value-a\"").expect("query");
    let mut hits = 0;
    loaded.query(&query, |_| {
        hits += 1;
        true
    });
    assert_eq!(hits, 1);

    let uuid_query = QueryExpr::new(&format!("uuid = \"{uuid}\"")).expect("uuid query");
    let mut uuid_hits = 0;
    loaded.query(&uuid_query, |_| {
        uuid_hits += 1;
        true
    });
    assert_eq!(uuid_hits, 1);
}

#[test]
fn native_bucket_snapshot_rejects_unaligned_buffer() {
    let mut idx = Index::new();
    let raw = produce_entry_with(1, LogLevel::Info, |mut fe| {
        "unaligned".encode_log_value_into(fe.key("msg"));
        "target".encode_log_value_into(fe.key("target"));
    });
    write_into(&mut idx, None, &raw);
    let (bytes, globals) = native_snapshot_bytes(&idx);

    let mut backing = vec![0u8; bytes.len() + 64];
    let aligned = (64 - ((backing.as_ptr() as usize) & 63)) & 63;
    let unaligned = if aligned == 0 { 1 } else { aligned - 1 };
    backing[unaligned..unaligned + bytes.len()].copy_from_slice(&bytes);
    let err = match LoadedBucket::borrow(&backing[unaligned..unaligned + bytes.len()], &globals, ValidationMode::Basic)
    {
        Ok(_) => panic!("must reject unaligned input"),
        Err(err) => err,
    };
    assert!(matches!(err, SnapshotLoadError::BufferNotAligned { .. }));
}

#[test]
fn native_bucket_snapshot_rejects_wrong_globals() {
    let mut idx = Index::new();
    let raw = produce_entry_with(1, LogLevel::Info, |mut fe| {
        "globals".encode_log_value_into(fe.key("msg"));
        "target-a".encode_log_value_into(fe.key("target"));
    });
    write_into(&mut idx, None, &raw);
    let (bytes, _globals) = native_snapshot_bytes(&idx);
    let wrong_globals = PersistentInterners::from_names(&[], &[], &["different-target"]).expect("wrong globals");

    let mut backing = vec![0u8; bytes.len() + 63];
    let start = (64 - ((backing.as_ptr() as usize) & 63)) & 63;
    backing[start..start + bytes.len()].copy_from_slice(&bytes);
    let err = match LoadedBucket::borrow(&backing[start..start + bytes.len()], &wrong_globals, ValidationMode::Basic) {
        Ok(_) => panic!("must reject mismatched globals"),
        Err(err) => err,
    };
    assert!(matches!(err, SnapshotLoadError::GlobalsMismatch));
}

#[test]
fn native_bucket_snapshot_rejects_bad_archetype_index() {
    let mut idx = Index::new();
    let raw = produce_entry_with(1, LogLevel::Info, |mut fe| {
        "offsets".encode_log_value_into(fe.key("msg"));
        "target".encode_log_value_into(fe.key("target"));
        "value".encode_log_value_into(fe.key("field"));
    });
    write_into(&mut idx, None, &raw);
    let (mut bytes, globals) = native_snapshot_bytes(&idx);

    const HEADER_LEN: usize = 136;
    const SECTION_LEN: usize = 24;
    const ARCHETYPE_INDEX_SECTION_INDEX: usize = 6;
    let arch_entry = HEADER_LEN + ARCHETYPE_INDEX_SECTION_INDEX * SECTION_LEN;
    let arch_section_start =
        u64::from_ne_bytes(bytes[arch_entry + 8..arch_entry + 16].try_into().unwrap()) as usize;
    // Point the first log at an archetype id far past the archetype count so
    // offset reconstruction must reject it.
    bytes[arch_section_start] = 0xFF;
    bytes[arch_section_start + 1] = 0xFF;

    let mut backing = vec![0u8; bytes.len() + 63];
    let start = (64 - ((backing.as_ptr() as usize) & 63)) & 63;
    backing[start..start + bytes.len()].copy_from_slice(&bytes);
    let err = match LoadedBucket::borrow(&backing[start..start + bytes.len()], &globals, ValidationMode::Basic) {
        Ok(_) => panic!("must reject invalid archetype index"),
        Err(err) => err,
    };
    assert!(matches!(err, SnapshotLoadError::InvalidReference("archetype id")));
}

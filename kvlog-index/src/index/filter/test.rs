use kvlog::encoding::LogFields;

use crate::index::test::{test_index, TestIndexWriter};
use crate::log;
use kvlog::{encoding::FieldBuffer, Encode, LogLevel, SpanInfo};
#[test]
fn forward_query_continuations() {
    let mut index = test_index();
    let reader = index.reader().clone();
    let mut forward_query = reader.forward_query(&[]);
    assert!(forward_query.next().is_none());

    let mut writer = TestIndexWriter::new(&mut index);
    crate::log!(writer; msg="Hello");

    assert!(forward_query.next().is_some());
    assert!(forward_query.next().is_none());
    crate::log!(writer; msg="ABC");
    assert_eq!(forward_query.next().unwrap().into_iter().next().unwrap().message(), b"ABC");
    crate::log!(writer; msg="Nice");

    assert!(forward_query.next().is_some());
    assert!(forward_query.next().is_none());
}

#[test]
fn bidirectional_query_from_starts_at_anchor_offset() {
    let mut index = test_index();
    let _first = index.write(1, LogLevel::Info, SpanInfo::None, None, LogFields::empty()).unwrap();
    let anchor = index.write(2, LogLevel::Info, SpanInfo::None, None, LogFields::empty()).unwrap();
    let _third = index.write(3, LogLevel::Info, SpanInfo::None, None, LogFields::empty()).unwrap();

    let reader = index.reader().clone();
    let (mut reverse, mut forward) = reader.bidirectional_query_from(&[], anchor).unwrap();

    let reverse_entries = reverse.next().unwrap();
    let reverse_timestamps = reverse_entries
        .into_iter()
        .map(|entry| entry.timestamp())
        .collect::<Vec<_>>();
    assert_eq!(reverse_timestamps, vec![2, 1]);

    let forward_entries = forward.next().unwrap();
    let forward_timestamps = forward_entries
        .into_iter()
        .map(|entry| entry.timestamp())
        .collect::<Vec<_>>();
    assert_eq!(forward_timestamps, vec![3]);
}

#[test]
fn forward_from_oldest_includes_oldest_live_bucket_after_wraparound() {
    let mut index = test_index();
    for timestamp in 1..=6 {
        index.write(timestamp, LogLevel::Info, SpanInfo::None, None, LogFields::empty()).unwrap();
        if timestamp != 6 {
            index.complete_bucket();
        }
    }

    let reader = index.reader().clone();
    let mut walker = reader.forward_query_from_oldest(&[]);
    let mut timestamps = Vec::new();
    while let Some(entries) = walker.next() {
        for entry in &entries {
            timestamps.push(entry.timestamp());
        }
    }

    assert_eq!(timestamps, vec![3, 4, 5, 6]);
}

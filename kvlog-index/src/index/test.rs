use std::{
    mem::ManuallyDrop,
    sync::{atomic::AtomicUsize, mpsc, Condvar},
    thread,
    time::{Duration, Instant},
};

use kvlog::{encoding::FieldBuffer, Encode};
use test::filter::Query;

use crate::index::filter::LevelFilter;

use self::filter::QueryFilter;

use super::*;

struct ReadyPool {
    pool: Vec<Box<Index>>,
    created: usize,
}
struct IndexPool {
    max: usize,
    cond: Condvar,
    ready: Mutex<ReadyPool>,
}

impl IndexPool {
    fn push(&self, index: Box<Index>) {
        {
            self.ready.lock().unwrap().pool.push(index);
        }
        self.cond.notify_one();
    }
    fn retire(&self) {
        {
            let mut ready = self.ready.lock().unwrap();
            ready.created -= 1;
        }
        self.cond.notify_one();
    }
    fn get(&self) -> PoolGuard {
        let mut ready = self.ready.lock().unwrap();
        if ready.pool.is_empty() && ready.created < self.max {
            ready.created += 1;
            drop(ready);
            return PoolGuard { index: ManuallyDrop::new(Box::new(Index::new())) };
        }
        while ready.pool.is_empty() {
            ready = self.cond.wait(ready).unwrap();
        }
        let mut index = ready.pool.pop().unwrap();
        // probably not good, hopefully previous tests are still touching it
        unsafe {
            index.clear_unchecked();
        }
        PoolGuard { index: ManuallyDrop::new(index) }
    }
}

static GLOBAL_TEST_POOL: IndexPool =
    IndexPool { max: 4, cond: Condvar::new(), ready: Mutex::new(ReadyPool { pool: Vec::new(), created: 0 }) };

pub struct PoolGuard {
    index: ManuallyDrop<Box<Index>>,
}

impl Drop for PoolGuard {
    fn drop(&mut self) {
        let index = unsafe { ManuallyDrop::take(&mut self.index) };
        let reusable = std::sync::Arc::strong_count(index.reader()) == 1
            && index.reader.buckets.iter().all(|bucket| bucket.ref_count.load(Ordering::Acquire) == 1);
        if reusable {
            GLOBAL_TEST_POOL.push(index)
        } else {
            GLOBAL_TEST_POOL.retire();
        }
    }
}
impl std::ops::Deref for PoolGuard {
    type Target = Index;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.index
    }
}
impl std::ops::DerefMut for PoolGuard {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.index
    }
}

pub(crate) fn test_index() -> PoolGuard {
    GLOBAL_TEST_POOL.get()
}

// merge to sorted slices into a large sorted slice quickly
fn merge_sorted(mut a: &[u64], mut b: &[u64]) -> Vec<u64> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let mut a = a.iter().copied();
    let mut b = b.iter().copied();
    let mut a_next = a.next();
    let mut b_next = b.next();
    loop {
        match (a_next, b_next) {
            (Some(a_val), Some(b_val)) => {
                if a_val < b_val {
                    result.push(a_val);
                    a_next = a.next();
                } else {
                    result.push(b_val);
                    b_next = b.next();
                }
            }
            (Some(a_val), None) => {
                result.push(a_val);
                result.extend(a);
                break;
            }
            (None, Some(b_val)) => {
                result.push(b_val);
                result.extend(b);
                break;
            }
            (None, None) => break,
        }
    }
    result
}

fn in_range(sorted: &[u64], min: u64, max: u64) -> &[u64] {
    if sorted.is_empty() {
        return &[];
    }
    let start = sorted.partition_point(|x| (*x < min));
    let end = sorted[start..].partition_point(|x| (*x <= max));
    &sorted[start..end + start]
}

#[test]
fn test_in_range() {
    assert_eq!(in_range(&[1, 2, 3], 20, 20), &[]);
    assert_eq!(in_range(&[1, 2, 3], 0, 0), &[]);
    assert_eq!(in_range(&[1, 2, 3], 1, 3), &[1, 2, 3]);
    assert_eq!(in_range(&[1, 2, 3], 1, 2), &[1, 2]);
    assert_eq!(in_range(&[1, 2, 3], 2, 3), &[2, 3]);
    assert_eq!(in_range(&[10, 20, 30, 40], 11, 29), &[20]);
}

#[test]
fn clear_unchecked_resets_reader_generation() {
    let mut index = test_index();
    index.write(1, LogLevel::Info, SpanInfo::None, None, LogFields::empty()).unwrap();
    index.complete_bucket();
    index.write(2, LogLevel::Info, SpanInfo::None, None, LogFields::empty()).unwrap();

    unsafe {
        index.clear_unchecked();
    }

    let weak = index.write(3, LogLevel::Info, SpanInfo::None, None, LogFields::empty()).unwrap();
    let reader = index.reader().clone();
    assert_eq!(reader.lastest_generation(), 0);
    let bucket = reader.newest_bucket().unwrap();
    let entry = bucket.upgrade(weak).unwrap();
    assert_eq!(entry.timestamp(), 3);
}

#[test]
fn weak_entry_upgrade_rejects_index_past_bucket_len() {
    let mut index = test_index();
    let weak = index.write(1, LogLevel::Info, SpanInfo::None, None, LogFields::empty()).unwrap();
    let stale = WeakLogEntry::new(weak.bucket_generation(), weak.index() + 1);
    let reader = index.reader().clone();

    let guard = reader.generation_guard();
    assert!(!unsafe { guard.is_alive(stale) });
    assert!(unsafe { guard.upgrade(stale) }.is_none());

    let bucket = reader.newest_bucket().unwrap();
    assert!(!bucket.is_alive(stale));
    assert!(bucket.upgrade(stale).is_none());
}

#[test]
fn level_and_time_filters() {
    let mut index = test_index();
    let mut rng = oorandom::Rand32::new(0xdeafbeaf);
    let mut level_buckets: [Vec<u64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut counter = 234..;
    for _ in 0..3 {
        for i in counter.by_ref().take(100_000) {
            let lvl = LogLevel::from_u8((rng.rand_u32() & 0b11) as u8).unwrap();
            let ts = (i * 1024) | ((rng.rand_u32() as u64) & 0x3ff);
            level_buckets[lvl as usize].push(ts);
            index.write(ts, lvl, SpanInfo::None, None, LogFields::empty()).unwrap();
        }
        index.complete_bucket();
    }
    let time_query = |min: u64, max: u64, level_mask: u8| -> Vec<u64> {
        let mut output = Vec::new();
        for i in 0..4 {
            if (1 << i) & level_mask != 0 {
                output.extend_from_slice(in_range(&level_buckets[i as usize], min, max));
            }
        }
        output.sort_unstable();
        output
    };
    let min_time = 234 * 1024;
    let max_time = counter.next().unwrap() * 1024;
    let time_rt = |r: f32| ((max_time - min_time) as f32 * r) as u64 + min_time;

    for (level, expected) in level_buckets.iter().enumerate() {
        let lvl = LogLevel::from_u8(level as u8).unwrap();
        let query = &[LevelFilter::empty().with(lvl).into()];
        for (entry, expected_timestamp) in index.reverse_query(query).zip(expected.iter().rev()) {
            if entry.timestamp() != *expected_timestamp {
                println!("{:p}", entry.bucket());
                println!("{}", entry.bucket().len.load(Ordering::Acquire));
                println!("{}", entry.index);
                println!("{:?}", entry.span_info());
                println!("{:?}", lvl);
                assert_eq!(entry.timestamp(), *expected_timestamp);
            }
        }
    }
    // todo test exactly values
    let tf = TimeFilter { min_utc_ns: time_rt(0.3), max_utc_ns: time_rt(0.5) };
    let expected = time_query(tf.min_utc_ns, tf.max_utc_ns, 0b101);
    assert!(expected.len() > 0);
    let query = &[LevelFilter { mask: 0b101 }.into(), tf.into()];
    let mut found = 0;
    for (entry, expected_timestamp) in index.reverse_query(query).zip(expected.iter().rev()) {
        found += 1;
        if entry.timestamp() != *expected_timestamp {
            println!("{}", entry.bucket().len.load(Ordering::Acquire));
            println!("{}", entry.index);
            println!("{:?}", entry.span_info());
            println!("FOUND: {}", found);
            assert_eq!(entry.timestamp(), *expected_timestamp);
        }
    }
    assert_eq!(found, expected.len());

    let mut found = 0;

    for (entry, expected_timestamp) in index.forward_query(query).zip(expected.iter()) {
        found += 1;
        if entry.timestamp() != *expected_timestamp {
            println!("{}", entry.bucket().len.load(Ordering::Acquire));
            println!("{}", entry.index);
            println!("{:?}", entry.span_info());
            println!("FOUND: {}", found);
            assert_eq!(entry.timestamp(), *expected_timestamp);
        }
    }
    assert_eq!(found, expected.len());
}

#[track_caller]
fn assert_eq_logs<'a, 'b>(
    input: impl IntoIterator<Item = LogEntry<'a>>,
    expected: impl IntoIterator<Item = &'b WeakLogEntry>,
) {
    let mut input_iter = input.into_iter();
    let mut expected_iter = expected.into_iter();
    for (input, expected) in input_iter.by_ref().zip(expected_iter.by_ref()) {
        assert_eq!(input.weak(), *expected, "Entry Mismatch");
    }
    let results_remaining = input_iter.take(512).count();
    assert!(results_remaining == 0, "Found {} more entries then expected", results_remaining);
    let expected_remaining = expected_iter.count();
    assert!(expected_remaining == 0, "Expected {} more entries", expected_remaining,);
}

pub struct TestIndexWriter<'a> {
    pub(crate) time: u64,
    pub(crate) buf: FieldBuffer,
    pub(crate) index: &'a mut Index,
}
impl TestIndexWriter<'_> {
    pub fn new(index: &'_ mut Index) -> TestIndexWriter<'_> {
        TestIndexWriter { time: 1, buf: FieldBuffer::default(), index }
    }
}

pub struct LogOptions {
    pub level: LogLevel,
    pub span: SpanInfo,
    pub service: Option<ServiceId>,
}
impl Default for LogOptions {
    fn default() -> Self {
        Self { level: LogLevel::Info, span: SpanInfo::None, service: None }
    }
}

#[macro_export]
macro_rules! log {
    ($buffer:ident ; $({$($meta:tt)*})? $($key:ident = $value:expr),*) => {{
        let time = $buffer.time;
        $buffer.time += 1;
        let fields = {
            let mut encoder = $buffer.buf.encoder();
            $( ($value).encode_log_value_into(encoder.key(stringify!($key)));)*
            encoder.fields()
        };
        let opt = $crate::index::test::LogOptions{$($($meta)* ,)? ..Default::default()};
        $buffer.index.write(time, opt.level, opt.span, opt.service, fields).unwrap()
    }};
}

#[test]
fn field_query() {
    let mut index = test_index();
    let mut rng = oorandom::Rand32::new(0xdeafbeaf);
    let mut writer = TestIndexWriter::new(&mut index);
    let w0 = log!(writer; msg="nice", cats="nice");
    let w1 = log!(writer; msg="nice hello", cats="what");
    let w2 = log!(writer; msg="todo, nice hello", dogs="what");

    let query = Query::expr(stringify!(msg.contains("hello"))).unwrap();
    assert_eq_logs(index.reverse_query(&query.filters), &[w2, w1]);
    // assert_eq_logs(index.forward_query(&query.filters), &[w1, w2]);

    let query = Query::expr(stringify!(cats = "what")).unwrap();
    assert_eq_logs(index.reverse_query(&query.filters), &[w1]);

    let query = Query::expr(stringify!(cats.exists())).unwrap();
    assert_eq_logs(index.reverse_query(&query.filters), &[w1, w0]);
}

#[test]
fn int_query() {
    let mut index = test_index();
    let mut rng = oorandom::Rand32::new(0xdeafbeaf);
    let mut writer = TestIndexWriter::new(&mut index);
    let w0 = log!(writer; msg="nice", count=3);
    let w1 = log!(writer; msg="nice hello", count=1);
    let w2 = log!(writer; msg="todo, nice hello", dogs="what");

    let query = Query::expr(stringify!(count: int = 1)).unwrap();

    assert_eq_logs(index.reverse_query(&query.filters), &[w1]);
}

#[test]
fn bucket_rollover_wakes_when_last_reader_drops() {
    let mut index = test_index();
    index.write(1, LogLevel::Info, SpanInfo::None, None, LogFields::empty()).unwrap();
    index.complete_bucket();
    index.write(2, LogLevel::Info, SpanInfo::None, None, LogFields::empty()).unwrap();

    let reader = index.reader().clone();
    let guard = reader.newest_bucket().unwrap();
    assert_eq!(guard.bucket.generation.load(Ordering::Acquire), 1);

    let target_bucket = &reader.buckets[1];
    let (done_tx, done_rx) = mpsc::channel();
    thread::scope(|scope| {
        let index = &mut index;
        let worker = scope.spawn(move || {
            for _ in 0..4 {
                index.complete_bucket();
            }
            done_tx.send(()).unwrap();
        });

        let start = Instant::now();
        while target_bucket.ref_count.load(Ordering::Acquire) != 1 {
            assert!(start.elapsed() < Duration::from_secs(2), "rollover did not wait on the active reader");
            thread::yield_now();
        }

        drop(guard);
        if done_rx.recv_timeout(Duration::from_secs(1)).is_err() {
            atomic_wait::wake_one(&target_bucket.ref_count);
            worker.join().unwrap();
            panic!("rollover was not woken by dropping the last reader");
        }

        worker.join().unwrap();
    });
}

#[test]
fn rollover_reinterns_message_in_new_bucket() {
    let mut index = test_index();
    let mut writer = TestIndexWriter::new(&mut index);

    let _ = log!(writer; msg="repeat");
    for _ in 0..4 {
        writer.index.complete_bucket();
    }
    let second = log!(writer; msg="repeat", rollover_clobber = "xxxxxx");

    let reader = writer.index.reader().clone();
    let bucket = reader.newest_bucket().unwrap();
    let entry = bucket.upgrade(second).unwrap();
    assert_eq!(entry.message(), b"repeat");
}

fn span_id_for(value: u64) -> kvlog::SpanID {
    kvlog::SpanID::from(std::num::NonZeroU64::new(value).unwrap())
}

#[test]
fn ancestor_chain_walks_within_bucket() {
    let mut index = test_index();
    let mut writer = TestIndexWriter::new(&mut index);
    let root = span_id_for(0xA0);
    let mid = span_id_for(0xA1);
    let leaf = span_id_for(0xA2);

    log!(writer; { span: SpanInfo::Start { span: root, parent: None } } msg = "root");
    log!(writer; { span: SpanInfo::Start { span: mid, parent: Some(root) } } msg = "mid");
    let leaf_entry = log!(writer;
        { span: SpanInfo::Start { span: leaf, parent: Some(mid) } } msg = "leaf");

    let reader = writer.index.reader().clone();
    let bucket = reader.newest_bucket().unwrap();
    let entry = bucket.upgrade(leaf_entry).unwrap();
    let chain = entry.ancestor_chain(super::MAX_CROSS_BUCKET_PARENT_DEPTH);
    assert_eq!(chain.as_slice(), &[root, mid]);
}

#[test]
fn ancestor_chain_replicates_across_bucket_rotation() {
    let mut index = test_index();
    let mut writer = TestIndexWriter::new(&mut index);
    let root = span_id_for(0xB0);
    let mid = span_id_for(0xB1);
    let leaf = span_id_for(0xB2);

    log!(writer; { span: SpanInfo::Start { span: root, parent: None } } msg = "root");
    log!(writer; { span: SpanInfo::Start { span: mid, parent: Some(root) } } msg = "mid");
    log!(writer; { span: SpanInfo::Start { span: leaf, parent: Some(mid) } } msg = "leaf-bucket0");
    writer.index.complete_bucket();

    // New bucket: a Current record on `leaf` should replicate the chain.
    let next_entry = log!(writer; { span: SpanInfo::Current { span: leaf } } msg = "leaf-bucket1");

    let reader = writer.index.reader().clone();
    let bucket = reader.newest_bucket().unwrap();
    let entry = bucket.upgrade(next_entry).unwrap();
    let chain = entry.ancestor_chain(super::MAX_CROSS_BUCKET_PARENT_DEPTH);
    assert_eq!(chain.as_slice(), &[root, mid]);

    // Replicated ancestors should be flagged.
    let mid_range = entry.span_range().unwrap();
    let parent_idx = mid_range.parent_index;
    assert_ne!(parent_idx, super::SPAN_PARENT_INDEX_NONE);
    let parent_sr = unsafe { &*entry.bucket().span_data.as_ptr().add(parent_idx as usize) };
    assert!(parent_sr.from_previous_bucket());
}

#[test]
fn ancestor_chain_truncates_at_replication_depth() {
    let mut index = test_index();
    let mut writer = TestIndexWriter::new(&mut index);
    // Build a deeper chain than MAX_CROSS_BUCKET_PARENT_DEPTH allows.
    let depth = super::MAX_CROSS_BUCKET_PARENT_DEPTH + 4;
    let spans: Vec<_> = (0..depth).map(|i| span_id_for(0x100 + i as u64)).collect();
    log!(writer; { span: SpanInfo::Start { span: spans[0], parent: None } } msg = "root");
    for i in 1..depth {
        log!(writer; { span: SpanInfo::Start { span: spans[i], parent: Some(spans[i - 1]) } } msg = "n");
    }
    writer.index.complete_bucket();
    let leaf_entry = log!(writer; { span: SpanInfo::Current { span: spans[depth - 1] } } msg = "leaf");

    let reader = writer.index.reader().clone();
    let bucket = reader.newest_bucket().unwrap();
    let entry = bucket.upgrade(leaf_entry).unwrap();
    let chain = entry.ancestor_chain(64);
    // Replication caps at MAX_CROSS_BUCKET_PARENT_DEPTH levels.
    assert_eq!(chain.len(), super::MAX_CROSS_BUCKET_PARENT_DEPTH);
}

#[test]
fn cross_bucket_stub_keeps_flag_after_record_lands() {
    // A span replicated as a cross-bucket parent stub keeps the
    // FROM_PREVIOUS_BUCKET flag even after a real record for that span
    // lands in the current bucket. Promotion only updates the atomic
    // mask fields; non-atomic fields stay as initialized.
    let mut index = test_index();
    let mut writer = TestIndexWriter::new(&mut index);
    let parent = span_id_for(0xD0);
    let child = span_id_for(0xD1);

    log!(writer; { span: SpanInfo::Start { span: parent, parent: None } } msg = "p");
    log!(writer; { span: SpanInfo::Start { span: child, parent: Some(parent) } } msg = "c");
    writer.index.complete_bucket();

    // New bucket: a record on `child` first replicates `parent` as a
    // stub. Then a record on `parent` itself promotes that stub.
    log!(writer; { span: SpanInfo::Current { span: child } } msg = "c2");
    let parent_record = log!(writer; { span: SpanInfo::Current { span: parent } } msg = "p2");

    let reader = writer.index.reader().clone();
    let bucket = reader.newest_bucket().unwrap();
    let entry = bucket.upgrade(parent_record).unwrap();
    let sr = entry.span_range().unwrap();
    // Flag remains set: parent has data in a previous bucket.
    assert!(sr.from_previous_bucket());
    // Mask was claimed atomically and now points at this record.
    let first = sr.first_mask.load(Ordering::Acquire);
    let last = sr.last_mask.load(Ordering::Acquire);
    assert_ne!(first, super::SPAN_MASK_NONE);
    assert_ne!(last, super::SPAN_MASK_NONE);
}

#[test]
fn fresh_span_has_no_previous_bucket_flag() {
    let mut index = test_index();
    let mut writer = TestIndexWriter::new(&mut index);
    let s = span_id_for(0xE0);
    let weak = log!(writer; { span: SpanInfo::Start { span: s, parent: None } } msg = "s");
    let reader = writer.index.reader().clone();
    let bucket = reader.newest_bucket().unwrap();
    let entry = bucket.upgrade(weak).unwrap();
    assert!(!entry.span_range().unwrap().from_previous_bucket());
}

#[test]
fn span_table_drops_entries_for_rotated_out_bucket() {
    let mut index = test_index();
    let mut writer = TestIndexWriter::new(&mut index);
    let span = span_id_for(0xC0);
    log!(writer; { span: SpanInfo::Start { span, parent: None } } msg = "x");
    assert!(writer.index.span_table.contains_key(&span));
    // BUCKET_COUNT rotations should evict the bucket originally holding the span.
    for _ in 0..super::BUCKET_COUNT {
        writer.index.complete_bucket();
    }
    assert!(!writer.index.span_table.contains_key(&span));
}

#[test]
fn oversized_interned_ranges_are_rejected() {
    let mut index = test_index();
    let too_large = vec![b'a'; u16::MAX as usize + 1];

    assert!(matches!(index.intern(&too_large), Err(MunchError::InvalidValue)));
    assert!(matches!(index.intern_msg(&too_large), Err(MunchError::InvalidValue)));
}

fn write_log_with_n_fields(writer: &mut TestIndexWriter<'_>, n: usize) -> WeakLogEntry {
    let time = writer.time;
    writer.time += 1;
    let fields = {
        let mut encoder = writer.buf.encoder();
        ("hello").encode_log_value_into(encoder.key("msg"));
        for i in 0..n {
            let name = format!("f{i:02}");
            (i as u32).encode_log_value_into(encoder.key(&name));
        }
        encoder.fields()
    };
    writer.index.write(time, LogLevel::Info, SpanInfo::None, None, fields).unwrap()
}

#[test]
fn wide_field_query_finds_each_position() {
    use crate::field_table::KeyID;

    let mut index = test_index();
    let mut writer = TestIndexWriter::new(&mut index);

    for n in [9usize, 16, super::archetype::FIELD_LANES] {
        let weak = write_log_with_n_fields(&mut writer, n);
        let reader = writer.index.reader().clone();
        let bucket = reader.newest_bucket().unwrap();
        let entry = bucket.upgrade(weak).unwrap();

        for i in 0..n {
            let name = format!("f{i:02}");
            let key = KeyID::try_from_str(&name).unwrap_or_else(|| panic!("missing key {name} for n={n}"));
            let field = entry
                .field_by_key_id(key)
                .unwrap_or_else(|| panic!("field_by_key_id missed `{name}` at position-of-N {n}"));
            assert_eq!(field.kind(), FieldKind::I60);
            assert!(entry.archetype().contains_key(key));
            assert!(entry.archetype().index_of(key).is_some());
        }

        let actual = entry.fields().count();
        assert_eq!(actual, n, "fields() iterator length mismatch for n={n}");
        assert_eq!(entry.raw_fields().len(), n);
        assert_eq!(entry.archetype().size as usize, n);
    }
}

#[test]
fn field_overflow_truncates_to_field_lanes() {
    use crate::field_table::KeyID;

    let mut index = test_index();
    let mut writer = TestIndexWriter::new(&mut index);

    let lanes = super::archetype::FIELD_LANES;
    let dropped = 6usize;
    let n = lanes + dropped;
    let weak = write_log_with_n_fields(&mut writer, n);
    let reader = writer.index.reader().clone();
    let bucket = reader.newest_bucket().unwrap();
    let entry = bucket.upgrade(weak).unwrap();

    assert_eq!(entry.archetype().size as usize, lanes, "archetype size must clamp to FIELD_LANES");
    assert_eq!(entry.raw_fields().len(), lanes, "raw_fields must stay consistent with archetype size");
    assert_eq!(entry.fields().count(), lanes, "fields() iterator must stop at FIELD_LANES");

    let mut found = 0usize;
    let mut missing = 0usize;
    for i in 0..n {
        let name = format!("f{i:02}");
        let key = KeyID::try_from_str(&name).unwrap_or_else(|| panic!("key {name} should have been interned"));
        if entry.field_by_key_id(key).is_some() {
            found += 1;
        } else {
            missing += 1;
        }
    }
    assert_eq!(found, lanes, "exactly FIELD_LANES of the {n} keys should be findable");
    assert_eq!(missing, dropped, "exactly {dropped} keys should have been truncated");
}

#[test]
fn archetype_hash_is_deterministic_for_same_keys() {
    let mut index = test_index();
    let mut writer = TestIndexWriter::new(&mut index);

    let w1 = log!(writer; msg="m", a=1u32, b=2u32, c=3u32);
    let w2 = log!(writer; msg="m", a=1u32, b=2u32, c=3u32);
    let reader = writer.index.reader().clone();
    let bucket = reader.newest_bucket().unwrap();
    let e1 = bucket.upgrade(w1).unwrap();
    let e2 = bucket.upgrade(w2).unwrap();

    assert_eq!(e1.raw_archetype(), e2.raw_archetype(), "identical key+meta logs should share an archetype");
    assert_eq!(e1.archetype().as_raw(), e2.archetype().as_raw());
}

#[test]
fn oversized_message_does_not_rotate_bucket() {
    let mut index = test_index();
    let too_large = vec![b'a'; u16::MAX as usize + 1];
    #[derive(Clone)]
    struct OversizedMsg<'a> {
        bytes: &'a [u8],
        done: bool,
    }
    impl<'a> Iterator for OversizedMsg<'a> {
        type Item = Result<(Key<'a>, Value<'a>), MunchError>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.done {
                return None;
            }
            self.done = true;
            Some(Ok((Key::Static(StaticKey::msg), Value::String(self.bytes))))
        }
    }

    let result = index.write(1, LogLevel::Info, SpanInfo::None, None, OversizedMsg { bytes: &too_large, done: false });

    assert!(matches!(result, Err(MunchError::InvalidValue)));
    assert_eq!(index.generation(), 0);
}

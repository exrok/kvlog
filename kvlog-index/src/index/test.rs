use std::{
    mem::ManuallyDrop,
    sync::{atomic::AtomicUsize, Condvar},
};

use kvlog::{encoding::FieldBuffer, Encode};
use test::filter::Query;

use self::filter::FieldFilter;

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
    fn get(&self) -> PoolGuard {
        let mut ready = self.ready.lock().unwrap();
        if ready.pool.is_empty() && ready.created < self.max {
            ready.created += 1;
            drop(ready);
            return PoolGuard {
                index: ManuallyDrop::new(Box::new(Index::new())),
            };
        }
        while ready.pool.is_empty() {
            ready = self.cond.wait(ready).unwrap();
        }
        let mut index = ready.pool.pop().unwrap();
        // probably not good, hopefully previous tests are still touching it
        unsafe {
            index.clear_unchecked();
        }
        PoolGuard {
            index: ManuallyDrop::new(index),
        }
    }
}

static GLOBAL_TEST_POOL: IndexPool = IndexPool {
    max: 4,
    cond: Condvar::new(),
    ready: Mutex::new(ReadyPool {
        pool: Vec::new(),
        created: 0,
    }),
};

pub struct PoolGuard {
    index: ManuallyDrop<Box<Index>>,
}

impl Drop for PoolGuard {
    fn drop(&mut self) {
        GLOBAL_TEST_POOL.push(unsafe { ManuallyDrop::take(&mut self.index) })
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
            index
                .write(ts, lvl, SpanInfo::None, LogFields::empty())
                .unwrap();
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
    let tf = TimeFilter {
        min_utc_ns: time_rt(0.3),
        max_utc_ns: time_rt(0.5),
    };
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
    assert!(
        results_remaining == 0,
        "Found {} more entries then expected",
        results_remaining
    );
    let expected_remaining = expected_iter.count();
    assert!(
        expected_remaining == 0,
        "Expected {} more entries",
        expected_remaining,
    );
}

pub struct TestIndexWriter<'a> {
    pub(crate) time: u64,
    pub(crate) buf: FieldBuffer,
    pub(crate) index: &'a mut Index,
}
impl TestIndexWriter<'_> {
    pub fn new(index: &'_ mut Index) -> TestIndexWriter<'_> {
        TestIndexWriter {
            time: 1,
            buf: FieldBuffer::default(),
            index,
        }
    }
}
#[macro_export]
macro_rules! log {
    ($buffer:ident $($level:ident)?; $($key:ident = $value:expr),*) => {{
        let time = $buffer.time;
        $buffer.time += 1;
        let fields = {
            let mut encoder = $buffer.buf.encoder();
            $( ($value).encode_log_value_into(encoder.key(stringify!($key)));)*
            encoder.fields()
        };
        #[allow(unused)]
        let mut level = LogLevel::Info;
        $(level = LogLevel::$level;)?
        $buffer.index.write(time, level, SpanInfo::None, fields).unwrap()
    }};
}

#[test]
fn field_query() {
    let mut index = test_index();
    let mut rng = oorandom::Rand32::new(0xdeafbeaf);
    let mut writer = TestIndexWriter::new(&mut index);
    let w0 = log!(writer; msg="nice", cats="nice");
    let w1 = log!(writer Info; msg="nice hello", cats="what");
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
    let w1 = log!(writer Info; msg="nice hello", count=1);
    let w2 = log!(writer; msg="todo, nice hello", dogs="what");

    let query = Query::expr(stringify!(count: int = 1)).unwrap();

    assert_eq_logs(index.reverse_query(&query.filters), &[w1]);
}

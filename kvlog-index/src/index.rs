use crate::{
    bloom::{BloomBuilder, BloomLine},
    field_table::{KeyID, KeyMap},
    query::QueryExpr,
    shared_interner::{LocalIntermentCache, SharedIntermentBuffer},
};
use ahash::RandomState;
use core::time;
pub use filter::{EntryCollection, Query};
use hashbrown::HashTable;
use hashbrown::{HashMap, HashSet};
use kvlog::{
    encoding::{Key, LogFields, MunchError, SpanInfo, StaticKey, Value},
    LogLevel, SpanID,
};
use std::{
    ptr::NonNull,
    sync::{
        atomic::{AtomicU32, AtomicUsize, Ordering},
        Arc, Mutex, MutexGuard,
    },
};
use uuid::Uuid;

#[cfg(test)]
mod test;

use self::filter::{ForwardQueryWalker, ForwardSegmentWalker, QueryStrategy};
pub use self::filter::{
    GeneralFilter, GeneralQuery, LevelFilter, ReverseQueryWalker, SpanCache, TimeFilter,
};
mod filter;
// use libc::MADV_HUGEPAGE;

#[derive(Default, Debug, Clone)]
struct BitSet {
    bytes: Vec<u64>,
}
impl BitSet {
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
    fn insert(&mut self, value: u16) {
        let index = value as usize >> 6;
        if index >= self.bytes.len() {
            self.bytes.resize(index as usize + 1, 0);
        }
        let offset = value as usize & 0b111111;
        self.bytes[index] |= 1 << offset;
    }
    fn contains(&self, value: u16) -> bool {
        let index = value as usize >> 6;
        if index >= self.bytes.len() {
            return false;
        }
        let offset = value as usize & 0b111111;
        self.bytes[index] & (1 << offset) != 0
    }
}

#[derive(PartialOrd, Ord)]
#[repr(C)]
pub struct Archetype {
    msg_offset: u32,
    msg_len: u16,
    target_id: u16,
    // first 4 bits is level
    // 5 bit is set when in span
    // 6-53 bits are a bloom filter of the field keys (not sure if I'll even use the blume)
    mask: u64,
    field_headers: [u16; 8],
}

impl PartialEq for Archetype {
    fn eq(&self, other: &Self) -> bool {
        self.as_raw() == other.as_raw()
    }
}

impl Eq for Archetype {}

impl std::hash::Hash for Archetype {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for chunk in self.as_raw() {
            chunk.hash(state);
        }
    }
}
impl Archetype {
    fn as_raw(&self) -> &[u64; 4] {
        unsafe { &*(self as *const Archetype as *const [u64; 4]) }
    }
    pub fn level(&self) -> LogLevel {
        match self.mask & 0b1111 {
            0b0001 => LogLevel::Debug,
            0b0010 => LogLevel::Info,
            0b0100 => LogLevel::Warn,
            0b1000 => LogLevel::Error,
            _ => LogLevel::Info,
        }
    }
    pub fn contains_key(&self, key: KeyID) -> bool {
        self.field_headers.contains(&key.0)
    }
    // safetey: Bucket must be for Archetype of same generation
    pub unsafe fn message<'a>(&self, bucket: &BucketGuard<'a>) -> &'a [u8] {
        bucket.bucket.data_unchecked(InternedRange {
            offset: self.msg_offset,
            data: 0,
            len: self.msg_len,
        })
    }
    pub fn level_in(&self, mask_filter: LevelFilter) -> bool {
        (self.mask as u8) & 0b1111 & mask_filter.mask != 0
    }
    fn new(
        msg: InternedRange,
        target_id: u16,
        level: LogLevel,
        in_span: bool,
        fields: &[Field],
    ) -> Archetype {
        let mut mask = 1u64 << level as u8;
        if in_span {
            mask |= 1 << 4;
        }
        let mut field_headers = [u16::MAX; 8];
        for (i, field) in fields.iter().enumerate().take(8) {
            let key = field.key_mask();
            let x1 = (key as u64).wrapping_mul(0x517cc1b727220a95) % 53;
            let x2 = (key as u64) % 53;
            mask |= 0b1_0_0000 << x1;
            mask |= 0b1_0_0000 << x2;
            //todo add kind so we can do even more pre-querying???
            field_headers[i] = key;
        }
        Archetype {
            msg_offset: msg.offset,
            msg_len: msg.len,
            target_id,
            mask,
            field_headers,
        }
    }
}
// enum Requirement {
//     None,
//     In(BitSet),
//     Is(u16),
// }

// Key Value in a log? Pair
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Field {
    pub(crate) raw: u64,
}

#[derive(Debug)]
#[repr(u8)]
pub enum FieldKind {
    None = 0,
    String = 1,
    Bytes = 2,
    I48 = 3,
    I64 = 4,
    U64 = 5,
    F32 = 6,
    Bool = 7,
    UUID = 8,
    Seconds = 9,
    Timestamp = 10,
    _Reserved2 = 11,
    _Reserved3 = 12,
    _Reserved4 = 13,
    _Reserved5 = 14,
    _Reserved6 = 15,
}

impl Field {
    fn kind(&self) -> FieldKind {
        unsafe { std::mem::transmute(((self.raw >> 48) as u8) & 0xf) }
    }

    fn new(key: u16, kind: FieldKind, value_mask: u64) -> Field {
        Field {
            raw: ((key as u64) << 52) | ((kind as u64) << 48) | value_mask,
        }
    }
    pub fn key_mask(&self) -> u16 {
        (self.raw >> 52) as u16
    }
    fn value_mask(&self) -> u64 {
        self.raw & ((1u64 << 48) - 1)
    }
    // fn max_field_bits_with_key(key: KeyID) -> u64 {
    //     ((key.raw() as u64) << 52) | (0xFFFF_FFFF_FFFF_FFFF >> (64 - 52))
    // }
    // fn static_key(self) -> Option<StaticKey> {
    //     StaticKey::from_u8(self.key_mask().try_into().ok()?)
    // }
    unsafe fn key(self) -> KeyID {
        unsafe { KeyID::new(self.key_mask()) }
    }
    unsafe fn value<'a>(self, bucket: &'a Bucket) -> Value<'a> {
        match self.kind() {
            FieldKind::String => {
                let range = InternedRange::from_field_mask(self.value_mask());
                Value::String(unsafe { bucket.data_unchecked(range) })
            }
            FieldKind::Bytes => {
                let range = InternedRange::from_field_mask(self.value_mask());
                Value::Bytes(unsafe { bucket.data_unchecked(range) })
            }
            FieldKind::I48 => Value::I64(i48::to_i64(self.value_mask())),
            FieldKind::I64 => {
                let value = unsafe { bucket.u64_unchecked(self.value_mask() as u32) };
                Value::I64(value as i64)
            }
            FieldKind::U64 => {
                let value = unsafe { bucket.u64_unchecked(self.value_mask() as u32) };
                Value::U64(value as u64)
            }
            FieldKind::F32 => Value::F32(f32::from_bits(self.value_mask() as u32)),
            FieldKind::Seconds => Value::Seconds(f32::from_bits(self.value_mask() as u32)),
            FieldKind::UUID => Value::UUID(uuid::Uuid::from_bytes(unsafe {
                *bucket.uuid_bytes(self.value_mask() as u32)
            })),
            FieldKind::None => Value::None,
            FieldKind::Bool => Value::Bool((self.value_mask() as u8) != 0),
            FieldKind::Timestamp => Value::Timestamp(
                // todo this should be improved with general timestamp support.
                kvlog::Timestamp::from_millisecond(self.value_mask() as i64),
            ),
            _ => Value::None,
        }
    }
}

impl std::fmt::Debug for Field {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Field({:?}: {:?} {:012X})",
            self.key_mask(),
            self.kind(),
            self.value_mask()
        )
    }
}

impl From<u8> for FieldKind {
    fn from(value: u8) -> Self {
        unsafe { std::mem::transmute(value & 0xf) }
    }
}
impl Field {}
// Current settings targets around ~1GB memory of RAM
// To hold around a million logs, per bucket
// todo make these runtime configurable
const BUCKET_LOG_SIZE: usize = (4096 * 256) - 1;
// const BUCKET_LOG_SIZE: usize = (4096) - 1;
const BUCKET_SPAN_RANGE_SIZE: usize = (BUCKET_LOG_SIZE + 1) / 3;
const BUCKET_FIELD_SIZE: usize = 8 * (BUCKET_LOG_SIZE + 1);
const BUCKET_DATA_SIZE: usize = BUCKET_LOG_SIZE * 128;
const BUCKET_MAX_MSG_SIZE: usize = u16::MAX as usize;
const BUCKET_MAX_TARGET_SIZE: usize = u16::MAX as usize;
const BUCKET_COUNT: usize = 4;
const TIME_RANGE_LOG_COUNT: usize = 4096;
const TIME_RANGE_LOG_MASK_SIZE: u32 = TIME_RANGE_LOG_COUNT.trailing_zeros();
const BUCKET_ARCHETYPE_SIZE: usize = u16::MAX as usize;

#[derive(Debug)]
pub struct SpanRange {
    pub id: SpanID,
    pub parent: Option<SpanID>,
    // 32nd bit indicates this entry was a Start Span
    pub first_mask: u32,
    // 32nd bit indicates this entry was a End Span
    pub last_mask: AtomicU32,
}
impl SpanRange {
    fn index_range(&self) -> std::ops::Range<u32> {
        (self.first_mask & 0x7FFF_FFFF)
            ..((self.last_mask.load(Ordering::Relaxed) & 0x7FFF_FFFF) + 1)
    }
    pub fn span_info(&self, entry: u32) -> SpanInfo {
        if self.first_mask ^ entry == (1u32 << 31) {
            SpanInfo::Start {
                span: self.id,
                parent: self.parent,
            }
        } else if self.last_mask.load(Ordering::Acquire) ^ entry == (1u32 << 31) {
            SpanInfo::End { span: self.id }
        } else {
            SpanInfo::Current { span: self.id }
        }
    }
}
#[repr(C)]
struct TimeRange {
    min_utc_ns: u64,
    max_utc_nc: u64,
}
#[derive(Clone, Copy, Default)]
pub struct TargetSummary {
    pub level_counts: [u32; 4],
}

pub struct IntermentMaps {
    general: HashTable<InternedRange>,
    uuid: HashTable<u32>,
    target_summaries: Vec<TargetSummary>,
    keys: KeyMap<KeyInfo>,
    indexed: usize,
    msgs: usize,
}
impl IntermentMaps {
    pub fn keys(&self) -> impl Iterator<Item = (KeyID, &KeyInfo)> + '_ {
        self.keys.iter().filter(|(_, info)| info.total > 0)
    }
    pub fn field_uuid(&self, bucket: &BucketGuard, key: KeyID, value: Uuid) -> Option<Field> {
        let index = *self.uuid.find(
            bucket.bucket.random_state.hash_one(value.as_bytes()),
            |t| unsafe { bucket.bucket.uuid_bytes(*t) == value.as_bytes() },
        )?;
        Some(Field::new(key.raw(), FieldKind::UUID, index as u64))
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct WeakLogEntry {
    index: u32,
    bucket_generation: u32,
}
// const BUCKET_LOG_SIZE: usize = 511;
// const BUCKET_FIELD_SIZE: usize = 8 * (BUCKET_LOG_SIZE + 1);
// const BUCKET_DATA_SIZE: usize = BUCKET_LOG_SIZE * 256;
// const BUCKET_COUNT: usize = 4;
pub struct Bucket {
    generation: AtomicU32,
    intern_map: Mutex<IntermentMaps>,
    data: NonNull<u8>,
    random_state: RandomState,
    field: NonNull<Field>,
    bloom: NonNull<BloomLine>,

    timerange: NonNull<TimeRange>,

    // // stores the message field, max 2^16 disinct per bucket
    // msg: NonNull<u16>,
    // msg_ptr: NonNull<InternedRange>,

    // stores the target field, max 2^16 disinct per bucket
    target: NonNull<u16>,

    span_index: NonNull<u32>,
    span_data: NonNull<SpanRange>,
    timestamp: NonNull<u64>,
    offset: NonNull<u32>,
    span_count: AtomicUsize,
    len: AtomicUsize,
    ref_count: AtomicU32,

    archetype: NonNull<Archetype>,
    archetype_index: NonNull<u16>,
    archetype_count: AtomicUsize,
}

#[derive(Clone, Copy)]
pub struct LogEntry<'a> {
    index: usize,
    bucket: &'a Bucket,
}
#[derive(Clone)]
pub struct Fields<'a> {
    fields: &'a [Field],
    bucket: &'a Bucket,
}

impl<'a> std::fmt::Debug for Fields<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(self.clone().into_iter().map(|(key, value)| (key, value)))
            .finish()
    }
}
impl<'a> Iterator for Fields<'a> {
    type Item = (KeyID, Value<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        let [field, rest @ ..] = self.fields else {
            return None;
        };
        self.fields = rest;
        Some(unsafe { (field.key(), field.value(self.bucket)) })
    }
}

impl<'a> LogEntry<'a> {
    fn slow_matches(&self, query: &QueryExpr) -> bool {
        if !query.simple.matches_raw(self.raw_bloom()) {
            return false;
        }
        'outer: for (key, predicate) in &query.fields {
            let Some(key) = KeyID::try_from_str(key) else {
                return false;
            };
            if key == StaticKey::msg {
                if !predicate.matches_text(self.message()) {
                    return false;
                }
                continue;
            }
            if key == StaticKey::target {
                todo!();
                //todo
                // if !predicate.matches_text(self.target()) {
                //     return false;
                // }
                continue;
            }
            for (field_key, value) in self.fields() {
                if field_key != key {
                    continue;
                }
                match value {
                    Value::String(text) => {
                        if !predicate.matches_text(text) {
                            return false;
                        }
                        continue 'outer;
                    }
                    Value::UUID(uuid) => {
                        if !predicate.matches_uuid(uuid) {
                            return false;
                        }
                        continue 'outer;
                    }
                    //todo
                    _ => return false,
                }
            }
            return false;
        }
        return true;
    }
    pub fn weak(&self) -> WeakLogEntry {
        WeakLogEntry {
            index: self.index as u32,
            bucket_generation: self.bucket.generation.load(Ordering::Relaxed),
        }
    }
    pub fn bucket(&self) -> &Bucket {
        self.bucket
    }
    pub fn bloom(&self) -> &BloomLine {
        unsafe { &*self.bucket.bloom.as_ptr().add(self.index) }
    }
    pub fn timestamp(&self) -> u64 {
        unsafe { *self.bucket.timestamp.as_ptr().add(self.index) }
    }
    // pub fn message_id(&self) -> u16 {
    //     unsafe { *self.bucket.msg.as_ptr().add(self.index) }
    // }
    pub fn target_id(&self) -> u16 {
        unsafe { *self.bucket.target.as_ptr().add(self.index) }
    }
    pub fn archetype(&self) -> &Archetype {
        unsafe {
            self.bucket
                .archetype(*self.bucket.archetype_index.as_ptr().add(self.index))
        }
    }
    pub fn message(&self) -> &[u8] {
        let archetype = self.archetype();
        let intern_range = InternedRange {
            offset: archetype.msg_offset,
            data: 0,
            len: archetype.msg_len,
        };
        unsafe { self.bucket.data_unchecked(intern_range) }
    }
    pub fn raw_bloom(&self) -> u8 {
        self.bloom().0
    }
    pub fn level(&self) -> LogLevel {
        self.bloom().level()
    }
    pub fn raw_fields(&self) -> &[Field] {
        let start = unsafe { *self.bucket.offset.as_ptr().add(self.index) };
        let end = unsafe { *self.bucket.offset.as_ptr().add(self.index + 1) };
        let field = self.bucket.field.as_ptr();
        unsafe { std::slice::from_raw_parts(field.add(start as usize), (end - start) as usize) }
    }

    /// Not inefficent for long running spans and will be incomplete when
    /// crossing buckets
    pub fn span_logs(&self) -> impl Iterator<Item = LogEntry<'_>> + '_ {
        let range = if let Some(range) = self.span_range() {
            range.index_range()
        } else {
            0..0
        };
        let span_index = unsafe { *self.bucket.span_index.as_ptr().add(self.index) };
        let indices = unsafe {
            std::slice::from_raw_parts(
                self.bucket.span_index.as_ptr().add(range.start as usize),
                (range.end - range.start) as usize,
            )
        };
        indices
            .iter()
            .enumerate()
            .filter(move |(_, index)| **index == span_index)
            .map(move |(index, _)| LogEntry {
                index: index + range.start as usize,
                bucket: self.bucket,
            })
    }
    pub fn fields(&self) -> Fields<'_> {
        Fields {
            fields: self.raw_fields(),
            bucket: self.bucket,
        }
    }

    pub fn raw_span_id(&self) -> u32 {
        let span_index = unsafe { *self.bucket.span_index.as_ptr().add(self.index) };
        span_index
    }
    pub fn span_range(&self) -> Option<&SpanRange> {
        let span_index = unsafe { *self.bucket.span_index.as_ptr().add(self.index) };
        if span_index == u32::MAX {
            return None;
        }
        Some(unsafe { &*self.bucket.span_data.as_ptr().add(span_index as usize) })
    }
    pub fn span_info(&self) -> SpanInfo {
        if let Some(range) = self.span_range() {
            range.span_info(self.index as u32)
        } else {
            SpanInfo::None
        }
    }
}

// struct InternMap<'a> {
//     guard: MutexGuard<'a, IntermentMaps>,
//     bucket: &'a Bucket,
// }
// impl<'a> InternMap<'a> {
//     fn get(&self, bytes: &[u8]) -> Option<InternedRange> {
//         let hash = self.bucket.random_state.hash_one(bytes);
//         self.guard
//             .general
//             .find(hash, |t| unsafe {
//                 t.data == 0 && self.bucket.data_unchecked(*t) == bytes
//             })
//             .copied()
//     }
//     fn get_key(&self, bytes: &[u8]) -> Option<InternedRange> {
//         let hash = self.bucket.random_state.hash_one(bytes);
//         self.guard
//             .general
//             .find(hash, |t| unsafe {
//                 t.data != 0 && self.bucket.data_unchecked(*t) == bytes
//             })
//             .copied()
//     }
// }
#[derive(Default)]
pub struct KeyInfo {
    total: u32,
    types: u32,
    values: HashTable<Field>,
}
impl KeyInfo {
    pub fn total(&self) -> usize {
        self.total as usize
    }
    pub fn kind_mask(&self) -> u32 {
        self.types
    }
    pub fn kinds(&self) -> impl Iterator<Item = FieldKind> {
        let value = self.types;
        (0..16)
            .filter(move |i| (value >> i) & 1 == 1)
            .map(|i| FieldKind::from(i))
    }
    pub unsafe fn values<'a, 'b: 'a>(
        &'a self,
        bucket: &'b Bucket,
    ) -> impl Iterator<Item = Value<'b>> + 'a + ExactSizeIterator {
        self.values
            .iter()
            .map(|value| unsafe { value.value(bucket) })
    }
}

impl Bucket {
    fn new(index: u32) -> Bucket {
        // todo optimize until a single alloc
        unsafe {
            let bucket = Bucket {
                intern_map: Mutex::new(IntermentMaps {
                    general: HashTable::with_capacity(4096),
                    uuid: HashTable::with_capacity(4096),
                    keys: KeyMap::default(),
                    target_summaries: Vec::new(),
                    indexed: 0,
                    msgs: 0,
                }),
                random_state: RandomState::new(),
                data: mmap_alloc(BUCKET_DATA_SIZE),
                field: mmap_alloc(BUCKET_FIELD_SIZE),
                bloom: mmap_alloc(BUCKET_LOG_SIZE + 1),
                target: mmap_alloc(BUCKET_LOG_SIZE + 1),
                timerange: mmap_alloc((BUCKET_LOG_SIZE + 1) / TIME_RANGE_LOG_COUNT),
                // msg_ptr: mmap_alloc(BUCKET_MAX_TARGET_SIZE),
                // msg: mmap_alloc(BUCKET_LOG_SIZE + 1),
                span_index: mmap_alloc(BUCKET_LOG_SIZE + 1),
                span_data: mmap_alloc(BUCKET_SPAN_RANGE_SIZE + 1),
                span_count: AtomicUsize::new(0),

                timestamp: mmap_alloc(BUCKET_LOG_SIZE + 1),
                offset: mmap_alloc(BUCKET_LOG_SIZE + 1),

                archetype: mmap_alloc(BUCKET_ARCHETYPE_SIZE),
                archetype_index: mmap_alloc(BUCKET_LOG_SIZE + 1),
                archetype_count: AtomicUsize::new(0),

                len: AtomicUsize::new(0),
                generation: AtomicU32::new(index),
                ref_count: AtomicU32::new(1), // 1 by default, to avoid calling wake
            };
            // first value of offset is always size
            bucket.offset.as_ptr().write(0);
            bucket
        }
    }
    unsafe fn clear_unchecked(&self, i: u32) {
        self.generation.store(i, Ordering::Release);
        self.archetype_count.store(0, Ordering::Release);
        self.span_count.store(0, Ordering::Release);
        self.len.store(0, Ordering::Release);
        self.offset.as_ptr().write(0);
        let intern = &mut *self.intern_map.lock().unwrap();
        intern.general.clear();
        intern.uuid.clear();
        intern.keys.clear();
        intern.indexed = 0;
        intern.msgs = 0;
    }

    unsafe fn archetype(&self, index: u16) -> &Archetype {
        unsafe { &*(self.archetype.as_ptr().add(index as usize)) }
    }
    unsafe fn uuid_bytes(&self, index: u32) -> &[u8; 16] {
        unsafe { &*(self.data.as_ptr().add(index as usize) as *const [u8; 16]) }
    }
    unsafe fn u64_unchecked(&self, offset: u32) -> u64 {
        unsafe { (self.data.as_ptr().add(offset as usize) as *const u64).read_unaligned() }
    }
    unsafe fn data_unchecked(&self, data: InternedRange) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr().add(data.offset as usize),
                data.len as usize,
            )
        }
    }
    pub fn read(&self) -> Option<BucketGuard<'_>> {
        if self.len.load(Ordering::Acquire) == 0 {
            return None;
        }

        self.ref_count.fetch_add(1, Ordering::AcqRel);
        let new_len = self.len.load(Ordering::Acquire);
        if new_len == 0 {
            if self.ref_count.fetch_sub(1, Ordering::AcqRel) == 0 {
                atomic_wait::wake_one(&self.ref_count);
            }
            return None;
        }
        // ensure that generation, loaded so we can used
        // relaxed later, while ref_count == 0 generation will
        // not change
        let _ = self.generation.load(Ordering::Acquire);
        Some(BucketGuard {
            bucket: self,
            len: new_len,
        })
    }
}

pub struct SavedEntryIndexReader<'e, 'b> {
    generation_guard: GenerationGuard<'b>,
    entries: &'e [WeakLogEntry],
}
impl<'e, 'b> SavedEntryIndexReader<'e, 'b> {
    pub unsafe fn project_alive(
        entries: &'e mut Vec<WeakLogEntry>,
        index: &'b IndexReader,
    ) -> Self {
        let gg = index.generation_guard();
        entries.retain(|entry| gg.is_alive(*entry));
        SavedEntryIndexReader {
            generation_guard: gg,
            entries: &*entries,
        }
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn get(&self, index: usize) -> Option<LogEntry<'_>> {
        unsafe { self.generation_guard.upgrade(*self.entries.get(index)?) }
    }
}

/// Todo optimize this like a lot....
pub struct GenerationGuard<'a> {
    buckets: [Option<BucketGuard<'a>>; BUCKET_COUNT],
}
impl<'a> GenerationGuard<'a> {
    pub unsafe fn upgrade(&self, weak: WeakLogEntry) -> Option<LogEntry<'_>> {
        if let Some(bucket) = &self.buckets[(weak.bucket_generation & 0b11) as usize] {
            bucket.upgrade(weak)
        } else {
            None
        }
    }
    pub fn targets_summary(&self) -> Vec<TargetSummary> {
        let mut summary: Vec<TargetSummary> = Vec::new();
        for bucket in self.buckets() {
            let maps = bucket.maps();
            if summary.is_empty() {
                summary = maps.target_summaries.clone();
                continue;
            }
            for (i, value) in maps.target_summaries.iter().enumerate() {
                if i >= summary.len() {
                    summary.resize(
                        i + 1,
                        TargetSummary {
                            level_counts: [0; 4],
                        },
                    );
                }
                let target = &mut summary[i];
                for (j, count) in value.level_counts.iter().enumerate() {
                    target.level_counts[j] += count;
                }
            }
        }

        summary
    }

    pub fn keys_summary(&self) -> HashMap<KeyID, KeySummary<'a>> {
        let mut keys: HashMap<KeyID, KeySummary<'a>> = HashMap::default();
        for bucket in self.buckets() {
            let maps = bucket.maps();
            for (key, info) in maps.keys() {
                match keys.entry(key) {
                    hashbrown::hash_map::Entry::Occupied(mut entry) => {
                        let summary = entry.get_mut();
                        summary.total += info.total();
                        summary.kind_mask |= info.kind_mask();
                        let values = unsafe { info.values(&bucket.bucket) };
                        if values.len() > 32 {
                            summary.values = None;
                            continue;
                        }
                        if let Some(all_values) = &mut summary.values {
                            all_values.extend(values.filter(|value| !value.is_large()));
                        }
                    }
                    hashbrown::hash_map::Entry::Vacant(entry) => {
                        let values = unsafe { info.values(&bucket.bucket) };
                        entry.insert(KeySummary {
                            total: info.total(),
                            kind_mask: info.kind_mask(),
                            values: if values.len() < 32 {
                                Some(values.filter(|value| !value.is_large()).collect())
                            } else {
                                None
                            },
                        });
                    }
                }
            }
        }
        keys
    }
    pub fn buckets(&self) -> impl Iterator<Item = &BucketGuard<'a>> {
        self.buckets.iter().filter_map(|x| x.as_ref())
    }
    pub unsafe fn is_alive(&self, weak: WeakLogEntry) -> bool {
        if let Some(bucket) = &self.buckets[(weak.bucket_generation & 0b11) as usize] {
            bucket.is_alive(weak)
        } else {
            false
        }
    }
}

pub struct BucketGuard<'a> {
    bucket: &'a Bucket,
    len: usize,
}
impl<'a> BucketGuard<'a> {
    pub fn entry_count(&self) -> usize {
        self.len
    }
    pub fn bucket(&self) -> &'a Bucket {
        self.bucket
    }
    pub fn is_alive(&self, log: WeakLogEntry) -> bool {
        self.bucket.generation.load(Ordering::Relaxed) == log.bucket_generation
    }
    pub fn renew(&mut self) {
        self.len = self.bucket.len.load(Ordering::Acquire);
    }
    pub fn reverse_time_range_skip(&self, offset: u32, range: TimeFilter) -> u32 {
        if self.len & (!(TIME_RANGE_LOG_COUNT - 1)) <= offset as usize {
            return offset;
        }
        let mut current = offset.saturating_sub(1) >> TIME_RANGE_LOG_MASK_SIZE;
        loop {
            let chunk_range = unsafe { &*self.bucket.timerange.as_ptr().add(current as usize) };
            if chunk_range.min_utc_ns <= range.max_utc_ns
                && chunk_range.max_utc_nc >= range.min_utc_ns
            {
                return ((current << TIME_RANGE_LOG_MASK_SIZE) + 4096).min(offset);
            }
            if current == 0 {
                return 0;
            }
            current -= 1;
        }
    }

    pub fn upgrade(&self, log: WeakLogEntry) -> Option<LogEntry<'a>> {
        if self.bucket.generation.load(Ordering::Relaxed) == log.bucket_generation {
            Some(LogEntry {
                bucket: self.bucket,
                index: log.index as usize,
            })
        } else {
            None
        }
    }
    pub fn span_indices(&self) -> &[u32] {
        let span = self.bucket.span_index.as_ptr();
        unsafe { std::slice::from_raw_parts(span, self.len) }
    }
    pub fn level_query(&self, mask: u8) -> impl Iterator<Item = LogEntry> + '_ {
        let query = BloomBuilder::default().query(mask);
        self.blooms()
            .iter()
            .enumerate()
            .filter(move |(_, bloom)| bloom.matches(&query))
            .map(|(index, _)| LogEntry {
                index,
                bucket: &self.bucket,
            })
    }

    // #[inline]
    ///todo vectorize more
    pub fn level_query2(&self, mask: u8) -> impl Iterator<Item = LogEntry> + '_ {
        let lmask = (mask as u64) * 0x01010101_01010101;
        let data = self.wide_blooms();
        //todo handle last little bit
        //todo handle endians
        data.iter().enumerate().flat_map(move |(i, batch)| {
            let mut tgx = batch & lmask;
            let mut b = 0;
            std::iter::from_fn(move || {
                while tgx != 0 {
                    let k = tgx & 0xff;
                    b += 1;
                    tgx >>= 8;
                    if k != 0 {
                        return Some(LogEntry {
                            index: ((i << 3) | b),
                            bucket: &self.bucket,
                        });
                    }
                }
                None
            })
        })
    }

    pub fn entries(&self) -> impl Iterator<Item = LogEntry> + DoubleEndedIterator + '_ {
        (0..self.len).map(|index| LogEntry {
            index,
            bucket: &self.bucket,
        })
    }
    // pub unsafe fn messages(&self) -> &[InternedRange] {
    //     let total_msg = {
    //         let intern = self.bucket.intern_map.lock().unwrap();
    //         intern.msgs
    //     };
    //     let msg_ptr = self.bucket.msg_ptr.as_ptr();
    //     unsafe { std::slice::from_raw_parts(msg_ptr, total_msg) }
    // }
    pub fn archetypes(&self) -> &[Archetype] {
        let amount = self.bucket.archetype_count.load(Ordering::Acquire);
        let ptr = self.bucket.archetype.as_ptr();
        unsafe { std::slice::from_raw_parts(ptr, amount) }
    }
    pub fn archetype_index(&self) -> &[u16] {
        let ptr = self.bucket.archetype_index.as_ptr();
        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }
    // pub fn message_ids(&self) -> &[u16] {
    //     let timestamp = self.bucket.msg.as_ptr();
    //     unsafe { std::slice::from_raw_parts(timestamp, self.len) }
    // }
    pub fn spans(&self) -> &[SpanRange] {
        let len = self.bucket.span_count.load(Ordering::Acquire);
        let span = self.bucket.span_data.as_ptr();
        unsafe { std::slice::from_raw_parts(span, len) }
    }
    fn wide_blooms(&self) -> &[u64] {
        let bloom = self.bucket.bloom.as_ptr();
        unsafe { std::slice::from_raw_parts(bloom as *const u64, self.len / 8) }
    }
    fn blooms(&self) -> &[BloomLine] {
        let bloom = self.bucket.bloom.as_ptr();
        unsafe { std::slice::from_raw_parts(bloom, self.len) }
    }
    pub fn logs(&self) -> impl Iterator<Item = (u64, BloomLine, &[Field])> + '_ {
        self.timestamps()
            .iter()
            .copied()
            .zip(self.blooms())
            .enumerate()
            .map(|(i, (ts, bloom))| (ts, *bloom, self.fields_of(i as u32)))
    }
    pub fn timestamps(&self) -> &[u64] {
        let timestamp = self.bucket.timestamp.as_ptr();
        unsafe { std::slice::from_raw_parts(timestamp, self.len) }
    }
    fn fields_of(&self, entry: u32) -> &[Field] {
        // todo optimize to make sure the first element in the offsets memmap is always 0
        if self.len == 0 {
            return &[];
        }
        let end = self.offsets()[(entry + 1) as usize];
        let start = self.offsets()[(entry) as usize];
        // println!("{}", start);
        // println!("{}", end);
        let field = self.bucket.field.as_ptr();
        unsafe { std::slice::from_raw_parts(field.add(start as usize), (end - start) as usize) }
    }
    fn offsets(&self) -> &[u32] {
        let offset = self.bucket.offset.as_ptr();
        unsafe { std::slice::from_raw_parts(offset, self.len + 1) }
    }
    pub fn maps(&self) -> MutexGuard<'_, IntermentMaps> {
        let mut map = self.bucket.intern_map.lock().unwrap();
        if map.indexed >= self.len {
            return map;
        };
        // println!("WHAT: {}", self.len);
        let (targets, level) = unsafe {
            let start = map.indexed;
            let len = self.len - map.indexed;
            (
                std::slice::from_raw_parts(self.bucket.target.as_ptr().add(start as usize), len),
                std::slice::from_raw_parts(self.bucket.bloom.as_ptr().add(start as usize), len),
            )
        };

        for (target, level) in targets.iter().zip(level) {
            let target = *target as usize;
            if target >= map.target_summaries.len() {
                map.target_summaries.resize(
                    target as usize + 1,
                    TargetSummary {
                        level_counts: [0; 4],
                    },
                );
            }
            map.target_summaries[target].level_counts[level.level() as usize] += 1;
        }

        let unprocessed_fields = unsafe {
            let start = *self.bucket.offset.as_ptr().add(map.indexed + 1);
            let end = *self.bucket.offset.as_ptr().add(self.len);
            std::slice::from_raw_parts(
                self.bucket.field.as_ptr().add(start as usize),
                (end - start) as usize,
            )
        };

        for field in unprocessed_fields {
            let key_info = map.keys.get_mut_or_default(unsafe { field.key() });
            key_info.total += 1;
            key_info.types |= 1u32 << field.kind() as u8;
            if key_info.values.len() < 32 {
                let hash = self.bucket.random_state.hash_one(field);
                use hashbrown::hash_table::Entry;
                if let Entry::Vacant(entry) = key_info.values.entry(
                    hash,
                    |v| v == field,
                    |value| self.bucket.random_state.hash_one(value),
                ) {
                    entry.insert(*field);
                }
            }
        }
        map.indexed = self.len;

        map
    }
}
impl<'a> Drop for BucketGuard<'a> {
    fn drop(&mut self) {
        if self.bucket.ref_count.fetch_sub(1, Ordering::AcqRel) == 0 {
            atomic_wait::wake_one(&self.bucket.ref_count);
        }
    }
}

pub struct IndexReader {
    pub targets: SharedIntermentBuffer,
    pub buckets: [Bucket; BUCKET_COUNT],
    generation: AtomicU32,
}
unsafe impl Send for IndexReader {}
unsafe impl Sync for IndexReader {}

#[derive(Debug)]
pub struct KeySummary<'a> {
    pub total: usize,
    pub kind_mask: u32,
    pub values: Option<HashSet<Value<'a>>>,
}

impl IndexReader {
    pub fn bidirectional_query<'a>(
        &'a self,
        filters: &'a [GeneralFilter],
    ) -> (ReverseQueryWalker<'a>, ForwardQueryWalker<'a>) {
        let mut reverse = self.reverse_query(filters);
        let forward = reverse.forward_query();
        (reverse, forward)
    }
    pub fn forward_query<'a>(&'a self, filters: &'a [GeneralFilter]) -> ForwardQueryWalker<'a> {
        let mut time_range = TimeFilter {
            min_utc_ns: 0,
            max_utc_ns: u64::MAX,
        };
        for filter in filters {
            if let GeneralFilter::Time(filter) = filter {
                time_range.min_utc_ns = time_range.min_utc_ns.max(filter.min_utc_ns);
                time_range.max_utc_ns = time_range.max_utc_ns.min(filter.max_utc_ns);
            }
        }
        if time_range.max_utc_ns != u64::MAX {
            time_range.min_utc_ns = time_range.min_utc_ns.max(1)
        }
        let max_generation = self.generation.load(Ordering::Acquire);
        let mut walker = ForwardQueryWalker {
            general_filter: filters,
            time_range,
            segments: ForwardSegmentWalker::new(max_generation.saturating_sub(4)),
            index: self,
            strategy: QueryStrategy::Simple,
        };
        walker
    }
    pub fn reverse_query<'a>(&'a self, filters: &'a [GeneralFilter]) -> ReverseQueryWalker<'a> {
        let mut time_range = TimeFilter {
            min_utc_ns: 0,
            max_utc_ns: u64::MAX,
        };
        for filter in filters {
            if let GeneralFilter::Time(filter) = filter {
                time_range.min_utc_ns = time_range.min_utc_ns.max(filter.min_utc_ns);
                time_range.max_utc_ns = time_range.max_utc_ns.min(filter.max_utc_ns);
            }
        }
        if time_range.max_utc_ns != u64::MAX {
            time_range.min_utc_ns = time_range.min_utc_ns.max(1)
        }
        let mut walker = ReverseQueryWalker {
            general_filter: filters,
            time_range,
            next_generation: self.generation.load(Ordering::Acquire),
            index: self,
            current_bucket: None,
            current_offset: 0,
            strategy: QueryStrategy::Simple,
            buffer: Vec::new(),
        };
        walker
    }
    pub fn generation_guard(&self) -> GenerationGuard<'_> {
        GenerationGuard {
            buckets: [
                self.buckets[0].read(),
                self.buckets[1].read(),
                self.buckets[2].read(),
                self.buckets[3].read(),
            ],
        }
    }
    pub fn lastest_generation(&self) -> u32 {
        self.generation.load(Ordering::Acquire)
    }
    pub fn read_newest_buckets(&self) -> impl Iterator<Item = BucketGuard<'_>> + '_ {
        let current_generation = self.generation.load(Ordering::Acquire);
        let current_bucket = current_generation & 0b11;
        (current_bucket + 1..current_bucket + 4 + 1)
            .rev()
            .filter_map(|i| {
                let bucket_index = i & 0b11;
                self.buckets[bucket_index as usize].read()
            })
    }
    pub fn query(&self, query: &QueryExpr, mut func: impl FnMut(LogEntry) -> bool) {
        for bucket in self.read_newest_buckets() {
            for entry in bucket.entries().rev() {
                if entry.slow_matches(query) {
                    if !func(entry) {
                        return;
                    }
                }
            }
        }
    }
}

// impl GenerationGuard {
//     fn upgrade(&self, weak: WeakLogEntry) -> LogEntry<'a> {}
// }

unsafe fn mmap_alloc<T: Sized>(amount: usize) -> NonNull<T> {
    let x = unsafe {
        libc::mmap(
            0 as *mut libc::c_void,
            amount * std::mem::size_of::<T>(),
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    } as *mut T;
    let szx = amount * std::mem::size_of::<T>();
    if szx > 4 * 1024 * 1024 {
        // unsafe {
        //     libc::madvise(x as *mut _, szx, MADV_HUGEPAGE);
        // }
    }
    NonNull::new(x).unwrap()
}

impl IndexReader {
    fn new() -> IndexReader {
        IndexReader {
            targets: SharedIntermentBuffer::with_capacity(8 * 1024 * 1024),
            buckets: [
                Bucket::new(0),
                Bucket::new(1),
                Bucket::new(2),
                Bucket::new(3),
            ],
            generation: AtomicU32::new(0),
        }
    }
}
#[derive(Copy, Clone)]
pub struct InternedRange {
    pub(crate) len: u16,
    pub(crate) data: u16,
    pub(crate) offset: u32,
}
impl InternedRange {
    fn from_field_mask(field_mask: u64) -> InternedRange {
        InternedRange {
            offset: field_mask as u32,
            len: (field_mask >> 32) as u16,
            data: 0,
        }
    }
    fn field_mask(&self) -> u64 {
        self.offset as u64 | ((self.len as u64) << 32)
    }
}

#[derive(Debug)]
pub struct BucketMemoryStats {
    pub log_count: usize,
    pub interned: usize,
    pub log_metadata: usize,
    pub fields: usize,
    pub custom_keys: usize,
    pub span: usize,
    pub general_map: usize,
    pub uuid_map: usize,
    pub messages: usize,
}
impl BucketMemoryStats {
    pub fn bytes_per_log(&self) -> f64 {
        (self.total() as f64) / self.log_count as f64
    }
    pub fn total(&self) -> usize {
        self.interned
            + self.log_metadata
            + self.fields
            + self.span
            + self.custom_keys
            + self.general_map
            + self.uuid_map
            + self.messages
    }
}

pub struct Index {
    fields_used: usize,
    data_used: usize,
    logs_used: usize,
    spans_used: usize,
    custom_keys: u16,
    generation: usize,
    general_intern_map: HashTable<InternedRange>,
    logdex_map: HashTable<u16>,
    msg_map: HashTable<InternedRange>,
    target_map: LocalIntermentCache,
    uuid_intern_map: HashTable<u32>,
    span_table: HashMap<SpanID, u32>,
    new_ranges: Vec<(InternedRange, u64)>,
    new_uuids: Vec<(u32, u64)>,
    dirty: bool,
    reader: Arc<IndexReader>,

    chunk_max_utc_ns: u64,
    chunk_min_utc_ns: u64,
}
unsafe impl Send for Index {}

#[allow(unused)]
pub const _: *const () = &Index::write::<'static, LogFields<'static>> as *const _ as _;
// Static keys are a maxium 127,
impl Index {
    pub fn reader(&self) -> &Arc<IndexReader> {
        &self.reader
    }

    pub fn forward_query<'b>(
        &'b self,
        filters: &'b [GeneralFilter],
    ) -> impl Iterator<Item = LogEntry<'b>> {
        let mut walker = self.reader().forward_query(filters);
        std::iter::from_fn(move || {
            let collection = walker.next()?;
            Some((
                // Safe because we hold the referce to the index now mutations will occour
                unsafe { &*(collection.bucket_generation as *const Bucket) },
                collection.entries,
            ))
        })
        .flat_map(|(bucket, indices)| {
            indices.into_iter().map(|index| LogEntry {
                bucket,
                index: index as usize,
            })
        })
    }
    pub fn reverse_query<'b>(
        &'b self,
        filters: &'b [GeneralFilter],
    ) -> impl Iterator<Item = LogEntry<'b>> {
        let mut walker = self.reader().reverse_query(filters);
        std::iter::from_fn(move || {
            let collection = walker.next()?;
            Some((
                // Safe because we hold the referce to the index now mutations will occour
                unsafe { &*(collection.bucket_generation as *const Bucket) },
                collection.entries,
            ))
        })
        .flat_map(|(bucket, indices)| {
            indices.into_iter().map(|index| LogEntry {
                bucket,
                index: index as usize,
            })
        })
    }
    pub(crate) fn complete_bucket(&mut self) {
        //todo handle bucket sizes
        self.fields_used = 0;
        self.data_used = 0;
        self.logs_used = 0;
        self.spans_used = 0;
        self.custom_keys = 0;
        self.general_intern_map.clear();
        self.uuid_intern_map.clear();
        self.span_table.clear();
        self.new_ranges.clear();
        self.new_uuids.clear();
        self.generation += 1;

        let bucket = &self.reader.buckets[self.generation & 0b11];
        bucket.len.store(0, Ordering::Release);
        bucket.span_count.store(0, Ordering::Release);

        let mut ref_count = bucket.ref_count.fetch_sub(1, Ordering::Acquire);
        while ref_count != 0 {
            atomic_wait::wait(&bucket.ref_count, ref_count);
            ref_count = bucket.ref_count.load(Ordering::Acquire);
        }
        {
            // TODO: Enforce, that these maps are never read when
            // len is 0, maybe we can use generational index, even/odd
            let mut bucket_maps = bucket.intern_map.lock().unwrap();
            bucket_maps.general.clear();
            bucket_maps.uuid.clear();
            bucket_maps.keys.clear();
        }
        bucket
            .generation
            .store(self.generation as u32, Ordering::Release);
        self.reader
            .generation
            .store(self.generation as u32, Ordering::Release);
        bucket.ref_count.fetch_add(1, Ordering::Release);
    }
    pub fn current_bucket_memory_used(&self) -> BucketMemoryStats {
        let interned_bytes = self.data_used;
        let per_log_bytes = self.logs_used
            * (
                std::mem::size_of::<BloomLine>()
            + std::mem::size_of::<u32>() //span_id
            + std::mem::size_of::<u64>() //timestamp
            + std::mem::size_of::<u32>() //offset
            + std::mem::size_of::<u16>() //target
            + std::mem::size_of::<u16>()
                //msg
            );
        let field_bytes = self.fields_used * std::mem::size_of::<Field>();
        let custom_key_bytes = (self.custom_keys as usize) * std::mem::size_of::<InternedRange>();
        let span_bytes = self.spans_used * std::mem::size_of::<SpanRange>();
        let intern_map_bytes =
            ((self.general_intern_map.len() * std::mem::size_of::<InternedRange>()) * 8) / 7;
        let uuid_map_bytes = ((self.general_intern_map.len() * std::mem::size_of::<u32>()) * 8) / 7;
        let message_bytes = self.msg_map.len() * std::mem::size_of::<InternedRange>();

        BucketMemoryStats {
            log_count: self.logs_used,
            interned: interned_bytes,
            log_metadata: per_log_bytes,
            fields: field_bytes,
            custom_keys: custom_key_bytes,
            span: span_bytes,
            messages: message_bytes,
            general_map: intern_map_bytes,
            uuid_map: uuid_map_bytes,
        }
    }
    pub fn new() -> Index {
        Index {
            fields_used: 0,
            data_used: 0,
            logs_used: 0,
            custom_keys: 0,
            generation: 0,
            spans_used: 0,
            new_ranges: Vec::default(),
            new_uuids: Vec::default(),
            dirty: false,
            logdex_map: HashTable::with_capacity(4096),
            general_intern_map: HashTable::with_capacity(4096),
            uuid_intern_map: HashTable::with_capacity(4096),
            target_map: LocalIntermentCache::default(),
            msg_map: HashTable::with_capacity(4096),
            span_table: HashMap::with_capacity(4096),
            reader: Arc::new(IndexReader::new()),
            chunk_max_utc_ns: u64::MIN,
            chunk_min_utc_ns: u64::MAX,
        }
    }
    pub unsafe fn clear_unchecked(&mut self) {
        self.fields_used = 0;
        self.data_used = 0;
        self.logs_used = 0;
        self.spans_used = 0;
        self.custom_keys = 0;
        self.general_intern_map.clear();
        self.uuid_intern_map.clear();
        self.span_table.clear();
        self.new_ranges.clear();
        self.new_uuids.clear();
        self.generation = 0;
        self.chunk_max_utc_ns = u64::MIN;
        self.chunk_min_utc_ns = u64::MAX;
        self.dirty = false;

        for (i, bucket) in self.reader.buckets.iter().enumerate() {
            unsafe { bucket.clear_unchecked(i as u32) }
        }
    }
    pub fn write<'a, T: Iterator<Item = Result<(Key<'a>, Value<'a>), MunchError>> + Clone>(
        &mut self,
        timestamp: u64,
        level: LogLevel,
        span_info: SpanInfo,
        fields: T,
    ) -> Result<WeakLogEntry, MunchError> {
        // bucket SADNESS

        // Todo make sure a munch error doesn't break anything doesn't
        // as long as we don't commit should just have to reset the counters,
        // Or defer updating the counters until we know the no error is possible.
        if self.write_current_to_bucket(timestamp, level, span_info.clone(), fields.clone())? {
            self.commit();
            return Ok(WeakLogEntry {
                bucket_generation: self.generation as u32,
                index: self.logs_used as u32 - 1,
            });
        }
        // Ran out of storage in this bucket lets move to the next one
        self.complete_bucket();
        if !self.write_current_to_bucket(timestamp, level, span_info, fields)? {
            // Todo make this a proper error
            panic!("Log too large to fit in a single bucket");
        }
        self.commit();
        Ok(WeakLogEntry {
            bucket_generation: self.generation as u32,
            index: self.logs_used as u32 - 1,
        })
    }
    fn write_current_to_bucket<'a>(
        &mut self,
        timestamp: u64,
        level: LogLevel,
        span_info: SpanInfo,
        mut fields: impl Iterator<Item = Result<(Key<'a>, Value<'a>), MunchError>> + Clone,
    ) -> Result<bool, MunchError> {
        if self.logs_used >= BUCKET_LOG_SIZE {
            return Ok(false);
        }
        let bucket = &self.reader.buckets[self.generation & 0b11];
        unsafe {
            bucket
                .timestamp
                .as_ptr()
                .add(self.logs_used)
                .write(timestamp);
        }
        let mut bloom = crate::bloom::BloomBuilder::default();
        let mut msg: &[u8] = b"";
        let mut target: &[u8] = b"";
        let fields_start = self.fields_used;
        for pair in fields {
            let (key, value) = pair?;
            let key_bits = match key {
                kvlog::encoding::Key::Static(key) => {
                    if key == StaticKey::msg {
                        if let Value::String(text) = value {
                            msg = text;
                        }
                        continue;
                    }
                    if key == StaticKey::target {
                        if let Value::String(text) = value {
                            target = text;
                        }
                        continue;
                    }

                    key as u16
                }
                kvlog::encoding::Key::Dynamic(key) => {
                    //todo handle utf8?
                    KeyID::intern(std::str::from_utf8(key).unwrap()).raw()
                }
            };
            // println!("{:?}", value);
            let (kind, value_bits) = match value {
                Value::String(text) => {
                    let Some(interned) = self.intern(text) else {
                        return Ok(false);
                    };
                    (FieldKind::String, interned.field_mask())
                }
                Value::Bytes(bytes) => {
                    let Some(interned) = self.intern(bytes) else {
                        return Ok(false);
                    };
                    (FieldKind::Bytes, interned.field_mask())
                }
                Value::I32(num) => (FieldKind::I48, i48::from_i64(num as i64)),
                Value::U32(num) => (FieldKind::I48, i48::from_i64(num as i64)),
                Value::I64(num) => {
                    if let Some(value) = i48::try_from_i64(num) {
                        (FieldKind::I48, value)
                    } else {
                        let Some(interned) = self.intern(&num.to_ne_bytes()) else {
                            return Ok(false);
                        };
                        (FieldKind::I64, interned.field_mask())
                    }
                }
                Value::U64(num) => {
                    if let Ok(signed) = i64::try_from(num) {
                        if let Some(value) = i48::try_from_i64(signed) {
                            (FieldKind::I48, value)
                        } else {
                            let Some(interned) = self.intern(&signed.to_ne_bytes()) else {
                                return Ok(false);
                            };
                            (FieldKind::I64, interned.field_mask())
                        }
                    } else {
                        let Some(interned) = self.intern(&num.to_ne_bytes()) else {
                            return Ok(false);
                        };
                        (FieldKind::U64, interned.field_mask())
                    }
                }
                Value::F32(float) => (FieldKind::F32, float.to_bits() as u64),
                Value::F64(float) => (FieldKind::F32, (float as f32).to_bits() as u64),
                Value::UUID(id) => {
                    let Some(interned) = self.intern_uuid(&id) else {
                        return Ok(false);
                    };
                    (FieldKind::UUID, interned as u64)
                }
                Value::Bool(value) => (FieldKind::Bool, value as u64),
                Value::None => (FieldKind::None, 0),
                Value::Seconds(float) => (FieldKind::Seconds, (float).to_bits() as u64),
                Value::Timestamp(ts) => {
                    // TODO have external time storage for things that don't fit.
                    // At first this seems quite bad, but atleast with the main user of kvlog
                    // they use millisecond position timestamps, and 48 bits will captures over a thousand years after epoch
                    let x = ts.as_millisecond_clamped().clamp(0, (1 << 48) - 1) as u64;
                    (FieldKind::Timestamp, x)
                }
            };
            let field = Field::new(key_bits, kind, value_bits);
            bloom.insert(field);
            if self.fields_used > BUCKET_FIELD_SIZE {
                return Ok(false);
            }
            unsafe {
                let bucket = &self.reader.buckets[self.generation & 0b11];
                bucket.field.as_ptr().add(self.fields_used).write(field);
            }
            self.fields_used += 1;
        }
        //todo don't require message and target
        let Some(msg_intern) = self.intern_msg(msg) else {
            return Ok(false);
        };
        let bucket = &self.reader.buckets[self.generation & 0b11];
        let target_intern = self.target_map.intern(&self.reader.targets, target);
        let fields = unsafe {
            std::slice::from_raw_parts_mut(
                bucket.field.as_ptr().add(fields_start),
                self.fields_used - fields_start,
            )
        };
        let dex = Archetype::new(
            msg_intern,
            target_intern,
            level,
            !span_info.is_none(),
            &fields,
        );
        let dex_id = {
            let hash = bucket.random_state.hash_one(&dex);
            let len = self.logdex_map.len();
            let entry = self.logdex_map.entry(
                hash,
                |v| unsafe { bucket.archetype(*v) == &dex },
                |t| bucket.random_state.hash_one(*t),
            );
            match entry {
                hashbrown::hash_table::Entry::Occupied(entry) => *entry.get(),
                hashbrown::hash_table::Entry::Vacant(entry) => {
                    if len >= BUCKET_ARCHETYPE_SIZE {
                        return Ok(false);
                    }
                    self.dirty = true;
                    unsafe { bucket.archetype.as_ptr().add(len).write(dex) }
                    entry.insert(len as u16);
                    len as u16
                }
            }
        };
        unsafe {
            bucket
                .archetype_index
                .as_ptr()
                .add(self.logs_used)
                .write(dex_id);
        }

        /// todod
        let fields = self.fields_used - fields_start;
        if fields > 1 {
            unsafe {
                let fields =
                    std::slice::from_raw_parts_mut(bucket.field.as_ptr().add(fields_start), fields);
                fields.sort_unstable_by_key(|f| f.raw);
            }
        }
        unsafe {
            bucket
                .bloom
                .as_ptr()
                .add(self.logs_used)
                .write(bloom.line(level));
        }
        unsafe {
            bucket
                .target
                .as_ptr()
                .add(self.logs_used)
                .write(target_intern);
        }
        // unsafe {
        //     bucket
        //         .msg
        //         .as_ptr()
        //         .add(self.logs_used)
        //         .write(msg_intern.data);
        // }

        let mut end_mask = 0;
        let mut start_mask = 0;
        let mut span_parent = None;
        let span = match span_info {
            SpanInfo::Start { span, parent } => {
                start_mask = 1u32 << 31;
                span_parent = parent;
                Some(span)
            }
            SpanInfo::Current { span } => Some(span),
            SpanInfo::End { span } => {
                end_mask = 1u32 << 31;
                Some(span)
            }
            SpanInfo::None => None,
        };
        if let Some(span) = span {
            use hashbrown::hash_map::Entry;
            let span_index = match self.span_table.entry(span) {
                Entry::Occupied(entry) => {
                    let id = *entry.get();
                    let span_range = unsafe { &*bucket.span_data.as_ptr().add(id as usize) };
                    span_range
                        .last_mask
                        .store((self.logs_used as u32) | end_mask, Ordering::Release);
                    id
                }
                Entry::Vacant(entry) => {
                    let id = self.spans_used as u32;
                    entry.insert(id);
                    if self.spans_used >= BUCKET_SPAN_RANGE_SIZE {
                        return Ok(false);
                    }
                    self.spans_used += 1;
                    unsafe {
                        bucket.span_data.as_ptr().add(id as usize).write(SpanRange {
                            id: span,
                            parent: span_parent,
                            first_mask: (self.logs_used as u32) | start_mask,
                            last_mask: AtomicU32::new((self.logs_used as u32) | end_mask),
                        })
                    };
                    id
                }
            };
            unsafe {
                bucket
                    .span_index
                    .as_ptr()
                    .add(self.logs_used)
                    .write(span_index);
            }
        } else {
            unsafe {
                bucket
                    .span_index
                    .as_ptr()
                    .add(self.logs_used)
                    .write(u32::MAX);
            }
        }
        self.chunk_min_utc_ns = timestamp.min(self.chunk_min_utc_ns);
        self.chunk_max_utc_ns = timestamp.max(self.chunk_max_utc_ns);

        if self.logs_used & (TIME_RANGE_LOG_COUNT - 1) == (TIME_RANGE_LOG_COUNT - 1) {
            unsafe {
                bucket
                    .timerange
                    .as_ptr()
                    .add(self.logs_used >> (TIME_RANGE_LOG_COUNT.trailing_zeros()))
                    .write(TimeRange {
                        min_utc_ns: self.chunk_min_utc_ns,
                        max_utc_nc: self.chunk_max_utc_ns,
                    });
            }
            self.chunk_max_utc_ns = u64::MIN;
            self.chunk_min_utc_ns = u64::MAX;
        }
        self.logs_used += 1;
        unsafe {
            bucket
                .offset
                .as_ptr()
                .add(self.logs_used)
                .write(self.fields_used as u32);
        }

        Ok(true)
    }
    fn commit(&mut self) {
        let bucket = &self.reader.buckets[self.generation & 0b11];
        if self.dirty || !self.new_ranges.is_empty() || !self.new_uuids.is_empty() {
            let mut map = bucket.intern_map.lock().unwrap();
            for (range, hash) in &self.new_ranges {
                map.general.insert_unique(*hash, *range, |t| {
                    bucket
                        .random_state
                        .hash_one(unsafe { bucket.data_unchecked(*t) })
                });
            }
            for (range, hash) in &self.new_uuids {
                map.uuid.insert_unique(*hash, *range, |t| {
                    bucket
                        .random_state
                        .hash_one(unsafe { bucket.uuid_bytes(*t) })
                });
            }
            map.msgs = self.msg_map.len();
            self.new_ranges.clear();
            self.new_uuids.clear();
            self.dirty = false;
            bucket
                .archetype_count
                .store(self.logdex_map.len(), Ordering::Release);
        }
        bucket.span_count.store(self.spans_used, Ordering::Release);
        bucket.len.store(self.logs_used, Ordering::Release);
    }
    fn intern_uuid(&mut self, id: &uuid::Uuid) -> Option<u32> {
        let bucket = &self.reader.buckets[self.generation & 0b11];
        let bytes = id.as_bytes();
        let hash = bucket.random_state.hash_one(bytes);
        let entry = self.uuid_intern_map.entry(
            hash,
            |v| unsafe { bucket.uuid_bytes(*v) == bytes },
            |t| {
                bucket
                    .random_state
                    .hash_one(unsafe { bucket.uuid_bytes(*t) })
            },
        );
        match entry {
            hashbrown::hash_table::Entry::Occupied(entry) => Some(*entry.get()),
            hashbrown::hash_table::Entry::Vacant(entry) => {
                if bytes.len() + self.data_used > BUCKET_DATA_SIZE {
                    return None;
                }
                let id = self.data_used as u32;
                self.new_uuids.push((id, hash));
                unsafe {
                    bucket
                        .data
                        .as_ptr()
                        .add(self.data_used)
                        .copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
                }
                entry.insert(id);
                self.data_used += bytes.len();
                Some(id)
            }
        }
    }

    fn intern_msg(&mut self, bytes: &[u8]) -> Option<InternedRange> {
        let bucket = &self.reader.buckets[self.generation & 0b11];
        let hash = bucket.random_state.hash_one(bytes);
        let len = self.msg_map.len();
        let entry = self.msg_map.entry(
            hash,
            |v| unsafe { bucket.data_unchecked(*v) == bytes },
            |t| {
                bucket
                    .random_state
                    .hash_one(unsafe { bucket.data_unchecked(*t) })
            },
        );
        match entry {
            hashbrown::hash_table::Entry::Occupied(entry) => Some(*entry.get()),
            hashbrown::hash_table::Entry::Vacant(entry) => {
                if bytes.len() + self.data_used > BUCKET_DATA_SIZE {
                    return None;
                }
                if len >= BUCKET_MAX_MSG_SIZE {
                    return None;
                }
                let range = InternedRange {
                    offset: self.data_used as u32,
                    data: len as u16,
                    len: bytes.len() as u16,
                };
                // unsafe {
                //     bucket.msg_ptr.as_ptr().add(len).write(range);
                // }
                self.dirty = true;
                unsafe {
                    bucket
                        .data
                        .as_ptr()
                        .add(self.data_used)
                        .copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
                }
                entry.insert(range);
                self.data_used += bytes.len();
                Some(range)
            }
        }
    }

    fn intern(&mut self, bytes: &[u8]) -> Option<InternedRange> {
        if bytes.is_empty() {
            return Some(InternedRange {
                len: 0,
                data: 0,
                offset: 0,
            });
        }
        let bucket = &self.reader.buckets[self.generation & 0b11];
        let hash = bucket.random_state.hash_one(bytes);
        let entry = self.general_intern_map.entry(
            hash,
            |v| unsafe { v.data == 0 && bucket.data_unchecked(*v) == bytes },
            |t| {
                bucket
                    .random_state
                    .hash_one(unsafe { bucket.data_unchecked(*t) })
            },
        );
        match entry {
            hashbrown::hash_table::Entry::Occupied(entry) => Some(*entry.get()),
            hashbrown::hash_table::Entry::Vacant(entry) => {
                if bytes.len() + self.data_used > BUCKET_DATA_SIZE {
                    return None;
                }
                let range = InternedRange {
                    offset: self.data_used as u32,
                    data: 0,
                    len: bytes.len() as u16,
                };
                self.new_ranges.push((range, hash));
                unsafe {
                    bucket
                        .data
                        .as_ptr()
                        .add(self.data_used)
                        .copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
                }
                entry.insert(range);
                self.data_used += bytes.len();
                Some(range)
            }
        }
    }
}
mod i48 {
    pub const MIN: i64 = -(1i64 << 47);
    pub const MAX: i64 = (1i64 << 47) - 1;
    const MASK: u64 = (1u64 << 48) - 1;
    fn sign_extend_i48(x: i64) -> i64 {
        let notherbits = std::mem::size_of_val(&x) as u32 * 8 - 48;
        x.wrapping_shl(notherbits).wrapping_shr(notherbits)
    }
    pub fn try_from_i64(value: i64) -> Option<u64> {
        if value >= MIN && value <= MAX {
            Some(from_i64(value))
        } else {
            None
        }
    }
    pub fn from_i64(value: i64) -> u64 {
        (value as u64) & MASK
    }
    pub fn to_i64(value: u64) -> i64 {
        sign_extend_i48(((value) & MASK) as i64)
    }
    #[cfg(test)]
    mod test {
        use super::*;
        #[test]
        fn i48_mapping() {
            for i in [MIN, MAX, -1, 43, 683, 0] {
                assert_eq!(i, to_i64(try_from_i64(i).unwrap()))
            }
        }
    }
}

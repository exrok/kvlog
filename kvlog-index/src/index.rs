use crate::{
    field_table::{KeyID, KeyMap},
    query::{query_parts::FieldKey, QueryExpr},
    shared_interner::{LocalIntermentCache, SharedIntermentBuffer},
};
use ahash::RandomState;
use archetype::ServiceId;
use core::time;
use f60::{f60_to_f64, f64_to_f60, F60};
pub use filter::{EntryCollection, Query};
use hashbrown::HashTable;
use hashbrown::{HashMap, HashSet};
use jsony::Jsony;
use smallvec::SmallVec;
pub mod f60;
use kvlog::{
    encoding::{Key, LogFields, MunchError, SpanInfo, StaticKey, Value},
    LogLevel, SpanID,
};
use std::{
    collections::VecDeque,
    fmt::Write,
    os::raw::c_void,
    ptr::NonNull,
    sync::{
        atomic::{AtomicU32, AtomicUsize, Ordering},
        Arc, Mutex, MutexGuard,
    },
    time::{SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

use crate::persistence::encoder::{self as stream_encoder, BucketLogStreamEncoder};
use crate::persistence::{DrainedLogBuffer, IndexConfig};
pub mod archetype;
pub use archetype::Archetype;

#[cfg(test)]
pub(crate) mod test;

use self::filter::{ForwardQueryWalker, ForwardSegmentWalker, QueryStrategy};
pub use self::filter::{GeneralFilter, GeneralQuery, ReverseQueryWalker, SpanCache, TimeFilter};
pub mod filter;
mod table_bucket;
pub use table_bucket::{
    BucketSnapshotScratch, BucketSnapshotSlices, GlobalInternError, LoadedBucket, PersistentInterners,
    SnapshotLoadError, SnapshotWriteError, ValidationMode,
};
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

// both query and field are given in sorted order
// the query inputs map to output_indices by index query[X] -> output_indices[X]
// output_indices contains the index of the field that has matching value of the query
fn inputs(query: &[u16; 4], fields: &[u16], output_indices: &mut [u16; 4]) {}

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

const FIELD_VALUE_MASK: u64 = (1u64 << 60) - 1;
const _: () = assert!(std::mem::size_of::<Field>() == 8);

#[derive(Debug, Jsony, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
#[jsony(ToStr)]
pub enum FieldKind {
    None = 0,
    String = 1,
    Bytes = 2,
    I60 = 3,
    I64 = 4,
    U64 = 5,
    F60 = 6,
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
    pub(crate) fn kind(&self) -> FieldKind {
        unsafe { std::mem::transmute(((self.raw >> 60) as u8) & 0xf) }
    }

    pub fn new(kind: FieldKind, value_mask: u64) -> Field {
        debug_assert_eq!(value_mask & !FIELD_VALUE_MASK, 0);
        Field { raw: ((kind as u64) << 60) | value_mask }
    }
    pub(crate) fn value_mask(&self) -> u64 {
        self.raw & FIELD_VALUE_MASK
    }
    pub(crate) unsafe fn as_f64<'a>(self, bucket: &'a Bucket) -> Option<f64> {
        match self.value(bucket) {
            Value::I64(value) => Some(value as f64),
            Value::U64(value) => Some(value as f64),
            Value::F64(value) => Some(value),
            _ => None,
        }
    }
    pub(crate) unsafe fn as_i64<'a>(self, bucket: &'a Bucket) -> Option<i64> {
        match self.value(bucket) {
            Value::I64(value) => Some(value),
            Value::F64(value) => Some(value as i64),
            _ => None,
        }
    }
    pub(crate) unsafe fn as_required_u64<'a>(self, bucket: &'a Bucket) -> Option<u64> {
        match self.value(bucket) {
            Value::U64(value) => {
                let value = unsafe { bucket.u64_unchecked(self.value_mask() as u32) };
                Some(value)
            }
            _ => None,
        }
    }
    pub(crate) unsafe fn as_bytes<'a>(self, bucket: &'a Bucket) -> Option<&'a [u8]> {
        match self.kind() {
            FieldKind::String | FieldKind::Bytes => {
                let range = InternedRange::from_field_mask(self.value_mask());
                Some(unsafe { bucket.data_unchecked(range) })
            }
            _ => None,
        }
    }
    pub(crate) unsafe fn as_text<'a>(self, bucket: &'a Bucket) -> Option<&'a [u8]> {
        match self.kind() {
            FieldKind::String => {
                let range = InternedRange::from_field_mask(self.value_mask());
                Some(unsafe { bucket.data_unchecked(range) })
            }
            _ => None,
        }
    }
    pub(crate) fn as_bool<'a>(self) -> Option<bool> {
        match self.kind() {
            FieldKind::Bool => Some((self.value_mask() as u8) != 0),
            _ => None,
        }
    }
    pub(crate) fn as_timestamp_ns<'a>(self) -> Option<i64> {
        match self.kind() {
            FieldKind::Timestamp => Some((self.value_mask() as i64).saturating_mul(1000_000)),
            _ => None,
        }
    }
    pub(crate) fn as_raw_f60_seconds<'a>(self) -> Option<F60> {
        match self.kind() {
            FieldKind::Seconds => Some(self.value_mask()),
            _ => None,
        }
    }
    pub(crate) unsafe fn value<'a>(self, bucket: &'a Bucket) -> Value<'a> {
        match self.kind() {
            FieldKind::String => {
                let range = InternedRange::from_field_mask(self.value_mask());
                Value::String(unsafe { bucket.data_unchecked(range) })
            }
            FieldKind::Bytes => {
                let range = InternedRange::from_field_mask(self.value_mask());
                Value::Bytes(unsafe { bucket.data_unchecked(range) })
            }
            FieldKind::I60 => Value::I64(i60::to_i64(self.value_mask())),
            FieldKind::I64 => {
                let value = unsafe { bucket.u64_unchecked(self.value_mask() as u32) };
                Value::I64(value as i64)
            }
            FieldKind::U64 => {
                let value = unsafe { bucket.u64_unchecked(self.value_mask() as u32) };
                Value::U64(value as u64)
            }
            FieldKind::F60 => Value::F64(f60_to_f64(self.value_mask())),
            FieldKind::Seconds => Value::Seconds(f60_to_f64(self.value_mask()) as f32),
            FieldKind::UUID => {
                Value::UUID(uuid::Uuid::from_bytes(unsafe { *bucket.uuid_bytes(self.value_mask() as u32) }))
            }
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
        f.write_str(self.kind().to_str())?;
        f.write_char('(');
        match self.kind() {
            FieldKind::String | FieldKind::Bytes => {
                let range = InternedRange::from_field_mask(self.value_mask());
                write!(f, "{{len: {}, offset: {}}}", range.len, range.offset)?;
            }
            FieldKind::I60 => {
                write!(f, "{}", i60::to_i64(self.value_mask()))?;
            }
            FieldKind::I64 => {
                write!(f, "{{offset: {}}}", self.value_mask())?;
            }
            FieldKind::U64 => {
                write!(f, "{{offset: {}}}", self.value_mask())?;
            }
            FieldKind::F60 => {
                write!(f, "{}", f60::f60_to_f64(self.value_mask()))?;
            }
            FieldKind::Bool => {
                write!(f, "{}", (self.value_mask() as u8) != 0)?;
            }
            FieldKind::UUID => {
                write!(f, "{}", self.value_mask() as u32)?;
            }
            FieldKind::Seconds => {
                write!(f, "{}", f60::f60_to_f64(self.value_mask()))?;
            }
            FieldKind::Timestamp => {
                let time = kvlog::Timestamp::from_millisecond(self.value_mask() as i64);
                write!(f, "{:?}", time)?;
            }
            _ => {}
        }
        f.write_char(')')
    }
}

impl From<u8> for FieldKind {
    fn from(value: u8) -> Self {
        unsafe { std::mem::transmute(value & 0xf) }
    }
}

fn sort_fields_by_key(fields: &mut [Field], keys: &mut [u16]) {
    debug_assert_eq!(fields.len(), keys.len());
    for i in 1..fields.len() {
        let key = keys[i];
        let field = fields[i];
        let mut j = i;
        while j > 0 && (keys[j - 1] > key || (keys[j - 1] == key && fields[j - 1].raw > field.raw)) {
            keys[j] = keys[j - 1];
            fields[j] = fields[j - 1];
            j -= 1;
        }
        keys[j] = key;
        fields[j] = field;
    }
}

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

/// Initial capacity reserved for a freshly-allocated active log buffer when
/// persistence is enabled. The caller is expected to swap in a larger
/// pre-sized buffer via [`Index::swap_log_buffer`].
const BUCKET_LOG_BUFFER_INITIAL_CAPACITY: usize = 4 * 1024;

/// Current wall-clock time in nanoseconds since the Unix epoch. Used to stamp
/// the close timestamp into a finalized log buffer. Returns 0 if the system
/// clock is set before the epoch.
fn system_time_now_ns() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

#[derive(Debug)]
#[repr(C)]
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
        (self.first_mask & 0x7FFF_FFFF)..((self.last_mask.load(Ordering::Relaxed) & 0x7FFF_FFFF) + 1)
    }
    fn true_index_range(&self) -> std::ops::Range<u32> {
        (self.first_mask & 0x7FFF_FFFF)..(self.last_mask.load(Ordering::Relaxed) & 0x7FFF_FFFF)
    }
    pub fn span_info(&self, entry: u32) -> SpanInfo {
        if self.first_mask ^ entry == (1u32 << 31) {
            SpanInfo::Start { span: self.id, parent: self.parent }
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
    pub(crate) keys: KeyMap<KeyInfo>,
    indexed: usize,
    msgs: usize,
    data_len: usize,
}
impl IntermentMaps {
    pub fn keys(&self) -> impl Iterator<Item = (KeyID, &KeyInfo)> + '_ {
        self.keys.iter().filter(|(_, info)| info.total > 0)
    }
    fn index_field_value(&mut self, bucket: &Bucket, field: Field) {
        match field.kind() {
            FieldKind::String | FieldKind::Bytes => {
                let range = InternedRange::from_field_mask(field.value_mask());
                if range.len == 0 && range.offset == 0 {
                    return;
                }
                let hash = bucket.random_state.hash_one(unsafe { bucket.data_unchecked(range) });
                let entry = self.general.entry(
                    hash,
                    |existing| same_range(*existing, range),
                    |existing| bucket.random_state.hash_one(unsafe { bucket.data_unchecked(*existing) }),
                );
                if let hashbrown::hash_table::Entry::Vacant(entry) = entry {
                    entry.insert(range);
                }
            }
            FieldKind::I64 | FieldKind::U64 => {
                let range = InternedRange { offset: field.value_mask() as u32, data: 0, len: 8 };
                let hash = bucket.random_state.hash_one(unsafe { bucket.data_unchecked(range) });
                let entry = self.general.entry(
                    hash,
                    |existing| same_range(*existing, range),
                    |existing| bucket.random_state.hash_one(unsafe { bucket.data_unchecked(*existing) }),
                );
                if let hashbrown::hash_table::Entry::Vacant(entry) = entry {
                    entry.insert(range);
                }
            }
            FieldKind::UUID => {
                let offset = field.value_mask() as u32;
                let hash = bucket.random_state.hash_one(unsafe { bucket.uuid_bytes(offset) });
                let entry = self.uuid.entry(
                    hash,
                    |existing| *existing == offset,
                    |existing| bucket.random_state.hash_one(unsafe { bucket.uuid_bytes(*existing) }),
                );
                if let hashbrown::hash_table::Entry::Vacant(entry) = entry {
                    entry.insert(offset);
                }
            }
            _ => {}
        }
    }
    pub fn field_u64(&self, bucket: &BucketGuard, key: KeyID, value: u64) -> Option<Field> {
        if let Ok(signed) = i64::try_from(value) {
            return self.field_i64(bucket, key, signed);
        }
        let value = &value.to_ne_bytes();
        let hash = bucket.bucket.random_state.hash_one(value);
        let range = *self.general.find(hash, |t| unsafe { bucket.bucket.data_unchecked(*t) == value })?;
        let (kind, mask) = (FieldKind::U64, range.field_mask());
        let field = Field::new(kind, mask);
        Some(field)
    }
    pub fn field_i64(&self, bucket: &BucketGuard, key: KeyID, value: i64) -> Option<Field> {
        let (kind, mask) = if let Some(value) = i60::try_from_i64(value) {
            (FieldKind::I60, value)
        } else {
            let value = &value.to_ne_bytes();
            let hash = bucket.bucket.random_state.hash_one(value);
            let range = *self.general.find(hash, |t| unsafe { bucket.bucket.data_unchecked(*t) == value })?;
            (FieldKind::I64, range.field_mask())
        };
        let field = Field::new(kind, mask);
        Some(field)
    }
    pub fn field_text(&self, bucket: &BucketGuard, key: KeyID, value: &[u8]) -> Option<Field> {
        let hash = bucket.bucket.random_state.hash_one(value);
        let index = *self.general.find(hash, |t| unsafe { bucket.bucket.data_unchecked(*t) == value })?;
        let field = Field::new(FieldKind::String, index.field_mask());
        Some(field)
    }
    pub fn field_uuid(&self, bucket: &BucketGuard, key: KeyID, value: Uuid) -> Option<Field> {
        let index = *self.uuid.find(bucket.bucket.random_state.hash_one(value.as_bytes()), |t| unsafe {
            bucket.bucket.uuid_bytes(*t) == value.as_bytes()
        })?;
        Some(Field::new(FieldKind::UUID, index as u64))
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct WeakLogEntry {
    pub(crate) index: u32,
    bucket_generation: u32,
}

impl WeakLogEntry {
    pub fn new(bucket_generation: u32, index: u32) -> WeakLogEntry {
        WeakLogEntry { index, bucket_generation }
    }
    pub fn bucket_generation(&self) -> u32 {
        self.bucket_generation
    }
    pub fn index(&self) -> u32 {
        self.index
    }
}

pub struct Bucket {
    generation: AtomicU32,
    intern_map: Mutex<IntermentMaps>,
    data: NonNull<u8>,
    pub(crate) random_state: RandomState,
    field: NonNull<Field>,

    timerange: NonNull<TimeRange>,

    span_index: NonNull<u32>,
    span_data: NonNull<SpanRange>,
    timestamp: NonNull<u64>,
    offset: NonNull<u32>,
    span_count: AtomicUsize,
    len: AtomicUsize,
    ref_count: AtomicU32,

    archetype: NonNull<archetype::Archetype>,
    archetype_index: NonNull<u16>,
    archetype_count: AtomicUsize,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct LogEntry<'a> {
    pub bucket: &'a Bucket,
    pub index: u32,
}
#[derive(Clone)]
pub struct Fields<'a> {
    fields: &'a [Field],
    keys: &'a [KeyID],
    bucket: &'a Bucket,
}

impl<'a> std::fmt::Debug for Fields<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.clone().into_iter().map(|(key, value)| (key, value))).finish()
    }
}
impl<'a> Iterator for Fields<'a> {
    type Item = (KeyID, Value<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        let [field, rest @ ..] = self.fields else {
            return None;
        };
        let [key, rest_keys @ ..] = self.keys else {
            return None;
        };
        self.fields = rest;
        self.keys = rest_keys;
        Some(unsafe { (*key, field.value(self.bucket)) })
    }
}
impl<'a> LogEntry<'a> {
    pub fn weak(&self) -> WeakLogEntry {
        WeakLogEntry { index: self.index as u32, bucket_generation: self.bucket.generation.load(Ordering::Relaxed) }
    }
    pub fn bucket(&self) -> &Bucket {
        self.bucket
    }
    pub fn timestamp(&self) -> u64 {
        unsafe { *self.bucket.timestamp.as_ptr().add(self.index as usize) }
    }
    pub fn target_id(&self) -> u16 {
        self.archetype().target_id
    }
    pub fn archetype(&self) -> &archetype::Archetype {
        unsafe { self.bucket.archetype(*self.bucket.archetype_index.as_ptr().add(self.index as usize)) }
    }
    pub fn message(&self) -> &[u8] {
        let archetype = self.archetype();
        let intern_range = InternedRange { offset: archetype.msg_offset, data: 0, len: archetype.msg_len };
        unsafe { self.bucket.data_unchecked(intern_range) }
    }
    pub fn raw_archetype(&self) -> u16 {
        unsafe { *self.bucket.archetype_index.as_ptr().add(self.index as usize) }
    }
    pub fn level_mask(&self) -> u8 {
        self.archetype().mask as u8 & 0xf
    }
    pub fn level(&self) -> LogLevel {
        self.archetype().level()
    }
    pub fn field_by_dyn_key(&self, key: FieldKey) -> Option<Field> {
        match key {
            FieldKey::New(name) => self.field_by_key_name(name),
            FieldKey::Seen(id) => self.field_by_key_id(id),
        }
    }
    pub fn field_by_key_name(&self, name: &str) -> Option<Field> {
        self.field_by_key_id(KeyID::try_from_str(name)?)
    }
    pub fn field_by_key_id(&self, key: KeyID) -> Option<Field> {
        let index = self.archetype().index_of(key)?;
        unsafe { Some(*self.get_field_unchecked(index)) }
    }
    pub fn get_field_unchecked(&self, index: usize) -> &Field {
        let start = unsafe { *self.bucket.offset.as_ptr().add(self.index as usize) };
        let field = self.bucket.field.as_ptr();
        unsafe { &*field.add(start as usize + index) }
    }
    pub fn raw_fields(&self) -> &[Field] {
        let start = unsafe { *self.bucket.offset.as_ptr().add(self.index as usize) };
        let end = unsafe { *self.bucket.offset.as_ptr().add(self.index as usize + 1) };
        let field = self.bucket.field.as_ptr();
        unsafe { std::slice::from_raw_parts(field.add(start as usize), (end - start) as usize) }
    }
    /// Not inefficent for long running spans and will be incomplete when
    /// crossing buckets
    pub fn span_logs(&self) -> impl Iterator<Item = LogEntry<'_>> + '_ {
        let range = if let Some(range) = self.span_range() { range.index_range() } else { 0..0 };
        let span_index = unsafe { *self.bucket.span_index.as_ptr().add(self.index as usize) };
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
            .map(move |(index, _)| LogEntry { index: (index + range.start as usize) as u32, bucket: self.bucket })
    }
    pub fn fields(&self) -> Fields<'_> {
        Fields { fields: self.raw_fields(), keys: self.archetype().field_keys(), bucket: self.bucket }
    }
    pub fn raw_span_id(&self) -> u32 {
        let span_index = unsafe { *self.bucket.span_index.as_ptr().add(self.index as usize) };
        span_index
    }
    pub fn span_ns_duration(&self) -> Option<u64> {
        let span_index = unsafe { *self.bucket.span_index.as_ptr().add(self.index as usize) };
        let range = self.span_range()?.true_index_range();
        unsafe {
            let start = self.bucket.timestamp.add(range.start as usize).read();
            let end = self.bucket.timestamp.add(range.end as usize).read();
            Some(end.abs_diff(start))
        }
    }
    pub fn span_range(&self) -> Option<&SpanRange> {
        let span_index = unsafe { *self.bucket.span_index.as_ptr().add(self.index as usize) };
        if span_index == u32::MAX {
            return None;
        }
        Some(unsafe { &*self.bucket.span_data.as_ptr().add(span_index as usize) })
    }
    pub fn parent_span_id(&self) -> Option<SpanID> {
        if let Some(range) = self.span_range() {
            range.parent
        } else {
            None
        }
    }
    pub fn span_id(&self) -> Option<SpanID> {
        if let Some(range) = self.span_range() {
            Some(range.id)
        } else {
            None
        }
    }
    pub fn span_info(&self) -> SpanInfo {
        if let Some(range) = self.span_range() {
            range.span_info(self.index as u32)
        } else {
            SpanInfo::None
        }
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub struct FieldKindSet {
    pub raw: u16,
}

impl std::fmt::Debug for FieldKindSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.raw == u16::MAX {
            f.write_str("ALL")
        } else {
            f.debug_set().entries(self.iter()).finish()
        }
    }
}
impl From<FieldKind> for FieldKindSet {
    fn from(kind: FieldKind) -> Self {
        let mut set = FieldKindSet::empty();
        set.insert(kind);
        set
    }
}
impl FieldKindSet {
    pub const NUMBERS: Self = FieldKindSet {
        raw: (1 << FieldKind::I60 as u8)
            | (1 << FieldKind::I64 as u8)
            | (1 << FieldKind::U64 as u8)
            | (1 << FieldKind::F60 as u8),
    };
    pub const fn empty() -> Self {
        FieldKindSet { raw: 0 }
    }
    pub const fn all() -> Self {
        FieldKindSet { raw: u16::MAX }
    }
    pub fn extend(&mut self, set: FieldKindSet) {
        self.raw |= set.raw;
    }
    pub fn contains_any(&self, set: FieldKindSet) -> bool {
        self.raw & set.raw != 0
    }
    pub fn insert(&mut self, kind: FieldKind) {
        self.raw |= 1 << kind as u8;
    }
    pub fn contains(&self, kind: FieldKind) -> bool {
        self.raw & (1 << kind as u8) != 0
    }
    pub fn iter(&self) -> impl Iterator<Item = FieldKind> {
        let value = self.raw;
        (0..16).filter(move |i| (value >> i) & 1 == 1).map(|i| FieldKind::from(i))
    }
}

#[derive(Default)]
pub struct KeyInfo {
    total: u32,
    pub(crate) kinds: FieldKindSet,
    values: HashTable<Field>,
}
impl KeyInfo {
    pub fn total(&self) -> usize {
        self.total as usize
    }
    pub fn kind_mask(&self) -> u32 {
        self.kinds.raw as u32
    }
    pub fn kinds(&self) -> impl Iterator<Item = FieldKind> {
        self.kinds.iter()
    }
    pub unsafe fn values<'a, 'b: 'a>(
        &'a self,
        bucket: &'b Bucket,
    ) -> impl Iterator<Item = Value<'b>> + 'a + ExactSizeIterator {
        self.values.iter().map(|value| unsafe { value.value(bucket) })
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
                    data_len: 0,
                }),
                random_state: RandomState::new(),
                data: mmap_alloc(BUCKET_DATA_SIZE),
                field: mmap_alloc(BUCKET_FIELD_SIZE),
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
        intern.target_summaries.clear();
        intern.indexed = 0;
        intern.msgs = 0;
        intern.data_len = 0;
    }

    unsafe fn archetype(&self, index: u16) -> &archetype::Archetype {
        unsafe { &*(self.archetype.as_ptr().add(index as usize)) }
    }
    unsafe fn uuid_bytes(&self, index: u32) -> &[u8; 16] {
        unsafe { &*(self.data.as_ptr().add(index as usize) as *const [u8; 16]) }
    }
    unsafe fn u64_unchecked(&self, offset: u32) -> u64 {
        unsafe { (self.data.as_ptr().add(offset as usize) as *const u64).read_unaligned() }
    }
    unsafe fn data_unchecked(&self, data: InternedRange) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr().add(data.offset as usize), data.len as usize) }
    }

    /// Bucket archetype at the given index. Crate-internal accessor used by
    /// the stream encoder.
    pub(crate) unsafe fn archetype_at(&self, index: u16) -> &archetype::Archetype {
        unsafe { &*self.archetype.as_ptr().add(index as usize) }
    }

    /// Archetype id assigned to the log entry at `log_index`.
    pub(crate) unsafe fn archetype_id_at(&self, log_index: u32) -> u16 {
        unsafe { *self.archetype_index.as_ptr().add(log_index as usize) }
    }

    /// Span id assigned to the log entry, or `u32::MAX` if no span.
    pub(crate) unsafe fn span_index_at(&self, log_index: u32) -> u32 {
        unsafe { *self.span_index.as_ptr().add(log_index as usize) }
    }

    /// SpanRange at the given span id.
    pub(crate) unsafe fn span_range_at(&self, span_id: u32) -> &SpanRange {
        unsafe { &*self.span_data.as_ptr().add(span_id as usize) }
    }

    /// Field slice for the log entry at `log_index`.
    pub(crate) unsafe fn fields_at(&self, log_index: u32) -> &[Field] {
        let start = unsafe { *self.offset.as_ptr().add(log_index as usize) } as usize;
        let end = unsafe { *self.offset.as_ptr().add(log_index as usize + 1) } as usize;
        unsafe { std::slice::from_raw_parts(self.field.as_ptr().add(start), end - start) }
    }

    /// Read interned bytes for a `[InternedRange]`. Used by the stream encoder.
    pub(crate) unsafe fn data_bytes(&self, range: InternedRange) -> &[u8] {
        unsafe { self.data_unchecked(range) }
    }

    /// Read 16 UUID bytes at the given data offset.
    pub(crate) unsafe fn uuid_at(&self, offset: u32) -> &[u8; 16] {
        unsafe { self.uuid_bytes(offset) }
    }

    /// Read an unaligned u64 at the given data offset (used for FieldKind::I64/U64).
    pub(crate) unsafe fn u64_at(&self, offset: u32) -> u64 {
        unsafe { self.u64_unchecked(offset) }
    }
    pub fn read(&self) -> Option<BucketGuard<'_>> {
        if self.len.load(Ordering::Acquire) == 0 {
            return None;
        }

        self.ref_count.fetch_add(1, Ordering::AcqRel);
        let new_len = self.len.load(Ordering::Acquire);
        if new_len == 0 {
            if self.ref_count.fetch_sub(1, Ordering::AcqRel) == 1 {
                atomic_wait::wake_one(&self.ref_count);
            }
            return None;
        }
        let span_count = self.span_count.load(Ordering::Acquire);
        let archetype_count = self.archetype_count.load(Ordering::Acquire);
        // ensure that generation, loaded so we can used
        // relaxed later, while ref_count == 0 generation will
        // not change
        let _ = self.generation.load(Ordering::Acquire);
        Some(BucketGuard { bucket: self, len: new_len, span_count, archetype_count })
    }
}

pub struct SavedEntryIndexReader<'e, 'b> {
    generation_guard: GenerationGuard<'b>,
    entries: &'e [WeakLogEntry],
}
impl<'e, 'b> SavedEntryIndexReader<'e, 'b> {
    pub unsafe fn project_alive(entries: &'e mut Vec<WeakLogEntry>, index: &'b IndexReader) -> Self {
        let gg = index.generation_guard();
        entries.retain(|entry| gg.is_alive(*entry));
        SavedEntryIndexReader { generation_guard: gg, entries: &*entries }
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
                    summary.resize(i + 1, TargetSummary { level_counts: [0; 4] });
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
    span_count: usize,
    archetype_count: usize,
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
            && (log.index as usize) < self.len
    }
    pub fn renew(&mut self) {
        self.len = self.bucket.len.load(Ordering::Acquire);
        self.span_count = self.bucket.span_count.load(Ordering::Acquire);
        self.archetype_count = self.bucket.archetype_count.load(Ordering::Acquire);
    }
    pub fn reverse_time_range_skip(&self, offset: u32, range: TimeFilter) -> u32 {
        if self.len & (!(TIME_RANGE_LOG_COUNT - 1)) <= offset as usize {
            return offset;
        }
        let mut current = offset.saturating_sub(1) >> TIME_RANGE_LOG_MASK_SIZE;
        loop {
            let chunk_range = unsafe { &*self.bucket.timerange.as_ptr().add(current as usize) };
            if chunk_range.min_utc_ns <= range.max_utc_ns && chunk_range.max_utc_nc >= range.min_utc_ns {
                return ((current << TIME_RANGE_LOG_MASK_SIZE) + 4096).min(offset);
            }
            if current == 0 {
                return 0;
            }
            current -= 1;
        }
    }

    pub fn upgrade(&self, log: WeakLogEntry) -> Option<LogEntry<'a>> {
        if self.bucket.generation.load(Ordering::Relaxed) == log.bucket_generation
            && (log.index as usize) < self.len
        {
            Some(LogEntry { bucket: self.bucket, index: log.index })
        } else {
            None
        }
    }
    pub fn span_indices(&self) -> &[u32] {
        let span = self.bucket.span_index.as_ptr();
        unsafe { std::slice::from_raw_parts(span, self.len) }
    }
    pub fn entries(&self) -> impl Iterator<Item = LogEntry> + DoubleEndedIterator + ExactSizeIterator + '_ {
        (0..self.len).map(|index| LogEntry { index: index as u32, bucket: &self.bucket })
    }

    pub fn archetypes(&self) -> &[archetype::Archetype] {
        let ptr = self.bucket.archetype.as_ptr();
        unsafe { std::slice::from_raw_parts(ptr, self.archetype_count) }
    }
    /// You must not use the returned slice after the bucket guard has been dropped
    pub unsafe fn archetypes_bucket_lifetime(&self) -> &'a [archetype::Archetype] {
        let ptr = self.bucket.archetype.as_ptr();
        unsafe { std::slice::from_raw_parts(ptr, self.archetype_count) }
    }
    pub fn archetype_index(&self) -> &[u16] {
        let ptr = self.bucket.archetype_index.as_ptr();
        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }

    pub fn spans(&self) -> &[SpanRange] {
        let span = self.bucket.span_data.as_ptr();
        unsafe { std::slice::from_raw_parts(span, self.span_count) }
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

        let offset_ptr = self.bucket.offset.as_ptr();
        let field_ptr = self.bucket.field.as_ptr();
        let archetype_index_ptr = self.bucket.archetype_index.as_ptr();

        for log_index in map.indexed..self.len {
            let arch = unsafe {
                let archetype_index = *archetype_index_ptr.add(log_index);
                self.bucket.archetype(archetype_index)
            };
            let target = arch.target_id as usize;
            if target >= map.target_summaries.len() {
                map.target_summaries.resize(target + 1, TargetSummary { level_counts: [0; 4] });
            }
            map.target_summaries[target].level_counts[arch.level() as usize] += 1;

            let (start, end) = unsafe { (*offset_ptr.add(log_index), *offset_ptr.add(log_index + 1)) };
            if start == end {
                continue;
            }
            let fields = unsafe { std::slice::from_raw_parts(field_ptr.add(start as usize), (end - start) as usize) };
            let keys = arch.field_keys();
            debug_assert_eq!(fields.len(), keys.len());

            for (field, key) in fields.iter().zip(keys) {
                map.index_field_value(self.bucket, *field);
                let key_info = map.keys.get_mut_or_default(*key);
                key_info.total += 1;
                key_info.kinds.insert(field.kind());
                if key_info.values.len() < 32 {
                    let hash = self.bucket.random_state.hash_one(field);
                    use hashbrown::hash_table::Entry;
                    if let Entry::Vacant(entry) =
                        key_info.values.entry(hash, |v| v == field, |value| self.bucket.random_state.hash_one(value))
                    {
                        entry.insert(*field);
                    }
                }
            }
        }
        map.indexed = self.len;

        map
    }
}
impl<'a> Drop for BucketGuard<'a> {
    fn drop(&mut self) {
        if self.bucket.ref_count.fetch_sub(1, Ordering::AcqRel) == 1 {
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

impl KeySummary<'_> {
    pub fn kinds(&self) -> impl Iterator<Item = FieldKind> {
        let value = self.kind_mask;
        (0..16).filter(move |i| (1 << i) & value != 0).map(|i| FieldKind::from(i))
    }
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
    pub fn bidirectional_query_from<'a>(
        &'a self,
        filters: &'a [GeneralFilter],
        offset: WeakLogEntry,
    ) -> Option<(ReverseQueryWalker<'a>, ForwardQueryWalker<'a>)> {
        let guard = self.generation_guard();
        if unsafe { !guard.is_alive(offset) } {
            return None;
        }
        drop(guard);

        let mut reverse = self.reverse_query(filters);
        reverse.seek_to(offset);
        let forward = reverse.forward_query();
        Some((reverse, forward))
    }
    pub fn forward_query<'a>(&'a self, filters: &'a [GeneralFilter]) -> ForwardQueryWalker<'a> {
        let mut time_range = TimeFilter { min_utc_ns: 0, max_utc_ns: u64::MAX };
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
    /// Creates a forward query walker starting from the oldest available bucket.
    ///
    /// Unlike [`forward_query`], which starts from recent buckets, this method
    /// begins iteration from generation 0, returning logs in insertion order
    /// starting from the oldest available entries.
    ///
    /// [`forward_query`]: Self::forward_query
    pub fn forward_query_from_oldest<'a>(&'a self, filters: &'a [GeneralFilter]) -> ForwardQueryWalker<'a> {
        let mut time_range = TimeFilter { min_utc_ns: 0, max_utc_ns: u64::MAX };
        for filter in filters {
            if let GeneralFilter::Time(filter) = filter {
                time_range.min_utc_ns = time_range.min_utc_ns.max(filter.min_utc_ns);
                time_range.max_utc_ns = time_range.max_utc_ns.min(filter.max_utc_ns);
            }
        }
        if time_range.max_utc_ns != u64::MAX {
            time_range.min_utc_ns = time_range.min_utc_ns.max(1)
        }
        ForwardQueryWalker {
            general_filter: filters,
            time_range,
            segments: ForwardSegmentWalker::new(0),
            index: self,
            strategy: QueryStrategy::Simple,
        }
    }
    pub fn reverse_query<'a>(&'a self, filters: &'a [GeneralFilter]) -> ReverseQueryWalker<'a> {
        let mut time_range = TimeFilter { min_utc_ns: 0, max_utc_ns: u64::MAX };
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
            frozen: false,
        };
        walker
    }
    pub fn generation_guard(&self) -> GenerationGuard<'_> {
        GenerationGuard {
            buckets: [self.buckets[0].read(), self.buckets[1].read(), self.buckets[2].read(), self.buckets[3].read()],
        }
    }
    pub fn lastest_generation(&self) -> u32 {
        self.generation.load(Ordering::Acquire)
    }
    pub fn newest_bucket(&self) -> Option<BucketGuard<'_>> {
        let current_generation = self.generation.load(Ordering::Acquire);
        let current_bucket = current_generation & 0b11;
        self.buckets[current_bucket as usize].read()
    }
    pub fn read_newest_buckets(&self) -> impl Iterator<Item = BucketGuard<'_>> + '_ {
        let current_generation = self.generation.load(Ordering::Acquire);
        let current_bucket = current_generation & 0b11;
        (current_bucket + 1..current_bucket + 4 + 1).rev().filter_map(|i| {
            let bucket_index = i & 0b11;
            self.buckets[bucket_index as usize].read()
        })
    }
    pub fn query(&self, query: &QueryExpr, mut func: impl FnMut(LogEntry) -> bool) {
        let mapper = self.targets.mapper();
        for bucket in self.read_newest_buckets() {
            for entry in bucket.entries().rev() {
                if query.pred().matches(entry, &mapper) {
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
    let raw = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            amount * std::mem::size_of::<T>(),
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if raw == libc::MAP_FAILED {
        panic!("mmap failed");
    }
    let x = raw as *mut T;
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
            buckets: [Bucket::new(0), Bucket::new(1), Bucket::new(2), Bucket::new(3)],
            generation: AtomicU32::new(0),
        }
    }
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct InternedRange {
    pub(crate) len: u16,
    pub(crate) data: u16,
    pub(crate) offset: u32,
}
impl InternedRange {
    pub(crate) fn from_field_mask(field_mask: u64) -> InternedRange {
        InternedRange { offset: field_mask as u32, len: (field_mask >> 32) as u16, data: 0 }
    }
    pub(crate) fn field_mask(&self) -> u64 {
        self.offset as u64 | ((self.len as u64) << 32)
    }
}

fn same_range(left: InternedRange, right: InternedRange) -> bool {
    left.offset == right.offset && left.len == right.len && left.data == right.data
}

#[derive(Debug)]
pub struct BucketMemoryStats {
    pub log_count: usize,
    pub total_fields: usize,
    pub custom_keys: usize,
    pub log_metadata_bytes: usize,
    pub span_bytes: usize,
    pub message_bytes: usize,
    pub interned_bytes: usize,
    pub general_map_bytes: usize,
    pub uuid_map_bytes: usize,
}
impl BucketMemoryStats {
    pub fn bytes_per_log(&self) -> f64 {
        (self.total() as f64) / self.log_count as f64
    }
    pub fn total(&self) -> usize {
        self.interned_bytes
            + self.log_metadata_bytes
            + self.total_fields
            + self.span_bytes
            + self.custom_keys
            + self.general_map_bytes
            + self.uuid_map_bytes
            + self.message_bytes
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
    archetype_map: HashTable<u16>,
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

    log_encoder: Option<BucketLogStreamEncoder>,
    active_log: Option<Vec<u8>>,
    pending_finalized: VecDeque<DrainedLogBuffer>,
    entries_in_current_bucket: u64,
}
unsafe impl Send for Index {}

#[allow(unused)]
pub const _: *const () = &Index::write::<'static, LogFields<'static>> as *const _ as _;
// Static keys are a maxium 127,
impl Index {
    pub fn generation(&self) -> usize {
        self.generation
    }
    pub fn reader(&self) -> &Arc<IndexReader> {
        &self.reader
    }

    pub fn forward_query<'b>(&'b self, filters: &'b [GeneralFilter]) -> impl Iterator<Item = LogEntry<'b>> {
        let mut walker = self.reader().forward_query(filters);
        std::iter::from_fn(move || {
            let collection = walker.next()?;
            Some((
                // Safe because we hold the referce to the index now mutations will occour
                unsafe { &*(collection.bucket_generation as *const Bucket) },
                collection.entries,
            ))
        })
        .flat_map(|(bucket, indices)| indices.into_iter().map(|index| LogEntry { bucket, index: index as u32 }))
    }
    pub fn reverse_query<'b>(&'b self, filters: &'b [GeneralFilter]) -> impl Iterator<Item = LogEntry<'b>> {
        let mut walker = self.reader().reverse_query(filters);
        std::iter::from_fn(move || {
            let collection = walker.next()?;
            Some((
                // Safe because we hold the referce to the index now mutations will occour
                unsafe { &*(collection.bucket_generation as *const Bucket) },
                collection.entries,
            ))
        })
        .flat_map(|(bucket, indices)| indices.into_iter().map(|index| LogEntry { bucket, index }))
    }
    pub fn complete_bucket(&mut self) {
        let persistence_was_on = self.active_log.is_some();
        if let Some(mut buf) = self.active_log.take() {
            let close_ts = system_time_now_ns();
            stream_encoder::write_footer(&mut buf, close_ts, self.entries_in_current_bucket);
            self.pending_finalized.push_back(DrainedLogBuffer {
                bytes: buf,
                generation: self.generation as u64,
                closed: true,
            });
        }
        self.entries_in_current_bucket = 0;
        if let Some(enc) = self.log_encoder.as_mut() {
            enc.reset();
        }

        //todo handle bucket sizes
        self.fields_used = 0;
        self.data_used = 0;
        self.logs_used = 0;
        self.spans_used = 0;
        self.custom_keys = 0;
        self.general_intern_map.clear();
        self.uuid_intern_map.clear();
        self.archetype_map.clear();
        self.msg_map.clear();
        self.span_table.clear();
        self.new_ranges.clear();
        self.new_uuids.clear();
        self.dirty = false;
        self.generation += 1;

        let bucket = &self.reader.buckets[self.generation & 0b11];
        bucket.len.store(0, Ordering::Release);
        bucket.span_count.store(0, Ordering::Release);
        bucket.archetype_count.store(0, Ordering::Release);

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
            bucket_maps.target_summaries.clear();
            bucket_maps.data_len = 0;
            bucket_maps.msgs = 0;
            bucket_maps.indexed = 0;
        }
        bucket.generation.store(self.generation as u32, Ordering::Release);
        self.reader.generation.store(self.generation as u32, Ordering::Release);
        bucket.ref_count.fetch_add(1, Ordering::Release);

        if persistence_was_on {
            let mut buf = Vec::with_capacity(BUCKET_LOG_BUFFER_INITIAL_CAPACITY);
            stream_encoder::write_header(&mut buf, self.generation as u64);
            self.active_log = Some(buf);
        }
    }
    pub fn current_bucket_memory_used(&self) -> BucketMemoryStats {
        let interned_bytes = self.data_used;
        let per_log_bytes = self.logs_used
            * (
                std::mem::size_of::<u32>() //span_id
            + std::mem::size_of::<u64>() //timestamp
            + std::mem::size_of::<u32>() //offset
            + std::mem::size_of::<u16>()
                //archetype
            );
        let field_bytes = self.fields_used * std::mem::size_of::<Field>();
        let custom_key_bytes = (self.custom_keys as usize) * std::mem::size_of::<InternedRange>();
        let span_bytes = self.spans_used * std::mem::size_of::<SpanRange>();
        let intern_map_bytes = ((self.general_intern_map.len() * std::mem::size_of::<InternedRange>()) * 8) / 7;
        let uuid_map_bytes = ((self.general_intern_map.len() * std::mem::size_of::<u32>()) * 8) / 7;
        let message_map_bytes = self.msg_map.len() * std::mem::size_of::<InternedRange>();

        BucketMemoryStats {
            log_count: self.logs_used,
            interned_bytes,
            log_metadata_bytes: per_log_bytes,
            total_fields: field_bytes,
            custom_keys: custom_key_bytes,
            span_bytes,
            message_bytes: message_map_bytes,
            general_map_bytes: intern_map_bytes,
            uuid_map_bytes,
        }
    }
    pub fn new() -> Index {
        Index::with_config(IndexConfig::default())
    }
    pub fn with_config(config: IndexConfig) -> Index {
        let (log_encoder, active_log) = if config.persistence_enabled {
            let mut buf = Vec::with_capacity(BUCKET_LOG_BUFFER_INITIAL_CAPACITY);
            stream_encoder::write_header(&mut buf, 0);
            (Some(BucketLogStreamEncoder::new()), Some(buf))
        } else {
            (None, None)
        };
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
            archetype_map: HashTable::with_capacity(4096),
            general_intern_map: HashTable::with_capacity(4096),
            uuid_intern_map: HashTable::with_capacity(4096),
            target_map: LocalIntermentCache::default(),
            msg_map: HashTable::with_capacity(4096),
            span_table: HashMap::with_capacity(4096),
            reader: Arc::new(IndexReader::new()),
            chunk_max_utc_ns: u64::MIN,
            chunk_min_utc_ns: u64::MAX,
            log_encoder,
            active_log,
            pending_finalized: VecDeque::new(),
            entries_in_current_bucket: 0,
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
        self.archetype_map.clear();
        self.msg_map.clear();
        self.span_table.clear();
        self.new_ranges.clear();
        self.new_uuids.clear();
        self.generation = 0;
        self.reader.generation.store(0, Ordering::Release);
        self.chunk_max_utc_ns = u64::MIN;
        self.chunk_min_utc_ns = u64::MAX;
        self.dirty = false;

        self.pending_finalized.clear();
        self.entries_in_current_bucket = 0;
        if let Some(enc) = self.log_encoder.as_mut() {
            enc.reset();
        }
        if let Some(mut buf) = self.active_log.take() {
            buf.clear();
            stream_encoder::write_header(&mut buf, 0);
            self.active_log = Some(buf);
        }

        for (i, bucket) in self.reader.buckets.iter().enumerate() {
            unsafe { bucket.clear_unchecked(i as u32) }
        }
    }

    /// Atomically swap the active log byte buffer for `fresh`. Returns the
    /// previously active buffer wrapped in a [`DrainedLogBuffer`], or `None`
    /// when persistence is disabled. The fresh buffer is cleared on entry so
    /// callers may reuse one from a pool.
    ///
    /// The encoder dedup state stays on the index across swaps. Mid-bucket
    /// swap returns a buffer with no footer; concatenating drained buffers
    /// in order reconstructs the complete stream file.
    pub fn swap_log_buffer(&mut self, mut fresh: Vec<u8>) -> Option<DrainedLogBuffer> {
        self.active_log.as_ref()?;
        fresh.clear();
        let previous = self.active_log.replace(fresh).expect("active_log presence checked above");
        Some(DrainedLogBuffer { bytes: previous, generation: self.generation as u64, closed: false })
    }

    /// Pop a finalized buffer from the pending queue. A finalized buffer ends
    /// with the file footer for its bucket. Drain in a loop until `None` after
    /// each rotation to avoid stalling the queue.
    pub fn take_finalized_log_buffer(&mut self) -> Option<DrainedLogBuffer> {
        self.pending_finalized.pop_front()
    }

    /// Generation of the bucket the active log buffer belongs to, if any.
    pub fn active_log_generation(&self) -> Option<u64> {
        self.active_log.as_ref().map(|_| self.generation as u64)
    }

    /// Byte length of the current active log buffer. Useful for sizing-based
    /// drain triggers from a writer thread.
    pub fn active_log_buffer_len(&self) -> Option<usize> {
        self.active_log.as_ref().map(|b| b.len())
    }

    /// Replay a stream slice into this index. Bypasses [`Index::write`]:
    /// decl frames pre-populate the existing dedup maps and Record frames
    /// write straight into bucket arrays. Returns the number of records
    /// applied.
    pub fn ingest(&mut self, bytes: &[u8]) -> Result<u64, crate::persistence::format::ReadError> {
        crate::persistence::replay::ingest(self, bytes)
    }

    /// Install a general string/bytes decl from a stream. Writes the bytes
    /// into bucket.data at `offset` (extending data_used), inserts the
    /// InternedRange into the local map and stages it for later commit into
    /// `bucket.intern_map.general`.
    pub(crate) fn install_string_decl(
        &mut self,
        offset: u32,
        len: u16,
        bytes: &[u8],
    ) -> Result<(), crate::persistence::format::ReadError> {
        let bucket = &self.reader.buckets[self.generation & 0b11];
        let end = offset as usize + len as usize;
        if end > BUCKET_DATA_SIZE {
            return Err(crate::persistence::format::ReadError::BucketCapacityExceeded);
        }
        unsafe {
            bucket.data.as_ptr().add(offset as usize).copy_from_nonoverlapping(bytes.as_ptr(), len as usize);
        }
        if end > self.data_used {
            self.data_used = end;
        }
        let range = InternedRange { offset, data: 0, len };
        let hash = bucket.random_state.hash_one(bytes);
        // Insert into the local in-progress map; commit will push into bucket.intern_map.general.
        self.general_intern_map
            .insert_unique(hash, range, |t| bucket.random_state.hash_one(unsafe { bucket.data_unchecked(*t) }));
        self.new_ranges.push((range, hash));
        self.dirty = true;
        Ok(())
    }

    /// Install a UUID decl: writes 16 bytes at the given offset and stages
    /// the index for the bucket UUID intern map.
    pub(crate) fn install_uuid_decl(
        &mut self,
        offset: u32,
        bytes: &[u8; 16],
    ) -> Result<(), crate::persistence::format::ReadError> {
        let bucket = &self.reader.buckets[self.generation & 0b11];
        let end = offset as usize + 16;
        if end > BUCKET_DATA_SIZE {
            return Err(crate::persistence::format::ReadError::BucketCapacityExceeded);
        }
        unsafe {
            bucket.data.as_ptr().add(offset as usize).copy_from_nonoverlapping(bytes.as_ptr(), 16);
        }
        if end > self.data_used {
            self.data_used = end;
        }
        let hash = bucket.random_state.hash_one(bytes);
        self.uuid_intern_map
            .insert_unique(hash, offset, |t| bucket.random_state.hash_one(unsafe { bucket.uuid_bytes(*t) }));
        self.new_uuids.push((offset, hash));
        self.dirty = true;
        Ok(())
    }

    /// Install a Msg decl: writes the bytes at the given offset and inserts
    /// directly into the msg_map.
    pub(crate) fn install_msg_decl(
        &mut self,
        msg_id: u16,
        offset: u32,
        len: u16,
        bytes: &[u8],
    ) -> Result<(), crate::persistence::format::ReadError> {
        let bucket = &self.reader.buckets[self.generation & 0b11];
        let end = offset as usize + len as usize;
        if end > BUCKET_DATA_SIZE {
            return Err(crate::persistence::format::ReadError::BucketCapacityExceeded);
        }
        unsafe {
            bucket.data.as_ptr().add(offset as usize).copy_from_nonoverlapping(bytes.as_ptr(), len as usize);
        }
        if end > self.data_used {
            self.data_used = end;
        }
        let range = InternedRange { offset, data: msg_id, len };
        let hash = bucket.random_state.hash_one(bytes);
        self.msg_map.insert_unique(hash, range, |t| bucket.random_state.hash_one(unsafe { bucket.data_unchecked(*t) }));
        self.dirty = true;
        Ok(())
    }

    /// Intern a target name into the local target cache and return its u16.
    /// Returns `None` if the shared target buffer is exhausted; callers must
    /// drop the offending record so a hostile or runaway producer cannot
    /// crash the ingest thread.
    pub(crate) fn install_target(&mut self, name: &[u8]) -> Option<u16> {
        self.target_map.intern(&self.reader.targets, name)
    }

    /// Install a SpanDecl: write the SpanRange to bucket.span_data and insert
    /// into span_table.
    pub(crate) fn install_span_decl(
        &mut self,
        span_id: u32,
        span: SpanID,
        parent: Option<SpanID>,
    ) -> Result<(), crate::persistence::format::ReadError> {
        if (span_id as usize) >= BUCKET_SPAN_RANGE_SIZE {
            return Err(crate::persistence::format::ReadError::BucketCapacityExceeded);
        }
        let bucket = &self.reader.buckets[self.generation & 0b11];
        // Initialize first_mask/last_mask to 0; the Record frame will update them
        // based on the span_kind reported per record.
        unsafe {
            bucket.span_data.as_ptr().add(span_id as usize).write(SpanRange {
                id: span,
                parent,
                first_mask: 0,
                last_mask: AtomicU32::new(0),
            });
        }
        if (span_id as usize) >= self.spans_used {
            self.spans_used = (span_id as usize) + 1;
        }
        self.span_table.insert(span, span_id);
        Ok(())
    }

    /// Install an ArchetypeDecl: build an Archetype struct and place it in
    /// bucket.archetype, then insert the new id into archetype_map.
    pub(crate) fn install_archetype_decl(
        &mut self,
        msg_id: u16,
        target_id: u16,
        level: LogLevel,
        in_span: bool,
        service: Option<ServiceId>,
        field_keys: &[u16],
    ) -> Result<(), crate::persistence::format::ReadError> {
        let bucket = &self.reader.buckets[self.generation & 0b11];
        // Look up msg's offset/len from msg_map keyed by msg_id (== InternedRange.data).
        // Iterate (msg_map is small; this is bounded by archetype-decl frequency, not records).
        let mut msg_offset = 0u32;
        let mut msg_len = 0u16;
        if let Some(found) = self.msg_map.iter().find(|r| r.data == msg_id) {
            msg_offset = found.offset;
            msg_len = found.len;
        }
        let archetype_id = self.archetype_map.len();
        if archetype_id >= BUCKET_ARCHETYPE_SIZE {
            return Err(crate::persistence::format::ReadError::BucketCapacityExceeded);
        }
        let mut mask = 1u32 << level as u8;
        if in_span {
            mask |= 1 << 4;
        }
        let mut field_headers = [u16::MAX; archetype::FIELD_LANES];
        for (i, &k) in field_keys.iter().enumerate().take(archetype::FIELD_LANES) {
            field_headers[i] = k;
        }
        let dex = Archetype {
            msg_offset,
            msg_len,
            target_id,
            mask,
            service,
            pad: 0,
            size: field_keys.len() as u16,
            field_headers,
        };
        unsafe {
            bucket.archetype.as_ptr().add(archetype_id).write(dex);
        }
        let archetype_id_u16 = archetype_id as u16;
        let hash = bucket.random_state.hash_one(unsafe { bucket.archetype(archetype_id_u16) });
        self.archetype_map
            .insert_unique(hash, archetype_id_u16, |t| bucket.random_state.hash_one(unsafe { bucket.archetype(*t) }));
        self.dirty = true;
        Ok(())
    }

    /// Read in_span flag from a previously-installed archetype.
    pub(crate) fn archetype_in_span_for(&self, archetype_id: u16) -> Option<bool> {
        if (archetype_id as usize) >= self.archetype_map.len() {
            return None;
        }
        let bucket = &self.reader.buckets[self.generation & 0b11];
        Some(unsafe { bucket.archetype(archetype_id) }.in_span())
    }

    /// Read field count from a previously-installed archetype.
    pub(crate) fn archetype_field_count_for(&self, archetype_id: u16) -> Option<u16> {
        if (archetype_id as usize) >= self.archetype_map.len() {
            return None;
        }
        let bucket = &self.reader.buckets[self.generation & 0b11];
        Some(unsafe { bucket.archetype(archetype_id) }.size)
    }

    /// Install a Record frame: write timestamp, archetype id, span_index,
    /// fields, offset entry. Bumps logs_used. Level and target are encoded in
    /// the archetype.
    pub(crate) fn install_record(
        &mut self,
        archetype_id: u16,
        timestamp: u64,
        span_kind: Option<crate::persistence::format::SpanKindTag>,
        span_id: Option<u32>,
        fields: &[Field],
    ) -> Result<(), crate::persistence::format::ReadError> {
        if self.logs_used >= BUCKET_LOG_SIZE {
            return Err(crate::persistence::format::ReadError::BucketCapacityExceeded);
        }
        if self.fields_used + fields.len() > BUCKET_FIELD_SIZE {
            return Err(crate::persistence::format::ReadError::BucketCapacityExceeded);
        }
        let bucket = &self.reader.buckets[self.generation & 0b11];
        let log_index = self.logs_used;

        unsafe {
            bucket.timestamp.as_ptr().add(log_index).write(timestamp);
            bucket.archetype_index.as_ptr().add(log_index).write(archetype_id);
        }

        // Span handling: update SpanRange.first_mask / last_mask based on span_kind.
        match (span_kind, span_id) {
            (Some(kind), Some(sid)) => {
                let sr = unsafe { &*bucket.span_data.as_ptr().add(sid as usize) };
                let raw_index = log_index as u32;
                use crate::persistence::format::SpanKindTag;
                match kind {
                    SpanKindTag::StartWithParent | SpanKindTag::Start => unsafe {
                        (bucket.span_data.as_ptr().add(sid as usize) as *mut SpanRange).write(SpanRange {
                            id: sr.id,
                            parent: sr.parent,
                            first_mask: raw_index | (1u32 << 31),
                            last_mask: AtomicU32::new(raw_index),
                        });
                    },
                    SpanKindTag::Current => {
                        sr.last_mask.store(raw_index, Ordering::Release);
                    }
                    SpanKindTag::End => {
                        sr.last_mask.store(raw_index | (1u32 << 31), Ordering::Release);
                    }
                }
                unsafe {
                    bucket.span_index.as_ptr().add(log_index).write(sid);
                }
            }
            _ => unsafe {
                bucket.span_index.as_ptr().add(log_index).write(u32::MAX);
            },
        }

        // Fields.
        let start = self.fields_used;
        for (i, field) in fields.iter().enumerate() {
            unsafe {
                bucket.field.as_ptr().add(start + i).write(*field);
            }
        }
        self.fields_used += fields.len();

        // Offset table: bucket.offset[log_index] is already set; bucket.offset[log_index+1] = fields_used.
        unsafe {
            bucket.offset.as_ptr().add(log_index + 1).write(self.fields_used as u32);
        }

        // Time range periodic write (mirroring write_current_to_bucket).
        self.chunk_min_utc_ns = timestamp.min(self.chunk_min_utc_ns);
        self.chunk_max_utc_ns = timestamp.max(self.chunk_max_utc_ns);
        if self.logs_used & (TIME_RANGE_LOG_COUNT - 1) == (TIME_RANGE_LOG_COUNT - 1) {
            unsafe {
                bucket
                    .timerange
                    .as_ptr()
                    .add(self.logs_used >> TIME_RANGE_LOG_COUNT.trailing_zeros())
                    .write(TimeRange { min_utc_ns: self.chunk_min_utc_ns, max_utc_nc: self.chunk_max_utc_ns });
            }
            self.chunk_max_utc_ns = u64::MIN;
            self.chunk_min_utc_ns = u64::MAX;
        }

        self.logs_used += 1;
        self.entries_in_current_bucket += 1;
        Ok(())
    }

    /// Drain any pending replay state into the bucket's intern maps.
    /// Mirrors `commit()` but is called once at the end of a replay run
    /// rather than per-record.
    pub(crate) fn flush_pending(&mut self) {
        let bucket = &self.reader.buckets[self.generation & 0b11];
        if self.dirty || !self.new_ranges.is_empty() || !self.new_uuids.is_empty() {
            let mut map = bucket.intern_map.lock().unwrap();
            for (range, hash) in &self.new_ranges {
                map.general.insert_unique(*hash, *range, |t| {
                    bucket.random_state.hash_one(unsafe { bucket.data_unchecked(*t) })
                });
            }
            for (range, hash) in &self.new_uuids {
                map.uuid
                    .insert_unique(*hash, *range, |t| bucket.random_state.hash_one(unsafe { bucket.uuid_bytes(*t) }));
            }
            map.msgs = self.msg_map.len();
            map.data_len = self.data_used;
            self.new_ranges.clear();
            self.new_uuids.clear();
            self.dirty = false;
            bucket.archetype_count.store(self.archetype_map.len(), Ordering::Release);
        }
        bucket.span_count.store(self.spans_used, Ordering::Release);
        bucket.len.store(self.logs_used, Ordering::Release);
    }
    pub fn write<'a, T: Iterator<Item = Result<(Key<'a>, Value<'a>), MunchError>> + Clone>(
        &mut self,
        timestamp: u64,
        level: LogLevel,
        span_info: SpanInfo,
        service_id: Option<ServiceId>,
        fields: T,
    ) -> Result<WeakLogEntry, MunchError> {
        // Todo make sure a munch error doesn't break anything doesn't
        // as long as we don't commit should just have to reset the counters,
        // Or defer updating the counters until we know the no error is possible.
        if self.write_current_to_bucket(timestamp, level, span_info.clone(), service_id, fields.clone())? {
            self.emit_record(timestamp, &span_info);
            self.commit();
            return Ok(WeakLogEntry { bucket_generation: self.generation as u32, index: self.logs_used as u32 - 1 });
        }
        // Ran out of storage in this bucket lets move to the next one
        self.complete_bucket();
        if !self.write_current_to_bucket(timestamp, level, span_info.clone(), service_id, fields)? {
            // A single record that won't fit in an empty bucket is malformed
            // input; previously this panicked and killed ingest. Surface it
            // as an error so the producer's bad record is dropped and the
            // service keeps ingesting subsequent records.
            return Err(MunchError::InvalidValue);
        }
        self.emit_record(timestamp, &span_info);
        self.commit();
        Ok(WeakLogEntry { bucket_generation: self.generation as u32, index: self.logs_used as u32 - 1 })
    }

    /// Emit the wire frames for the log entry just written into the bucket.
    /// No-op when persistence is disabled. Called between
    /// `write_current_to_bucket` and `commit` so the encoder can read the
    /// just-staged entries from `new_ranges` / `new_uuids` before they are
    /// drained.
    fn emit_record(&mut self, timestamp: u64, span_info: &SpanInfo) {
        let Some(enc) = &mut self.log_encoder else {
            return;
        };
        let Some(buf) = &mut self.active_log else {
            return;
        };
        let log_index = (self.logs_used - 1) as u32;
        let bucket = &self.reader.buckets[self.generation & 0b11];
        let msg_count = self.msg_map.len();
        let span_count = self.span_table.len();
        let new_ranges: &[_] = &self.new_ranges;
        let new_uuids: &[_] = &self.new_uuids;
        let msg_map = &self.msg_map;
        let targets = &self.reader.targets;
        stream_encoder::record(
            enc, buf, bucket, new_ranges, new_uuids, msg_map, msg_count, span_count, targets, log_index, timestamp,
            span_info,
        );
        self.entries_in_current_bucket += 1;
    }
    fn write_current_to_bucket<'a>(
        &mut self,
        timestamp: u64,
        level: LogLevel,
        span_info: SpanInfo,
        service_id: Option<ServiceId>,
        mut fields: impl Iterator<Item = Result<(Key<'a>, Value<'a>), MunchError>> + Clone,
    ) -> Result<bool, MunchError> {
        if self.logs_used >= BUCKET_LOG_SIZE {
            return Ok(false);
        }
        let bucket = &self.reader.buckets[self.generation & 0b11];
        unsafe {
            bucket.timestamp.as_ptr().add(self.logs_used).write(timestamp);
        }
        let mut msg: &[u8] = b"";
        let mut target: &[u8] = b"";
        let fields_start = self.fields_used;
        let mut field_keys: SmallVec<[u16; archetype::FIELD_LANES]> = SmallVec::new();
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
                    // A non-UTF-8 dynamic key was previously a panic. Surface
                    // it as a record-level error so a malformed record cannot
                    // kill the ingest thread.
                    let Ok(name) = std::str::from_utf8(key) else {
                        return Err(MunchError::InvalidKey);
                    };
                    KeyID::intern(name).raw()
                }
            };
            if field_keys.len() >= archetype::FIELD_LANES {
                continue;
            }
            // println!("{:?}", value);
            let (kind, value_bits) = match value {
                Value::String(text) => {
                    let Some(interned) = self.intern(text)? else {
                        return Ok(false);
                    };
                    (FieldKind::String, interned.field_mask())
                }
                Value::Bytes(bytes) => {
                    let Some(interned) = self.intern(bytes)? else {
                        return Ok(false);
                    };
                    (FieldKind::Bytes, interned.field_mask())
                }
                Value::I32(num) => (FieldKind::I60, i60::from_i64(num as i64)),
                Value::U32(num) => (FieldKind::I60, i60::from_i64(num as i64)),
                Value::I64(num) => {
                    if let Some(value) = i60::try_from_i64(num) {
                        (FieldKind::I60, value)
                    } else {
                        let Some(interned) = self.intern(&num.to_ne_bytes())? else {
                            return Ok(false);
                        };
                        (FieldKind::I64, interned.field_mask())
                    }
                }
                Value::U64(num) => {
                    if let Ok(signed) = i64::try_from(num) {
                        if let Some(value) = i60::try_from_i64(signed) {
                            (FieldKind::I60, value)
                        } else {
                            let Some(interned) = self.intern(&signed.to_ne_bytes())? else {
                                return Ok(false);
                            };
                            (FieldKind::I64, interned.field_mask())
                        }
                    } else {
                        let Some(interned) = self.intern(&num.to_ne_bytes())? else {
                            return Ok(false);
                        };
                        (FieldKind::U64, interned.field_mask())
                    }
                }
                Value::F32(float) => (FieldKind::F60, f64_to_f60(float as f64)),
                Value::F64(float) => (FieldKind::F60, f64_to_f60(float)),
                Value::UUID(id) => {
                    let Some(interned) = self.intern_uuid(&id) else {
                        return Ok(false);
                    };
                    (FieldKind::UUID, interned as u64)
                }
                Value::Bool(value) => (FieldKind::Bool, value as u64),
                Value::None => (FieldKind::None, 0),
                Value::Seconds(float) => (FieldKind::Seconds, f64_to_f60(float as f64)),
                Value::Timestamp(ts) => {
                    let x = ts.as_millisecond_clamped().clamp(0, (1i64 << 60) - 1) as u64;
                    (FieldKind::Timestamp, x)
                }
            };
            let field = Field::new(kind, value_bits);
            if self.fields_used >= BUCKET_FIELD_SIZE {
                return Ok(false);
            }
            field_keys.push(key_bits);
            unsafe {
                let bucket = &self.reader.buckets[self.generation & 0b11];
                bucket.field.as_ptr().add(self.fields_used).write(field);
            }
            self.fields_used += 1;
        }
        //todo don't require message and target
        let Some(msg_intern) = self.intern_msg(msg)? else {
            return Ok(false);
        };
        let bucket = &self.reader.buckets[self.generation & 0b11];
        // Drop the record (rather than panic) if the global target buffer is
        // full. `MunchError::InvalidValue` lets callers surface the failure
        // and continue ingesting subsequent records.
        let Some(target_intern) = self.target_map.intern(&self.reader.targets, target) else {
            return Err(MunchError::InvalidValue);
        };
        let fields = unsafe {
            std::slice::from_raw_parts_mut(bucket.field.as_ptr().add(fields_start), self.fields_used - fields_start)
        };
        sort_fields_by_key(fields, &mut field_keys);
        let dex =
            archetype::Archetype::new(msg_intern, target_intern, level, !span_info.is_none(), service_id, &field_keys);
        let dex_id = {
            let hash = bucket.random_state.hash_one(&dex);
            let len = self.archetype_map.len();
            let entry = self.archetype_map.entry(
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
            bucket.archetype_index.as_ptr().add(self.logs_used).write(dex_id);
        }

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
                    span_range.last_mask.store((self.logs_used as u32) | end_mask, Ordering::Release);
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
                bucket.span_index.as_ptr().add(self.logs_used).write(span_index);
            }
        } else {
            unsafe {
                bucket.span_index.as_ptr().add(self.logs_used).write(u32::MAX);
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
                    .write(TimeRange { min_utc_ns: self.chunk_min_utc_ns, max_utc_nc: self.chunk_max_utc_ns });
            }
            self.chunk_max_utc_ns = u64::MIN;
            self.chunk_min_utc_ns = u64::MAX;
        }
        self.logs_used += 1;
        unsafe {
            bucket.offset.as_ptr().add(self.logs_used).write(self.fields_used as u32);
        }

        Ok(true)
    }
    fn commit(&mut self) {
        let bucket = &self.reader.buckets[self.generation & 0b11];
        if self.dirty || !self.new_ranges.is_empty() || !self.new_uuids.is_empty() {
            let mut map = bucket.intern_map.lock().unwrap();
            for (range, hash) in &self.new_ranges {
                map.general.insert_unique(*hash, *range, |t| {
                    bucket.random_state.hash_one(unsafe { bucket.data_unchecked(*t) })
                });
            }
            for (range, hash) in &self.new_uuids {
                map.uuid
                    .insert_unique(*hash, *range, |t| bucket.random_state.hash_one(unsafe { bucket.uuid_bytes(*t) }));
            }
            map.msgs = self.msg_map.len();
            map.data_len = self.data_used;
            self.new_ranges.clear();
            self.new_uuids.clear();
            self.dirty = false;
            bucket.archetype_count.store(self.archetype_map.len(), Ordering::Release);
        }
        bucket.span_count.store(self.spans_used, Ordering::Release);
        bucket.len.store(self.logs_used, Ordering::Release);
        if let Some(enc) = self.log_encoder.as_mut() {
            enc.after_commit();
        }
    }
    fn intern_uuid(&mut self, id: &uuid::Uuid) -> Option<u32> {
        let bucket = &self.reader.buckets[self.generation & 0b11];
        let bytes = id.as_bytes();
        let hash = bucket.random_state.hash_one(bytes);
        let entry = self.uuid_intern_map.entry(
            hash,
            |v| unsafe { bucket.uuid_bytes(*v) == bytes },
            |t| bucket.random_state.hash_one(unsafe { bucket.uuid_bytes(*t) }),
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
                    bucket.data.as_ptr().add(self.data_used).copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
                }
                entry.insert(id);
                self.data_used += bytes.len();
                Some(id)
            }
        }
    }

    fn intern_msg(&mut self, bytes: &[u8]) -> Result<Option<InternedRange>, MunchError> {
        if bytes.len() > u16::MAX as usize {
            return Err(MunchError::InvalidValue);
        }
        let bucket = &self.reader.buckets[self.generation & 0b11];
        let hash = bucket.random_state.hash_one(bytes);
        let len = self.msg_map.len();
        let entry = self.msg_map.entry(
            hash,
            |v| unsafe { bucket.data_unchecked(*v) == bytes },
            |t| bucket.random_state.hash_one(unsafe { bucket.data_unchecked(*t) }),
        );
        match entry {
            hashbrown::hash_table::Entry::Occupied(entry) => Ok(Some(*entry.get())),
            hashbrown::hash_table::Entry::Vacant(entry) => {
                if bytes.len() + self.data_used > BUCKET_DATA_SIZE {
                    return Ok(None);
                }
                if len >= BUCKET_MAX_MSG_SIZE {
                    return Ok(None);
                }
                let range = InternedRange { offset: self.data_used as u32, data: len as u16, len: bytes.len() as u16 };
                // unsafe {
                //     bucket.msg_ptr.as_ptr().add(len).write(range);
                // }
                self.dirty = true;
                unsafe {
                    bucket.data.as_ptr().add(self.data_used).copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
                }
                entry.insert(range);
                self.data_used += bytes.len();
                Ok(Some(range))
            }
        }
    }

    fn intern(&mut self, bytes: &[u8]) -> Result<Option<InternedRange>, MunchError> {
        if bytes.is_empty() {
            return Ok(Some(InternedRange { len: 0, data: 0, offset: 0 }));
        }
        if bytes.len() > u16::MAX as usize {
            return Err(MunchError::InvalidValue);
        }
        let bucket = &self.reader.buckets[self.generation & 0b11];
        let hash = bucket.random_state.hash_one(bytes);
        let entry = self.general_intern_map.entry(
            hash,
            |v| unsafe { v.data == 0 && bucket.data_unchecked(*v) == bytes },
            |t| bucket.random_state.hash_one(unsafe { bucket.data_unchecked(*t) }),
        );
        match entry {
            hashbrown::hash_table::Entry::Occupied(entry) => Ok(Some(*entry.get())),
            hashbrown::hash_table::Entry::Vacant(entry) => {
                if bytes.len() + self.data_used > BUCKET_DATA_SIZE {
                    return Ok(None);
                }
                let range = InternedRange { offset: self.data_used as u32, data: 0, len: bytes.len() as u16 };
                self.new_ranges.push((range, hash));
                unsafe {
                    bucket.data.as_ptr().add(self.data_used).copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
                }
                entry.insert(range);
                self.data_used += bytes.len();
                Ok(Some(range))
            }
        }
    }
}
pub(crate) mod i60 {
    pub const MIN: i64 = -(1i64 << 59);
    pub const MAX: i64 = (1i64 << 59) - 1;
    const MASK: u64 = (1u64 << 60) - 1;
    pub fn try_from_i64(value: i64) -> Option<u64> {
        if value >= MIN && value <= MAX {
            Some(from_i64(value))
        } else {
            None
        }
    }
    pub fn from_i64(value: i64) -> u64 {
        (value.wrapping_sub(MIN) as u64)
    }
    pub fn to_i64(value: u64) -> i64 {
        (value & MASK).wrapping_add(MIN as u64) as i64
    }
    #[cfg(test)]
    mod test {
        use super::*;
        #[test]
        fn i60_mapping() {
            for i in [MIN, MAX, -1, 43, 683, 0] {
                assert_eq!(i, to_i64(try_from_i64(i).unwrap()))
            }
        }
    }
}

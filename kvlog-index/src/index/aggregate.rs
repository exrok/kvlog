//! Group-by aggregation primitives.

use hashbrown::HashTable;
use kvlog::encoding::Value;

use crate::field_table::KeyID;

use super::archetype::{Archetype, FIELD_LANES};
use super::{Bucket, BucketGuard, Field, InternedRange, LogEntry, ReverseQueryWalker};

/// Maximum group-by columns supported. Matches the libra-monitoring UI cap.
pub const MAX_GROUP_COLUMNS: usize = 4;

/// Origin of a single group-by column.
///
/// `MissingField` denotes a key that no log has ever used (the index has
/// never assigned it a `KeyID`). Every column for that source resolves
/// to the missing sentinel, identical to fields absent from a given
/// archetype.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GroupKeySource {
    Field(KeyID),
    MissingField,
    Target,
    Message,
    Level,
}

/// Compiled group-by spec. Cheap to clone; built once per analysis request.
#[derive(Clone, Debug)]
pub struct GroupBySpec {
    keys: [GroupKeySource; MAX_GROUP_COLUMNS],
    len: usize,
}

impl Default for GroupBySpec {
    fn default() -> Self {
        Self { keys: [GroupKeySource::MissingField; MAX_GROUP_COLUMNS], len: 0 }
    }
}

impl GroupBySpec {
    pub fn from_sources(sources: &[GroupKeySource]) -> Self {
        let mut spec = Self::default();
        for source in sources.iter().take(MAX_GROUP_COLUMNS) {
            spec.keys[spec.len] = *source;
            spec.len += 1;
        }
        spec
    }
    pub fn columns(&self) -> &[GroupKeySource] {
        &self.keys[..self.len]
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn len(&self) -> usize {
        self.len
    }
}

/// A 32-byte fixed-size group key — four `u64` lanes carrying raw bits.
///
/// Lane encoding by source, with no kind tags or bucket-id stuffing:
///
/// | Source                    | Lane value                                              |
/// |---------------------------|---------------------------------------------------------|
/// | `Field(k)` present        | `field.raw` — verbatim 8 bytes                          |
/// | `Field(k)` missing        | `0` (acts as a sentinel — real `Field.raw` always       |
/// | / `MissingField`          |  has a non-zero kind tag in the top 4 bits)             |
/// | `Target`                  | `archetype.target_id as u64`                            |
/// | `Message`                 | `(msg_offset:32 << 16) \| (msg_len:16)`                 |
/// | `Level`                   | `archetype.mask & 0b1111`                               |
#[derive(Copy, Clone, Eq, PartialEq, Default, Debug)]
pub struct GroupKey {
    pub lanes: [u64; MAX_GROUP_COLUMNS],
}

impl std::hash::Hash for GroupKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.lanes.hash(state);
    }
}

/// Per-column plan slot. `Field(pos)` reads the entry's `pos`-th raw
/// field. `Constant(value)` is a precomputed per-archetype lane value
/// (target id, message offset+len, or level mask) so the hot loop never
/// touches the archetype. `Missing` produces a 0 lane.
#[derive(Copy, Clone)]
enum PlanCol {
    Missing,
    Field(u8),
    Constant(u64),
}

#[derive(Copy, Clone)]
struct ArchetypePlan {
    cols: [PlanCol; MAX_GROUP_COLUMNS],
}

impl Default for ArchetypePlan {
    fn default() -> Self {
        Self { cols: [PlanCol::Missing; MAX_GROUP_COLUMNS] }
    }
}

const PLAN_CACHE_SIZE: usize = 32;

struct PlanCache {
    tags: [u16; PLAN_CACHE_SIZE],
    plans: [ArchetypePlan; PLAN_CACHE_SIZE],
}

impl PlanCache {
    fn new() -> Self {
        Self { tags: [u16::MAX; PLAN_CACHE_SIZE], plans: [ArchetypePlan::default(); PLAN_CACHE_SIZE] }
    }
    fn clear(&mut self) {
        self.tags = [u16::MAX; PLAN_CACHE_SIZE];
    }
    /// Hit path is one tag compare; the returned `&ArchetypePlan` aliases the
    /// cache slot itself so the caller never spills the 64-byte plan into a
    /// fresh stack temp on each row. Archetype is read only on miss, so cached
    /// rows never touch the archetype array.
    #[inline]
    fn lookup_lazy(&mut self, raw_archetype: u16, entry: LogEntry<'_>, spec: &GroupBySpec) -> &ArchetypePlan {
        let slot = (raw_archetype as usize) & (PLAN_CACHE_SIZE - 1);
        if self.tags[slot] != raw_archetype {
            self.plans[slot] = build_plan(entry.archetype(), spec);
            self.tags[slot] = raw_archetype;
        }
        unsafe { self.plans.get_unchecked(slot) }
    }
}

fn build_plan(archetype: &Archetype, spec: &GroupBySpec) -> ArchetypePlan {
    let mut plan = ArchetypePlan::default();
    for (col, src) in spec.columns().iter().enumerate() {
        plan.cols[col] = match *src {
            GroupKeySource::Field(key) => match archetype.index_of(key) {
                Some(idx) => PlanCol::Field(idx as u8),
                None => PlanCol::Missing,
            },
            GroupKeySource::MissingField => PlanCol::Missing,
            GroupKeySource::Target => PlanCol::Constant(archetype.target_id as u64),
            GroupKeySource::Message => {
                PlanCol::Constant(((archetype.msg_offset as u64) << 16) | (archetype.msg_len as u64))
            }
            GroupKeySource::Level => PlanCol::Constant((archetype.mask as u64) & 0b1111),
        };
    }
    plan
}

#[inline]
fn extract(entry: LogEntry<'_>, plan: &ArchetypePlan, n_cols: usize) -> GroupKey {
    let fields = entry.raw_fields();
    let mut key = GroupKey::default();
    for col in 0..n_cols {
        key.lanes[col] = match plan.cols[col] {
            PlanCol::Missing => 0,
            PlanCol::Field(pos) => unsafe { fields.get_unchecked(pos as usize).raw },
            PlanCol::Constant(v) => v,
        };
    }
    key
}

/// One batch of entries with their precomputed group keys, all from a
/// single bucket. The caller compares `bucket` pointers across calls to
/// detect bucket transitions.
pub struct GroupBatch<'a> {
    pub bucket: &'a Bucket,
    pub entries: &'a [(u32, GroupKey)],
}

/// Drives a [`ReverseQueryWalker`] and emits per-batch group keys
/// alongside log indices, all without per-row allocation. Mirrors the
/// per-bucket specialization in [`super::filter`] but with a non-generic
/// `next()` to keep monomorphic bloat down — the consumer's per-row body
/// lives in its own loop over the returned batch.
pub struct AggregateScanner<'a> {
    walker: ReverseQueryWalker<'a>,
    spec: GroupBySpec,
    plan_cache: PlanCache,
    last_bucket: Option<*const Bucket>,
    n_cols: usize,
    out: Vec<(u32, GroupKey)>,
}

impl<'a> AggregateScanner<'a> {
    pub fn new(walker: ReverseQueryWalker<'a>, spec: GroupBySpec) -> Self {
        let n_cols = spec.len();
        Self { walker, spec, plan_cache: PlanCache::new(), last_bucket: None, n_cols, out: Vec::with_capacity(256) }
    }

    pub fn spec(&self) -> &GroupBySpec {
        &self.spec
    }

    pub fn release_bucket_reclamation_lock(&mut self) {
        self.walker.release_bucket_reclamation_lock();
    }

    pub fn next(&mut self) -> Option<GroupBatch<'_>> {
        let collection = self.walker.next()?;
        let bucket = collection.bucket_generation;
        let bucket_ptr = bucket as *const Bucket;
        if self.last_bucket != Some(bucket_ptr) {
            self.plan_cache.clear();
            self.last_bucket = Some(bucket_ptr);
        }
        self.out.clear();
        self.out.reserve(collection.entries.len());
        let n_cols = self.n_cols;
        let spec = &self.spec;
        for &index in &collection.entries {
            let entry = LogEntry { bucket, index };
            let raw_arch = entry.raw_archetype();
            let plan = self.plan_cache.lookup_lazy(raw_arch, entry, spec);
            let key = extract(entry, plan, n_cols);
            self.out.push((index, key));
        }
        Some(GroupBatch { bucket, entries: &self.out })
    }
}

/// Bench helper: walk every entry in a single bucket through the same
/// per-archetype plan cache used by [`AggregateScanner`]. No filtering
/// every entry is emitted. Used by the kvlog-bench harness to measure
/// raw group-key extraction throughput without the walker on top.
pub fn for_each_in_bucket<F>(bucket: &BucketGuard<'_>, spec: &GroupBySpec, mut emit: F)
where
    F: FnMut(LogEntry<'_>, &GroupKey),
{
    let mut cache = PlanCache::new();
    let n_cols = spec.len();
    for entry in bucket.entries() {
        let raw_arch = entry.raw_archetype();
        let plan = cache.lookup_lazy(raw_arch, entry, spec);
        let key = extract(entry, plan, n_cols);
        emit(entry, &key);
    }
}

impl GroupKey {
    /// Raw u16 target id stored in a `Target` lane.
    #[inline]
    pub fn target_id(&self, col: usize) -> u16 {
        self.lanes[col] as u16
    }

    /// Raw 4-bit level mask stored in a `Level` lane.
    #[inline]
    pub fn level_mask(&self, col: usize) -> u8 {
        (self.lanes[col] as u8) & 0b1111
    }

    /// True if a `Field` lane was missing on the entry's archetype (or
    /// the source was `Field(None)`). A real `Field.raw` always carries
    /// a non-zero kind tag in the top 4 bits, so a zero lane is
    /// unambiguous.
    #[inline]
    pub fn is_field_missing(&self, col: usize) -> bool {
        self.lanes[col] == 0
    }

    /// Decode a `Field` lane back to its `Value`. The bucket must be the
    /// one that emitted this lane.
    ///
    /// # Safety
    /// `bucket` must be the live bucket whose generation produced this
    /// `GroupKey`, and the relevant data offsets must still be valid.
    /// In practice that means the caller is holding a `BucketGuard` for
    /// the bucket through the decode.
    #[inline]
    pub unsafe fn decode_field<'a>(&self, col: usize, bucket: &'a Bucket) -> Value<'a> {
        let field = Field { raw: self.lanes[col] };
        unsafe { field.value(bucket) }
    }

    /// Decode a `Message` lane to its raw bytes in `bucket.data`.
    ///
    /// # Safety
    /// Same constraint as [`Self::decode_field`].
    #[inline]
    pub unsafe fn decode_message<'a>(&self, col: usize, bucket: &'a Bucket) -> &'a [u8] {
        let lane = self.lanes[col];
        let len = (lane & 0xFFFF) as u16;
        let offset = (lane >> 16) as u32;
        let range = InternedRange { offset, data: 0, len };
        unsafe { bucket.data_unchecked(range) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_archetype() -> Archetype {
        let mut arch: Archetype = unsafe { std::mem::zeroed() };
        arch.field_headers = [u16::MAX; FIELD_LANES];
        arch
    }

    #[test]
    fn spec_truncates_to_max_columns() {
        let sources: Vec<GroupKeySource> = (0..10).map(|_| GroupKeySource::MissingField).collect();
        let spec = GroupBySpec::from_sources(&sources);
        assert_eq!(spec.len(), MAX_GROUP_COLUMNS);
    }

    #[test]
    fn missing_field_lane_is_sentinel() {
        let arch = empty_archetype();
        let spec = GroupBySpec::from_sources(&[GroupKeySource::MissingField]);
        let plan = build_plan(&arch, &spec);
        assert!(matches!(plan.cols[0], PlanCol::Missing));
    }

    #[test]
    fn meta_sources_precompute_to_constants() {
        let mut arch = empty_archetype();
        arch.target_id = 7;
        arch.msg_offset = 0;
        arch.msg_len = 0;
        arch.mask = 0b0010;
        let spec = GroupBySpec::from_sources(&[GroupKeySource::Target, GroupKeySource::Level, GroupKeySource::Message]);
        let plan = build_plan(&arch, &spec);
        assert!(matches!(plan.cols[0], PlanCol::Constant(7)));
        assert!(matches!(plan.cols[1], PlanCol::Constant(0b0010)));
        assert!(matches!(plan.cols[2], PlanCol::Constant(0)));
    }

    #[test]
    fn group_key_lanes_default_to_zero() {
        let key = GroupKey::default();
        assert_eq!(key.lanes, [0u64; MAX_GROUP_COLUMNS]);
    }

    use crate::index::test::{test_index, TestIndexWriter};
    use kvlog::Encode;
    use std::collections::HashMap;

    fn run_scan(index: &crate::index::Index, spec: GroupBySpec) -> Vec<(GroupKey, u32)> {
        let reader = index.reader();
        let walker = reader.reverse_query(&[]);
        let mut scanner = AggregateScanner::new(walker, spec);
        let mut counts: HashMap<GroupKey, u32> = HashMap::new();
        while let Some(batch) = scanner.next() {
            for (_idx, key) in batch.entries {
                *counts.entry(*key).or_insert(0) += 1;
            }
        }
        let mut v: Vec<(GroupKey, u32)> = counts.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }

    #[test]
    fn aggregate_groups_by_handler_field() {
        let mut index = test_index();
        let mut writer = TestIndexWriter::new(&mut index);
        crate::log!(writer; msg = "Response", handler = "list");
        crate::log!(writer; msg = "Response", handler = "list");
        crate::log!(writer; msg = "Response", handler = "list");
        crate::log!(writer; msg = "Response", handler = "create");
        crate::log!(writer; msg = "Response", handler = "create");

        let handler_key = KeyID::try_from_str("handler").unwrap();
        let spec = GroupBySpec::from_sources(&[GroupKeySource::Field(handler_key)]);
        let groups = run_scan(&index, spec);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].1, 3); // list
        assert_eq!(groups[1].1, 2); // create
                                    // Both groups must have non-zero (a real interned String field).
        assert_ne!(groups[0].0.lanes[0], 0);
        assert_ne!(groups[1].0.lanes[0], 0);
        assert_ne!(groups[0].0.lanes[0], groups[1].0.lanes[0]);
    }

    #[test]
    fn aggregate_groups_by_message() {
        let mut index = test_index();
        let mut writer = TestIndexWriter::new(&mut index);
        crate::log!(writer; msg = "alpha");
        crate::log!(writer; msg = "alpha");
        crate::log!(writer; msg = "beta");

        let spec = GroupBySpec::from_sources(&[GroupKeySource::Message]);
        let groups = run_scan(&index, spec);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].1, 2); // alpha
        assert_eq!(groups[1].1, 1); // beta
    }

    #[test]
    fn aggregate_missing_field_collapses_to_zero_lane() {
        let mut index = test_index();
        let mut writer = TestIndexWriter::new(&mut index);
        crate::log!(writer; msg = "x", handler = "a");
        crate::log!(writer; msg = "x"); // no handler
        crate::log!(writer; msg = "x"); // no handler

        let handler_key = KeyID::try_from_str("handler").unwrap();
        let spec = GroupBySpec::from_sources(&[GroupKeySource::Field(handler_key)]);
        let groups = run_scan(&index, spec);
        assert_eq!(groups.len(), 2);
        // Two missing entries dominate.
        assert_eq!(groups[0].1, 2);
        assert_eq!(groups[0].0.lanes[0], 0); // missing sentinel
        assert_eq!(groups[1].1, 1);
        assert_ne!(groups[1].0.lanes[0], 0);
    }

    #[test]
    fn decode_field_lane_round_trips() {
        let mut index = test_index();
        let mut writer = TestIndexWriter::new(&mut index);
        crate::log!(writer; msg = "x", handler = "alpha");
        crate::log!(writer; msg = "x", handler = "beta");

        let handler_key = KeyID::try_from_str("handler").unwrap();
        let spec = GroupBySpec::from_sources(&[GroupKeySource::Field(handler_key)]);
        let reader = index.reader();
        let walker = reader.reverse_query(&[]);
        let mut scanner = AggregateScanner::new(walker, spec);
        let mut decoded: Vec<String> = Vec::new();
        while let Some(batch) = scanner.next() {
            for (_idx, key) in batch.entries {
                let value = unsafe { key.decode_field(0, batch.bucket) };
                if let kvlog::encoding::Value::String(b) = value {
                    decoded.push(String::from_utf8_lossy(b).into_owned());
                }
            }
        }
        decoded.sort();
        assert_eq!(decoded, vec!["alpha".to_string(), "beta".to_string()]);
    }
}

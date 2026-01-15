use crate::{
    accel::{collect_indices_of_values_in_set_reverse, MicroBloom, U16Set},
    query::{
        self,
        query_parts::{FieldTest, OptimizationInfo, Pred, PredBuilder},
        QueryParseError,
    },
};

use super::*;
pub mod archetype_filter;
mod field_filter;
mod level_filter;
mod time_filter;
mod vm_filter;
use crate::shared_interner::Mapper;
use ahash::HashSet;
pub use archetype_filter::specialize_on_archetypes;
use archetype_filter::{build_pred_addr_map, ArchetypePrefilter, Match};
use bumpalo::Bump;
pub use field_filter::QueryFilter;
pub use level_filter::LevelFilter;
pub use time_filter::TimeFilter;
pub use vm_filter::QueryVm;
use vm_filter::VmFilter;
#[cfg(test)]
mod test;

#[derive(Debug, Clone)]
pub struct TargetLevelFilter {
    mask: Vec<u8>,
}

impl TargetLevelFilter {
    pub fn from_compact(max: u16, default: u16, pairs: &[u16]) -> TargetLevelFilter {
        let mut mask = vec![default as u8 & 0b1111; (max as usize + 1)];
        for a in pairs.chunks_exact(2) {
            let [target_id, set] = a else {
                unreachable!();
            };
            let Some(collect) = mask.get_mut(*target_id as usize) else {
                continue;
            };
            *collect = (*set as u8) & 0b1111;
        }
        TargetLevelFilter { mask }
    }
    pub fn matches_archtype(&self, entry: &archetype::Archetype) -> bool {
        let Some(collect) = self.mask.get(entry.target_id as usize) else {
            return false;
        };
        *collect & entry.mask as u8 != 0
    }
    pub fn matches(&self, entry: LogEntry) -> bool {
        let Some(collect) = self.mask.get(entry.target_id() as usize) else {
            return false;
        };
        *collect & entry.raw_bloom() != 0
    }
}

#[derive(Debug, Clone)]
pub struct TargetFilter {
    pub predicate: BitSet,
}
impl TargetFilter {
    pub fn matches(&self, entry: LogEntry) -> bool {
        self.predicate.contains(entry.target_id())
    }
}

pub struct WeakEntryCollection {
    bucket_generation: u32,
    entries: Vec<u32>,
}

pub struct EntryCollection<'a> {
    pub(crate) bucket_generation: &'a Bucket,
    pub(crate) entries: Vec<u32>,
}

impl<'a> EntryCollection<'a> {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}
pub struct EntryIterator<'a, 'b> {
    entries: std::slice::Iter<'a, u32>,
    bucket: &'b Bucket,
}
impl<'a, 'b> Iterator for EntryIterator<'a, 'b> {
    type Item = LogEntry<'b>;

    fn next(&mut self) -> Option<Self::Item> {
        self.entries.next().map(|index| LogEntry { index: *index, bucket: self.bucket })
    }
}
impl<'a, 'b> IntoIterator for &'a EntryCollection<'b> {
    type Item = LogEntry<'b>;
    type IntoIter = EntryIterator<'a, 'b>;

    fn into_iter(self) -> Self::IntoIter {
        EntryIterator { entries: self.entries.iter(), bucket: self.bucket_generation }
    }
}
macro_rules! impl_filter_set {
    (pub enum $ident:ident {
        $($variant:ident($type: ty)),* $(,)?
    }) => {
        #[derive(Debug, Clone)]
        pub enum $ident { $($variant($type)),* }
        $(impl From<$type> for $ident {
            fn from(ty: $type) -> $ident { $ident::$variant(ty) }
        })*
        // impl $ident {
        //     pub fn matches(&self, entry: LogEntry, mapper: &Mapper) -> bool {
        //         match self { $( $ident::$variant(ty) => ty.matches(entry, mapper),)* }
        //     }
        // }
    };
}
impl GeneralFilter {
    pub fn matches(&self, entry: LogEntry, mapper: &Mapper) -> bool {
        match self {
            GeneralFilter::Time(filter) => filter.matches(entry),
            GeneralFilter::LevelFilter(filter) => filter.matches(entry),
            GeneralFilter::Query(filter) => filter.query.pred().matches(entry, mapper),
            GeneralFilter::TargetFilter(filter) => filter.matches(entry),
            GeneralFilter::TargetLevelFilter(filter) => filter.matches(entry),
        }
    }
}

impl_filter_set! {
    pub enum GeneralFilter {
        Time(TimeFilter), // handled by time index
        LevelFilter(LevelFilter), // handled by archetype index
        Query(QueryFilter), // existence handled by archetype index
        TargetFilter(TargetFilter), //  handled by archetype index
        TargetLevelFilter(TargetLevelFilter), //  handled by archetype index
    }
}

#[derive(Default)]
enum MaybeLevelFilter {
    #[default]
    None,
    Global(LevelFilter),
    PerTarget(TargetLevelFilter),
}

#[derive(Default)]
pub struct QueryBuilder {
    time_filter: TimeFilter,
    level_filter: MaybeLevelFilter,
    //todo improve
    query: QueryExpr,
}
impl QueryBuilder {
    pub fn min_timestamp_ns(&mut self, tm: u64) {
        self.time_filter.min_utc_ns = tm;
    }
    pub fn max_timestamp_ns(&mut self, tm: u64) {
        self.time_filter.max_utc_ns = tm;
    }
    pub fn with_compact_level_filter_v1(&mut self, level_filter: &[u16]) -> &mut Self {
        match level_filter {
            [] => (),
            [single] => {
                self.level_filter = MaybeLevelFilter::Global(LevelFilter { mask: *single as u8 & 0b1111 });
            }
            [max, default, pairs @ ..] => {
                self.level_filter = MaybeLevelFilter::PerTarget(TargetLevelFilter::from_compact(*max, *default, pairs));
            }
        }
        self
    }
    pub fn with_levels(&mut self, level_filter: LevelFilter) -> &mut Self {
        self.level_filter = MaybeLevelFilter::Global(level_filter);
        self
    }
    pub fn with_expr(&mut self, query: &str) -> Result<&mut Self, QueryParseError> {
        self.with_parsed_expr(QueryExpr::new(query)?);
        Ok(self)
    }
    pub fn with_parsed_expr(&mut self, query: QueryExpr) -> &mut Self {
        self.query = query;
        self
    }
    pub fn build(&mut self) -> Query {
        let mut filters = Vec::<GeneralFilter>::new();
        if self.time_filter != TimeFilter::default() {
            filters.push(std::mem::take(&mut self.time_filter).into());
        }
        match std::mem::take(&mut self.level_filter) {
            MaybeLevelFilter::None => (),
            MaybeLevelFilter::Global(level_filter) => {
                if level_filter != LevelFilter::all() {
                    filters.push(level_filter.into());
                }
            }
            MaybeLevelFilter::PerTarget(target_level_filter) => {
                filters.push(target_level_filter.into());
            }
        }
        // if let Some(predicate) = self.target.take() {
        //     filters.push(TargetFilter { predicate }.into());
        // }
        if !matches!(self.query.pred(), PredBuilder::True) {
            filters.push(QueryFilter { query: Arc::new(std::mem::take(&mut self.query)) }.into());
        }
        // if !self.fields.is_empty() {
        //     self.fields.sort_by_key(|(key, _)| key.0);
        //     filters.push(GeneralFilter::Field(FieldFilter { fields: std::mem::take(&mut self.fields).into() }));
        // }
        Query { filters }
    }
}

pub struct Query {
    pub filters: Vec<GeneralFilter>,
}

impl Query {
    pub fn builder() -> QueryBuilder {
        QueryBuilder::default()
    }
    pub fn expr(string: &str) -> Result<Query, QueryParseError> {
        Ok(Self::builder().with_expr(string)?.build())
    }
}

fn match_all(filters: &[GeneralFilter], entry: LogEntry, mapper: &Mapper) -> bool {
    for filter in filters {
        if !filter.matches(entry, mapper) {
            return false;
        }
    }
    true
}

pub enum QueryStrategy {
    Simple,
    Archetype { archetypes: Box<U16Set> },
    ArchetypeWithFilters { archetypes: Box<U16Set>, filters: Vec<GeneralFilter> },
    ArchetypeWithVm { archetypes: Box<U16Set>, vm: VmFilter },
    ArchetypeContainsField { archetypes: Box<U16Set>, field: Field },
    Empty,
}
impl QueryStrategy {
    fn name(&self) -> &'static str {
        match self {
            QueryStrategy::Simple => "Simple",
            QueryStrategy::Archetype { .. } => "Archetype",
            QueryStrategy::ArchetypeWithFilters { .. } => "ArchetypeWithFilters",
            QueryStrategy::ArchetypeWithVm { .. } => "ArchetypeWithVm",
            QueryStrategy::ArchetypeContainsField { .. } => "ArchetypeContainsField",
            QueryStrategy::Empty => "Empty",
        }
    }
}

pub struct QueryOffset {
    pub index: u32,
    pub generation: u32,
}

pub struct GeneralQuery {
    pub filter: Vec<GeneralFilter>,
    pub offset: Option<QueryOffset>,
}

unsafe impl<'a> Send for ForwardQueryWalker<'a> {}
pub struct ForwardQueryWalker<'a> {
    pub(crate) general_filter: &'a [GeneralFilter],
    pub(crate) time_range: TimeFilter,
    pub(crate) index: &'a IndexReader,
    pub(crate) segments: ForwardSegmentWalker<'a>,
    pub(crate) strategy: QueryStrategy,
}

enum QueryContinuation<'a> {
    Active { bucket: &'a Bucket },
    Reload { bucket: &'a Bucket, previous_summary_count: u32 },
    New { bucket: &'a Bucket },
    Empty,
}

pub struct ForwardSegmentWalker<'a> {
    pub(crate) bucket: Option<BucketGuard<'a>>,
    pub(crate) generation: u32,
    // all for the current generation
    pub(crate) processed: u32,
    pub(crate) summary_count: u32,
    pub(crate) log_count: u32,
}
impl<'a> ForwardSegmentWalker<'a> {
    pub fn new(initial_generation: u32) -> ForwardSegmentWalker<'a> {
        ForwardSegmentWalker {
            bucket: None,
            generation: initial_generation,
            processed: 0,
            summary_count: 0,
            log_count: 0,
        }
    }
    pub fn new_offset(initial_generation: u32, amount: u32) -> ForwardSegmentWalker<'a> {
        ForwardSegmentWalker {
            bucket: None,
            generation: initial_generation,
            processed: amount,
            summary_count: 0,
            log_count: 0,
        }
    }
    fn release(&mut self) {
        self.bucket = None;
    }
    //guarded bucket
    fn next(&mut self, index: &'a IndexReader) -> QueryContinuation<'a> {
        if let Some(bucket) = &mut self.bucket {
            if self.processed < bucket.len as u32 {
                return QueryContinuation::Active { bucket: bucket.bucket };
            }
            bucket.renew();
            if self.processed < bucket.len as u32 {
                let ret =
                    QueryContinuation::Reload { bucket: bucket.bucket, previous_summary_count: self.summary_count };
                self.log_count = bucket.len as u32;
                self.summary_count = bucket.bucket.archetype_count.load(Ordering::Relaxed) as u32;
                return ret;
            }
        } else if let Some(bucket_guard) = index.buckets[(self.generation & 0b11) as usize].read() {
            if bucket_guard.bucket.generation.load(Ordering::Relaxed) == self.generation {
                let bucket = bucket_guard.bucket;
                if self.processed < bucket_guard.len as u32 {
                    let ret = QueryContinuation::Reload { bucket, previous_summary_count: self.summary_count };
                    self.log_count = bucket_guard.len as u32;
                    self.summary_count = bucket.archetype_count.load(Ordering::Relaxed) as u32;
                    return ret;
                }
            }
        }

        loop {
            // TODO: need handle continuing from Empty
            if self.generation >= index.lastest_generation() {
                return QueryContinuation::Empty;
            }
            self.generation += 1;
            self.processed = 0;
            let Some(new_bucket) = index.buckets[(self.generation & 0b11) as usize].read() else {
                if self.generation == index.lastest_generation() {
                    return QueryContinuation::Empty;
                } else {
                    continue;
                }
                // panic!(
                //     "{} No Value but, {}",
                //     self.generation,
                //     index.lastest_generation()
                // );
                // // return QueryContinuation::Empty;
                // // Not sure exactly what needs to happen here need to read the code
            };
            if new_bucket.bucket.generation.load(Ordering::Relaxed) != self.generation {
                self.generation += 1;
                continue;
            }
            self.summary_count = new_bucket.bucket.archetype_count.load(Ordering::Relaxed) as u32;
            self.log_count = new_bucket.len as u32;
            let ret = QueryContinuation::New { bucket: new_bucket.bucket };
            self.bucket = Some(new_bucket);
            self.processed = 0;
            return ret;
        }
    }
}

enum ForwardQueryState {
    Frozen { current_offset: u32, current_limit: u32 },
}

impl<'a> ForwardQueryWalker<'a> {
    // important don't make `'_` be `'a` because then could escape bucket guard
    pub fn next(&mut self) -> Option<EntryCollection<'_>> {
        let bucket = match self.segments.next(self.index) {
            QueryContinuation::Active { bucket } => bucket,
            QueryContinuation::Reload { bucket, previous_summary_count } => bucket,
            QueryContinuation::New { bucket } => bucket,
            QueryContinuation::Empty => {
                return None;
            }
        };
        match &self.strategy {
            QueryStrategy::Empty => {
                let mut entries: Vec<u32> = Vec::new();
                self.segments.processed = self.segments.log_count;
                return Some(EntryCollection { bucket_generation: bucket, entries });
            }
            QueryStrategy::Archetype { .. } => {
                todo!();
            }
            QueryStrategy::ArchetypeWithVm { .. } => {
                todo!();
            }
            QueryStrategy::ArchetypeWithFilters { .. } => {
                todo!();
            }
            QueryStrategy::ArchetypeContainsField { .. } => {
                todo!();
            }
            QueryStrategy::Simple => {
                let mapper = self.index.targets.mapper();
                let mut entries: Vec<u32> = Vec::new();
                let end = (self.segments.processed + 4096).min(self.segments.log_count);
                for i in self.segments.processed..end {
                    let entry = LogEntry { bucket, index: i };
                    if match_all(self.general_filter, entry, &mapper) {
                        entries.push(i);
                        if entries.len() > 255 {
                            self.segments.processed = i + 1;
                            return Some(EntryCollection { bucket_generation: bucket, entries });
                        }
                    }
                }
                self.segments.processed = end;
                Some(EntryCollection { bucket_generation: bucket, entries })
            }
        }
    }
}

unsafe impl<'a> Send for ReverseQueryWalker<'a> {}
pub struct ReverseQueryWalker<'a> {
    pub(crate) general_filter: &'a [GeneralFilter],
    pub(crate) time_range: TimeFilter,
    pub(crate) next_generation: u32,
    pub(crate) index: &'a IndexReader,
    pub(crate) current_bucket: Option<BucketGuard<'a>>,
    pub(crate) current_offset: u32,
    pub(crate) strategy: QueryStrategy,
    pub(crate) buffer: Vec<u32>,
    pub(crate) frozen: bool,
}

#[derive(Default)]
pub struct SpanCache {
    used_spans: hashbrown::HashSet<u32>,
}
impl SpanCache {
    pub fn expand<'a>(&mut self, entries: EntryCollection<'a>) -> EntryCollection<'a> {
        let mut output: Vec<u32> = Vec::new();
        for entry in &entries {
            let raw_span_id = entry.raw_span_id();
            if raw_span_id == u32::MAX {
                continue;
            }
            if !self.used_spans.insert(raw_span_id) {
                continue;
            }
            let span_range = unsafe { &*entries.bucket_generation.span_data.as_ptr().add(raw_span_id as usize) };
            let field_range = span_range.index_range();
            let span_id_slice = unsafe {
                std::slice::from_raw_parts(
                    entries.bucket_generation.span_index.as_ptr().add(field_range.start as usize),
                    field_range.len(),
                )
            };
            let start = entries.bucket_generation.span_index.as_ptr();
            let mut found = 0;
            //todo setup context system.
            for span_id in span_id_slice.into_iter().rev() {
                if *span_id != raw_span_id {
                    continue;
                }
                found += 1;
                output.push(unsafe { (span_id as *const u32).offset_from(start) as u32 });
                // if found > 8 {
                //     break;
                // }
            }
        }
        EntryCollection { bucket_generation: entries.bucket_generation, entries: output }
    }
}

pub struct ReverseQuerySpanGrouper<'a> {
    used_spans: hashbrown::HashSet<u32>,
    walker: ReverseQueryWalker<'a>,
}
impl<'a> ReverseQuerySpanGrouper<'a> {
    pub fn next(&mut self) -> Option<EntryCollection<'_>> {
        let entries = self.walker.next()?;
        let mut output: Vec<u32> = Vec::new();
        for entry in &entries {
            let raw_span_id = entry.raw_span_id();
            if raw_span_id == u32::MAX {
                continue;
            }
            if !self.used_spans.insert(raw_span_id) {
                continue;
            }
            let span_range = unsafe { &*entries.bucket_generation.span_data.as_ptr().add(raw_span_id as usize) };
            let field_range = span_range.index_range();
            let span_id_slice = unsafe {
                std::slice::from_raw_parts(
                    entries.bucket_generation.span_index.as_ptr().add(field_range.start as usize),
                    field_range.len(),
                )
            };
            let start = entries.bucket_generation.span_index.as_ptr();
            let mut found = 0;
            //todo setup context system.
            for span_id in span_id_slice.into_iter().rev() {
                if *span_id != raw_span_id {
                    continue;
                }
                found += 1;
                output.push(unsafe { (span_id as *const u32).offset_from(start) as u32 });
                // if found > 8 {
                //     break;
                // }
            }
        }
        Some(EntryCollection { bucket_generation: entries.bucket_generation, entries: output })
    }
}

fn optimized_general_filters_assuming_matching_archetype_and_no_query(
    unoptimized: &[GeneralFilter],
) -> Vec<GeneralFilter> {
    let mut optimized: Vec<GeneralFilter> = Vec::new();
    for filter in unoptimized {
        match filter {
            GeneralFilter::TargetFilter(..) => continue,
            GeneralFilter::LevelFilter(..) => continue,
            GeneralFilter::Query(filter) => continue,
            other => optimized.push(other.clone()),
        }
    }
    optimized
}

fn specialize_with_archetype_index(
    bucket: &BucketGuard<'_>,
    general_filters: &[GeneralFilter],
    reader: &IndexReader,
) -> QueryStrategy {
    // return QueryStrategy::Simple;
    if general_filters.is_empty() {
        return QueryStrategy::Simple;
    }
    let mut possible_archetypes: Box<U16Set> = Default::default();

    let ibump = Bump::new();

    let bump = &ibump;
    let mut query_filter = None;
    for filter in general_filters {
        if let GeneralFilter::Query(QueryFilter { query }) = filter {
            query_filter = Some(query.pred());
            // todo check if mulitple of them
            break;
        }
    }

    if let Some(pred) = query_filter {
        let maps = bucket.maps();
        let optimized = pred.build_with_opt(bump, bucket, &maps);
        drop(maps);
        match optimized {
            query::query_parts::PredBuildResult::AlwaysTrue => (),
            query::query_parts::PredBuildResult::AlwaysFalse => {
                return QueryStrategy::Empty;
            }
            query::query_parts::PredBuildResult::Ok(mut pred) => {
                let mut filter = ArchetypePrefilter {
                    pred_addr_map: build_pred_addr_map(&pred),
                    always_mask: [0; 2],
                    bucket,
                    targets: reader.targets.mapper(),
                };

                let mut always_mask = [u64::MAX; 2];
                let mut found = 0;
                let archetypes = bucket.archetypes();
                'search: for (archetype_id, archetype) in archetypes.iter().enumerate() {
                    for filter in general_filters {
                        match filter {
                            // one day maybe filtering by time
                            GeneralFilter::Time(_) => (),
                            GeneralFilter::LevelFilter(filter) => {
                                if !archetype.level_in(*filter) {
                                    continue 'search;
                                }
                            }
                            GeneralFilter::Query(_) => (),
                            GeneralFilter::TargetLevelFilter(filter) => {
                                if !filter.matches_archtype(&archetype) {
                                    continue 'search;
                                }
                            }
                            GeneralFilter::TargetFilter(filter) => {
                                if !filter.predicate.contains(archetype.target_id) {
                                    continue 'search;
                                }
                            }
                        }
                    }
                    filter.always_mask = [0; 2];
                    if filter.implicate(&pred, archetype) != Match::AlwaysFalse {
                        possible_archetypes.insert(archetype_id as u16);
                        always_mask[0] &= filter.always_mask[0];
                        always_mask[1] &= filter.always_mask[1];
                        found += 1;
                    }
                }
                if found == 0 {
                    return QueryStrategy::Empty;
                }
                if always_mask != [0; 2] {
                    let mut opt = OptimizationInfo { pred_addr_trival_map: Default::default() };
                    for (&key, &mask) in &filter.pred_addr_map {
                        if mask & always_mask[0] != 0 {
                            opt.pred_addr_trival_map.insert(key, false);
                        } else if mask & always_mask[1] != 0 {
                            opt.pred_addr_trival_map.insert(key, true);
                        }
                    }
                    pred = match pred.reduce(bump, &opt) {
                        crate::query::query_parts::PredBuildResult::AlwaysTrue => {
                            kvlog::debug!(
                                "Using pure archetype filter",
                                ratio = (found as f32 / archetypes.len() as f32)
                            );
                            // todo add trival query straget?
                            // if archetypes.len() == found {
                            //     return QueryStrategy::AlwaysTrue;
                            // }

                            return QueryStrategy::Archetype { archetypes: possible_archetypes };
                        }
                        // this should never happen I don't think, unless possible archetypes is empty.
                        crate::query::query_parts::PredBuildResult::AlwaysFalse => return QueryStrategy::Empty,
                        crate::query::query_parts::PredBuildResult::Ok(pred) => pred,
                    }
                }

                if let Pred::Field(key, FieldTest::EqRaw(false, field)) = pred {
                    // Don't know if this is still faster compared to the VM as currently implemented
                    return QueryStrategy::ArchetypeContainsField { archetypes: possible_archetypes, field };
                }

                match QueryVm::compile(bump, pred, bucket, reader.targets.mapper()) {
                    Ok(vm) => {
                        println!("{:?}", pred);
                        vm.print();
                        let vm_filter = VmFilter {
                            filter: unsafe { std::mem::transmute::<QueryVm, QueryVm<'static>>(vm) },
                            bump: ibump,
                        };

                        return QueryStrategy::ArchetypeWithVm { archetypes: possible_archetypes, vm: vm_filter };
                    }
                    Err(err) => {
                        kvlog::info!("VM failed to compiler query", ?err);
                        // TODO choose more optimial
                        return QueryStrategy::Simple;
                    }
                }
            }
        }
    }

    'search: for (archetype_id, archetype) in bucket.archetypes().iter().enumerate() {
        for filter in general_filters {
            match filter {
                // one day maybe filtering by time
                GeneralFilter::Time(_) => continue,
                GeneralFilter::LevelFilter(filter) => {
                    if !archetype.level_in(*filter) {
                        continue 'search;
                    }
                }
                GeneralFilter::Query(_) => continue,
                GeneralFilter::TargetLevelFilter(filter) => {
                    if !filter.matches_archtype(&archetype) {
                        continue 'search;
                    }
                }
                GeneralFilter::TargetFilter(filter) => {
                    if !filter.predicate.contains(archetype.target_id) {
                        continue 'search;
                    }
                }
            }
        }
        possible_archetypes.insert(archetype_id as u16);
    }
    if possible_archetypes.is_empty() {
        return QueryStrategy::Empty;
    }
    let remaining_filters = optimized_general_filters_assuming_matching_archetype_and_no_query(general_filters);
    if remaining_filters.is_empty() {
        return QueryStrategy::Archetype { archetypes: possible_archetypes };
    }

    return QueryStrategy::ArchetypeWithFilters { archetypes: possible_archetypes, filters: remaining_filters };
}

impl<'a> ReverseQueryWalker<'a> {
    pub fn grouped_by_spans(self) -> ReverseQuerySpanGrouper<'a> {
        ReverseQuerySpanGrouper { used_spans: hashbrown::HashSet::default(), walker: self }
    }
    fn specialize(&mut self, bucket: &BucketGuard<'a>) {
        self.strategy = specialize_with_archetype_index(bucket, &self.general_filter, self.index);
        kvlog::info!("Specialized Strategy", strategy = self.strategy.name());
    }
    pub fn release_bucket_reclamation_lock(&mut self) {
        if let Some(bucket) = self.current_bucket.take() {
            self.next_generation = bucket.bucket.generation.load(Ordering::Relaxed);
            self.frozen = true;
        }
    }

    fn load_bucket(&mut self) -> bool {
        if let Some(bucket) = &self.current_bucket {
            if self.time_range.min_utc_ns != 0 {
                self.current_offset = bucket.reverse_time_range_skip(self.current_offset, self.time_range.clone());
            }
            if self.current_offset != 0 {
                return true;
            }
        };
        loop {
            if let Some(new_bucket) = self.index.buckets[(self.next_generation & 0b11) as usize].read() {
                if new_bucket.bucket.generation.load(Ordering::Relaxed) != self.next_generation {
                    return false;
                }
                if self.frozen {
                    self.current_offset = (new_bucket.len as u32).min(self.current_offset);
                    self.frozen = false;
                } else {
                    self.current_offset = new_bucket.len as u32;
                }
                if self.time_range.min_utc_ns != 0 {
                    self.current_offset =
                        new_bucket.reverse_time_range_skip(self.current_offset, self.time_range.clone());
                }
                if self.current_offset != 0 {
                    self.next_generation = self.next_generation.wrapping_sub(1);
                    self.specialize(&new_bucket);
                    self.current_bucket = Some(new_bucket);
                    return true;
                }
            }
            if self.next_generation == 0 {
                return false;
            }
            self.next_generation -= 1;
        }
    }
    pub fn forward_query(&mut self) -> ForwardQueryWalker<'a> {
        //maybe?
        let segments = if self.load_bucket() {
            ForwardSegmentWalker::new_offset(self.next_generation.wrapping_add(1), self.current_offset)
        } else {
            ForwardSegmentWalker::new(self.next_generation)
        };
        ForwardQueryWalker {
            general_filter: &self.general_filter,
            time_range: self.time_range.clone(),
            segments,
            index: &self.index,
            strategy: QueryStrategy::Simple,
        }
    }
    // important don't make `'_` be `'a` because then could escape bucket guard
    pub fn next(&mut self) -> Option<EntryCollection<'_>> {
        if !self.load_bucket() {
            return None;
        }
        let Some(bucket) = &self.current_bucket else { unreachable!() };

        match &mut self.strategy {
            QueryStrategy::Empty => {
                let mut entries: Vec<u32> = Vec::new();
                self.current_offset = 0;
                return Some(EntryCollection { bucket_generation: bucket.bucket, entries });
            }
            QueryStrategy::Archetype { archetypes } => {
                let mut entries: Vec<u32> = Vec::new();
                let until = self.current_offset.saturating_sub(4096 * 16) as usize;
                self.current_offset = collect_indices_of_values_in_set_reverse(
                    archetypes,
                    bucket.archetype_index(),
                    until as usize..self.current_offset as usize,
                    &mut entries,
                    256,
                ) as u32;
                Some(EntryCollection { bucket_generation: bucket.bucket, entries })
            }
            QueryStrategy::ArchetypeContainsField { archetypes: set, field: exact } => {
                let mut entries: Vec<u32> = Vec::new();
                let mut end = self.current_offset as usize;
                let until = self.current_offset.saturating_sub(4096 * 4) as usize;
                while until < end {
                    self.buffer.clear();
                    end = collect_indices_of_values_in_set_reverse(
                        set,
                        bucket.archetype_index(),
                        until as usize..end as usize,
                        &mut self.buffer,
                        256,
                    );
                    for i in &self.buffer {
                        let entry = LogEntry { bucket: bucket.bucket, index: *i };
                        //todo try contains
                        if entry.raw_fields().contains(exact) {
                            entries.push(*i);
                        }
                    }
                    if entries.len() > 255 {
                        self.current_offset = end as u32;
                        return Some(EntryCollection { bucket_generation: bucket.bucket, entries });
                    }
                }
                self.current_offset = end as u32;
                Some(EntryCollection { bucket_generation: bucket.bucket, entries })
            }
            QueryStrategy::ArchetypeWithVm { archetypes: set, vm } => {
                let mut entries: Vec<u32> = Vec::new();
                let mut end = self.current_offset as usize;
                let until = self.current_offset.saturating_sub(4096 * 4) as usize;
                while until < end {
                    self.buffer.clear();
                    end = collect_indices_of_values_in_set_reverse(
                        set,
                        bucket.archetype_index(),
                        until as usize..end as usize,
                        &mut self.buffer,
                        256,
                    );
                    for i in &self.buffer {
                        let entry = LogEntry { bucket: bucket.bucket, index: *i };
                        if vm.query().matches(entry) {
                            entries.push(*i);
                        }
                    }
                    if entries.len() > 255 {
                        self.current_offset = end as u32;
                        return Some(EntryCollection { bucket_generation: bucket.bucket, entries });
                    }
                }
                self.current_offset = end as u32;
                Some(EntryCollection { bucket_generation: bucket.bucket, entries })
            }
            QueryStrategy::ArchetypeWithFilters { archetypes: set, filters } => {
                let mapper = self.index.targets.mapper();
                let mut entries: Vec<u32> = Vec::new();
                let mut end = self.current_offset as usize;
                let until = self.current_offset.saturating_sub(4096 * 4) as usize;
                while until < end {
                    self.buffer.clear();
                    end = collect_indices_of_values_in_set_reverse(
                        set,
                        bucket.archetype_index(),
                        until as usize..end as usize,
                        &mut self.buffer,
                        256,
                    );
                    for i in &self.buffer {
                        let entry = LogEntry { bucket: bucket.bucket, index: *i };
                        if match_all(filters, entry, &mapper) {
                            entries.push(*i);
                        }
                    }
                    if entries.len() > 255 {
                        self.current_offset = end as u32;
                        return Some(EntryCollection { bucket_generation: bucket.bucket, entries });
                    }
                }
                self.current_offset = end as u32;
                Some(EntryCollection { bucket_generation: bucket.bucket, entries })
            }
            QueryStrategy::Simple => {
                let mapper = self.index.targets.mapper();
                let mut entries: Vec<u32> = Vec::new();
                for i in (self.current_offset.saturating_sub(4096)..self.current_offset).rev() {
                    let entry = LogEntry { bucket: bucket.bucket, index: i };
                    if match_all(self.general_filter, entry, &mapper) {
                        entries.push(i);
                        if entries.len() > 255 {
                            self.current_offset = i;
                            return Some(EntryCollection { bucket_generation: bucket.bucket, entries });
                        }
                    }
                }
                self.current_offset = self.current_offset.saturating_sub(4096);
                Some(EntryCollection { bucket_generation: bucket.bucket, entries })
            }
        }
    }
}

#[derive(Default, Clone)]
pub struct KeyMapCache {
    pub query: [u16; 8],
    index_cache: [(u16, [u8; 8]); 32],
    pub comps: u32,
    pub len: u32,
}
impl KeyMapCache {
    #[inline]
    pub fn lookup(&mut self, entry: LogEntry) -> &[u8; 8] {
        let archetype_index = entry.raw_archetype();
        let (current, bucket) = &mut self.index_cache[(archetype_index & 0b11111) as usize];
        if *current == archetype_index {
            return bucket;
        }

        *current = archetype_index;
        *bucket = [u8::MAX; 8];
        unsafe {
            std::hint::assert_unchecked(self.len <= 8);
            load_inputs(&self.query[0..self.len as usize], archetype_index, entry.bucket, bucket);
        }
        return bucket;
    }
    pub fn new(keys: &[KeyID]) -> Box<KeyMapCache> {
        unsafe {
            let ptr = std::alloc::alloc_zeroed(std::alloc::Layout::new::<KeyMapCache>());
            if ptr.is_null() {
                std::alloc::handle_alloc_error(std::alloc::Layout::new::<KeyMapCache>());
            }
            let mut cache = Box::from_raw(ptr as *mut KeyMapCache);
            for (i, key) in keys.iter().enumerate() {
                cache.query[i] = key.0;
            }
            for entry in &mut cache.index_cache {
                entry.0 = u16::MAX;
            }
            cache.len = keys.len().max(8) as u32;
            cache
        }
    }
    // #[inline]
    // pub fn lookup(&mut self, entry: LogEntry) -> &[u8; 8] {
}

#[cold]
#[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "bmi2"))]
unsafe fn load_inputs(query: &[u16], raw_archetype: u16, bucket: &Bucket, output_indices: &mut [u8; 8]) {
    std::hint::assert_unchecked(query.len() <= 8);
    let arch = bucket.archetype(raw_archetype);
    use core::arch::x86_64::*;

    let data_ptr = arch.field_headers.as_ptr() as *const __m128i;
    let simd_data: __m128i = _mm_loadu_si128(data_ptr);
    for (i, key) in query.iter().enumerate() {
        let simd_value: __m128i = _mm_set1_epi16(*key as i16);
        let cmp_result: __m128i = _mm_cmpeq_epi16(simd_data, simd_value);
        let mask: i32 = _mm_movemask_epi8(cmp_result);

        if mask != 0 {
            let lsb_bit_index: u32 = _tzcnt_u32(mask as u32);
            // The u16 element index is half the bit index.
            let index = (lsb_bit_index / 2);
            output_indices[i] = index as u8;
        }
    }
}

#[cold]
#[cfg(not(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "bmi2")))]
unsafe fn load_inputs(query: &[u16], raw_archetype: u16, bucket: &Bucket, output_indices: &mut [u8; 8]) {
    std::hint::assert_unchecked(query.len() <= 8);
    let fields = bucket.archetype(raw_archetype);
    for (i, key) in query.iter().enumerate() {
        if let Some(j) = fields.index_of(KeyID(*key)) {
            output_indices[i] = j as u8;
        }
    }
}

fn inputs(query: &[u16; 8], fields: &[u16; 8], output_indices: &mut [u8; 8]) {
    let mut query_idx: usize = 0; // Pointer for the query array
    let mut field_idx: usize = 0; // Pointer for the fields slice

    // Iterate while there are elements to process in both the query array
    // and the fields slice.
    while query_idx < 8 && field_idx < fields.len() {
        if query[query_idx] ^ fields[field_idx] <= 0xf {
            // Found a match: query[query_idx] is equal to fields[field_idx].
            // Store the index from the fields slice.
            output_indices[query_idx] = field_idx as u8;
            query_idx += 1;
            field_idx += 1;
        } else if query[query_idx] < fields[field_idx] {
            // The current query element is smaller than the current fields element.
            // Since both arrays are sorted, query[query_idx] cannot be found
            // at or after the current position in fields.
            // Its output_indices entry remains u16::MAX (set during initialization).
            // Move to the next element in the query array.
            output_indices[query_idx] = u8::MAX;
            query_idx += 1;
        } else {
            // query[query_idx] > fields[field_idx]
            // The current query element is larger than the current fields element.
            // We need to find a larger or equal element in the fields slice.
            // Move to the next element in the fields slice.
            field_idx += 1;
        }
    }
}

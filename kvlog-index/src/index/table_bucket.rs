use std::{
    marker::PhantomData,
    mem::{align_of, size_of},
    ptr::NonNull,
};

use crate::field_table::MIN_DYN_KEY;
use crate::shared_interner::{LocalIntermentCache, Mapper, SharedIntermentBuffer};

use super::*;

const SNAPSHOT_MAGIC: [u8; 4] = *b"KVBS";
const SNAPSHOT_VERSION: u32 = 6;
const SNAPSHOT_ENDIAN_MARKER: u32 = 0x0102_0304;
const SNAPSHOT_ALIGNMENT: usize = 64;
const SNAPSHOT_HEADER_LEN: usize = 136;
const SNAPSHOT_SECTION_LEN: usize = 24;

const GLOBAL_MAGIC: [u8; 4] = *b"KVGI";
const GLOBAL_VERSION: u32 = 1;
const GLOBAL_HEADER_LEN: usize = 44;

const SECTION_DATA: u32 = 1;
const SECTION_FIELD: u32 = 2;
const SECTION_SPAN: u32 = 3;
const SECTION_ARCHETYPE: u32 = 4;
const SECTION_SPAN_INDEX: u32 = 5;
const SECTION_TIMESTAMP: u32 = 6;
const SECTION_ARCHETYPE_INDEX: u32 = 8;
const SECTION_TIME_RANGE: u32 = 9;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug)]
pub enum SnapshotWriteError {
    MissingGlobalTarget(u16),
    MissingGlobalKey(u16),
    MissingGlobalService(u8),
    LengthOverflow,
}

impl std::fmt::Display for SnapshotWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnapshotWriteError::MissingGlobalTarget(id) => {
                write!(f, "snapshot target id {id} is not present in globals")
            }
            SnapshotWriteError::MissingGlobalKey(id) => write!(f, "snapshot key id {id} is not present in globals"),
            SnapshotWriteError::MissingGlobalService(id) => {
                write!(f, "snapshot service id {id} is not present in globals")
            }
            SnapshotWriteError::LengthOverflow => write!(f, "snapshot length does not fit in u64"),
        }
    }
}

impl std::error::Error for SnapshotWriteError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationMode {
    Basic,
    Full,
}

#[derive(Debug)]
pub enum SnapshotLoadError {
    BufferNotAligned { required: usize },
    TruncatedHeader,
    InvalidMagic,
    UnsupportedVersion(u32),
    EndianMismatch,
    LayoutMismatch,
    GlobalsMismatch,
    InvalidHeader,
    InvalidSectionTable,
    InvalidSection(u32),
    InvalidLength,
    CountExceeded(&'static str),
    InvalidOffsetTable,
    InvalidReference(&'static str),
    InvalidGlobalReference(&'static str),
    InvalidTimeRange,
    InvalidSpan,
}

impl std::fmt::Display for SnapshotLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnapshotLoadError::BufferNotAligned { required } => {
                write!(f, "snapshot buffer is not aligned to {required} bytes")
            }
            SnapshotLoadError::TruncatedHeader => write!(f, "snapshot is truncated before the header"),
            SnapshotLoadError::InvalidMagic => write!(f, "snapshot magic mismatch"),
            SnapshotLoadError::UnsupportedVersion(version) => write!(f, "unsupported snapshot version {version}"),
            SnapshotLoadError::EndianMismatch => write!(f, "snapshot endian marker does not match this process"),
            SnapshotLoadError::LayoutMismatch => write!(f, "snapshot native layout fingerprint does not match"),
            SnapshotLoadError::GlobalsMismatch => write!(f, "snapshot globals fingerprint does not match"),
            SnapshotLoadError::InvalidHeader => write!(f, "snapshot header is invalid"),
            SnapshotLoadError::InvalidSectionTable => write!(f, "snapshot section table is invalid"),
            SnapshotLoadError::InvalidSection(kind) => write!(f, "snapshot section {kind} is invalid"),
            SnapshotLoadError::InvalidLength => write!(f, "snapshot byte length is invalid"),
            SnapshotLoadError::CountExceeded(name) => write!(f, "snapshot {name} count exceeds bucket limits"),
            SnapshotLoadError::InvalidOffsetTable => write!(f, "snapshot offset table is invalid"),
            SnapshotLoadError::InvalidReference(name) => write!(f, "snapshot contains invalid {name} reference"),
            SnapshotLoadError::InvalidGlobalReference(name) => {
                write!(f, "snapshot contains invalid global {name} reference")
            }
            SnapshotLoadError::InvalidTimeRange => write!(f, "snapshot timerange section is invalid"),
            SnapshotLoadError::InvalidSpan => write!(f, "snapshot span section is invalid"),
        }
    }
}

impl std::error::Error for SnapshotLoadError {}

#[derive(Debug)]
pub enum GlobalInternError {
    InvalidMagic,
    UnsupportedVersion(u32),
    EndianMismatch,
    Truncated,
    InvalidUtf8,
    FingerprintMismatch,
    RawKeyMismatch { expected: u16, actual: u16 },
    RawServiceMismatch { expected: u8, actual: u8 },
    RawTargetMismatch { expected: u16, actual: u16 },
    TooManyNames,
}

impl std::fmt::Display for GlobalInternError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GlobalInternError::InvalidMagic => write!(f, "global interner magic mismatch"),
            GlobalInternError::UnsupportedVersion(version) => {
                write!(f, "unsupported global interner version {version}")
            }
            GlobalInternError::EndianMismatch => write!(f, "global interner endian marker does not match"),
            GlobalInternError::Truncated => write!(f, "global interner data is truncated"),
            GlobalInternError::InvalidUtf8 => write!(f, "global interner name is not valid utf-8"),
            GlobalInternError::FingerprintMismatch => write!(f, "global interner fingerprint mismatch"),
            GlobalInternError::RawKeyMismatch { expected, actual } => {
                write!(f, "dynamic key raw id mismatch: expected {expected}, got {actual}")
            }
            GlobalInternError::RawServiceMismatch { expected, actual } => {
                write!(f, "service raw id mismatch: expected {expected}, got {actual}")
            }
            GlobalInternError::RawTargetMismatch { expected, actual } => {
                write!(f, "target raw id mismatch: expected {expected}, got {actual}")
            }
            GlobalInternError::TooManyNames => write!(f, "global interner contains too many names"),
        }
    }
}

impl std::error::Error for GlobalInternError {}

/// Persisted global interner state required by native bucket snapshots.
///
/// Dynamic key and service names are validated against the process-global
/// interners in raw-id order. Target names are loaded into an owned
/// `SharedIntermentBuffer`, also in raw-id order.
pub struct PersistentInterners {
    dynamic_keys: Vec<String>,
    services: Vec<String>,
    targets: SharedIntermentBuffer,
    target_names: Vec<String>,
    fingerprint: u64,
}

impl PersistentInterners {
    pub fn from_names(
        dynamic_keys: &[&str],
        services: &[&str],
        targets: &[&str],
    ) -> Result<PersistentInterners, GlobalInternError> {
        Self::from_owned(
            dynamic_keys.iter().map(|s| (*s).to_owned()).collect(),
            services.iter().map(|s| (*s).to_owned()).collect(),
            targets.iter().map(|s| (*s).to_owned()).collect(),
        )
    }

    pub fn capture_from_reader(reader: &IndexReader) -> PersistentInterners {
        let dynamic_keys = KeyID::known_dynamic().into_iter().map(|(_, name)| name.to_owned()).collect();
        let services = ServiceId::known().map(|service| service.as_str().to_owned()).collect();
        let mut targets: Vec<(u16, String)> = reader.targets.iter().map(|(id, name)| (id, name.to_owned())).collect();
        targets.sort_by_key(|(id, _)| *id);
        let target_names = targets.into_iter().map(|(_, name)| name).collect();
        Self::from_owned(dynamic_keys, services, target_names).expect("captured globals must validate")
    }

    pub fn decode(bytes: &[u8]) -> Result<PersistentInterners, GlobalInternError> {
        let mut cursor = 0usize;
        if bytes.len() < GLOBAL_HEADER_LEN {
            return Err(GlobalInternError::Truncated);
        }
        if read_fixed::<4>(bytes, &mut cursor)? != GLOBAL_MAGIC {
            return Err(GlobalInternError::InvalidMagic);
        }
        let version = read_u32_ne(bytes, &mut cursor)?;
        if version != GLOBAL_VERSION {
            return Err(GlobalInternError::UnsupportedVersion(version));
        }
        if read_u32_ne(bytes, &mut cursor)? != SNAPSHOT_ENDIAN_MARKER {
            return Err(GlobalInternError::EndianMismatch);
        }
        let fingerprint = read_u64_ne(bytes, &mut cursor)?;
        let dynamic_count = read_u64_ne(bytes, &mut cursor)? as usize;
        let service_count = read_u64_ne(bytes, &mut cursor)? as usize;
        let target_count = read_u64_ne(bytes, &mut cursor)? as usize;
        if cursor != GLOBAL_HEADER_LEN {
            return Err(GlobalInternError::Truncated);
        }

        let dynamic_keys = read_name_list(bytes, &mut cursor, dynamic_count)?;
        let services = read_name_list(bytes, &mut cursor, service_count)?;
        let targets = read_name_list(bytes, &mut cursor, target_count)?;
        if cursor != bytes.len() {
            return Err(GlobalInternError::Truncated);
        }

        let globals = Self::from_owned(dynamic_keys, services, targets)?;
        if globals.fingerprint != fingerprint {
            return Err(GlobalInternError::FingerprintMismatch);
        }
        Ok(globals)
    }

    pub fn encode(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(&GLOBAL_MAGIC);
        push_u32(out, GLOBAL_VERSION);
        push_u32(out, SNAPSHOT_ENDIAN_MARKER);
        push_u64(out, self.fingerprint);
        push_u64(out, self.dynamic_keys.len() as u64);
        push_u64(out, self.services.len() as u64);
        push_u64(out, self.target_names.len() as u64);
        write_name_list(out, &self.dynamic_keys);
        write_name_list(out, &self.services);
        write_name_list(out, &self.target_names);
    }

    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    pub fn target_mapper(&self) -> Mapper<'_> {
        self.targets.mapper()
    }

    fn from_owned(
        dynamic_keys: Vec<String>,
        services: Vec<String>,
        target_names: Vec<String>,
    ) -> Result<PersistentInterners, GlobalInternError> {
        if dynamic_keys.len() > (u16::MAX - MIN_DYN_KEY) as usize
            || services.len() > u8::MAX as usize
            || target_names.len() > u16::MAX as usize + 1
        {
            return Err(GlobalInternError::TooManyNames);
        }

        for (index, name) in dynamic_keys.iter().enumerate() {
            let expected = MIN_DYN_KEY + index as u16;
            let actual = KeyID::intern(name).raw();
            if actual != expected {
                return Err(GlobalInternError::RawKeyMismatch { expected, actual });
            }
        }

        for (index, name) in services.iter().enumerate() {
            let expected = (index + 1) as u8;
            let actual = ServiceId::intern(name).as_u8();
            if actual != expected {
                return Err(GlobalInternError::RawServiceMismatch { expected, actual });
            }
        }

        let targets = SharedIntermentBuffer::with_capacity(target_data_capacity(&target_names));
        let mut target_cache = LocalIntermentCache::default();
        for (index, name) in target_names.iter().enumerate() {
            let expected = index as u16;
            // The buffer is sized exactly for these names, so overflow on
            // load is itself a corruption / mismatch case rather than a
            // runtime ingest condition.
            let actual = target_cache
                .intern(&targets, name.as_bytes())
                .ok_or(GlobalInternError::TooManyNames)?;
            if actual != expected {
                return Err(GlobalInternError::RawTargetMismatch { expected, actual });
            }
        }

        let fingerprint = globals_fingerprint(&dynamic_keys, &services, &target_names);
        Ok(PersistentInterners { dynamic_keys, services, targets, target_names, fingerprint })
    }

    fn has_key_raw(&self, raw: u16) -> bool {
        if raw < MIN_DYN_KEY {
            return KeyID::try_raw_to_str(raw).is_some();
        }
        let index = (raw - MIN_DYN_KEY) as usize;
        let Some(expected) = self.dynamic_keys.get(index) else {
            return false;
        };
        KeyID::try_raw_to_str(raw) == Some(expected.as_str())
    }

    fn has_service_raw(&self, raw: u8) -> bool {
        if raw == 0 {
            return true;
        }
        let Some(expected) = self.services.get(raw as usize - 1) else {
            return false;
        };
        ServiceId::known().any(|service| service.as_u8() == raw && service.as_str() == expected)
    }

    fn has_target_raw(&self, raw: u16) -> bool {
        self.target_names.get(raw as usize).is_some()
    }
}

pub struct BucketSnapshotScratch {
    prefix: Vec<u8>,
    timestamps: Vec<u8>,
    padding: [u8; SNAPSHOT_ALIGNMENT],
}

impl Default for BucketSnapshotScratch {
    fn default() -> Self {
        BucketSnapshotScratch {
            prefix: Vec::new(),
            timestamps: Vec::new(),
            padding: [0; SNAPSHOT_ALIGNMENT],
        }
    }
}

pub struct BucketSnapshotSlices<'a> {
    pub parts: Vec<&'a [u8]>,
    pub byte_len: usize,
    pub required_load_align: usize,
}

impl<'a> BucketSnapshotSlices<'a> {
    pub fn iter(&self) -> impl Iterator<Item = &'a [u8]> + '_ {
        self.parts.iter().copied()
    }
}

#[derive(Clone, Copy)]
struct SectionPlan {
    kind: u32,
    align: usize,
    offset: usize,
    len: usize,
}

#[derive(Clone, Copy)]
struct ParsedHeader {
    total_len: usize,
    layout_fingerprint: u64,
    globals_fingerprint: u64,
    generation: u64,
    log_count: usize,
    field_count: usize,
    data_len: usize,
    span_count: usize,
    archetype_count: usize,
    timerange_count: usize,
    general_range_count: usize,
    uuid_offset_count: usize,
    msg_count: usize,
}

pub struct LoadedBucket<'a> {
    bucket: Bucket,
    globals: &'a PersistentInterners,
    // Decompressed offsets reconstructed from each log's archetype size.
    // The bucket's `offset` pointer aliases this allocation, so it must
    // outlive the bucket. Boxed to keep the address stable across moves.
    _offsets: Box<[u32]>,
    // Decoded timestamps reconstructed from the byte-shuffled zigzag-delta
    // stream. The bucket's `timestamp` pointer aliases this allocation.
    _timestamps: Box<[u64]>,
    _bytes: PhantomData<&'a [u8]>,
}

impl<'a> LoadedBucket<'a> {
    pub fn borrow(
        bytes: &'a [u8],
        globals: &'a PersistentInterners,
        mode: ValidationMode,
    ) -> Result<LoadedBucket<'a>, SnapshotLoadError> {
        if (bytes.as_ptr() as usize) & (SNAPSHOT_ALIGNMENT - 1) != 0 {
            return Err(SnapshotLoadError::BufferNotAligned { required: SNAPSHOT_ALIGNMENT });
        }

        let (header, sections) = parse_snapshot_header(bytes, globals)?;
        validate_counts(&header)?;

        let data = section_bytes(bytes, sections[0])?;
        let fields = typed_section::<Field>(bytes, sections[1], header.field_count)?;
        let spans = typed_section::<SpanRange>(bytes, sections[2], header.span_count)?;
        let archetypes = typed_section::<archetype::Archetype>(bytes, sections[3], header.archetype_count)?;
        let span_indices = typed_section::<u32>(bytes, sections[4], header.log_count)?;
        let timestamp_bytes = section_bytes(bytes, sections[5])?;
        let archetype_indices = typed_section::<u16>(bytes, sections[6], header.log_count)?;
        let timeranges = typed_section::<TimeRange>(bytes, sections[7], header.timerange_count)?;

        if data.len() != header.data_len {
            return Err(SnapshotLoadError::InvalidSection(SECTION_DATA));
        }
        let random_state = RandomState::new();
        let intern_maps = new_loaded_intern_maps(header.data_len, header.msg_count);
        validate_archetypes(archetypes, globals, header.data_len)?;
        let offsets = decode_offsets(archetype_indices, archetypes, header.field_count)?;
        let timestamps = decode_timestamps(timestamp_bytes, header.log_count)?;
        validate_records(fields, span_indices, data, header.span_count)?;
        validate_spans(spans, header.log_count)?;
        if mode == ValidationMode::Full {
            validate_timeranges(timeranges, &timestamps)?;
        }

        let bucket = Bucket {
            generation: AtomicU32::new(header.generation as u32),
            intern_map: Mutex::new(intern_maps),
            data: ptr_for_slice(data),
            random_state,
            field: ptr_for_slice(fields),
            timerange: ptr_for_slice(timeranges),
            span_index: ptr_for_slice(span_indices),
            span_data: ptr_for_slice(spans),
            span_count: AtomicUsize::new(header.span_count),
            timestamp: ptr_for_slice(&timestamps),
            offset: ptr_for_slice(&offsets),
            len: AtomicUsize::new(header.log_count),
            ref_count: AtomicU32::new(1),
            archetype: ptr_for_slice(archetypes),
            archetype_index: ptr_for_slice(archetype_indices),
            archetype_count: AtomicUsize::new(header.archetype_count),
        };

        let loaded =
            LoadedBucket { bucket, globals, _offsets: offsets, _timestamps: timestamps, _bytes: PhantomData };
        Ok(loaded)
    }

    pub fn read(&self) -> Option<BucketGuard<'_>> {
        self.bucket.read()
    }

    pub fn target_mapper(&self) -> Mapper<'_> {
        self.globals.target_mapper()
    }

    pub fn query(&self, query: &QueryExpr, mut func: impl FnMut(LogEntry<'_>) -> bool) {
        let mapper = self.globals.target_mapper();
        let Some(bucket) = self.read() else {
            return;
        };
        for entry in bucket.entries().rev() {
            if query.pred().matches(entry, &mapper) && !func(entry) {
                return;
            }
        }
    }
}

impl<'a> BucketGuard<'a> {
    pub fn snapshot_slices<'b>(
        &'b self,
        scratch: &'b mut BucketSnapshotScratch,
        globals: &'b PersistentInterners,
    ) -> Result<BucketSnapshotSlices<'b>, SnapshotWriteError> {
        validate_bucket_globals(self, globals)?;

        scratch.prefix.clear();

        let spans = self.spans();
        let archetypes = self.archetypes();
        let field_count = unsafe { self.bucket.offset.add(self.len).read() } as usize;
        let timerange_count = self.len / TIME_RANGE_LOG_COUNT;
        let data_len;
        let msg_count;
        {
            let maps = self.bucket.intern_map.lock().unwrap();
            data_len = maps.data_len;
            msg_count = maps.msgs;
        }

        encode_timestamps(self.timestamps(), &mut scratch.timestamps);

        let sections = build_section_plan(&[
            (SECTION_DATA, 1, data_len),
            (SECTION_FIELD, align_of::<Field>(), checked_bytes::<Field>(field_count)?),
            (SECTION_SPAN, align_of::<SpanRange>(), checked_bytes::<SpanRange>(spans.len())?),
            (
                SECTION_ARCHETYPE,
                align_of::<archetype::Archetype>(),
                checked_bytes::<archetype::Archetype>(archetypes.len())?,
            ),
            (SECTION_SPAN_INDEX, align_of::<u32>(), checked_bytes::<u32>(self.len)?),
            (SECTION_TIMESTAMP, 1, scratch.timestamps.len()),
            (SECTION_ARCHETYPE_INDEX, align_of::<u16>(), checked_bytes::<u16>(self.len)?),
            (SECTION_TIME_RANGE, align_of::<TimeRange>(), checked_bytes::<TimeRange>(timerange_count)?),
        ])?;
        let total_len = sections.last().map(|section| section.offset + section.len).unwrap_or(SNAPSHOT_HEADER_LEN);

        write_snapshot_header(
            &mut scratch.prefix,
            &sections,
            total_len,
            globals.fingerprint(),
            self.bucket.generation.load(Ordering::Relaxed) as u64,
            self.len,
            field_count,
            data_len,
            spans.len(),
            archetypes.len(),
            timerange_count,
            0,
            0,
            msg_count,
        );
        let first_offset = sections.first().map(|section| section.offset).unwrap_or(total_len);
        scratch.prefix.extend_from_slice(&scratch.padding[..first_offset - scratch.prefix.len()]);

        let data = unsafe { std::slice::from_raw_parts(self.bucket.data.as_ptr(), data_len) };
        let fields = unsafe { std::slice::from_raw_parts(self.bucket.field.as_ptr(), field_count) };
        let span_indices = self.span_indices();
        let archetype_indices = self.archetype_index();
        let timeranges = unsafe { std::slice::from_raw_parts(self.bucket.timerange.as_ptr(), timerange_count) };

        let section_bytes: [&[u8]; 8] = [
            data,
            unsafe { slice_as_bytes(fields) },
            unsafe { slice_as_bytes(spans) },
            unsafe { slice_as_bytes(archetypes) },
            unsafe { slice_as_bytes(span_indices) },
            scratch.timestamps.as_slice(),
            unsafe { slice_as_bytes(archetype_indices) },
            unsafe { slice_as_bytes(timeranges) },
        ];

        let mut parts = Vec::with_capacity(1 + sections.len() * 2);
        parts.push(scratch.prefix.as_slice());
        for (index, section) in sections.iter().enumerate() {
            parts.push(section_bytes[index]);
            let end = section.offset + section.len;
            let next = sections.get(index + 1).map(|next| next.offset).unwrap_or(total_len);
            let padding = next - end;
            if padding != 0 {
                parts.push(&scratch.padding[..padding]);
            }
        }

        Ok(BucketSnapshotSlices { parts, byte_len: total_len, required_load_align: SNAPSHOT_ALIGNMENT })
    }
}

fn validate_bucket_globals(bucket: &BucketGuard<'_>, globals: &PersistentInterners) -> Result<(), SnapshotWriteError> {
    for archetype in bucket.archetypes() {
        if !globals.has_target_raw(archetype.target_id) {
            return Err(SnapshotWriteError::MissingGlobalTarget(archetype.target_id));
        }
        let service = archetype.raw_service();
        if !globals.has_service_raw(service) {
            return Err(SnapshotWriteError::MissingGlobalService(service));
        }
        for key in archetype.field_keys() {
            if !globals.has_key_raw(key.raw()) {
                return Err(SnapshotWriteError::MissingGlobalKey(key.raw()));
            }
        }
    }
    Ok(())
}

fn build_section_plan(specs: &[(u32, usize, usize)]) -> Result<Vec<SectionPlan>, SnapshotWriteError> {
    let mut current = SNAPSHOT_HEADER_LEN + specs.len() * SNAPSHOT_SECTION_LEN;
    let mut sections = Vec::with_capacity(specs.len());
    for &(kind, align, len) in specs {
        let offset = align_up(current, align);
        sections.push(SectionPlan { kind, align, offset, len });
        current = offset.checked_add(len).ok_or(SnapshotWriteError::LengthOverflow)?;
    }
    Ok(sections)
}

fn checked_bytes<T>(len: usize) -> Result<usize, SnapshotWriteError> {
    len.checked_mul(size_of::<T>()).ok_or(SnapshotWriteError::LengthOverflow)
}

fn write_snapshot_header(
    out: &mut Vec<u8>,
    sections: &[SectionPlan],
    total_len: usize,
    globals_fingerprint: u64,
    generation: u64,
    log_count: usize,
    field_count: usize,
    data_len: usize,
    span_count: usize,
    archetype_count: usize,
    timerange_count: usize,
    general_range_count: usize,
    uuid_offset_count: usize,
    msg_count: usize,
) {
    out.extend_from_slice(&SNAPSHOT_MAGIC);
    push_u32(out, SNAPSHOT_VERSION);
    push_u32(out, SNAPSHOT_ENDIAN_MARKER);
    push_u32(out, SNAPSHOT_ALIGNMENT as u32);
    push_u32(out, sections.len() as u32);
    push_u32(out, 0);
    push_u64(out, (SNAPSHOT_HEADER_LEN + sections.len() * SNAPSHOT_SECTION_LEN) as u64);
    push_u64(out, total_len as u64);
    push_u64(out, layout_fingerprint());
    push_u64(out, globals_fingerprint);
    push_u64(out, generation);
    push_u64(out, log_count as u64);
    push_u64(out, field_count as u64);
    push_u64(out, data_len as u64);
    push_u64(out, span_count as u64);
    push_u64(out, archetype_count as u64);
    push_u64(out, timerange_count as u64);
    push_u64(out, general_range_count as u64);
    push_u64(out, uuid_offset_count as u64);
    push_u64(out, msg_count as u64);
    debug_assert_eq!(out.len(), SNAPSHOT_HEADER_LEN);
    for section in sections {
        push_u32(out, section.kind);
        push_u32(out, section.align as u32);
        push_u64(out, section.offset as u64);
        push_u64(out, section.len as u64);
    }
}

fn parse_snapshot_header(
    bytes: &[u8],
    globals: &PersistentInterners,
) -> Result<(ParsedHeader, Vec<SectionPlan>), SnapshotLoadError> {
    if bytes.len() < SNAPSHOT_HEADER_LEN {
        return Err(SnapshotLoadError::TruncatedHeader);
    }
    let mut cursor = 0usize;
    if read_fixed_load::<4>(bytes, &mut cursor)? != SNAPSHOT_MAGIC {
        return Err(SnapshotLoadError::InvalidMagic);
    }
    let version = read_u32_ne_load(bytes, &mut cursor)?;
    if version != SNAPSHOT_VERSION {
        return Err(SnapshotLoadError::UnsupportedVersion(version));
    }
    if read_u32_ne_load(bytes, &mut cursor)? != SNAPSHOT_ENDIAN_MARKER {
        return Err(SnapshotLoadError::EndianMismatch);
    }
    let required_align = read_u32_ne_load(bytes, &mut cursor)? as usize;
    if required_align != SNAPSHOT_ALIGNMENT {
        return Err(SnapshotLoadError::InvalidHeader);
    }
    let section_count = read_u32_ne_load(bytes, &mut cursor)? as usize;
    let _reserved = read_u32_ne_load(bytes, &mut cursor)?;
    let header_len = read_u64_ne_load(bytes, &mut cursor)? as usize;
    let total_len = read_u64_ne_load(bytes, &mut cursor)? as usize;
    let layout = read_u64_ne_load(bytes, &mut cursor)?;
    if layout != layout_fingerprint() {
        return Err(SnapshotLoadError::LayoutMismatch);
    }
    let globals_fingerprint = read_u64_ne_load(bytes, &mut cursor)?;
    if globals_fingerprint != globals.fingerprint() {
        return Err(SnapshotLoadError::GlobalsMismatch);
    }
    let generation = read_u64_ne_load(bytes, &mut cursor)?;
    let log_count = read_u64_ne_load(bytes, &mut cursor)? as usize;
    let field_count = read_u64_ne_load(bytes, &mut cursor)? as usize;
    let data_len = read_u64_ne_load(bytes, &mut cursor)? as usize;
    let span_count = read_u64_ne_load(bytes, &mut cursor)? as usize;
    let archetype_count = read_u64_ne_load(bytes, &mut cursor)? as usize;
    let timerange_count = read_u64_ne_load(bytes, &mut cursor)? as usize;
    let general_range_count = read_u64_ne_load(bytes, &mut cursor)? as usize;
    let uuid_offset_count = read_u64_ne_load(bytes, &mut cursor)? as usize;
    let msg_count = read_u64_ne_load(bytes, &mut cursor)? as usize;
    if cursor != SNAPSHOT_HEADER_LEN || section_count != expected_sections().len() {
        return Err(SnapshotLoadError::InvalidHeader);
    }
    if generation > u32::MAX as u64 || total_len != bytes.len() {
        return Err(SnapshotLoadError::InvalidLength);
    }
    if header_len != SNAPSHOT_HEADER_LEN + section_count * SNAPSHOT_SECTION_LEN || header_len > bytes.len() {
        return Err(SnapshotLoadError::InvalidHeader);
    }

    let mut sections = Vec::with_capacity(section_count);
    let expected = expected_sections();
    let mut previous_end = header_len;
    for (expected_kind, expected_align) in expected {
        let kind = read_u32_ne_load(bytes, &mut cursor)?;
        let align = read_u32_ne_load(bytes, &mut cursor)? as usize;
        let offset = read_u64_ne_load(bytes, &mut cursor)? as usize;
        let len = read_u64_ne_load(bytes, &mut cursor)? as usize;
        if kind != expected_kind || align != expected_align {
            return Err(SnapshotLoadError::InvalidSection(kind));
        }
        if align == 0 || align > SNAPSHOT_ALIGNMENT || offset < previous_end || offset % align != 0 {
            return Err(SnapshotLoadError::InvalidSection(kind));
        }
        let end = offset.checked_add(len).ok_or(SnapshotLoadError::InvalidLength)?;
        if end > total_len {
            return Err(SnapshotLoadError::InvalidLength);
        }
        previous_end = end;
        sections.push(SectionPlan { kind, align, offset, len });
    }
    if cursor != header_len {
        return Err(SnapshotLoadError::InvalidSectionTable);
    }

    Ok((
        ParsedHeader {
            total_len,
            layout_fingerprint: layout,
            globals_fingerprint,
            generation,
            log_count,
            field_count,
            data_len,
            span_count,
            archetype_count,
            timerange_count,
            general_range_count,
            uuid_offset_count,
            msg_count,
        },
        sections,
    ))
}

fn validate_counts(header: &ParsedHeader) -> Result<(), SnapshotLoadError> {
    if header.log_count > BUCKET_LOG_SIZE {
        return Err(SnapshotLoadError::CountExceeded("log"));
    }
    if header.field_count > BUCKET_FIELD_SIZE {
        return Err(SnapshotLoadError::CountExceeded("field"));
    }
    if header.data_len > BUCKET_DATA_SIZE {
        return Err(SnapshotLoadError::CountExceeded("data"));
    }
    if header.span_count > BUCKET_SPAN_RANGE_SIZE {
        return Err(SnapshotLoadError::CountExceeded("span"));
    }
    if header.archetype_count > BUCKET_ARCHETYPE_SIZE {
        return Err(SnapshotLoadError::CountExceeded("archetype"));
    }
    if header.timerange_count != header.log_count / TIME_RANGE_LOG_COUNT {
        return Err(SnapshotLoadError::InvalidTimeRange);
    }
    if header.general_range_count != 0 || header.uuid_offset_count != 0 {
        return Err(SnapshotLoadError::InvalidHeader);
    }
    Ok(())
}

fn new_loaded_intern_maps(data_len: usize, msg_count: usize) -> IntermentMaps {
    IntermentMaps {
        general: HashTable::new(),
        uuid: HashTable::new(),
        target_summaries: Vec::new(),
        keys: KeyMap::default(),
        indexed: 0,
        msgs: msg_count,
        data_len,
    }
}

fn validate_archetypes(
    archetypes: &[archetype::Archetype],
    globals: &PersistentInterners,
    data_len: usize,
) -> Result<(), SnapshotLoadError> {
    for archetype in archetypes {
        if archetype.size as usize > archetype::FIELD_LANES {
            return Err(SnapshotLoadError::InvalidReference("archetype field count"));
        }
        let level = archetype.mask & 0xF;
        if !matches!(level, 0x1 | 0x2 | 0x4 | 0x8) || archetype.mask & !0x1F != 0 {
            return Err(SnapshotLoadError::InvalidReference("archetype level mask"));
        }
        if archetype.msg_offset as usize + archetype.msg_len as usize > data_len {
            return Err(SnapshotLoadError::InvalidReference("archetype message"));
        }
        if !globals.has_target_raw(archetype.target_id) {
            return Err(SnapshotLoadError::InvalidGlobalReference("target"));
        }
        if !globals.has_service_raw(archetype.raw_service()) {
            return Err(SnapshotLoadError::InvalidGlobalReference("service"));
        }

        let mut previous_key = None;
        for index in 0..archetype::FIELD_LANES {
            let raw = archetype.field_headers[index];
            if index < archetype.size as usize {
                if !globals.has_key_raw(raw) {
                    return Err(SnapshotLoadError::InvalidGlobalReference("key"));
                }
                if previous_key.is_some_and(|prev| prev > raw) {
                    return Err(SnapshotLoadError::InvalidReference("archetype key ordering"));
                }
                previous_key = Some(raw);
            } else if raw != u16::MAX {
                return Err(SnapshotLoadError::InvalidReference("archetype key tail"));
            }
        }
    }
    Ok(())
}

/// Encode the timestamp column as zigzag-encoded i64 deltas in ULEB128 varints.
///
/// The first record's predecessor is implicitly 0 so its delta is the absolute
/// timestamp itself. Each subsequent delta is `ts[i] - ts[i - 1]` as a signed
/// i64 (handles the rare out-of-order pair) then zigzag-encoded. ULEB128 yields
/// 1 byte for |delta| <= 63, 2 for <= 8K, 3 for <= 1M (~ms), 5 for typical
/// nanosecond minute-gaps, 9-10 for the absolute first record.
///
/// Uses raw pointer writes into a pre-reserved buffer to avoid per-byte Vec
/// capacity checks and length updates on the hot loop.
fn encode_timestamps(timestamps: &[u64], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(timestamps.len() * 10);
    unsafe {
        let base = out.as_mut_ptr();
        let mut p = base;
        let mut prev: u64 = 0;
        for &ts in timestamps {
            let delta = (ts as i64).wrapping_sub(prev as i64);
            let mut zig = ((delta << 1) ^ (delta >> 63)) as u64;
            while zig >= 0x80 {
                p.write((zig as u8) | 0x80);
                p = p.add(1);
                zig >>= 7;
            }
            p.write(zig as u8);
            p = p.add(1);
            prev = ts;
        }
        out.set_len(p.offset_from(base) as usize);
    }
}

/// Reconstruct the timestamp column from the varint zigzag-delta stream.
///
/// Sequentially decodes each varint via raw pointer reads, undoes the zigzag
/// and accumulates a running sum. Errors if the stream is truncated, exceeds
/// the 10-byte u64 ceiling, or has trailing bytes after `log_count` records.
fn decode_timestamps(encoded: &[u8], log_count: usize) -> Result<Box<[u64]>, SnapshotLoadError> {
    let mut out = vec![0u64; log_count].into_boxed_slice();
    let start = encoded.as_ptr();
    let end = unsafe { start.add(encoded.len()) };
    let mut prev: u64 = 0;
    let mut p = start;
    for slot in out.iter_mut() {
        let mut zig: u64 = 0;
        let mut shift: u32 = 0;
        loop {
            if p >= end {
                return Err(SnapshotLoadError::InvalidSection(SECTION_TIMESTAMP));
            }
            let byte = unsafe { *p };
            p = unsafe { p.add(1) };
            zig |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift >= 64 {
                return Err(SnapshotLoadError::InvalidSection(SECTION_TIMESTAMP));
            }
        }
        let delta = ((zig >> 1) as i64) ^ -((zig & 1) as i64);
        let ts = (prev as i64).wrapping_add(delta) as u64;
        *slot = ts;
        prev = ts;
    }
    if p != end {
        return Err(SnapshotLoadError::InvalidSection(SECTION_TIMESTAMP));
    }
    Ok(out)
}

/// Reconstruct the absolute offset table from each log's archetype size.
///
/// `offsets[i + 1] - offsets[i]` equals the field count of the archetype that
/// log `i` points at, so the table is rebuilt by summing `archetype.size` per
/// log. Each archetype id must be in range and the running sum must match the
/// header's field count. The returned slice always holds at least one element
/// so callers may dereference index 0.
fn decode_offsets(
    archetype_indices: &[u16],
    archetypes: &[archetype::Archetype],
    field_count: usize,
) -> Result<Box<[u32]>, SnapshotLoadError> {
    let mut offsets = Vec::with_capacity(archetype_indices.len() + 1);
    offsets.push(0u32);
    let mut sum: u32 = 0;
    for &archetype_id in archetype_indices {
        let Some(archetype) = archetypes.get(archetype_id as usize) else {
            return Err(SnapshotLoadError::InvalidReference("archetype id"));
        };
        sum += archetype.size as u32;
        offsets.push(sum);
    }
    if sum as usize != field_count {
        return Err(SnapshotLoadError::InvalidOffsetTable);
    }
    Ok(offsets.into_boxed_slice())
}

fn validate_records(
    fields: &[Field],
    span_indices: &[u32],
    data: &[u8],
    span_count: usize,
) -> Result<(), SnapshotLoadError> {
    for field in fields {
        validate_field(*field, data)?;
    }
    for &span_id in span_indices {
        if span_id != u32::MAX && span_id as usize >= span_count {
            return Err(SnapshotLoadError::InvalidReference("span id"));
        }
    }
    Ok(())
}

fn validate_field(field: Field, data: &[u8]) -> Result<(), SnapshotLoadError> {
    match field.kind() {
        FieldKind::String | FieldKind::Bytes => {
            let range = InternedRange::from_field_mask(field.value_mask());
            if range.len == 0 && range.offset == 0 {
                return Ok(());
            }
            if range.offset as usize + range.len as usize > data.len() {
                return Err(SnapshotLoadError::InvalidReference("string/bytes field"));
            }
        }
        FieldKind::I64 | FieldKind::U64 => {
            let range = InternedRange { offset: field.value_mask() as u32, data: 0, len: 8 };
            if range.offset as usize + 8 > data.len() {
                return Err(SnapshotLoadError::InvalidReference("large integer field"));
            }
        }
        FieldKind::UUID => {
            let offset = field.value_mask() as u32;
            if offset as usize + 16 > data.len() {
                return Err(SnapshotLoadError::InvalidReference("uuid field"));
            }
        }
        FieldKind::None
        | FieldKind::I60
        | FieldKind::F60
        | FieldKind::Bool
        | FieldKind::Seconds
        | FieldKind::Timestamp
        | FieldKind::_Reserved2
        | FieldKind::_Reserved3
        | FieldKind::_Reserved4
        | FieldKind::_Reserved5
        | FieldKind::_Reserved6 => {}
    }
    Ok(())
}

fn validate_spans(spans: &[SpanRange], log_count: usize) -> Result<(), SnapshotLoadError> {
    for span in spans {
        let first = span.first_mask & 0x7FFF_FFFF;
        let last = span.last_mask.load(Ordering::Relaxed) & 0x7FFF_FFFF;
        if first as usize >= log_count || last as usize >= log_count || first > last {
            return Err(SnapshotLoadError::InvalidSpan);
        }
    }
    Ok(())
}

fn validate_timeranges(timeranges: &[TimeRange], timestamps: &[u64]) -> Result<(), SnapshotLoadError> {
    for (index, timerange) in timeranges.iter().enumerate() {
        let start = index * TIME_RANGE_LOG_COUNT;
        let end = start + TIME_RANGE_LOG_COUNT;
        let Some(chunk) = timestamps.get(start..end) else {
            return Err(SnapshotLoadError::InvalidTimeRange);
        };
        let min = chunk.iter().copied().min().unwrap_or(u64::MAX);
        let max = chunk.iter().copied().max().unwrap_or(u64::MIN);
        if timerange.min_utc_ns != min || timerange.max_utc_nc != max {
            return Err(SnapshotLoadError::InvalidTimeRange);
        }
    }
    Ok(())
}

fn expected_sections() -> [(u32, usize); 8] {
    [
        (SECTION_DATA, 1),
        (SECTION_FIELD, align_of::<Field>()),
        (SECTION_SPAN, align_of::<SpanRange>()),
        (SECTION_ARCHETYPE, align_of::<archetype::Archetype>()),
        (SECTION_SPAN_INDEX, align_of::<u32>()),
        (SECTION_TIMESTAMP, 1),
        (SECTION_ARCHETYPE_INDEX, align_of::<u16>()),
        (SECTION_TIME_RANGE, align_of::<TimeRange>()),
    ]
}

fn section_bytes(bytes: &[u8], section: SectionPlan) -> Result<&[u8], SnapshotLoadError> {
    bytes.get(section.offset..section.offset + section.len).ok_or(SnapshotLoadError::InvalidLength)
}

fn typed_section<T>(bytes: &[u8], section: SectionPlan, count: usize) -> Result<&[T], SnapshotLoadError> {
    if section.len != count * size_of::<T>() {
        return Err(SnapshotLoadError::InvalidSection(section.kind));
    }
    if (bytes.as_ptr() as usize + section.offset) % align_of::<T>() != 0 {
        return Err(SnapshotLoadError::InvalidSection(section.kind));
    }
    let ptr = if count == 0 {
        NonNull::<T>::dangling().as_ptr()
    } else {
        unsafe { bytes.as_ptr().add(section.offset) as *const T }
    };
    Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
}

fn ptr_for_slice<T>(slice: &[T]) -> NonNull<T> {
    if slice.is_empty() {
        NonNull::dangling()
    } else {
        NonNull::new(slice.as_ptr() as *mut T).unwrap()
    }
}

fn range_bytes(data: &[u8], range: InternedRange) -> &[u8] {
    &data[range.offset as usize..range.offset as usize + range.len as usize]
}

unsafe fn slice_as_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice)) }
}

fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_ne_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_ne_bytes());
}

fn read_fixed<const N: usize>(bytes: &[u8], cursor: &mut usize) -> Result<[u8; N], GlobalInternError> {
    let end = cursor.checked_add(N).ok_or(GlobalInternError::Truncated)?;
    let slice = bytes.get(*cursor..end).ok_or(GlobalInternError::Truncated)?;
    *cursor = end;
    Ok(slice.try_into().unwrap())
}

fn read_u32_ne(bytes: &[u8], cursor: &mut usize) -> Result<u32, GlobalInternError> {
    Ok(u32::from_ne_bytes(read_fixed(bytes, cursor)?))
}

fn read_u64_ne(bytes: &[u8], cursor: &mut usize) -> Result<u64, GlobalInternError> {
    Ok(u64::from_ne_bytes(read_fixed(bytes, cursor)?))
}

fn read_fixed_load<const N: usize>(bytes: &[u8], cursor: &mut usize) -> Result<[u8; N], SnapshotLoadError> {
    let end = cursor.checked_add(N).ok_or(SnapshotLoadError::InvalidLength)?;
    let slice = bytes.get(*cursor..end).ok_or(SnapshotLoadError::TruncatedHeader)?;
    *cursor = end;
    Ok(slice.try_into().unwrap())
}

fn read_u32_ne_load(bytes: &[u8], cursor: &mut usize) -> Result<u32, SnapshotLoadError> {
    Ok(u32::from_ne_bytes(read_fixed_load(bytes, cursor)?))
}

fn read_u64_ne_load(bytes: &[u8], cursor: &mut usize) -> Result<u64, SnapshotLoadError> {
    Ok(u64::from_ne_bytes(read_fixed_load(bytes, cursor)?))
}

fn read_name_list(bytes: &[u8], cursor: &mut usize, count: usize) -> Result<Vec<String>, GlobalInternError> {
    let mut names = Vec::with_capacity(count);
    for _ in 0..count {
        let len = read_u32_ne(bytes, cursor)? as usize;
        let end = cursor.checked_add(len).ok_or(GlobalInternError::Truncated)?;
        let slice = bytes.get(*cursor..end).ok_or(GlobalInternError::Truncated)?;
        let text = std::str::from_utf8(slice).map_err(|_| GlobalInternError::InvalidUtf8)?;
        names.push(text.to_owned());
        *cursor = end;
    }
    Ok(names)
}

fn write_name_list(out: &mut Vec<u8>, names: &[String]) {
    for name in names {
        push_u32(out, name.len() as u32);
        out.extend_from_slice(name.as_bytes());
    }
}

fn target_data_capacity(names: &[String]) -> usize {
    names.iter().map(|name| name.len()).sum::<usize>().max(1)
}

fn globals_fingerprint(dynamic_keys: &[String], services: &[String], targets: &[String]) -> u64 {
    let mut hash = FNV_OFFSET;
    hash = hash_bytes(hash, b"kvlog-index-globals-v1");
    hash = hash_u64(hash, dynamic_keys.len() as u64);
    for (index, name) in dynamic_keys.iter().enumerate() {
        hash = hash_u64(hash, (MIN_DYN_KEY as usize + index) as u64);
        hash = hash_bytes(hash, name.as_bytes());
    }
    hash = hash_u64(hash, services.len() as u64);
    for (index, name) in services.iter().enumerate() {
        hash = hash_u64(hash, (index + 1) as u64);
        hash = hash_bytes(hash, name.as_bytes());
    }
    hash = hash_u64(hash, targets.len() as u64);
    for (index, name) in targets.iter().enumerate() {
        hash = hash_u64(hash, index as u64);
        hash = hash_bytes(hash, name.as_bytes());
    }
    hash
}

fn layout_fingerprint() -> u64 {
    let mut hash = FNV_OFFSET;
    hash = hash_bytes(hash, b"kvlog-index-native-bucket-v1");
    hash = hash_bytes(hash, env!("CARGO_PKG_VERSION").as_bytes());
    for value in [
        SNAPSHOT_VERSION as u64,
        size_of::<Field>() as u64,
        align_of::<Field>() as u64,
        size_of::<SpanRange>() as u64,
        align_of::<SpanRange>() as u64,
        size_of::<TimeRange>() as u64,
        align_of::<TimeRange>() as u64,
        size_of::<archetype::Archetype>() as u64,
        align_of::<archetype::Archetype>() as u64,
        size_of::<InternedRange>() as u64,
        align_of::<InternedRange>() as u64,
        BUCKET_LOG_SIZE as u64,
        BUCKET_FIELD_SIZE as u64,
        BUCKET_DATA_SIZE as u64,
        BUCKET_SPAN_RANGE_SIZE as u64,
        BUCKET_ARCHETYPE_SIZE as u64,
        TIME_RANGE_LOG_COUNT as u64,
        archetype::FIELD_LANES as u64,
    ] {
        hash = hash_u64(hash, value);
    }
    hash
}

fn hash_u64(hash: u64, value: u64) -> u64 {
    hash_bytes(hash, &value.to_ne_bytes())
}

fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    hash = hash_u64_inner(hash, bytes.len() as u64);
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn hash_u64_inner(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_ne_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

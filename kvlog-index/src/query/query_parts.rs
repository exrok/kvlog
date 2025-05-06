use std::alloc::Layout;
use std::mem::ManuallyDrop;
use std::num::ParseFloatError;
use std::sync::Arc;

use bumpalo::collections::Vec as BumpVec;
use bumpalo::Bump;
use hashbrown::HashMap;
use kvlog::encoding::Value;
use uuid::Uuid;

mod literal;
pub mod parser;
mod range_eval;
mod timestamp_eval;
use crate::index::f48::{self, f48_to_f64, F48};
use crate::index::{i48, Bucket, BucketGuard, Field, FieldKind, FieldKindSet, IntermentMaps, LogEntry};
use crate::keyset::KeySet;
use crate::shared_interner::Mapper;
use literal::{EvalError, LiteralBinOp, LiteralParseError};
use parser::Expr;
use parser::{ErrorContext, ErrorKind, InternalQueryError, Of, QueryPredicates, Span};
use ra_ap_rustc_lexer::{Cursor, LiteralKind as RaLiteralKind};
use ra_ap_rustc_lexer::{Token as LexerToken, TokenKind};

use crate::field_table::KeyID;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LiteralKind {
    Null = 0,
    String = 1,
    Bytes = 2,
    Integer = 3,
    Float = 4,
    Duration = 5,
    List = 6,
    Bool = 7,
}

pub type LiteralKindBitSet = u16;
const fn literal_type_set(mut kinds: &[FieldKind]) -> FieldKindBitSet {
    let mut set = 0;
    while let [head, rest @ ..] = kinds {
        set |= 1 << (*head as u8);
        kinds = rest;
    }
    set
}

pub type FieldKindBitSet = u16;
const fn type_set(mut kinds: &[FieldKind]) -> FieldKindBitSet {
    let mut set = 0;
    while let [head, rest @ ..] = kinds {
        set |= 1 << (*head as u8);
        kinds = rest;
    }
    set
}

const ANY_FLOAT_TYPE: FieldKindBitSet = type_set(&[FieldKind::F48]);
const ANY_INTEGER_TYPE: FieldKindBitSet = type_set(&[FieldKind::I48, FieldKind::I64, FieldKind::U64]);
const ANY_NUMBER_TYPE: FieldKindBitSet = ANY_FLOAT_TYPE | ANY_INTEGER_TYPE;

/// Note, ranges are inclusive.
/// Timestamps are represented at durations from unix EPOCH
/// If a value U64Eq only used if value would not fit into U64Eq
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum FieldTest<'bump> {
    Defined(bool),
    RangeRaw { negated: bool, min: Field, max: Field },
    EqRaw(bool, Field),
    AnyRaw(bool, &'bump [Field]),
    BytesEqual(bool, &'bump [u8]),

    TextEqual(bool, &'bump str),
    TextAny(bool, &'bump [&'bump str]),
    TextContains(bool, &'bump str),
    TextStartsWith(bool, &'bump str),
    TextEndWith(bool, &'bump str),

    IsTrue(bool),
    NullValue(bool),
    FiniteFloat(bool),
    Type(bool, FieldKindBitSet),
    I64Eq(bool, i64),
    U64Eq(bool, u64),
    FloatRange { negated: bool, min: f64, max: f64 },
    DurationRange { negated: bool, min_seconds: F48, max_seconds: F48 },
    TimeRange { negated: bool, min_ns: i64, max_ns: i64 },
}

impl<'bump> FieldTest<'bump> {
    pub fn matches_text(&self, bytes: &[u8]) -> bool {
        match *self {
            FieldTest::Defined(negated) => return !negated,
            FieldTest::BytesEqual(negated, text) => (bytes == text) ^ negated,
            FieldTest::TextEqual(negated, text) => (bytes == text.as_bytes()) ^ negated,
            FieldTest::TextAny(negated, items) => {
                for item in items {
                    if bytes == item.as_bytes() {
                        return !negated;
                    }
                }
                return negated;
            }
            FieldTest::TextContains(negated, text) => {
                if memchr::memmem::find(bytes, text.as_bytes()).is_some() {
                    return !negated;
                }
                return negated;
            }
            FieldTest::TextStartsWith(negated, text) => {
                if bytes.starts_with(text.as_bytes()) {
                    return !negated;
                }
                return negated;
            }
            FieldTest::TextEndWith(negated, text) => {
                if bytes.ends_with(text.as_bytes()) {
                    return !negated;
                }
                return negated;
            }
            // not sure what we should do if it's negated, really all these other cases should have been errors
            _ => return self.is_negated(),
        }
    }
    unsafe fn matches(&self, bucket: &Bucket, field: Field) -> bool {
        match *self {
            FieldTest::Defined(negated) => return !negated,
            FieldTest::RangeRaw { negated, min, max } => {
                if (field.raw >= min.raw) & (field.raw <= max.raw) {
                    !negated
                } else {
                    negated
                }
            }
            FieldTest::EqRaw(negated, raw) => {
                if field == raw {
                    !negated
                } else {
                    negated
                }
            }
            FieldTest::AnyRaw(negated, raws) => {
                for raw in raws {
                    if field == *raw {
                        return !negated;
                    }
                }
                return negated;
            }
            FieldTest::BytesEqual(negated, bytes) => {
                if let Some(field_bytes) = unsafe { field.as_bytes(bucket) } {
                    if field_bytes == bytes {
                        return !negated;
                    }
                }
                return negated;
            }
            FieldTest::TextEqual(negated, text) => {
                match unsafe { field.value(bucket) } {
                    kvlog::encoding::Value::String(field_text) => {
                        if field_text == text.as_bytes() {
                            return !negated;
                        }
                    }
                    kvlog::encoding::Value::UUID(uuid) => {
                        if let Ok(value) = text.parse::<Uuid>() {
                            if value == uuid {
                                return !negated;
                            }
                        }
                    }
                    _ => (),
                }
                return negated;
            }
            FieldTest::TextAny(negated, texts) => {
                match unsafe { field.value(bucket) } {
                    kvlog::encoding::Value::String(field_text) => {
                        for text in texts {
                            if field_text == text.as_bytes() {
                                return !negated;
                            }
                        }
                    }
                    kvlog::encoding::Value::UUID(uuid) => {
                        for text in texts {
                            if let Ok(value) = text.parse::<Uuid>() {
                                if value == uuid {
                                    return !negated;
                                }
                            }
                        }
                    }
                    _ => (),
                }
                return negated;
            }
            FieldTest::TextContains(negated, text) => {
                if let Some(field_text) = unsafe { field.as_text(bucket) } {
                    if memchr::memmem::find(field_text, text.as_bytes()).is_some() {
                        return !negated;
                    }
                }
                return negated;
            }
            FieldTest::TextStartsWith(negated, text) => {
                if let Some(field_text) = unsafe { field.as_text(bucket) } {
                    if memchr::arch::all::is_prefix(field_text, text.as_bytes()) {
                        return !negated;
                    }
                }
                return negated;
            }
            FieldTest::TextEndWith(negated, text) => {
                if let Some(field_text) = unsafe { field.as_text(bucket) } {
                    if memchr::arch::all::is_suffix(field_text, text.as_bytes()) {
                        return !negated;
                    }
                }
                return negated;
            }
            FieldTest::IsTrue(negated) => {
                if let Some(value) = field.as_bool() {
                    return value != negated;
                }
                return negated;
            }
            FieldTest::NullValue(negated) => {
                if field.kind() == FieldKind::None {
                    return !negated;
                } else {
                    return negated;
                }
            }
            FieldTest::FiniteFloat(negated) => {
                // note currently value is always f64 and never f32
                if let Value::F64(value) = unsafe { field.value(bucket) } {
                    if value.is_finite() {
                        return !negated;
                    }
                }
                return negated;
            }
            FieldTest::Type(negated, types) => {
                if (FieldKindSet { raw: types }.contains(field.kind())) {
                    return !negated;
                } else {
                    negated
                }
            }
            FieldTest::I64Eq(negated, value) => {
                match unsafe { field.value(bucket) } {
                    Value::U32(v) => {
                        if v as i64 == value {
                            return !negated;
                        }
                    }
                    Value::I32(v) => {
                        if v as i64 == value {
                            return !negated;
                        }
                    }
                    Value::I64(v) => {
                        if v == value {
                            return !negated;
                        }
                    }
                    Value::U64(value_u64) => {
                        if value >= 0 && value_u64 == (value as u64) {
                            return !negated;
                        }
                    }
                    _ => (),
                }
                return negated;
            }
            FieldTest::U64Eq(negated, value) => {
                if let Some(field_value) = unsafe { field.as_required_u64(bucket) } {
                    if field_value == value {
                        return !negated;
                    }
                }
                return negated;
            }
            FieldTest::FloatRange { negated, min, max } => {
                if let Some(value) = unsafe { field.as_f64(bucket) } {
                    if value >= min && value <= max {
                        return !negated;
                    }
                }
                return negated;
            }
            FieldTest::DurationRange { negated, min_seconds, max_seconds } => {
                if let Some(value) = unsafe { field.as_raw_f48_seconds() } {
                    if value >= min_seconds && value <= max_seconds {
                        return !negated;
                    }
                }
                return negated;
            }
            FieldTest::TimeRange { negated, min_ns, max_ns } => {
                if let Some(ns) = unsafe { field.as_timestamp_ns() } {
                    if ns >= min_ns && ns <= max_ns {
                        return !negated;
                    }
                }
                return negated;
            }
        }
    }
    fn copy_to_other_arena<'o>(&self, bump: &'o Bump) -> FieldTest<'o> {
        use FieldTest::*;
        match self {
            RangeRaw { negated, min, max } => RangeRaw { negated: *negated, min: *min, max: *max },
            EqRaw(negated, raw) => EqRaw(*negated, *raw),
            AnyRaw(negated, raw) => AnyRaw(*negated, bump.alloc_slice_copy(raw)),
            Defined(negated) => Defined(*negated),
            BytesEqual(negated, bytes) => BytesEqual(*negated, bump.alloc_slice_copy(bytes)),
            TextEqual(negated, text) => TextEqual(*negated, bump.alloc_str(text)),
            TextAny(negated, texts) => {
                TextAny(*negated, { bump.alloc_slice_fill_iter(texts.iter().map(|text| &*bump.alloc_str(*text))) })
            }
            TextContains(negated, text) => TextContains(*negated, bump.alloc_str(text)),
            TextStartsWith(negated, text) => TextStartsWith(*negated, bump.alloc_str(text)),
            TextEndWith(negated, text) => TextEndWith(*negated, bump.alloc_str(text)),
            IsTrue(negated) => IsTrue(*negated),
            NullValue(negated) => NullValue(*negated),
            FiniteFloat(negated) => FiniteFloat(*negated),
            Type(negated, types) => Type(*negated, *types),
            I64Eq(negated, value) => I64Eq(*negated, *value),
            U64Eq(negated, value) => U64Eq(*negated, *value),
            FloatRange { negated, min, max } => FloatRange { negated: *negated, min: *min, max: *max },
            DurationRange { negated, min_seconds, max_seconds } => {
                DurationRange { negated: *negated, min_seconds: *min_seconds, max_seconds: *max_seconds }
            }
            TimeRange { negated, min_ns, max_ns } => TimeRange { negated: *negated, min_ns: *min_ns, max_ns: *max_ns },
        }
    }
    #[no_mangle]
    fn get_negated_mut(&mut self) -> &mut bool {
        // This match should compile out
        use FieldTest::*;
        match self {
            Defined(negated)
            | BytesEqual(negated, _)
            | RangeRaw { negated, .. }
            | EqRaw(negated, _)
            | AnyRaw(negated, _)
            | TextEqual(negated, _)
            | TextAny(negated, _)
            | TextContains(negated, _)
            | TextStartsWith(negated, _)
            | TextEndWith(negated, _)
            | IsTrue(negated)
            | NullValue(negated)
            | FiniteFloat(negated)
            | Type(negated, _)
            | I64Eq(negated, _)
            | U64Eq(negated, _)
            | FloatRange { negated, .. }
            | TimeRange { negated, .. }
            | DurationRange { negated, .. } => negated,
        }
    }
    #[no_mangle]
    pub fn is_negated(&self) -> bool {
        // This match should compile out
        use FieldTest::*;
        match self {
            Defined(negated)
            | RangeRaw { negated, .. }
            | BytesEqual(negated, _)
            | TextEqual(negated, _)
            | AnyRaw(negated, _)
            | EqRaw(negated, _)
            | TextAny(negated, _)
            | TextContains(negated, _)
            | TextStartsWith(negated, _)
            | TextEndWith(negated, _)
            | IsTrue(negated)
            | NullValue(negated)
            | FiniteFloat(negated)
            | Type(negated, _)
            | I64Eq(negated, _)
            | U64Eq(negated, _)
            | FloatRange { negated, .. }
            | TimeRange { negated, .. }
            | DurationRange { negated, .. } => *negated,
        }
    }
    fn negate(&mut self) {
        *self.get_negated_mut() ^= true;
    }
}

pub type FieldPredicate<'bump> = (bool, FieldTest<'bump>);

const fn Is<'bump>(test: FieldTest<'bump>) -> (bool, FieldTest<'bump>) {
    (true, test)
}

const fn IsNot<'bump>(test: FieldTest<'bump>) -> (bool, FieldTest<'bump>) {
    (false, test)
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum FieldKey<'a> {
    Seen(KeyID),
    New(&'a str),
}
impl<'a> FieldKey<'a> {
    pub fn as_str(&self) -> &'a str {
        match self {
            FieldKey::Seen(key) => key.as_str(),
            FieldKey::New(key) => key,
        }
    }
    pub fn id(&self) -> Option<KeyID> {
        match self {
            FieldKey::Seen(key) => Some(*key),
            FieldKey::New(key) => KeyID::try_from_str(key),
        }
    }
}
impl<'a> From<&'a str> for FieldKey<'a> {
    fn from(value: &'a str) -> Self {
        match KeyID::try_from_str(value) {
            Some(key) => FieldKey::Seen(key),
            None => FieldKey::New(value),
        }
    }
}

pub type Bumpy<'a, T> = ManuallyDrop<BumpVec<'a, T>>;

#[derive(Copy, Clone, Debug)]
pub enum Pred<'b> {
    Field(KeyID, FieldTest<'b>),
    FieldOr(KeyID, &'b [FieldTest<'b>]),
    FieldAnd(KeyID, &'b [FieldTest<'b>]),
    And(&'b [Pred<'b>]),
    Or(&'b [Pred<'b>]),
    SpanIs(bool, u64),
    ParentSpanIs(bool, u64),
    HasSpan(bool),
    HasParentSpan(bool),
    SpanDurationRange { negated: bool, min_ns: u64, max_ns: u64 },
    TimestampRange { negated: bool, min_ns: u64, max_ns: u64 },
    LevelMask(bool, u8),
    Target(FieldTest<'b>),
    Service(FieldTest<'b>),
    Message(FieldTest<'b>),
}

#[derive(Default)]
pub struct OptimizationInfo {
    pub pred_addr_trival_map: HashMap<usize, bool>,
}
impl OptimizationInfo {
    pub fn test_is_always(&self, field_test: &FieldTest) -> Option<bool> {
        self.pred_addr_trival_map.get(&<*const FieldTest>::addr(field_test)).copied()
    }
    pub fn predicate_is_always(&self, pred: &Pred) -> Option<bool> {
        self.pred_addr_trival_map.get(&<*const Pred>::addr(pred)).copied()
    }
}

impl<'b> Pred<'b> {
    pub fn matches(&self, log: LogEntry, target_mapper: &Mapper) -> bool {
        match self {
            Pred::Field(key, field_test) => {
                if let Some(field) = log.field_by_key_id(*key) {
                    unsafe { field_test.matches(log.bucket(), field) }
                } else {
                    field_test.is_negated()
                }
            }
            Pred::FieldOr(key, tests) => {
                if let Some(field) = log.field_by_key_id(*key) {
                    for test in tests.iter() {
                        if unsafe { test.matches(log.bucket(), field) } {
                            return true;
                        }
                    }
                };
                return false;
            }
            Pred::FieldAnd(key, tests) => {
                if let Some(field) = log.field_by_key_id(*key) {
                    for test in tests.iter() {
                        if !unsafe { test.matches(log.bucket(), field) } {
                            return false;
                        }
                    }
                    return true;
                };
                return false;
            }
            Pred::And(preds) => {
                for test in preds.iter() {
                    if !test.matches(log, target_mapper) {
                        return false;
                    }
                }
                return true;
            }
            Pred::Or(preds) => {
                for test in preds.iter() {
                    if test.matches(log, target_mapper) {
                        return true;
                    }
                }
                return false;
            }
            Pred::SpanIs(negated, span) => {
                if let Some(field_span) = log.span_id() {
                    if field_span.as_u64() == *span {
                        return !negated;
                    }
                }
                return *negated;
            }
            Pred::ParentSpanIs(negated, span) => {
                if let Some(field_parent_span) = log.parent_span_id() {
                    if field_parent_span.as_u64() == *span {
                        return !negated;
                    }
                }
                return *negated;
            }
            Pred::HasSpan(negated) => {
                if log.span_range().is_none() {
                    return !*negated;
                } else {
                    return *negated;
                }
            }
            Pred::HasParentSpan(negated) => {
                if log.parent_span_id().is_some() {
                    return !negated;
                }
                return *negated;
            }
            Pred::SpanDurationRange { negated, min_ns, max_ns } => {
                if let Some(duration) = log.span_ns_duration() {
                    if duration >= *min_ns && duration <= *max_ns {
                        return !*negated;
                    }
                }
                return *negated;
            }
            Pred::TimestampRange { negated, min_ns, max_ns } => {
                let timestamp = log.timestamp();
                if timestamp >= *min_ns && timestamp <= *max_ns {
                    return !*negated;
                };
                return *negated;
            }
            Pred::LevelMask(negated, mask) => {
                if log.level_mask() & mask != 0 {
                    return !*negated;
                }
                return *negated;
            }
            Pred::Target(field_test) => {
                if let Some(text) = target_mapper.get(log.target_id()) {
                    unsafe { field_test.matches_text(text.as_bytes()) }
                } else {
                    field_test.is_negated()
                }
            }
            Pred::Message(field_test) => return field_test.matches_text(log.message()),
            Pred::Service(field_test) => {
                if let Some(service_id) = log.archetype().service() {
                    unsafe { field_test.matches_text(service_id.as_str().as_bytes()) }
                } else {
                    field_test.is_negated()
                }
            }
        }
    }
    pub fn reduce(&self, bump: &'b Bump, opt: &OptimizationInfo) -> PredBuildResult<'b> {
        use PredBuildResult as R;
        match opt.predicate_is_always(self) {
            Some(true) => return R::AlwaysTrue,
            Some(false) => return R::AlwaysFalse,
            None => (),
        }
        match self {
            Pred::FieldOr(key_id, field_tests) => {
                let mut tests = *field_tests;
                while let [head, rest @ ..] = tests {
                    match opt.test_is_always(head) {
                        Some(true) => return R::AlwaysTrue,
                        Some(false) => tests = rest,
                        None => break,
                    }
                }
                while let [rest @ .., tail] = tests {
                    match opt.test_is_always(tail) {
                        Some(true) => return R::AlwaysTrue,
                        Some(false) => tests = rest,
                        None => break,
                    }
                }
                if tests.iter().any(|test| opt.test_is_always(test).is_some()) {
                    let mut bump = BumpVec::<FieldTest<'b>>::with_capacity_in(tests.len(), bump);
                    for test in tests {
                        match opt.test_is_always(test) {
                            Some(true) => return R::AlwaysTrue,
                            Some(false) => continue,
                            None => bump.push(*test),
                        }
                    }
                    tests = bump.into_bump_slice();
                }

                if tests.is_empty() {
                    return R::AlwaysFalse;
                } else if tests.len() == 1 {
                    return R::Ok(Pred::Field(*key_id, tests[0]));
                } else {
                    return R::Ok(Pred::FieldOr(*key_id, tests));
                }
            }
            Pred::FieldAnd(key_id, field_tests) => {
                let mut tests = *field_tests;
                while let [head, rest @ ..] = tests {
                    match opt.test_is_always(head) {
                        Some(false) => return R::AlwaysFalse,
                        Some(true) => tests = rest,
                        None => break,
                    }
                }
                while let [rest @ .., tail] = tests {
                    match opt.test_is_always(tail) {
                        Some(false) => return R::AlwaysFalse,
                        Some(true) => tests = rest,
                        None => break,
                    }
                }
                if tests.iter().any(|test| opt.test_is_always(test).is_some()) {
                    let mut bump = BumpVec::<FieldTest<'b>>::with_capacity_in(tests.len(), bump);
                    for test in tests {
                        match opt.test_is_always(test) {
                            Some(false) => return R::AlwaysFalse,
                            Some(true) => continue,
                            None => bump.push(*test),
                        }
                    }
                    tests = bump.into_bump_slice();
                }

                if tests.is_empty() {
                    return R::AlwaysFalse;
                } else if tests.len() == 1 {
                    return R::Ok(Pred::Field(*key_id, tests[0]));
                } else {
                    return R::Ok(Pred::FieldAnd(*key_id, tests));
                }
            }
            Pred::And(preds) => {
                let mut count = 0;
                unsafe {
                    let alloc = bump
                        .alloc_layout(Layout::from_size_align_unchecked(
                            preds.len() * std::mem::size_of::<Pred>(),
                            align_of::<Pred>(),
                        ))
                        .cast::<Pred>();
                    let mut count = 0;
                    for item in preds.iter() {
                        match item.reduce(bump, opt) {
                            R::AlwaysTrue => continue,
                            R::AlwaysFalse => return R::AlwaysFalse,
                            R::Ok(pred) => alloc.add(count).write(pred),
                        }
                        count += 1;
                    }
                    if count == 0 {
                        return R::AlwaysTrue;
                    } else if count == 1 {
                        return R::Ok(unsafe { alloc.read() });
                    } else {
                        return R::Ok(Pred::And(std::slice::from_raw_parts(alloc.as_ptr(), count)));
                    }
                }
            }
            Pred::Or(preds) => {
                let mut count = 0;
                unsafe {
                    let alloc = bump
                        .alloc_layout(Layout::from_size_align_unchecked(
                            preds.len() * std::mem::size_of::<Pred>(),
                            align_of::<Pred>(),
                        ))
                        .cast::<Pred>();
                    let mut count = 0;
                    for item in preds.iter() {
                        match item.reduce(bump, opt) {
                            R::AlwaysTrue => return R::AlwaysTrue,
                            R::AlwaysFalse => continue,
                            R::Ok(pred) => alloc.add(count).write(pred),
                        }
                        count += 1;
                    }
                    if count == 0 {
                        return R::AlwaysFalse;
                    } else if count == 1 {
                        return R::Ok(unsafe { alloc.read() });
                    } else {
                        return R::Ok(Pred::Or(std::slice::from_raw_parts(alloc.as_ptr(), count)));
                    }
                }
            }
            _ => R::Ok(*self),
        }
    }
}

pub enum PredBuilder<'b> {
    True,
    False,
    Field(FieldKey<'b>, FieldTest<'b>),
    FieldOr(FieldKey<'b>, Bumpy<'b, FieldTest<'b>>),
    FieldAnd(FieldKey<'b>, Bumpy<'b, FieldTest<'b>>),
    And(Bumpy<'b, PredBuilder<'b>>),
    Or(Bumpy<'b, PredBuilder<'b>>),
    SpanIs(bool, u64),
    ParentSpanIs(bool, u64),
    HasSpan(bool),
    HasParentSpan(bool),
    SpanDurationRange { negated: bool, min_ns: u64, max_ns: u64 },
    TimestampRange { negated: bool, min_ns: u64, max_ns: u64 },
    LevelMask(bool, u8),
    Target(FieldTest<'b>),
    Service(FieldTest<'b>),
    Message(FieldTest<'b>),
}

impl<'b> std::fmt::Debug for PredBuilder<'b> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::True => write!(f, "True"),
            Self::False => write!(f, "False"),
            Self::Field(key, test) => {
                write!(f, "Field({:?}, {:?})", key.as_str(), test)
            }
            Self::FieldOr(arg0, arg1) => f.debug_tuple("FieldOr").field(arg0).field(&arg1.iter()).finish(),
            Self::FieldAnd(arg0, arg1) => f.debug_tuple("FieldAnd").field(arg0).field(arg1).finish(),
            Self::And(arg0) => {
                f.write_str("And ");
                f.debug_set().entries(arg0.as_slice()).finish()
            }
            Self::Or(arg0) => {
                f.write_str("Or ");
                f.debug_set().entries(arg0.as_slice()).finish()
            }
            Self::SpanIs(arg0, arg1) => f.debug_tuple("SpanIs").field(arg0).field(arg1).finish(),
            Self::ParentSpanIs(arg0, arg1) => f.debug_tuple("ParentSpanIs").field(arg0).field(arg1).finish(),
            Self::HasSpan(arg0) => f.debug_tuple("HasSpan").field(arg0).finish(),
            Self::HasParentSpan(arg0) => f.debug_tuple("HasParentSpan").field(arg0).finish(),
            Self::SpanDurationRange { negated, min_ns, max_ns } => f
                .debug_struct("SpanDurationRange")
                .field("negated", negated)
                .field("min_ns", min_ns)
                .field("max_ns", max_ns)
                .finish(),
            Self::TimestampRange { negated, min_ns, max_ns } => f
                .debug_struct("TimestampRange")
                .field("negated", negated)
                .field("min_ns", &f48_to_f64(*min_ns))
                .field("max_ns", &f48_to_f64(*max_ns))
                .finish(),
            Self::LevelMask(arg0, arg1) => f.debug_tuple("LevelMask").field(arg0).field(arg1).finish(),
            Self::Target(arg1) => f.debug_tuple("Target").field(arg1).finish(),
            Self::Service(arg1) => f.debug_tuple("Service").field(arg1).finish(),
            Self::Message(arg1) => f.debug_tuple("Message").field(arg1).finish(),
        }
    }
}

#[derive(Debug)]
pub enum PredBuildResult<'b> {
    AlwaysTrue,
    AlwaysFalse,
    Ok(Pred<'b>),
}
impl<'b> PredBuildResult<'b> {
    pub fn pred(self) -> Option<Pred<'b>> {
        match self {
            PredBuildResult::AlwaysTrue => None,
            PredBuildResult::AlwaysFalse => None,
            PredBuildResult::Ok(pred) => Some(pred),
        }
    }
}

pub enum FieldTestResult<'b> {
    AlwaysTrue,
    AlwaysFalse,
    Ok(FieldTest<'b>),
}

fn opt_copy_field_tes<'o>(
    key: KeyID,
    kinds: FieldKindSet,
    field: &FieldTest,
    bump: &'o Bump,
    bucket: &BucketGuard,
    maps: &IntermentMaps,
) -> FieldTestResult<'o> {
    use FieldTest::*;
    use FieldTestResult as R;
    match field {
        Defined(negated) => R::Ok(Defined(*negated)),
        RangeRaw { negated, min, max } => {
            if kinds.contains_any(FieldKindSet::NUMBERS) {
                return R::Ok(RangeRaw { negated: *negated, min: *min, max: *max });
            }
            if *negated {
                R::AlwaysTrue
            } else {
                R::AlwaysFalse
            }
        }
        EqRaw(negated, raw) => R::Ok(EqRaw(*negated, *raw)),
        AnyRaw(negated, raw) => R::Ok(AnyRaw(*negated, bump.alloc_slice_copy(raw))),
        BytesEqual(negated, bytes) => R::Ok(BytesEqual(*negated, bump.alloc_slice_copy(bytes))),
        TextEqual(negated, text) => {
            let mut single = None;
            if kinds.contains(FieldKind::UUID) {
                if let Ok(uuid) = text.parse::<Uuid>() {
                    if let Some(field) = maps.field_uuid(&bucket, key, uuid) {
                        single = Some(field)
                    }
                }
            }

            if kinds.contains(FieldKind::String) {
                if let Some(field) = maps.field_text(&bucket, key, text.as_bytes()) {
                    if let Some(other) = single {
                        return R::Ok(AnyRaw(*negated, bump.alloc([other, field])));
                    } else {
                        single = Some(field);
                    }
                }
            }

            if let Some(raw) = single {
                return R::Ok(EqRaw(*negated, raw));
            } else if *negated {
                return R::AlwaysTrue;
            } else {
                return R::AlwaysFalse;
            }
        }
        TextAny(negated, texts) => {
            let mut options = Vec::new();
            if kinds.contains(FieldKind::UUID) {
                for text in *texts {
                    if let Ok(uuid) = text.parse::<Uuid>() {
                        if let Some(field) = maps.field_uuid(&bucket, key, uuid) {
                            options.push(field)
                        }
                    }
                }
            }

            if kinds.contains(FieldKind::String) {
                for text in *texts {
                    if let Some(field) = maps.field_text(&bucket, key, text.as_bytes()) {
                        options.push(field)
                    }
                }
            }
            match options.as_slice() {
                [] => {
                    if *negated {
                        return R::AlwaysTrue;
                    } else {
                        return R::AlwaysFalse;
                    }
                }
                [single] => R::Ok(EqRaw(*negated, *single)),
                many => R::Ok(AnyRaw(*negated, bump.alloc_slice_copy(many))),
            }
        }
        TextContains(negated, text) => {
            if kinds.contains(FieldKind::String) {
                R::Ok(TextContains(*negated, bump.alloc_str(text)))
            } else if *negated {
                R::AlwaysTrue
            } else {
                R::AlwaysFalse
            }
        }
        TextStartsWith(negated, text) => {
            if kinds.contains(FieldKind::String) {
                R::Ok(TextStartsWith(*negated, bump.alloc_str(text)))
            } else if *negated {
                R::AlwaysTrue
            } else {
                R::AlwaysFalse
            }
        }
        TextEndWith(negated, text) => {
            if kinds.contains(FieldKind::String) {
                R::Ok(TextEndWith(*negated, bump.alloc_str(text)))
            } else if *negated {
                R::AlwaysTrue
            } else {
                R::AlwaysFalse
            }
        }
        IsTrue(negated) => R::Ok(IsTrue(*negated)),
        NullValue(negated) => R::Ok(NullValue(*negated)),
        FiniteFloat(negated) => R::Ok(FiniteFloat(*negated)),
        Type(negated, types) => R::Ok(Type(*negated, *types)),
        I64Eq(negated, value) => {
            if kinds.contains_any(FieldKindSet::NUMBERS) {
                if let Some(field) = maps.field_i64(bucket, key, *value) {
                    return R::Ok(EqRaw(*negated, field));
                }
            }
            if *negated {
                R::AlwaysTrue
            } else {
                R::AlwaysFalse
            }
        }
        U64Eq(negated, value) => {
            if kinds.contains_any(FieldKindSet::NUMBERS) {
                if let Some(field) = maps.field_u64(bucket, key, *value) {
                    return R::Ok(EqRaw(*negated, field));
                }
            }
            if *negated {
                R::AlwaysTrue
            } else {
                R::AlwaysFalse
            }
        }
        FloatRange { negated, min, max } => {
            if kinds == FieldKindSet::from(FieldKind::I48) {
                return R::Ok(RangeRaw {
                    negated: *negated,
                    min: Field::new(
                        key.0,
                        FieldKind::I48,
                        if *min < (i48::MIN as f64) { 0 } else { i48::from_i64(min.ceil() as i64) },
                    ),
                    max: Field::new(
                        key.0,
                        FieldKind::I48,
                        if *max > (i48::MAX as f64) { (1u64 << 48) - 1 } else { i48::from_i64(max.floor() as i64) },
                    ),
                });
            } else if kinds == FieldKindSet::from(FieldKind::F48) {
                return R::Ok(RangeRaw {
                    negated: *negated,
                    min: Field::new(key.0, FieldKind::F48, f48::f64_to_f48(*min)),
                    max: Field::new(key.0, FieldKind::F48, f48::f64_to_f48(*max)),
                });
            } else {
                R::Ok(FloatRange { negated: *negated, min: *min, max: *max })
            }
        }
        DurationRange { negated, min_seconds, max_seconds } => {
            if kinds.contains(FieldKind::Seconds) {
                return R::Ok(RangeRaw {
                    negated: *negated,
                    min: Field::new(key.0, FieldKind::Seconds, *min_seconds & ((1 << 48) - 1)),
                    max: Field::new(key.0, FieldKind::Seconds, *max_seconds & ((1 << 48) - 1)),
                });
            }
            if *negated {
                R::AlwaysTrue
            } else {
                R::AlwaysFalse
            }
        }
        TimeRange { negated, min_ns, max_ns } => {
            if kinds.contains(FieldKind::Timestamp) {
                return R::Ok(RangeRaw {
                    negated: *negated,
                    min: Field::new(key.0, FieldKind::Timestamp, (min_ns / 1_000_000).clamp(0, (1 << 48) - 1) as u64),
                    max: Field::new(key.0, FieldKind::Timestamp, (max_ns / 1_000_000).clamp(0, (1 << 48) - 1) as u64),
                });
            }
            if *negated {
                R::AlwaysTrue
            } else {
                R::AlwaysFalse
            }
        }
    }
}

impl<'b> PredBuilder<'b> {
    pub fn matches(&self, log: LogEntry, target_mapper: &Mapper) -> bool {
        match self {
            PredBuilder::True => return true,
            PredBuilder::False => return false,
            PredBuilder::Field(key, field_test) => {
                if let Some(field) = log.field_by_dyn_key(*key) {
                    unsafe { field_test.matches(log.bucket(), field) }
                } else {
                    field_test.is_negated()
                }
            }
            PredBuilder::FieldOr(key, tests) => {
                if let Some(field) = log.field_by_dyn_key(*key) {
                    for test in tests.iter() {
                        if unsafe { test.matches(log.bucket(), field) } {
                            return true;
                        }
                    }
                };
                return false;
            }
            PredBuilder::FieldAnd(key, tests) => {
                if let Some(field) = log.field_by_dyn_key(*key) {
                    for test in tests.iter() {
                        if !unsafe { test.matches(log.bucket(), field) } {
                            return false;
                        }
                    }
                    return true;
                };
                return false;
            }
            PredBuilder::And(preds) => {
                for test in preds.iter() {
                    if !test.matches(log, target_mapper) {
                        return false;
                    }
                }
                return true;
            }
            PredBuilder::Or(preds) => {
                for test in preds.iter() {
                    if test.matches(log, target_mapper) {
                        return true;
                    }
                }
                return false;
            }
            PredBuilder::SpanIs(negated, span) => {
                if let Some(field_span) = log.span_id() {
                    if field_span.as_u64() == *span {
                        return !negated;
                    }
                }
                return *negated;
            }
            PredBuilder::ParentSpanIs(negated, span) => {
                if let Some(field_parent_span) = log.parent_span_id() {
                    if field_parent_span.as_u64() == *span {
                        return !negated;
                    }
                }
                return *negated;
            }
            PredBuilder::HasSpan(negated) => {
                if log.span_range().is_none() {
                    return !*negated;
                } else {
                    return *negated;
                }
            }
            PredBuilder::HasParentSpan(negated) => {
                if log.parent_span_id().is_some() {
                    return !negated;
                }
                return *negated;
            }
            PredBuilder::SpanDurationRange { negated, min_ns, max_ns } => {
                if let Some(duration) = log.span_ns_duration() {
                    if duration >= *min_ns && duration <= *max_ns {
                        return !*negated;
                    }
                }
                return *negated;
            }
            PredBuilder::TimestampRange { negated, min_ns, max_ns } => {
                let timestamp = log.timestamp();
                if timestamp >= *min_ns && timestamp <= *max_ns {
                    return !*negated;
                };
                return *negated;
            }
            PredBuilder::LevelMask(negated, mask) => {
                if log.level_mask() & mask != 0 {
                    return !*negated;
                }
                return *negated;
            }
            PredBuilder::Target(field_test) => {
                if let Some(text) = target_mapper.get(log.target_id()) {
                    unsafe { field_test.matches_text(text.as_bytes()) }
                } else {
                    return field_test.is_negated();
                }
            }
            PredBuilder::Message(field_test) => return field_test.matches_text(log.message()),
            PredBuilder::Service(field_test) => {
                if let Some(service) = log.archetype().service() {
                    unsafe { field_test.matches_text(service.as_str().as_bytes()) }
                } else {
                    return field_test.is_negated();
                }
            }
        }
    }
    pub fn build_with_opt<'o>(
        &self,
        bump: &'o Bump,
        bucket: &BucketGuard,
        maps: &IntermentMaps,
    ) -> PredBuildResult<'o> {
        use PredBuildResult as R;
        match self {
            PredBuilder::True => R::AlwaysTrue,
            PredBuilder::False => R::AlwaysFalse,
            PredBuilder::Field(key, test) => {
                if let Some(id) = key.id() {
                    let types = maps.keys.get(id).map(|info| info.kinds).unwrap_or_default();
                    match opt_copy_field_tes(id, types, test, bump, bucket, maps) {
                        FieldTestResult::AlwaysTrue => R::AlwaysTrue,
                        FieldTestResult::AlwaysFalse => R::AlwaysFalse,
                        FieldTestResult::Ok(field_test) => R::Ok(Pred::Field(id, field_test)),
                    }
                } else if test.is_negated() {
                    R::AlwaysTrue
                } else {
                    R::AlwaysFalse
                }
            }
            PredBuilder::FieldOr(key, items) => {
                let Some(id) = key.id() else {
                    return PredBuildResult::AlwaysFalse;
                };
                let types = maps.keys.get(id).map(|info| info.kinds).unwrap_or_default();
                unsafe {
                    let ptr = bump
                        .alloc_layout(Layout::from_size_align_unchecked(
                            size_of::<FieldTest>() * items.len(),
                            align_of::<FieldTest>(),
                        ))
                        .cast::<FieldTest<'o>>();
                    let mut len = 0;
                    for test in items.iter() {
                        match opt_copy_field_tes(id, types, test, bump, bucket, maps) {
                            FieldTestResult::AlwaysTrue => return R::AlwaysTrue,
                            FieldTestResult::AlwaysFalse => continue,
                            FieldTestResult::Ok(field_test) => {
                                ptr.add(len).write(field_test);
                                len += 1;
                            }
                        }
                    }
                    if len == 0 {
                        return R::AlwaysFalse;
                    } else if len == 1 {
                        return R::Ok(Pred::Field(id, unsafe { ptr.read() }));
                    }
                    R::Ok(Pred::FieldOr(id, std::slice::from_raw_parts(ptr.as_ptr(), len)))
                }
            }
            PredBuilder::FieldAnd(key, items) => {
                let Some(id) = key.id() else {
                    for item in items.iter() {
                        if !item.is_negated() {
                            return PredBuildResult::AlwaysFalse;
                        }
                    }
                    return PredBuildResult::AlwaysTrue;
                };
                let types = maps.keys.get(id).map(|info| info.kinds).unwrap_or_default();
                unsafe {
                    let ptr = bump
                        .alloc_layout(Layout::from_size_align_unchecked(
                            size_of::<FieldTest>() * items.len(),
                            align_of::<FieldTest>(),
                        ))
                        .cast::<FieldTest<'o>>();
                    let mut len = 0;
                    for test in items.iter() {
                        match opt_copy_field_tes(id, types, test, bump, bucket, maps) {
                            FieldTestResult::AlwaysFalse => return R::AlwaysFalse,
                            FieldTestResult::AlwaysTrue => continue,
                            FieldTestResult::Ok(field_test) => {
                                ptr.add(len).write(field_test);
                                len += 1;
                            }
                        }
                    }
                    if len == 0 {
                        return R::AlwaysTrue;
                    } else if len == 1 {
                        return R::Ok(Pred::Field(id, unsafe { ptr.read() }));
                    }
                    R::Ok(Pred::FieldAnd(id, std::slice::from_raw_parts(ptr.as_ptr(), len)))
                }
            }
            PredBuilder::And(items) => {
                let mut count = 0;
                unsafe {
                    let alloc = bump
                        .alloc_layout(Layout::from_size_align_unchecked(
                            items.len() * std::mem::size_of::<Pred>(),
                            align_of::<Pred>(),
                        ))
                        .cast::<Pred>();
                    let mut count = 0;
                    for item in items.iter() {
                        match item.build_with_opt(bump, bucket, maps) {
                            R::AlwaysTrue => continue,
                            R::AlwaysFalse => return R::AlwaysFalse,
                            R::Ok(pred) => alloc.add(count).write(pred),
                        }
                        count += 1;
                    }
                    if count == 0 {
                        return R::AlwaysTrue;
                    } else if count == 1 {
                        return R::Ok(unsafe { alloc.read() });
                    } else {
                        return R::Ok(Pred::And(std::slice::from_raw_parts(alloc.as_ptr(), count)));
                    }
                }
            }
            PredBuilder::Or(items) => {
                let mut count = 0;
                unsafe {
                    let alloc = bump
                        .alloc_layout(Layout::from_size_align_unchecked(
                            items.len() * std::mem::size_of::<Pred>(),
                            align_of::<Pred>(),
                        ))
                        .cast::<Pred>();
                    let mut count = 0;
                    for item in items.iter() {
                        match item.build_with_opt(bump, bucket, maps) {
                            R::AlwaysTrue => return R::AlwaysTrue,
                            R::AlwaysFalse => continue,
                            R::Ok(pred) => alloc.add(count).write(pred),
                        }
                        count += 1;
                    }
                    if count == 0 {
                        return R::AlwaysTrue;
                    } else if count == 1 {
                        return R::Ok(unsafe { alloc.read() });
                    } else {
                        return R::Ok(Pred::Or(std::slice::from_raw_parts(alloc.as_ptr(), count)));
                    }
                }
            }
            PredBuilder::SpanIs(negated, span_id) => R::Ok(Pred::SpanIs(*negated, *span_id)),
            PredBuilder::ParentSpanIs(negated, span_id) => R::Ok(Pred::ParentSpanIs(*negated, *span_id)),
            PredBuilder::HasSpan(negated) => R::Ok(Pred::HasSpan(*negated)),
            PredBuilder::HasParentSpan(negated) => R::Ok(Pred::HasParentSpan(*negated)),
            PredBuilder::SpanDurationRange { negated, min_ns, max_ns: mas_ns } => {
                R::Ok(Pred::SpanDurationRange { negated: *negated, min_ns: *min_ns, max_ns: *mas_ns })
            }
            PredBuilder::TimestampRange { negated, min_ns, max_ns: mas_ns } => {
                R::Ok(Pred::TimestampRange { negated: *negated, min_ns: *min_ns, max_ns: *mas_ns })
            }
            PredBuilder::LevelMask(negated, mask) => R::Ok(Pred::LevelMask(*negated, *mask)),
            PredBuilder::Target(test) => R::Ok(Pred::Target(test.copy_to_other_arena(bump))),
            PredBuilder::Service(test) => R::Ok(Pred::Service(test.copy_to_other_arena(bump))),
            PredBuilder::Message(test) => R::Ok(Pred::Message(test.copy_to_other_arena(bump))),
        }
    }
    // crated a more compact version of PredBuilder in Bump copying all the data
    //
    pub fn build<'o>(&self, bump: &'o Bump) -> PredBuildResult<'o> {
        use PredBuildResult as R;
        match self {
            PredBuilder::True => R::AlwaysTrue,
            PredBuilder::False => R::AlwaysFalse,
            PredBuilder::Field(key, test) => {
                if let Some(id) = key.id() {
                    R::Ok(Pred::Field(id, test.copy_to_other_arena(bump)))
                } else if test.is_negated() {
                    R::AlwaysTrue
                } else {
                    R::AlwaysFalse
                }
            }
            PredBuilder::FieldOr(key, items) => {
                let Some(id) = key.id() else {
                    return PredBuildResult::AlwaysFalse;
                };
                R::Ok(Pred::FieldOr(
                    id,
                    bump.alloc_slice_fill_iter(items.iter().map(|item| item.copy_to_other_arena(bump))),
                ))
            }
            PredBuilder::FieldAnd(key, items) => {
                let Some(id) = key.id() else {
                    return PredBuildResult::AlwaysFalse;
                };
                R::Ok(Pred::FieldAnd(
                    id,
                    bump.alloc_slice_fill_iter(items.iter().map(|item| item.copy_to_other_arena(bump))),
                ))
            }
            PredBuilder::And(items) => {
                let mut count = 0;
                unsafe {
                    let alloc = bump
                        .alloc_layout(Layout::from_size_align_unchecked(
                            items.len() * std::mem::size_of::<Pred>(),
                            align_of::<Pred>(),
                        ))
                        .cast::<Pred>();
                    let mut count = 0;
                    for item in items.iter() {
                        match item.build(bump) {
                            R::AlwaysTrue => continue,
                            R::AlwaysFalse => return R::AlwaysFalse,
                            R::Ok(pred) => alloc.add(count).write(pred),
                        }
                        count += 1;
                    }
                    if count == 0 {
                        return R::AlwaysTrue;
                    } else if count == 1 {
                        return R::Ok(unsafe { alloc.read() });
                    } else {
                        return R::Ok(Pred::And(std::slice::from_raw_parts(alloc.as_ptr(), count)));
                    }
                }
            }
            PredBuilder::Or(items) => {
                let mut count = 0;
                unsafe {
                    let alloc = bump
                        .alloc_layout(Layout::from_size_align_unchecked(
                            items.len() * std::mem::size_of::<Pred>(),
                            align_of::<Pred>(),
                        ))
                        .cast::<Pred>();
                    let mut count = 0;
                    for item in items.iter() {
                        match item.build(bump) {
                            R::AlwaysTrue => return R::AlwaysTrue,
                            R::AlwaysFalse => continue,
                            R::Ok(pred) => alloc.add(count).write(pred),
                        }
                        count += 1;
                    }
                    if count == 0 {
                        return R::AlwaysTrue;
                    } else if count == 1 {
                        return R::Ok(unsafe { alloc.read() });
                    } else {
                        return R::Ok(Pred::Or(std::slice::from_raw_parts(alloc.as_ptr(), count)));
                    }
                }
            }
            PredBuilder::SpanIs(negated, span_id) => R::Ok(Pred::SpanIs(*negated, *span_id)),
            PredBuilder::ParentSpanIs(negated, span_id) => R::Ok(Pred::ParentSpanIs(*negated, *span_id)),
            PredBuilder::HasSpan(negated) => R::Ok(Pred::HasSpan(*negated)),
            PredBuilder::HasParentSpan(negated) => R::Ok(Pred::HasParentSpan(*negated)),
            PredBuilder::SpanDurationRange { negated, min_ns, max_ns: mas_ns } => {
                R::Ok(Pred::SpanDurationRange { negated: *negated, min_ns: *min_ns, max_ns: *mas_ns })
            }
            PredBuilder::TimestampRange { negated, min_ns, max_ns: mas_ns } => {
                R::Ok(Pred::TimestampRange { negated: *negated, min_ns: *min_ns, max_ns: *mas_ns })
            }
            PredBuilder::LevelMask(negated, mask) => R::Ok(Pred::LevelMask(*negated, *mask)),
            PredBuilder::Target(test) => R::Ok(Pred::Target(test.copy_to_other_arena(bump))),
            PredBuilder::Service(test) => R::Ok(Pred::Service(test.copy_to_other_arena(bump))),
            PredBuilder::Message(test) => R::Ok(Pred::Message(test.copy_to_other_arena(bump))),
        }
    }
    pub(crate) fn negate(&mut self) {
        match self {
            PredBuilder::True => *self = PredBuilder::False,
            PredBuilder::False => *self = PredBuilder::True,
            PredBuilder::FieldOr(key, items) => {
                for item in items.iter_mut() {
                    item.negate();
                }
                let PredBuilder::FieldOr(key, items) = std::mem::replace(self, PredBuilder::True) else {
                    unreachable!()
                };
                *self = PredBuilder::FieldAnd(key, items);
            }
            PredBuilder::FieldAnd(key, items) => {
                for item in items.iter_mut() {
                    item.negate();
                }
                let PredBuilder::FieldAnd(key, items) = std::mem::replace(self, PredBuilder::True) else {
                    unreachable!()
                };
                *self = PredBuilder::FieldOr(key, items);
            }
            PredBuilder::Or(items) => {
                for item in items.iter_mut() {
                    item.negate()
                }
                let PredBuilder::Or(items) = std::mem::replace(self, PredBuilder::True) else { unreachable!() };
                *self = PredBuilder::And(items);
            }
            PredBuilder::And(items) => {
                for item in items.iter_mut() {
                    item.negate()
                }
                let PredBuilder::And(items) = std::mem::replace(self, PredBuilder::True) else { unreachable!() };
                *self = PredBuilder::Or(items);
            }
            PredBuilder::Field(_, field)
            | PredBuilder::Target(field)
            | PredBuilder::Service(field)
            | PredBuilder::Message(field) => {
                field.negate();
            }
            PredBuilder::SpanIs(negated, _)
            | PredBuilder::ParentSpanIs(negated, _)
            | PredBuilder::HasSpan(negated)
            | PredBuilder::HasParentSpan(negated)
            | PredBuilder::SpanDurationRange { negated, .. }
            | PredBuilder::TimestampRange { negated, .. }
            | PredBuilder::LevelMask(negated, _) => {
                *negated ^= true;
            }
        }
    }
}

enum KeySetPredicate<'a> {
    And(KeySet),
    Or(&'a [KeySet]),
}
struct Data {
    data: HashMap<KeyID, u8>,
}

macro_rules! throw {
    ($error_name:ident $({ $($fields:tt)* })? $(( $($tuple:tt)* ))? $(@ $ctx: expr)?) => {
        return Err(InternalQueryError::from((
            ErrorKind::$error_name $({$($fields)*})?  $(($($tuple)*))?
            $(, $ctx)?
        )))
    };
}

pub struct QueryParseError {
    pub kind: ErrorKind,
    pub span: Option<Span>,
    pub token_kind: Option<Of>,
    pub source_text: String,
}

impl QueryParseError {
    pub fn from_internal(err: InternalQueryError, source: &str) -> Self {
        match err.context {
            ErrorContext::Token(token) => QueryParseError {
                kind: err.kind,
                span: Some(token.span),
                token_kind: Some(token.of),
                source_text: source.to_string(),
            },
            ErrorContext::Range(span) => {
                QueryParseError { kind: err.kind, span: Some(span), token_kind: None, source_text: source.to_string() }
            }
            ErrorContext::None => {
                QueryParseError { kind: err.kind, span: None, token_kind: None, source_text: source.to_string() }
            }
        }
    }
}
pub fn parse_query<'b>(bump: &'b Bump, input: &str) -> Result<PredBuilder<'b>, QueryParseError> {
    let input = input.trim();
    if input.is_empty() {
        return Ok(PredBuilder::True);
    }
    match parser::Parser::init(bump, input).and_then(|mut p| {
        let (expr, span) = p.parse_expr(0)?;
        p.expr_to_predicates(expr)
    }) {
        Ok(predicates) => Ok(predicates),
        Err(err) => Err(QueryParseError::from_internal(err, input)),
    }
}

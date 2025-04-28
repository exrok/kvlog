use std::{fmt::Display, num::NonZeroU64};

use crate::{timestamp::Timestamp, LogLevel, SpanID, Timer};
use uuid::Uuid;
pub struct Encoder {
    pub(crate) buffer: Vec<u8>,
}

impl Encoder {
    pub const fn new() -> Encoder {
        Encoder { buffer: Vec::new() }
    }
    pub fn clear(&mut self) {
        self.buffer.clear()
    }
    pub fn with_capacity(capacity: usize) -> Encoder {
        Encoder {
            buffer: Vec::with_capacity(capacity),
        }
    }
    pub fn bytes(&self) -> &[u8] {
        &self.buffer
    }
    pub fn append(&mut self, log_level: LogLevel, timestamp_nano: u64) -> FieldEncoder<'_> {
        let start_offset = self.buffer.len();
        self.buffer.extend_from_slice(&[0u8; 4]);
        self.buffer.extend_from_slice(&timestamp_nano.to_le_bytes());
        self.buffer.push(log_level as u8);
        FieldEncoder {
            encoder: self,
            start: start_offset,
        }
    }
    pub fn append_now(&mut self, log_level: LogLevel) -> FieldEncoder<'_> {
        self.append(log_level, now())
    }
}

pub struct FieldBuffer {
    encoder: Encoder,
}
impl Default for FieldBuffer {
    fn default() -> Self {
        FieldBuffer {
            encoder: Encoder::new(),
        }
    }
}

impl FieldBuffer {
    pub fn clear(&mut self) {
        self.encoder.buffer.clear();
    }
    pub fn encoder(&mut self) -> FieldEncoder<'_> {
        let start = self.encoder.buffer.len();
        self.encoder.buffer.extend_from_slice(&[0u8; 4]);
        FieldEncoder {
            start: start,
            encoder: &mut self.encoder,
        }
    }
}

#[repr(C)]
pub struct FieldEncoder<'a> {
    start: usize,
    encoder: &'a mut Encoder,
}

impl<'a> Drop for FieldEncoder<'a> {
    fn drop(&mut self) {
        let len = self.encoder.buffer.len() - self.start - 4;
        self.encoder.buffer[self.start..self.start + 4]
            .copy_from_slice(&((MAGIC_BYTE << 24) | (len as u32)).to_le_bytes());
    }
}

impl<'a> FieldEncoder<'a> {
    fn append_u64le(&mut self, value: u64) {
        self.encoder.buffer.extend_from_slice(&value.to_le_bytes());
    }
    pub fn fields(self) -> LogFields<'a> {
        let (start, encoder) = unsafe { std::mem::transmute::<_, (usize, &'a mut Encoder)>(self) };

        LogFields {
            parser: ReverseDecoder {
                bytes: &encoder.buffer[start + 4..],
            },
        }
    }
    pub fn start_span(mut self, span: SpanID) {
        let mut flag = 0b0100_0000;
        flag |= SpanInfoKind::Start as u8;
        self.append_u64le(span.inner.get());
        self.encoder.buffer.push(flag);
    }
    pub fn start_span_with_parent(mut self, span: SpanID, parent: Option<SpanID>) {
        let mut flag = 0b0100_0000;
        if let Some(parent) = parent {
            flag |= SpanInfoKind::StartWithParent as u8;
            self.append_u64le(parent.inner.get());
        } else {
            flag |= SpanInfoKind::Start as u8;
        }
        self.append_u64le(span.inner.get());
        self.encoder.buffer.push(flag);
    }
    pub fn apply_span(mut self, span: SpanID) {
        let flag = 0b0100_0000 | (SpanInfoKind::Current as u8);
        self.append_u64le(span.inner.get());
        self.encoder.buffer.push(flag);
    }
    pub fn end_span(mut self, span: SpanID) {
        let flag = 0b0100_0000 | (SpanInfoKind::End as u8);
        self.append_u64le(span.inner.get());
        self.encoder.buffer.push(flag);
    }
    pub fn apply_span_info(self, info: SpanInfo) {
        match info {
            SpanInfo::Start { span, parent } => self.start_span_with_parent(span, parent),
            SpanInfo::Current { span } => self.apply_span(span),
            SpanInfo::End { span } => self.end_span(span),
            SpanInfo::None => (),
        }
    }
    pub fn apply_current_span(self) {
        if let Some(span) = crate::SpanID::current() {
            self.apply_span(span)
        }
    }
    #[inline]
    pub fn key(&mut self, key: &str) -> ValueEncoder<'_> {
        if let Some(key) = get_static_key(key) {
            self.encoder.buffer.push(key as u8);
        } else {
            if key.len() > 127 {
                panic!("Key too large");
            }
            self.encoder.buffer.extend_from_slice(key.as_bytes());
            self.encoder.buffer.push(0x80 | (key.len() as u8));
        }
        ValueEncoder {
            encoder: self.encoder,
        }
    }
    #[inline]
    pub fn dynamic_key(&mut self, key: &str) -> ValueEncoder<'_> {
        if key.len() > 127 {
            panic!("Key too large");
        }
        self.encoder.buffer.extend_from_slice(key.as_bytes());
        self.encoder.buffer.push(0x80 | (key.len() as u8));
        ValueEncoder {
            encoder: self.encoder,
        }
    }
    #[inline]
    #[doc(hidden)]
    pub fn raw_key(&mut self, key: u8) -> ValueEncoder<'_> {
        self.encoder.buffer.push(key);
        ValueEncoder {
            encoder: self.encoder,
        }
    }
    #[inline]
    pub fn static_key(&mut self, key: StaticKey) -> ValueEncoder<'_> {
        self.encoder.buffer.push(key as u8);
        ValueEncoder {
            encoder: self.encoder,
        }
    }
}

pub struct ValueEncoder<'a> {
    pub(crate) encoder: &'a mut Encoder,
}
impl<'a> ValueEncoder<'a> {
    pub fn value_via_debug<T: std::fmt::Debug>(self, value: &T) {
        let start = self.encoder.buffer.len();
        use std::io::Write;
        let _ = write!(&mut self.encoder.buffer, "{:?}", value);
        let len = self.encoder.buffer.len() - start;
        if len <= 127 {
            self.encoder.buffer.push((len as u8) | 0x80)
        } else {
            let len = if len > u16::MAX as usize {
                self.encoder.buffer.truncate(start + (u16::MAX as usize));
                u16::MAX
            } else {
                len as u16
            };
            self.encoder.buffer.extend_from_slice(&len.to_le_bytes());
            self.encoder.buffer.push(ValueKind::String as u8);
        }
    }
    pub fn value_via_display<T: std::fmt::Display>(self, value: &T) {
        let start = self.encoder.buffer.len();
        use std::io::Write;
        let _ = write!(&mut self.encoder.buffer, "{}", value);
        let len = self.encoder.buffer.len() - start;
        if len <= 127 {
            self.encoder.buffer.push((len as u8) | 0x80)
        } else {
            let len = if len > u16::MAX as usize {
                self.encoder.buffer.truncate(start + (u16::MAX as usize));
                u16::MAX
            } else {
                len as u16
            };
            self.encoder.buffer.extend_from_slice(&len.to_le_bytes());
            self.encoder.buffer.push(ValueKind::String as u8);
        }
    }
}

macro_rules! builtin_keys {
    ($($key: ident),* $(,)?) => {
        #[allow(non_camel_case_types)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(u8)]
        pub enum StaticKey{
            $($key,)*
        }
        impl StaticKey {
            pub const NAMES: &'static [&'static str] = &[
                $(stringify!($key),)*
            ];
            pub fn from_u8(a: u8) -> Option<StaticKey> {
                if (a as usize) < Self::NAMES.len() {
                    Some(unsafe{
                    std::mem::transmute(a)
                    })
                } else {
                    None
                }
            }
            pub fn get_key(&self) -> &'static str {
                match self {
                    $(StaticKey::$key => stringify!($key),)*
                }
            }
            pub fn as_str(self) -> &'static str {
                *Self::NAMES.get(self as usize).unwrap()
            }
            pub fn u8_to_string(byte: u8) -> Option<&'static str> {
                Self::NAMES.get(byte as usize).copied()
            }
        }
        pub fn get_static_key(key: &str) -> Option<StaticKey> {
            match key {
                $(stringify!($key) => Some(StaticKey::$key),)*
                _ => None
            }
        }
    };
}
builtin_keys! {
    msg,
    err,
    error,
    cause,
    method,
    status,
    size,
    time,
    count,
    total,
    ms,
    id,
    user_id,
    object_id,
    caller,
    target,
    duration,
    timezone,
    content_type,
    conn_id,
    path,
    length,
    on,
    kind,
    sensor_id,
    handler,
    timestamp,
    elapsed,
    camera_id,
    system_id,
    next
}

pub trait Encode {
    fn encode_log_value_into(&self, output: ValueEncoder<'_>);
}

impl Encode for SpanID {
    fn encode_log_value_into(&self, output: ValueEncoder<'_>) {
        self.inner.get().encode_log_value_into(output);
    }
}

impl Encode for NonZeroU64 {
    fn encode_log_value_into(&self, output: ValueEncoder<'_>) {
        self.get().encode_log_value_into(output);
    }
}
impl<T: Encode> Encode for Option<T> {
    fn encode_log_value_into(&self, output: ValueEncoder<'_>) {
        if let Some(value) = self {
            value.encode_log_value_into(output);
        } else {
            output.encoder.buffer.push(ValueKind::None as u8)
        }
    }
}

impl Encode for String {
    fn encode_log_value_into(&self, output: ValueEncoder<'_>) {
        self.as_str().encode_log_value_into(output)
    }
}

impl Encode for Box<str> {
    fn encode_log_value_into(&self, output: ValueEncoder<'_>) {
        self.as_ref().encode_log_value_into(output)
    }
}

impl Encode for char {
    fn encode_log_value_into(&self, output: ValueEncoder<'_>) {
        let mut buf = [0u8; 4];
        self.encode_utf8(&mut buf).encode_log_value_into(output);
    }
}
impl Encode for bool {
    fn encode_log_value_into(&self, output: ValueEncoder<'_>) {
        output.encoder.buffer.push(if *self {
            ValueKind::True as u8
        } else {
            ValueKind::False as u8
        })
    }
}

#[derive(PartialEq, Clone, Copy, PartialOrd)]
pub enum Value<'a> {
    String(&'a [u8]),
    Bytes(&'a [u8]),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
    F64(f64),
    UUID(Uuid),
    Bool(bool),
    Seconds(f32),
    Timestamp(Timestamp),
    None,
}

impl Value<'_> {
    pub fn is_large(&self) -> bool {
        match self {
            Value::String(slice) | Value::Bytes(slice) => slice.len() > 48,
            _ => false,
        }
    }
}

impl<'a> std::fmt::Debug for Value<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(arg0) => write!(f, "String(\"{}\")", arg0.escape_ascii()),
            Self::Bytes(arg0) => f.debug_tuple("Bytes").field(arg0).finish(),
            Self::I32(arg0) => f.debug_tuple("I32").field(arg0).finish(),
            Self::U32(arg0) => f.debug_tuple("U32").field(arg0).finish(),
            Self::I64(arg0) => f.debug_tuple("I64").field(arg0).finish(),
            Self::U64(arg0) => f.debug_tuple("U64").field(arg0).finish(),
            Self::F32(arg0) => f.debug_tuple("F32").field(arg0).finish(),
            Self::F64(arg0) => f.debug_tuple("F64").field(arg0).finish(),
            Self::UUID(arg0) => f.debug_tuple("UUID").field(arg0).finish(),
            Self::Bool(arg0) => f.debug_tuple("Bool").field(arg0).finish(),
            Self::Seconds(arg0) => f.debug_tuple("Seconds").field(arg0).finish(),
            Self::Timestamp(arg0) => f.debug_tuple("Timestamp").field(arg0).finish(),
            Self::None => write!(f, "None"),
        }
    }
}

impl<'a> Eq for Value<'a> {}

impl<'a> std::hash::Hash for Value<'a> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Value::String(s) => s.hash(state),
            Value::Bytes(b) => b.hash(state),
            Value::I32(i) => i.hash(state),
            Value::U32(u) => u.hash(state),
            Value::I64(i) => i.hash(state),
            Value::U64(u) => u.hash(state),
            Value::F32(n) => n.to_bits().hash(state),
            Value::F64(n) => n.to_bits().hash(state),
            Value::UUID(u) => u.hash(state),
            Value::Bool(u) => u.hash(state),
            Value::Seconds(u) => u.to_bits().hash(state),
            Value::Timestamp(u) => u.hash(state),
            Value::None => {}
        }
    }
}
impl<'a> Display for Value<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::String(s) => s.escape_ascii().fmt(f),
            Value::Bytes(b) => b.escape_ascii().fmt(f),
            Value::I32(i) => i.fmt(f),
            Value::U32(u) => u.fmt(f),
            Value::I64(i) => i.fmt(f),
            Value::U64(u) => u.fmt(f),
            Value::F32(n) => n.fmt(f),
            Value::F64(n) => n.fmt(f),
            Value::UUID(u) => u.fmt(f),
            Value::Bool(u) => u.fmt(f),
            Value::Seconds(u) => fmt_seconds(*u, f),
            Value::Timestamp(t) => t.fmt(f),
            Value::None => f.write_str("None"),
        }
    }
}

impl Encode for Timestamp {
    fn encode_log_value_into(&self, ValueEncoder { encoder }: ValueEncoder<'_>) {
        encoder
            .buffer
            .extend_from_slice(&self.seconds.to_le_bytes());
        encoder.buffer.extend_from_slice(&self.nanos.to_le_bytes());
        encoder.buffer.push(ValueKind::Timestamp as u8);
    }
}

#[cfg(feature = "jiff-02")]
impl Encode for jiff::Timestamp {
    fn encode_log_value_into(&self, ValueEncoder { encoder }: ValueEncoder<'_>) {
        let seconds = self.as_second();
        let nanos = self.subsec_nanosecond();
        encoder.buffer.extend_from_slice(&seconds.to_le_bytes());
        encoder.buffer.extend_from_slice(&nanos.to_le_bytes());
        encoder.buffer.push(ValueKind::Timestamp as u8);
    }
}

fn decode_timestamp(muncher: &mut ReverseDecoder) -> Result<Timestamp, MunchError> {
    let nanos = u32::from_le_bytes(*muncher.pop_array()?);
    let seconds = i64::from_le_bytes(*muncher.pop_array()?);
    Ok(Timestamp::new_clamped(seconds, nanos))
}

fn decode_value<'a>(muncher: &mut ReverseDecoder<'a>) -> Result<Value<'a>, MunchError> {
    let value_kind = muncher.pop()?;
    let value = if value_kind & 0x80 != 0 {
        let len = value_kind as usize & 0x7F;
        Value::String(muncher.pop_slice(len)?)
    } else {
        let value_kind = ValueKind::from_u8(value_kind).ok_or(MunchError::InvalidValueKind)?;
        match value_kind {
            ValueKind::String => {
                let len = u16::from_le_bytes(*muncher.pop_array()?);
                Value::String(muncher.pop_slice(len as usize)?)
            }
            ValueKind::Bytes => {
                let len = u16::from_le_bytes(*muncher.pop_array()?);
                Value::Bytes(muncher.pop_slice(len as usize)?)
            }
            ValueKind::I32 => Value::I32(i32::from_le_bytes(*muncher.pop_array()?)),
            ValueKind::U32 => Value::U32(u32::from_le_bytes(*muncher.pop_array()?)),
            ValueKind::True => Value::Bool(true),
            ValueKind::False => Value::Bool(false),
            ValueKind::I64 => Value::I64(i64::from_le_bytes(*muncher.pop_array()?)),
            ValueKind::U64 => Value::U64(u64::from_le_bytes(*muncher.pop_array()?)),
            ValueKind::F32 => Value::F32(f32::from_le_bytes(*muncher.pop_array()?)),
            ValueKind::F64 => Value::F64(f64::from_le_bytes(*muncher.pop_array()?)),
            ValueKind::UUID => Value::UUID(uuid::Uuid::from_bytes(*muncher.pop_array()?)),
            ValueKind::None => Value::None,
            ValueKind::Seconds => Value::Seconds(f32::from_le_bytes(*muncher.pop_array()?)),
            ValueKind::Timestamp => Value::Timestamp(decode_timestamp(muncher)?),
        }
    };
    Ok(value)
}

impl<'a> Encode for Value<'a> {
    fn encode_log_value_into(&self, output: ValueEncoder<'_>) {
        match self {
            Value::String(s) => {
                let buf = &mut output.encoder.buffer;
                let len = s.len().min(u16::MAX as usize);
                buf.extend_from_slice(&s[..len]);
                if len > 127 {
                    buf.extend_from_slice(&(len as u16).to_le_bytes());
                    buf.push(ValueKind::String as u8);
                } else {
                    buf.push(0x80 | (s.len() as u8));
                }
            }
            Value::Bytes(b) => b.encode_log_value_into(output),
            Value::I32(i) => i.encode_log_value_into(output),
            Value::U32(u) => u.encode_log_value_into(output),
            Value::I64(i) => i.encode_log_value_into(output),
            Value::U64(u) => u.encode_log_value_into(output),
            Value::F32(n) => n.encode_log_value_into(output),
            Value::F64(n) => n.encode_log_value_into(output),
            Value::UUID(u) => u.encode_log_value_into(output),
            Value::Bool(u) => u.encode_log_value_into(output),
            Value::Seconds(f) => encode_seconds(*f, output),
            Value::Timestamp(f) => f.encode_log_value_into(output),
            Value::None => None::<u8>.encode_log_value_into(output),
        }
    }
}

pub enum ValueKind {
    String = 0x10,
    Bytes = 0x11,
    I32 = 0x12,
    U32 = 0x13,
    I64 = 0x14,
    U64 = 0x15,
    F32 = 0x16,
    F64 = 0x17,
    UUID = 0x18,
    True = 0x19,
    False = 0x1A,
    None = 0x1B,
    Seconds = 0x1C,
    Timestamp = 0x1D,
}
impl ValueKind {
    fn from_u8(value: u8) -> Option<ValueKind> {
        match value {
            0x10 => Some(ValueKind::String),
            0x11 => Some(ValueKind::Bytes),
            0x12 => Some(ValueKind::I32),
            0x13 => Some(ValueKind::U32),
            0x14 => Some(ValueKind::I64),
            0x15 => Some(ValueKind::U64),
            0x16 => Some(ValueKind::F32),
            0x17 => Some(ValueKind::F64),
            0x18 => Some(ValueKind::UUID),
            0x19 => Some(ValueKind::True),
            0x1A => Some(ValueKind::False),
            0x1B => Some(ValueKind::None),
            0x1C => Some(ValueKind::Seconds),
            0x1D => Some(ValueKind::Timestamp),
            _ => None,
        }
    }
}

pub struct BStr<'a>(pub &'a [u8]);
impl<'a> Encode for BStr<'a> {
    fn encode_log_value_into(&self, output: ValueEncoder<'_>) {
        let buf = &mut output.encoder.buffer;
        let len = self.0.len().min(u16::MAX as usize);
        buf.extend_from_slice(&self.0[..len]);
        if len > 127 {
            buf.extend_from_slice(&(len as u16).to_le_bytes());
            buf.push(ValueKind::String as u8);
        } else {
            buf.push(0x80 | (self.0.len() as u8));
        }
    }
}

impl Encode for str {
    fn encode_log_value_into(&self, output: ValueEncoder<'_>) {
        let buf = &mut output.encoder.buffer;
        let len = self.as_bytes().len().min(u16::MAX as usize);
        buf.extend_from_slice(&self.as_bytes()[..len]);
        if len > 127 {
            buf.extend_from_slice(&(len as u16).to_le_bytes());
            buf.push(ValueKind::String as u8);
        } else {
            buf.push(0x80 | (self.len() as u8));
        }
    }
}

impl Encode for &str {
    fn encode_log_value_into(&self, output: ValueEncoder<'_>) {
        (*self).encode_log_value_into(output)
    }
}

impl Encode for Timer {
    fn encode_log_value_into(&self, output: ValueEncoder<'_>) {
        encode_seconds(self.instant.elapsed().as_secs_f32(), output)
    }
}
impl Encode for Uuid {
    fn encode_log_value_into(&self, output: ValueEncoder<'_>) {
        output.encoder.buffer.extend_from_slice(self.as_bytes());
        output.encoder.buffer.push(ValueKind::UUID as u8);
    }
}
impl Encode for [u8] {
    fn encode_log_value_into(&self, output: ValueEncoder<'_>) {
        let buf = &mut output.encoder.buffer;
        let len = self.len().min(u16::MAX as usize);
        buf.extend_from_slice(&self[..len]);
        buf.extend_from_slice(&(len as u16).to_le_bytes());
        output.encoder.buffer.push(ValueKind::Bytes as u8);
    }
}

pub struct Seconds(pub f32);

impl Encode for Seconds {
    fn encode_log_value_into(&self, output: ValueEncoder<'_>) {
        encode_seconds(self.0, output)
    }
}

fn encode_seconds(value: f32, output: ValueEncoder<'_>) {
    output
        .encoder
        .buffer
        .extend_from_slice(&value.to_le_bytes());
    output.encoder.buffer.push(ValueKind::Seconds as u8);
}
macro_rules! impl_le_bytes_log_value {
    ($($variant:tt : $type:tt),*) => {
        $(
            impl Encode for $type {
                fn encode_log_value_into(&self, output: ValueEncoder<'_>) {
                    output.encoder.buffer.extend_from_slice(&self.to_le_bytes());
                    output.encoder.buffer.push(ValueKind::$variant as u8);
                }
            }
        )*
    };
}

impl_le_bytes_log_value! {
    I32: i32,
    I64: i64,
    U32: u32,
    U64: u64,
    F32: f32,
    F64: f64
}

macro_rules! impl_as_log_value {
    ($($type:tt as $as_type:tt),* $(,)?) => {
        $(
            impl Encode for $type {
                fn encode_log_value_into(&self, output: ValueEncoder<'_>) {
                    (*self as $as_type).encode_log_value_into(output);
                }
            }
        )*
    };
}

impl_as_log_value! {
    i8 as i32,
    u8 as u32,
    i16 as i32,
    u16 as u32,
    usize as u64,
    isize as i64,
}
const MAGIC_BYTE: u32 = 0b1110_0001;
#[derive(Debug)]
pub enum MunchError {
    Eof,
    EofOnHeader,
    EofOnFields,
    InvalidKey,
    InvalidValueKind,
    InvalidLogLevel,
    InvalidValue,
    InvalidString,
    InvalidTimestamp,
    MissingMagicByte,
    InvalidSpanID,
}

#[derive(Clone)]
pub struct ReverseDecoder<'a> {
    pub bytes: &'a [u8],
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SpanInfo {
    Start {
        span: SpanID,
        parent: Option<SpanID>,
    },
    Current {
        span: SpanID,
    },
    End {
        span: SpanID,
    },
    None,
}

impl SpanInfo {
    pub fn is_none(&self) -> bool {
        matches!(self, SpanInfo::None)
    }
}
enum SpanInfoKind {
    StartWithParent = 0,
    Start = 1,
    Current = 2,
    End = 3,
}
impl SpanInfoKind {
    fn from_u8(value: u8) -> SpanInfoKind {
        match value & 0b11 {
            0 => SpanInfoKind::StartWithParent,
            1 => SpanInfoKind::Start,
            2 => SpanInfoKind::Current,
            3 => SpanInfoKind::End,
            _ => unreachable!(),
        }
    }
}
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Key<'a> {
    Static(StaticKey),
    Dynamic(&'a [u8]),
}
impl<'a> From<&'a str> for Key<'a> {
    fn from(value: &'a str) -> Self {
        if let Some(key) = get_static_key(value) {
            Key::Static(key)
        } else {
            Key::Dynamic(value.as_bytes())
        }
    }
}
impl<'a> PartialEq<&str> for Key<'a> {
    fn eq(&self, other: &&str) -> bool {
        match self {
            Key::Static(key) => key.as_str() == *other,
            Key::Dynamic(value) => *value == other.as_bytes(),
        }
    }
}
impl<'a> PartialEq<str> for Key<'a> {
    fn eq(&self, other: &str) -> bool {
        match self {
            Key::Static(key) => key.as_str() == other,
            Key::Dynamic(value) => *value == other.as_bytes(),
        }
    }
}

impl<'a> PartialEq<StaticKey> for Key<'a> {
    fn eq(&self, other: &StaticKey) -> bool {
        match self {
            Key::Static(key) => key == other,
            Key::Dynamic(_) => false,
        }
    }
}

impl<'a> std::fmt::Display for Key<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(text) = self.as_str() {
            f.write_str(text)
        } else {
            write!(f, "{}", self.as_bytes().escape_ascii())
        }
    }
}

impl<'a> Key<'a> {
    pub fn as_bytes(&self) -> &'a [u8] {
        match self {
            Key::Static(key) => key.as_str().as_bytes(),
            Key::Dynamic(key) => key,
        }
    }
    pub fn as_str(&self) -> Option<&'a str> {
        match self {
            Key::Static(key) => Some(key.as_str()),
            Key::Dynamic(key) => std::str::from_utf8(key).ok(),
        }
    }
}

impl<'a> ReverseDecoder<'a> {
    pub fn pop_array<const N: usize>(&mut self) -> Result<&'a [u8; N], MunchError> {
        if let Some(value) = self.bytes.len().checked_sub(N) {
            let ptr = self.bytes.as_ptr();
            self.bytes = unsafe { std::slice::from_raw_parts(ptr, value) };
            Ok(unsafe { &*(ptr.add(value) as *const [u8; N]) })
        } else {
            Err(MunchError::Eof)
        }
    }
    pub fn pop_slice(&mut self, len: usize) -> Result<&'a [u8], MunchError> {
        if let Some(value) = self.bytes.len().checked_sub(len) {
            let ptr = self.bytes.as_ptr();
            self.bytes = unsafe { std::slice::from_raw_parts(ptr, value) };
            Ok(unsafe { std::slice::from_raw_parts(ptr.add(value), len) })
        } else {
            Err(MunchError::Eof)
        }
    }
    pub fn pop(&mut self) -> Result<u8, MunchError> {
        if let Some(value) = self.bytes.len().checked_sub(1) {
            let ptr = self.bytes.as_ptr();
            self.bytes = unsafe { std::slice::from_raw_parts(ptr, value) };
            Ok(unsafe { *ptr.add(value) })
        } else {
            Err(MunchError::Eof)
        }
    }
    pub fn pop_header(&mut self) -> Result<(usize, u64, LogLevel), MunchError> {
        let mut muncher = self.clone();
        let level = LogLevel::from_u8(muncher.pop()?).ok_or(MunchError::InvalidLogLevel)?;
        let timestamp = u64::from_le_bytes(*muncher.pop_array()?);
        let header = u32::from_le_bytes(*muncher.pop_array()?);
        if header >> 24 != MAGIC_BYTE {
            return Err(MunchError::MissingMagicByte);
        }
        let len = (header & 0x00FF_FFFF) as usize;
        *self = muncher;
        Ok((len, timestamp, level))
    }
    pub fn pop_span_info(&mut self) -> Result<SpanInfo, MunchError> {
        let mut muncher = self.clone();
        let Ok(flag) = muncher.pop() else {
            return Ok(SpanInfo::None);
        };
        if flag & 0b1100_0000 != 0b0100_0000 {
            return Ok(SpanInfo::None);
        }
        let span = SpanID {
            inner: NonZeroU64::new(u64::from_le_bytes(*muncher.pop_array()?))
                .ok_or(MunchError::InvalidSpanID)?,
        };
        let span_info = match SpanInfoKind::from_u8(flag) {
            SpanInfoKind::StartWithParent => SpanInfo::Start {
                span,
                parent: Some(SpanID {
                    inner: NonZeroU64::new(u64::from_le_bytes(*muncher.pop_array()?))
                        .ok_or(MunchError::InvalidSpanID)?,
                }),
            },
            SpanInfoKind::Start => SpanInfo::Start { span, parent: None },
            SpanInfoKind::Current => SpanInfo::Current { span },
            SpanInfoKind::End => SpanInfo::End { span },
        };
        *self = muncher;
        Ok(span_info)
    }

    pub fn pop_key(&mut self) -> Result<Key<'a>, MunchError> {
        let mut muncher = self.clone();
        let key_header = muncher.pop()?;
        let key = if key_header < 0x80 {
            match StaticKey::from_u8(key_header) {
                Some(key) => Key::Static(key),
                None => return Err(MunchError::InvalidKey),
            }
        } else {
            let len = key_header & 0x7F;
            Key::Dynamic(muncher.pop_slice(len as usize)?)
        };
        *self = muncher;
        Ok(key)
    }
    pub fn pop_value(&mut self) -> Result<Value<'a>, MunchError> {
        let mut muncher = self.clone();
        let value = decode_value(&mut muncher)?;
        *self = muncher;
        Ok(value)
    }
    pub fn pop_key_value(&mut self) -> Result<(Key<'a>, Value<'a>), MunchError> {
        // Only commit, if parsing succeeds;
        let mut muncher = self.clone();
        let value = decode_value(&mut muncher)?;
        let key_header = muncher.pop()?;
        let key = if key_header < 0x80 {
            match StaticKey::from_u8(key_header) {
                Some(key) => Key::Static(key),
                None => return Err(MunchError::InvalidKey),
            }
        } else {
            let len = key_header & 0x7F;
            Key::Dynamic(muncher.pop_slice(len as usize)?)
        };
        *self = muncher;
        Ok((key, value))
    }
}

pub(crate) fn now() -> u64 {
    let now = std::time::SystemTime::now();
    let duration = now.duration_since(std::time::UNIX_EPOCH).unwrap();
    duration.as_nanos() as u64
}

pub fn munch_log_with_span<'a>(
    input: &mut &'a [u8],
) -> Result<(u64, LogLevel, SpanInfo, LogFields<'a>), MunchError> {
    if input.len() < (4 + 8 + 1) {
        return Err(MunchError::EofOnHeader);
    }
    let header = u32::from_le_bytes(input[..4].try_into().unwrap());
    if header >> 24 != MAGIC_BYTE {
        return Err(MunchError::MissingMagicByte);
    }
    let len = header & 0x00FF_FFFF;
    let timestamp = u64::from_le_bytes(input[4..12].try_into().unwrap());
    let Some(log_level) = LogLevel::from_u8(input[12]) else {
        return Err(MunchError::InvalidLogLevel);
    };
    let values = &input
        .get(13..(4 + len as usize))
        .ok_or(MunchError::EofOnFields)?;
    *input = &input[(4 + len as usize)..];
    let mut parser = ReverseDecoder { bytes: values };
    let span_info = parser.pop_span_info()?;
    Ok((timestamp, log_level, span_info, LogFields { parser }))
}

pub fn decode<'a>(
    mut bytes: &'a [u8],
) -> impl Iterator<Item = Result<(u64, LogLevel, SpanInfo, LogFields<'a>), MunchError>> + 'a {
    std::iter::from_fn(move || {
        if bytes.is_empty() {
            return None;
        }
        Some(munch_log_with_span(&mut bytes))
    })
}

#[derive(Clone)]
pub struct LogFields<'a> {
    pub parser: ReverseDecoder<'a>,
}

impl LogFields<'static> {
    pub fn empty() -> LogFields<'static> {
        LogFields {
            parser: ReverseDecoder { bytes: &[] },
        }
    }
}

impl<'a> Iterator for LogFields<'a> {
    type Item = Result<(Key<'a>, Value<'a>), MunchError>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.parser.bytes.is_empty() {
            return None;
        }
        Some(self.parser.pop_key_value())
    }
}
fn fmt_seconds(s: f32, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    #[repr(u8)]
    enum Unit {
        S,
        Ms,
        Us,
        Ns,
    }
    use Unit::*;
    let slog = ((s.log10() * 2.0) + 0.000001) as i32;
    static SIZE_TABLE: &[(u8, Unit, f32); 17] = &[
        (0, Ns, 1000_000_000.0),
        (2, Us, 1000_000.0),
        (2, Us, 1000_000.0),
        (1, Us, 1000_000.0),
        (1, Us, 1000_000.0),
        (0, Us, 1000_000.0),
        (0, Us, 1000_000.0),
        (2, Ms, 1000.0),
        (2, Ms, 1000.0),
        (1, Ms, 1000.0),
        (1, Ms, 1000.0),
        (0, Ms, 1000.0),
        (2, S, 1.0),
        (2, S, 1.0),
        (1, S, 1.0),
        (1, S, 1.0),
        (0u8, S, 1.0),
    ];
    let (prec, unit, factor) = &SIZE_TABLE[(slog + 12).clamp(0, 16) as usize];
    write!(
        f,
        "{secs:.prec$}{unit}",
        prec = *prec as usize,
        unit = match unit {
            S => "s",
            Ms => "ms",
            Us => "µs",
            Ns => "ns",
        },
        secs = s * factor
    )
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn fmt() {
        let value4 = 10000.0f64;
        let value3 = 19000.0f64;
        let value2 = 33333.0f64;
        let value1 = 88888.0f64;
        for i in 1..13 {
            let value = value1 / (10.0f64).powi(i);
            println!("{:20} => {}", value as f32, Value::Seconds(value as f32));
            let value = value2 / (10.0f64).powi(i);
            println!("{:20} => {}", value as f32, Value::Seconds(value as f32));
            let value = value3 / (10.0f64).powi(i);
            println!("{:20} => {}", value as f32, Value::Seconds(value as f32));
            let value = value4 / (10.0f64).powi(i);
            println!("{:20} => {}\n", value as f32, Value::Seconds(value as f32));
        }
    }
}

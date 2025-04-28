use std::{
    cell::Cell,
    mem::ManuallyDrop,
    num::NonZeroU64,
    sync::{atomic::AtomicU64, MutexGuard},
};

use collector::LogQueue;
pub use encoding::BStr;
pub use encoding::{Encode, SpanInfo, ValueEncoder};
pub use kvlog_macros::emit_log;
mod spanning;
mod timestamp;
pub use spanning::Spanning;
pub use timestamp::Timestamp;

pub struct Timer {
    instant: std::time::Instant,
}

impl Timer {
    pub fn start() -> Timer {
        Timer {
            instant: std::time::Instant::now(),
        }
    }
}

#[macro_export]
macro_rules! info { ($($tt:tt)*) => { $crate::emit_log!(Info, $($tt)*) }; }
#[macro_export]
macro_rules! warn { ($($tt:tt)*) => { $crate::emit_log!(Warn, $($tt)*) }; }
#[macro_export]
macro_rules! trace { ($($tt:tt)*) => { $crate::emit_log!(Trace, $($tt)*) }; }
#[cfg(feature = "debug")]
#[macro_export]
macro_rules! debug { ($($tt:tt)*) => { $crate::emit_log!(Debug, $($tt)*) }; }
#[cfg(not(feature = "debug"))]
#[macro_export]
macro_rules! debug {
    ($($tt:tt)*) => {};
}
#[macro_export]
macro_rules! error { ($($tt:tt)*) => { $crate::emit_log!(Error, $($tt)*) }; }

thread_local! {
    static CURRENT_SPAN_ID: Cell<u64> = const { Cell::new(0) };
}

static NEXT_SPAN_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpanID {
    inner: NonZeroU64,
}
impl std::fmt::Debug for SpanID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016X}", self.inner.get())
    }
}

impl From<NonZeroU64> for SpanID {
    fn from(inner: NonZeroU64) -> Self {
        SpanID { inner }
    }
}

pub struct EnteredSpan {
    previous: u64,
}

pub fn global_logger() -> MutexGuard<'static, LogQueue> {
    match crate::collector::LOG_WRITER.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

impl Drop for EnteredSpan {
    fn drop(&mut self) {
        CURRENT_SPAN_ID.set(self.previous)
    }
}

pub fn continue_span<T: Sized>(t: T) -> Spanning<T> {
    Spanning {
        inner: ManuallyDrop::new(t),
        span: SpanID::current_or_next().inner,
    }
}

impl SpanID {
    pub fn as_u64(&self) -> u64 {
        self.inner.get()
    }
    pub fn next() -> SpanID {
        next_span_id()
    }
    #[inline]
    pub fn spanning<T: Sized>(&self, t: T) -> Spanning<T> {
        Spanning {
            inner: ManuallyDrop::new(t),
            span: self.inner,
        }
    }
    pub fn current_or_next() -> SpanID {
        if let Some(value) = NonZeroU64::new(CURRENT_SPAN_ID.with(|foo| foo.get())) {
            SpanID { inner: value }
        } else {
            return SpanID::next();
        }
    }
    pub fn current() -> Option<SpanID> {
        if let Some(value) = NonZeroU64::new(CURRENT_SPAN_ID.with(|foo| foo.get())) {
            Some(SpanID { inner: value })
        } else {
            None
        }
    }
    pub fn enter(&self) -> EnteredSpan {
        EnteredSpan {
            previous: CURRENT_SPAN_ID.replace(self.inner.get()),
        }
    }
}

#[cold]
fn update_span_id_from_entropy() -> SpanID {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut s = DefaultHasher::new();
    // Entropy: ASLR & Caller Location
    (((&s) as *const _) as usize).hash(&mut s);
    // Entropy: Time
    encoding::now().hash(&mut s);
    // Entropy: PID
    std::process::id().hash(&mut s);

    // Note: the ParentSpanSuffixTable currently relies on the high bit always being set
    // so that even after zeroing the first 12 bits the span id is still non-zero.
    let bit = unsafe { NonZeroU64::new_unchecked(s.finish() | 0x8000_0000_0000_0000) };
    // Plus one as the SPAN_ID counter stores next ID.
    NEXT_SPAN_ID.store(bit.get() + 1, std::sync::atomic::Ordering::Relaxed);
    SpanID { inner: bit }
}
pub fn set_next_span_counter(value: u64) {
    NEXT_SPAN_ID.store(value, std::sync::atomic::Ordering::Relaxed);
}

fn next_span_id() -> SpanID {
    let id = NEXT_SPAN_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if id < (1u64 << 63) {
        update_span_id_from_entropy()
    } else {
        unsafe {
            SpanID {
                inner: NonZeroU64::new_unchecked(id),
            }
        }
    }
}

pub fn set_global_span_id(x: SpanID) {
    CURRENT_SPAN_ID.set(x.inner.get())
}

pub fn clear_global_span_id() {
    CURRENT_SPAN_ID.set(0);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn from_u8(byte: u8) -> Option<LogLevel> {
        match byte {
            0 => Some(LogLevel::Debug),
            1 => Some(LogLevel::Info),
            2 => Some(LogLevel::Warn),
            3 => Some(LogLevel::Error),
            _ => None,
        }
    }
}

pub mod collector;
pub mod encoding;

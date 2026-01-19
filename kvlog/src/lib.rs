//! High-performance structured binary logging for Rust applications.
//!
//! kvlog provides macros for emitting structured log messages with key-value pairs,
//! optimized for high throughput and fast compile times. Log messages are encoded
//! in a compact binary format with nanosecond-resolution timestamps.
//!
//! # Log Messages
//!
//! Each log message contains:
//! - A nanosecond-resolution timestamp
//! - An optional fixed string message
//! - A target (Rust module path)
//! - A series of key-value pairs
//! - Optional span information for distributed tracing
//!
//! # Log Levels
//!
//! - [`debug!`]: Development-only logging, requires the `debug` feature flag
//! - [`info!`]: Provides useful context about application state
//! - [`warn!`]: Something bad happened but it was expected
//! - [`error!`]: Something bad happened unexpectedly, typically requires intervention
//!
//! # Examples
//!
//! Basic logging with key-value pairs:
//!
//! ```
//! # let status = 200u16;
//! # let object_id = 42u64;
//! // Simple message
//! kvlog::info!("Request completed");
//!
//! // With key-value pairs (shorthand: `status` is equivalent to `status = status`)
//! kvlog::info!("Request completed", status, object_id);
//! ```
//!
//! Using format specifiers:
//!
//! ```
//! # let err = "connection refused";
//! # let config = vec![1, 2, 3];
//! // %expr uses Display formatting
//! kvlog::error!("Failed to connect", %err);
//!
//! // ?expr uses Debug formatting
//! kvlog::warn!("Unexpected config", ?config);
//! ```
//!
//! Conditional fields:
//!
//! ```
//! # let response_length: Option<usize> = Some(1024);
//! kvlog::info!(
//!     "Response sent",
//!     if let Some(len) = response_length { length = len }
//! );
//! ```
//!
//! # Spans
//!
//! kvlog supports building trees of spans for distributed tracing. Unlike other
//! tracing systems, logs are emitted immediately with correlation IDs that track
//! parent-child relationships.
//!
//! ## Basic Span Usage
//!
//! ```
//! use kvlog::SpanID;
//!
//! let span = SpanID::next();
//! kvlog::info!("Starting operation", span.start = span);
//! kvlog::info!("Processing", span.current = span);
//! kvlog::info!("Operation complete", span.end = span);
//! ```
//!
//! ## Nested Spans
//!
//! Use `span.parent = Some(parent_id)` to create hierarchical span trees:
//!
//! ```
//! use kvlog::SpanID;
//!
//! let request_span = SpanID::next();
//! kvlog::info!("Request received", span.start = request_span);
//!
//! // Create a child span for database work
//! let db_span = SpanID::next();
//! kvlog::info!("Querying database", span.start = db_span, span.parent = Some(request_span));
//! kvlog::info!("Query complete", span.end = db_span);
//!
//! kvlog::info!("Request complete", span.end = request_span);
//! ```
//!
//! ## Span Guards
//!
//! Use [`SpanID::enter`] to set a thread-local span context. This is useful
//! when calling functions that should inherit the current span:
//!
//! ```
//! use kvlog::SpanID;
//!
//! fn process_item() {
//!     // This log automatically includes the current span context
//!     kvlog::info!("Processing item");
//! }
//!
//! let span = SpanID::next();
//! let _guard = span.enter();  // Set span as current
//! kvlog::info!("Starting work", span.start = span);
//! process_item();  // Logs here inherit the span
//! kvlog::info!("Work complete", span.end = span);
//! // Guard dropped, previous span context restored
//! ```
//!
//! # Default Test Logger
//!
//! When no collector is explicitly initialized, kvlog automatically emits logs
//! to stdout in a human-readable format with colors. This works seamlessly with
//! Rust's test runner - use `cargo test -- --nocapture` to see log output:
//!
//! ```
//! #[test]
//! fn my_test() {
//!     // No setup needed - logs are automatically printed to stdout
//!     kvlog::info!("Test starting", test_id = 42);
//!     // ... test logic ...
//! }
//! ```
//!
//! This zero-configuration behavior is ideal for development and debugging.
//!
//! # Collector Setup
//!
//! For production use, initialize a log collector to capture and output logs
//! with better performance and output control:
//!
//! ```no_run
//! let _guard = kvlog::spawn_collector_from_env(Some("my-service"), false);
//! kvlog::info!("Application started");
//! // Guard must be held for the duration of logging
//! ```

use std::{
    cell::Cell,
    mem::ManuallyDrop,
    num::NonZeroU64,
    path::PathBuf,
    str::FromStr,
    sync::{atomic::AtomicU64, MutexGuard as StdMutexGuard},
};

use collector::{LogQueue, LoggerGuard};
pub use collector::LogBuffer;
pub use encoding::BStr;
pub use encoding::{Encode, SpanInfo, ValueEncoder};
mod mutex;
mod spanning;
mod timestamp;
pub use mutex::{Mutex, MutexGuard};
pub use spanning::Spanning;
pub use timestamp::Timestamp;

/// Simple stopwatch for measuring elapsed time in log messages.
///
/// When encoded into a log message, the timer outputs the elapsed duration
/// since it was started, formatted in human-readable units (seconds, milliseconds,
/// microseconds, or nanoseconds).
///
/// # Examples
///
/// ```
/// let timer = kvlog::Timer::start();
/// // ... do some work ...
/// kvlog::info!("Operation completed", elapsed = timer);
/// ```
pub struct Timer {
    instant: std::time::Instant,
}

impl Timer {
    /// Starts a new timer from the current instant.
    pub fn start() -> Timer {
        Timer {
            instant: std::time::Instant::now(),
        }
    }
}

pub use kvlog_macros::{error, info, warn};

#[cfg(feature = "debug")]
#[macro_export]
pub use kvlog_macros::debug;
#[cfg(not(feature = "debug"))]
#[macro_export]
macro_rules! debug {
    ($($tt:tt)*) => {};
}

thread_local! {
    static CURRENT_SPAN_ID: Cell<u64> = const { Cell::new(0) };
}

static NEXT_SPAN_ID: AtomicU64 = AtomicU64::new(0);

/// Unique identifier for correlating related log messages.
///
/// Spans allow building trees of related operations for distributed tracing.
/// Each span has a unique ID that can be used to track parent-child relationships
/// between log messages.
///
/// # Span Lifecycle
///
/// - `span.start = id` - Mark the beginning of a span
/// - `span.current = id` - Log within an active span
/// - `span.end = id` - Mark the end of a span
/// - `span.parent = Some(parent_id)` - Establish parent-child relationship (use with `span.start`)
///
/// # Examples
///
/// Basic span usage:
///
/// ```
/// use kvlog::SpanID;
///
/// let span = SpanID::next();
/// kvlog::info!("Starting request", span.start = span);
/// kvlog::info!("Processing", span.current = span);
/// kvlog::info!("Request complete", span.end = span);
/// ```
///
/// Nested spans with parent-child relationships:
///
/// ```
/// use kvlog::SpanID;
///
/// let parent = SpanID::next();
/// kvlog::info!("Parent operation", span.start = parent);
///
/// let child = SpanID::next();
/// kvlog::info!("Child operation", span.start = child, span.parent = Some(parent));
/// kvlog::info!("Child complete", span.end = child);
///
/// kvlog::info!("Parent complete", span.end = parent);
/// ```
///
/// Using [`SpanID::enter`] for automatic span context:
///
/// ```
/// use kvlog::SpanID;
///
/// fn do_work() {
///     kvlog::info!("Working"); // Inherits current span context
/// }
///
/// let span = SpanID::next();
/// let _guard = span.enter();
/// kvlog::info!("Starting", span.start = span);
/// do_work();  // This log will be associated with the span
/// kvlog::info!("Done", span.end = span);
/// ```
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

/// Guard that restores the previous span context when dropped.
///
/// Created by calling [`SpanID::enter`]. While this guard exists, the associated
/// span is set as the current thread-local span context.
pub struct EnteredSpan {
    previous: u64,
}

/// Returns a lock guard to the global log queue.
///
/// This function is primarily intended for advanced use cases where direct
/// access to the logging infrastructure is needed.
#[doc(hidden)]
pub fn global_logger() -> StdMutexGuard<'static, LogQueue> {
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

/// Wraps a value with the current span context, or creates a new span if none exists.
///
/// Returns a [`Spanning<T>`] wrapper that carries span context. When the wrapped
/// value is a [`Future`][std::future::Future], polling it will automatically set
/// the span context for any logs emitted during execution.
///
/// # Examples
///
/// ```
/// async fn my_async_fn() {
///     kvlog::info!("Within span context");
/// }
///
/// let spanned_future = kvlog::continue_span(my_async_fn());
/// // When awaited, logs will include the span context
/// ```
pub fn continue_span<T: Sized>(t: T) -> Spanning<T> {
    Spanning {
        inner: ManuallyDrop::new(t),
        span: SpanID::current_or_next().inner,
    }
}

impl SpanID {
    /// Enters this span, setting it as the current thread-local span context.
    ///
    /// Returns an [`EnteredSpan`] guard that restores the previous span context
    /// when dropped. While the guard exists, logs emitted on this thread will
    /// automatically include this span if no explicit span is provided.
    ///
    /// # Examples
    ///
    /// ```
    /// use kvlog::SpanID;
    ///
    /// let span = SpanID::next();
    /// {
    ///     let _guard = span.enter();
    ///     kvlog::info!("This log automatically includes the span");
    /// }
    /// // Previous span context is restored here
    /// ```
    pub fn enter(&self) -> EnteredSpan {
        EnteredSpan {
            previous: CURRENT_SPAN_ID.replace(self.inner.get()),
        }
    }

    /// Returns the raw `u64` representation of this span ID.
    pub fn as_u64(&self) -> u64 {
        self.inner.get()
    }

    /// Generates a new unique span ID.
    ///
    /// Span IDs are generated using a combination of entropy sources to ensure
    /// uniqueness across processes and restarts.
    pub fn next() -> SpanID {
        next_span_id()
    }

    /// Wraps a value with this span's context.
    ///
    /// Returns a [`Spanning<T>`] wrapper that carries this span's context. When
    /// the wrapped value is a [`Future`][std::future::Future], polling it will
    /// automatically set this span as the current context.
    ///
    /// # Examples
    ///
    /// ```
    /// use kvlog::SpanID;
    ///
    /// async fn process_request() {
    ///     kvlog::info!("Processing"); // Automatically includes span context
    /// }
    ///
    /// let span = SpanID::next();
    /// let future = span.spanning(process_request());
    /// ```
    #[inline]
    pub fn spanning<T: Sized>(&self, t: T) -> Spanning<T> {
        Spanning {
            inner: ManuallyDrop::new(t),
            span: self.inner,
        }
    }

    /// Returns the current thread-local span, or generates a new one if none exists.
    pub fn current_or_next() -> SpanID {
        if let Some(value) = NonZeroU64::new(CURRENT_SPAN_ID.with(|foo| foo.get())) {
            SpanID { inner: value }
        } else {
            return SpanID::next();
        }
    }

    /// Returns the current thread-local span, if one is set.
    ///
    /// Returns [`None`] if no span context is currently active on this thread.
    pub fn current() -> Option<SpanID> {
        if let Some(value) = NonZeroU64::new(CURRENT_SPAN_ID.with(|foo| foo.get())) {
            Some(SpanID { inner: value })
        } else {
            None
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
/// Sets the counter used for generating span IDs.
///
/// This is primarily useful for testing or when deterministic span IDs are needed.
#[doc(hidden)]
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

/// Sets the current thread-local span context.
///
/// Prefer using [`SpanID::enter`] which provides automatic cleanup via RAII.
#[doc(hidden)]
pub fn set_global_span_id(x: SpanID) {
    CURRENT_SPAN_ID.set(x.inner.get())
}

/// Clears the current thread-local span context.
///
/// Prefer using [`SpanID::enter`] which provides automatic cleanup via RAII.
#[doc(hidden)]
pub fn clear_global_span_id() {
    CURRENT_SPAN_ID.set(0);
}

/// The severity level of a log message.
///
/// Levels are ordered from least to most severe: Debug < Info < Warn < Error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Development-only logging, requires the `debug` feature flag.
    ///
    /// Debug logs are completely compiled out in release builds unless the
    /// `debug` feature is enabled, ensuring zero runtime overhead.
    Debug,
    /// Informational messages that provide useful context about application state.
    Info,
    /// Warnings about expected but potentially problematic conditions.
    Warn,
    /// Errors indicating unexpected failures that typically require intervention.
    Error,
}

impl LogLevel {
    /// Converts a byte value to a log level.
    ///
    /// Returns [`None`] if the byte does not correspond to a valid log level.
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

impl CollectorConfig {
    pub fn with_service_name(self, name: &str) -> Self {
        match self {
            CollectorConfig::SocketOrStdout { socket, .. } => CollectorConfig::SocketOrStdout {
                service_name: name.to_string(),
                socket,
            },
            _ => self,
        }
    }
}

/// Spawn background log collector thread with configuration from the env var:
///     `KVLOG_COLLECTOR_CONFIG`
/// Using [CollectorConfig::default()] if the env var can't be read. If quiet
/// is not set to true, the selected logging configuration will be printed to
/// stdout.
///
/// See [CollectorConfig] for details on the configuration.
///
/// Note: When the `LoggerGuard` returned from this function is dropped
/// the spawned collector will be flushed and terminated.
///
/// # Example:
/// ```
/// let _guard = kvlog::spawn_collector_from_env(Some("ServiceName"), false);
/// kvlog::info!("Hello World");
/// ```
#[must_use]
pub fn spawn_collector_from_env(service_name: Option<&str>, quiet: bool) -> LoggerGuard {
    let (mut collector, printstmt) = if let Ok(value) = std::env::var("KVLOG_COLLECTOR_CONFIG") {
        match value.parse::<CollectorConfig>() {
            Ok(value) => (
                value,
                "configuration from the KVLOG_COLLECTOR_CONFIG environment variable",
            ),
            Err(err) => {
                if !quiet {
                    println!("KVLOG: Error parsing KVLOG_COLLECTOR_CONFIG environment variable\n value: `{}`\n error: {}", value, err);
                }
                (
                    CollectorConfig::default(),
                    "default configuration after an error parsing",
                )
            }
        }
    } else {
        (
            CollectorConfig::default(),
            "default configuration (KVLOG_COLLECTOR_CONFIG not set)",
        )
    };
    if let Some(service_name) = service_name {
        collector = collector.with_service_name(service_name);
    }

    if !quiet {
        println!("KVLOG: Using the {printstmt}: {:#?}", collector)
    }
    collector.spawn()
}

impl std::default::Default for CollectorConfig {
    fn default() -> Self {
        CollectorConfig::SocketOrStdout {
            service_name: String::new(),
            socket: collector::DEFAULT_COLLECTOR_SOCKET_PATH.into(),
        }
    }
}

/// Configuration for how log messages should be collected.
///
/// This enum specifies the destination for log output. It implements the
/// `FromStr` trait, allowing it to be parsed from a string representation.
///
/// # String Format
///
/// The expected string format is: `[SERVICE_NAME@]KIND[:PATH]`
///
/// - `KIND`: (Required) Specifies the collection method. Must be one of:
///   - `Stdout`: Log to standard output.
///   - `Directory`: Log to files within a specified directory.
///   - `SingleFile`: Log to a single specified file.
///   - `SocketOrStdout`: Log to a Unix domain socket, falling back to standard output.
/// - `PATH`: (Optional, separated by `:`) The file system path for relevant kinds.
///   - *Required* and must not be empty for `Directory` and `SingleFile`.
///   - *Optional* for `SocketOrStdout`. If omitted or empty, it defaults to
///     `collector::DEFAULT_COLLECTOR_SOCKET_PATH`.
///   - Ignored for `Stdout`.
/// - `SERVICE_NAME`: (Optional, separated by `@`) A name for the service, used only
///   by `SocketOrStdout`. If omitted, it defaults to an empty string.
///
/// # Examples of String Parsing
///
/// - `"Stdout"` parses to `CollectorConfig::Stdout`
/// - `"Directory:/var/log/my_app"` parses to `CollectorConfig::Directory { path: "/var/log/my_app".into() }`
/// - `"SingleFile:/tmp/app.log"` parses to `CollectorConfig::SingleFile { path: "/tmp/app.log".into() }`
/// - `"service1@SocketOrStdout:/tmp/service1.sock"` parses to `CollectorConfig::SocketOrStdout { service_name: "service1".into(), socket: "/tmp/service1.sock".into() }`
/// - `"SocketOrStdout"` parses to `CollectorConfig::SocketOrStdout { service_name: "".into(), socket: collector::DEFAULT_COLLECTOR_SOCKET_PATH.into() }` // Using default path and empty service name
/// - `"service2@SocketOrStdout"` parses to `CollectorConfig::SocketOrStdout { service_name: "service2".into(), socket: collector::DEFAULT_COLLECTOR_SOCKET_PATH.into() }` // Using default path
///
#[derive(PartialEq, Eq, Clone, Debug)]
pub enum CollectorConfig {
    /// Log messages are written directly to standard output (stdout).
    Stdout,
    /// Log messages are written to timestamped files within the specified directory.
    /// The `path` must be provided when parsing from a string.
    Directory { path: PathBuf },
    /// All log messages are written to the single specified file.
    /// The `path` must be provided when parsing from a string.
    SingleFile { path: PathBuf },
    /// Log messages are sent over a Unix domain socket specified by `socket`.
    /// If the socket connection fails or is unavailable, logs fall back to standard output.
    /// The `service_name` is used for identification purposes with the collector service.
    /// When parsing from a string, `service_name` is optional (defaults to empty)
    /// and `socket` path is optional (defaults to `collector::DEFAULT_COLLECTOR_SOCKET_PATH`).
    SocketOrStdout {
        service_name: String,
        socket: PathBuf,
    },
}

impl FromStr for CollectorConfig {
    type Err = &'static str;

    fn from_str(mut input: &str) -> Result<Self, Self::Err> {
        let service_name = if let Some((service_name, rest)) = input.split_once('@') {
            input = rest;
            Some(service_name)
        } else {
            None
        };
        let (kind, path) = input.split_once(':').unwrap_or((input, ""));
        match kind {
            "Stdout" => Ok(CollectorConfig::Stdout),
            "Directory" => {
                if path.is_empty() {
                    Err("Directory path is required and must not be empty")
                } else {
                    Ok(CollectorConfig::Directory {
                        path: PathBuf::from(path),
                    })
                }
            }
            "SingleFile" => {
                if path.is_empty() {
                    Err("SingleFile path is required and must not be empty")
                } else {
                    Ok(CollectorConfig::SingleFile {
                        path: PathBuf::from(path),
                    })
                }
            }
            "SocketOrStdout" => Ok(CollectorConfig::SocketOrStdout {
                service_name: service_name.unwrap_or_default().into(),
                socket: if path.is_empty() {
                    collector::DEFAULT_COLLECTOR_SOCKET_PATH.into()
                } else {
                    path.into()
                },
            }),
            _ => Err("Invalid collector type"),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn parse_collector_config() {
        assert_eq!(
            CollectorConfig::SingleFile {
                path: "/tmp/output.log".into()
            },
            "SingleFile:/tmp/output.log".parse().unwrap(),
        );
        assert_eq!(
            CollectorConfig::Directory {
                path: "/tmp/logs/".into()
            },
            "Directory:/tmp/logs/".parse().unwrap(),
        );
        assert!("SingleFile:".parse::<CollectorConfig>().is_err());
        assert!("SingleFile".parse::<CollectorConfig>().is_err());
        assert!("Directory:".parse::<CollectorConfig>().is_err());
        assert!("Directory".parse::<CollectorConfig>().is_err());
        assert_eq!(CollectorConfig::Stdout, "Stdout".parse().unwrap(),);

        assert!("SocketOrStdout:".parse::<CollectorConfig>().is_ok());
        assert!("SocketOrStdout".parse::<CollectorConfig>().is_ok());

        assert_eq!(
            CollectorConfig::SocketOrStdout {
                service_name: String::new(),
                socket: collector::DEFAULT_COLLECTOR_SOCKET_PATH.into()
            },
            "SocketOrStdout".parse().unwrap(),
        );

        assert_eq!(
            CollectorConfig::SocketOrStdout {
                service_name: "aname".to_string(),
                socket: collector::DEFAULT_COLLECTOR_SOCKET_PATH.into()
            },
            "aname@SocketOrStdout".parse().unwrap(),
        );
        assert_eq!(
            CollectorConfig::SocketOrStdout {
                service_name: "another_name".to_string(),
                socket: "/tmp/my.kvlog.sock".into()
            },
            "another_name@SocketOrStdout:/tmp/my.kvlog.sock"
                .parse()
                .unwrap(),
        );
    }
}

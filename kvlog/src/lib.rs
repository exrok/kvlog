use std::{
    cell::Cell,
    mem::ManuallyDrop,
    num::NonZeroU64,
    path::PathBuf,
    str::FromStr,
    sync::{atomic::AtomicU64, MutexGuard},
};

use collector::{LogQueue, LoggerGuard};
pub use encoding::BStr;
pub use encoding::{Encode, SpanInfo, ValueEncoder};
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

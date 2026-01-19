//! Log collection and output infrastructure.
//!
//! This module provides functions for initializing log collectors that process
//! and output log messages. A collector runs on a background thread and handles
//! batching, formatting, and writing logs to various destinations.
//!
//! # Quick Start
//!
//! The easiest way to set up logging is with [`init_stdout_logger`]:
//!
//! ```no_run
//! let _guard = kvlog::collector::init_stdout_logger();
//! kvlog::info!("Application started");
//! // Logs are written to stdout
//! ```
//!
//! # Collector Types
//!
//! - [`init_stdout_logger`]: Writes formatted logs to stdout
//! - [`init_file_logger`]: Appends binary logs to a single file
//! - [`init_directory_logger`]: Writes to rotating files in a directory
//! - [`init_closure_logger`]: Custom processing via a closure
//!
//! # Guard Lifetime
//!
//! All initialization functions return a [`LoggerGuard`] that must be held
//! for the duration of logging. When the guard is dropped, the collector
//! thread is flushed and terminated.

use std::alloc::Layout;
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use std::{fs::File, io::IsTerminal};

use std::sync::{Condvar, Mutex};
use std::thread;

use crate::encoding::{now, Encoder, LogFields, MunchError, SpanInfo, StaticKey};
use crate::{CollectorConfig, LogLevel, SpanID};

/// Wrapper around raw encoded log bytes providing helpful utilities.
///
/// This type provides access to the raw log buffer and methods for iterating
/// over decoded logs or swapping the buffer contents.
pub struct LogBuffer<'a> {
    buffer: &'a mut Vec<u8>,
}

impl<'a> LogBuffer<'a> {
    /// Create a new LogBuffer wrapping a mutable reference to a Vec.
    pub fn new(buffer: &'a mut Vec<u8>) -> Self {
        LogBuffer { buffer }
    }

    /// Get a reference to the raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    /// Get a mutable reference to the raw bytes.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.buffer
    }

    /// Get the underlying Vec reference.
    pub fn as_vec(&self) -> &Vec<u8> {
        self.buffer
    }

    /// Get a mutable reference to the underlying Vec.
    pub fn as_vec_mut(&mut self) -> &mut Vec<u8> {
        self.buffer
    }

    /// Get the length of the buffer in bytes.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Iterate over decoded logs in the buffer.
    ///
    /// Each item yields the timestamp, log level, span info, and fields iterator.
    pub fn iter(
        &self,
    ) -> impl Iterator<Item = Result<(u64, LogLevel, SpanInfo, LogFields<'_>), MunchError>> + '_
    {
        crate::encoding::decode(&self.buffer)
    }

    /// Swap the internal buffer with another Vec.
    ///
    /// This is useful for reusing allocations - swap in an empty or pre-allocated
    /// Vec and process the returned data.
    pub fn swap(&mut self, other: &mut Vec<u8>) {
        std::mem::swap(self.buffer, other);
    }

    /// Take the buffer contents, leaving an empty Vec in place.
    ///
    /// Returns the original buffer contents.
    pub fn take(&mut self) -> Vec<u8> {
        std::mem::take(self.buffer)
    }

    /// Replace the buffer contents with the provided Vec.
    ///
    /// Returns the old buffer contents.
    pub fn replace(&mut self, new: Vec<u8>) -> Vec<u8> {
        std::mem::replace(self.buffer, new)
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

#[derive(PartialEq, Eq, Copy, Clone)]
pub enum SyncState {
    Flushing,
    Writing,
    Waiting,
}

#[allow(unused)]
pub struct LogQueue {
    pub encoder: Encoder,
    pub sync_state: SyncState,
    pub logfmt: Vec<u8>,
    pub exiting: bool,
    pub handle: Option<thread::JoinHandle<()>>,
    pub parent_table: Option<Box<ParentSpanSuffixCache>>,
}

impl LogQueue {
    pub fn poke(&mut self) {
        if self.sync_state != SyncState::Writing {
            if let Some(handle) = &self.handle {
                handle.thread().unpark();
            } else {
                let parent_table = self
                    .parent_table
                    .get_or_insert_with(|| ParentSpanSuffixCache::new_boxed());
                print!(
                    "{}",
                    pretty_print_buffer(&mut self.logfmt, &mut *parent_table, &self.encoder.buffer)
                );
                self.encoder.buffer.clear();
            }
        }
    }
}

pub static FLUSHING: Condvar = Condvar::new();

pub static LOG_WRITER: Mutex<LogQueue> = Mutex::new(LogQueue {
    encoder: Encoder::new(),
    logfmt: Vec::new(),
    sync_state: SyncState::Waiting,
    exiting: false,
    handle: None,
    parent_table: None,
});

const BUFFER_CAPACITY: usize = 4096;

pub fn mini_time(out: &mut Vec<u8>, timestamp_nanos: u64) {
    crate::Timestamp::raw_millisecond_iso_in_vec((timestamp_nanos / 1000_000) as i64, out);
}

const SPAN_COLORS: &[&str] = &[
    "\x1b[48;5;183m\x1b[38;5;16m",
    "\x1b[48;5;16m\x1b[38;5;183m",
    "\x1b[48;5;194m\x1b[38;5;16m",
    "\x1b[48;5;254m\x1b[38;5;16m",
    "\x1b[48;5;93m\x1b[38;5;254m",
    "\x1b[48;5;150m\x1b[38;5;16m",
    "\x1b[48;5;225m\x1b[38;5;16m",
    "\x1b[48;5;223m\x1b[38;5;16m",
    "\x1b[48;5;16m\x1b[38;5;223m",
];

/// Guaranteed to Append valid UTF-8

#[cold]
pub fn pretty_print_buffer<'a>(
    temp: &'a mut Vec<u8>,
    parents: &mut ParentSpanSuffixCache,
    buffer: &[u8],
) -> &'a str {
    temp.clear();
    for log in crate::encoding::decode(buffer) {
        let (timestamp, log_level, span_info, fields) = log.expect("Log Decoding failed");
        format_statement_with_colors(temp, parents, timestamp, log_level, span_info, fields);
    }
    // Safety: We cleared `temp` and then only valid strings are appended
    unsafe { std::str::from_utf8_unchecked(temp) }
}

#[repr(transparent)]
pub struct ParentSpanSuffixCache {
    /// Each entry stores the highest 52 bits of the child span, and is index
    /// byte the other 12 bits of child span. The lowest 12 bits of parent span
    /// are stored in the remaining space of the item.
    table: [u64; 4096],
}

impl ParentSpanSuffixCache {
    pub fn new_boxed() -> Box<ParentSpanSuffixCache> {
        // Since `SpanID`s always have the high bit set even after zeroing the first
        // 12 bits, it's still non-zero, hence the zeroed table will never match any
        // spans and can be used to represent the empty state.
        unsafe {
            let ptr = std::alloc::alloc_zeroed(Layout::new::<ParentSpanSuffixCache>());
            if ptr.is_null() {
                std::alloc::handle_alloc_error(Layout::new::<ParentSpanSuffixCache>());
            }
            Box::from_raw(ptr as *mut ParentSpanSuffixCache)
        }
    }
    /// Get the lowest 12 bits of the parent span id for the given span id.
    ///
    /// NOTE: If enough child-parent insertions have occurred a previous association
    /// may have been overwritten causing this to now return None. However, as the
    /// is designed for printing the screen, where we only display the last 3 digits
    /// anyway the Span ID would already be ambiguous. Moreover, child spans are
    /// typically short lived so such a case would be exceedingly rare.
    pub fn get(&self, child: SpanID) -> Option<u16> {
        let raw = child.as_u64();
        let value = self.table[raw as usize & 0xfff];
        if value ^ raw > 0xfff {
            None
        } else {
            Some((value & 0xfff) as u16)
        }
    }
    /// See [Self::get] for limitatinos.
    pub fn insert(&mut self, child: SpanID, parent: SpanID) {
        let rchild = child.as_u64();
        let index = rchild & 0xfff;
        self.table[index as usize] = (rchild ^ index) | (parent.as_u64() & 0xfff);
    }
}

pub fn format_statement_with_colors(
    buffer: &mut Vec<u8>,
    parent_table: &mut ParentSpanSuffixCache,
    timestamp: u64,
    log_level: crate::LogLevel,
    span_info: crate::SpanInfo,
    fields: crate::encoding::LogFields<'_>,
) {
    let level_text = match log_level {
        crate::LogLevel::Debug => b"\x1b[38;5;246mDEBUG",
        crate::LogLevel::Info => b"\x1b[38;5;246m INFO",
        crate::LogLevel::Warn => b"\x1b[38;5;227m WARN",
        crate::LogLevel::Error => b"\x1b[38;5;197mERROR",
    };
    buffer.extend_from_slice(level_text);
    buffer.push(b' ');
    mini_time(buffer, timestamp);
    buffer.extend_from_slice(b"\x1b[0m");
    let span = match span_info {
        crate::SpanInfo::Start { span, parent } => {
            if let Some(parent) = parent {
                parent_table.insert(span, parent);
            }
            Some(('>', span))
        }
        crate::SpanInfo::Current { span } => Some((' ', span)),
        crate::SpanInfo::End { span } => Some(('<', span)),
        crate::SpanInfo::None => None,
    };
    if let Some((chr, id)) = span {
        if let Some(parent) = parent_table.get(id) {
            let color = SPAN_COLORS[(parent & 0x0fff) as usize % SPAN_COLORS.len()];
            let _ = write!(buffer, " {color} {:03X} \x1b[0m", parent & 0x0FFF);
        }
        let color = SPAN_COLORS[((id.inner.get() as usize) & 0x0fff) % SPAN_COLORS.len()];
        let _ = write!(
            buffer,
            " {color}{chr}{:03X}{chr}\x1b[0m",
            id.inner.get() as u16 & 0x0FFF
        );
    }
    for field in fields {
        let (key, value) = field.unwrap();
        if key == StaticKey::target {
            write!(buffer, " \x1b[38;5;116m@{value}\x1b[0m").ok();
        } else if key == StaticKey::msg {
            write!(buffer, " {value}").ok();
        } else {
            if key == StaticKey::err {
                buffer.extend_from_slice(b" \x1b[38;5;197m");
            } else {
                buffer.extend_from_slice(b" \x1b[38;5;110m");
            }

            buffer.extend_from_slice(key.as_bytes());
            buffer.extend_from_slice(b"\x1b[38;5;249m=\x1b[0m");
            match value {
                crate::encoding::Value::String(value) => {
                    buffer.extend_from_slice(b"\x1b[38;5;193m");
                    buffer.extend_from_slice(value);
                    buffer.extend_from_slice(b"\x1b[0m");
                }
                crate::encoding::Value::Bytes(bytes) => {
                    write!(buffer, "\x1b[38;5;47m{}\x1b[0m", bytes.escape_ascii()).ok();
                }
                crate::encoding::Value::Bool(boolean) => {
                    write!(buffer, "\x1b[38;5;182m{boolean}\x1b[0m").ok();
                }
                crate::encoding::Value::F64(num) => {
                    write!(buffer, "{num:.3}").ok();
                }
                other => {
                    write!(buffer, "{other}").ok();
                }
            }
        }
    }
    buffer.push(b'\n');
}

fn stdout_sync_thread() {
    let mut max_size: usize = 0;
    let mut buffer = Vec::<u8>::with_capacity(BUFFER_CAPACITY);
    let mut wait_flushers = false;
    loop {
        let mut queue = LOG_WRITER.lock().unwrap();
        if queue.sync_state == SyncState::Flushing {
            wait_flushers = true;
        }
        if queue.encoder.buffer.is_empty() {
            queue.sync_state = SyncState::Waiting;
            let exiting = queue.exiting;
            if wait_flushers {
                wait_flushers = false;
                FLUSHING.notify_all();
            }
            drop(queue);
            if exiting {
                return;
            }
            std::thread::park();
            continue;
        }
        buffer.clear();
        std::mem::swap(&mut buffer, &mut queue.encoder.buffer);
        if buffer.len() > max_size {
            max_size = buffer.len();
        }
        queue.sync_state = SyncState::Writing;
        drop(queue);
        {
            let mut stdout = std::io::stdout().lock();
            for log in crate::encoding::decode(&buffer) {
                let (timestamp, log_level, _span_info, fields) = log.expect("Log Decoding failed");
                let level_text = match log_level {
                    crate::LogLevel::Debug => "DEBUG",
                    crate::LogLevel::Info => "INFO",
                    crate::LogLevel::Warn => "WARN",
                    crate::LogLevel::Error => "ERROR",
                };
                write!(&mut stdout, "{} {}", level_text, timestamp).ok();
                for field in fields {
                    let (key, value) = field.unwrap();
                    write!(&mut stdout, " {}={value}", key.as_bytes().escape_ascii()).ok();
                }
                stdout.write(b"\n").ok();
            }
        }
    }
}

const INIT_MAGIC: u64 = 0xF745_119Eu64 << 32;
const MAGIC: u64 = 0x8910_FC0Eu64 << 32;
pub const DEFAULT_COLLECTOR_SOCKET_PATH: &str = "/tmp/.kvlog_collector.sock";

pub struct SmartCollector {
    current_socket_modified_at: Option<SystemTime>,
    last_polled: std::time::Instant,
    fmtbuf: Vec<u8>,
    socket_path: PathBuf,
    stream: Option<UnixStream>,
    parents: Box<ParentSpanSuffixCache>,
    service_name: String,
}
fn attach_prelude(buffer: &mut [u8]) -> &[u8] {
    if buffer.len() < 9 {
        return &[];
    }
    let len = buffer.len() - 8;
    buffer[..8].copy_from_slice(&u64::to_le_bytes((len as u64) | MAGIC));
    buffer
}

impl SmartCollector {
    fn new(socket_path: PathBuf, service_name: String) -> SmartCollector {
        let mut collector = SmartCollector {
            current_socket_modified_at: None,
            last_polled: std::time::Instant::now(),
            fmtbuf: Vec::new(),
            socket_path,
            stream: None,
            parents: ParentSpanSuffixCache::new_boxed(),
            service_name,
        };
        collector.try_connect_stream(Duration::ZERO);
        collector
    }
    fn try_connect_stream(&mut self, retry_duration: Duration) -> Option<&mut UnixStream> {
        // work around borrow checker limitation (so no if let here)
        if self.stream.is_none() {
            let now = std::time::Instant::now();
            if now.duration_since(self.last_polled) < retry_duration {
                return None;
            }
            self.last_polled = now;
            let metadata = match std::fs::metadata(&self.socket_path) {
                Ok(metadata) => metadata,
                Err(_err) => {
                    return None;
                }
            };
            // best effort optimization to avoid trying re-connect which is more
            // expansive then a stat call.
            if let Some(modified_at) = metadata.modified().ok() {
                if Some(modified_at) == self.current_socket_modified_at {
                    return None;
                }
                self.current_socket_modified_at = Some(modified_at);
            }
            self.stream = match UnixStream::connect(&self.socket_path) {
                Ok(mut stream) => {
                    if !self.service_name.is_empty() {
                        let buflen = self.fmtbuf.len();
                        //todo guard against huge service name
                        let prefix = INIT_MAGIC | ((self.service_name.len() as u64) & 0xffff_ffff);
                        self.fmtbuf.extend_from_slice(&u64::to_le_bytes(prefix));
                        self.fmtbuf.extend_from_slice(self.service_name.as_bytes());
                        let res = stream.write_all(&self.fmtbuf[buflen..]);
                        self.fmtbuf.truncate(buflen);
                        if res.is_err() {
                            return None;
                        }
                    }

                    Some(stream)
                }
                Err(_err) => None,
            }
        }
        self.stream.as_mut()
    }
    fn output(&mut self, buffer: &mut [u8]) {
        if let Some(stream) = self.try_connect_stream(Duration::from_millis(500)) {
            if let Err(err) = stream.write_all(attach_prelude(buffer)) {
                println!("stream write failure {:?}", err);
                use crate as kvlog;
                {
                    use kvlog::encoding::Encode;
                    let mut log = kvlog::global_logger();
                    let mut fields = log.encoder.append_now(kvlog::LogLevel::Warn);
                    fields.raw_key(1).value_via_display(&err);
                    fields
                        .dynamic_key("socket")
                        .value_via_debug(&self.socket_path);
                    (module_path!()).encode_log_value_into(fields.raw_key(15));
                    ("KVLog collector connection closed").encode_log_value_into(fields.raw_key(0));
                    fields.apply_current_span();
                    log.poke();
                };
                self.stream = None;
                self.current_socket_modified_at = None;
            } else {
                return;
            }
        }
        let output = pretty_print_buffer(&mut self.fmtbuf, &mut self.parents, &buffer[8..]);
        let _ = std::io::stdout().write_all(output.as_bytes());
    }
}

pub(crate) fn hybrid_local_thread(socket: PathBuf, service_name: String) {
    let mut max_size: usize = 0;
    let mut buffer = Vec::<u8>::with_capacity(BUFFER_CAPACITY);
    buffer.extend_from_slice(&MAGIC.to_le_bytes());
    let mut wait_flushers = false;
    {
        let mut queue = LOG_WRITER.lock().unwrap();
        if queue.encoder.buffer.is_empty() {
            queue.encoder.buffer.extend_from_slice(&MAGIC.to_le_bytes());
        } else {
            //todo guard against multiple initlizations
            queue.encoder.buffer.splice(0..0, MAGIC.to_le_bytes());
        }
    }
    let mut stream = SmartCollector::new(socket, service_name);
    loop {
        let mut queue = LOG_WRITER.lock().unwrap();
        if queue.sync_state == SyncState::Flushing {
            wait_flushers = true;
        }
        if queue.encoder.buffer.len() == 8 {
            queue.sync_state = SyncState::Waiting;
            let exiting = queue.exiting;
            if wait_flushers {
                wait_flushers = false;
                FLUSHING.notify_all();
            }
            drop(queue);
            if exiting {
                return;
            }
            std::thread::park();
            continue;
        }
        buffer.truncate(8);
        std::mem::swap(&mut buffer, &mut queue.encoder.buffer);
        if buffer.len() > max_size {
            max_size = buffer.len();
        }
        queue.sync_state = SyncState::Writing;
        drop(queue);
        stream.output(&mut buffer);
    }
}
fn stdout_sync_thread_pretty() {
    let mut parents = ParentSpanSuffixCache::new_boxed();
    let mut max_size: usize = 0;
    let mut buffer = Vec::<u8>::with_capacity(BUFFER_CAPACITY);
    let mut wait_flushers = false;
    let mut logfmt = Vec::<u8>::with_capacity(4096 * 2);
    loop {
        let mut queue = LOG_WRITER.lock().unwrap();
        if queue.sync_state == SyncState::Flushing {
            wait_flushers = true;
        }
        if queue.encoder.buffer.is_empty() {
            queue.sync_state = SyncState::Waiting;
            let exiting = queue.exiting;
            if wait_flushers {
                wait_flushers = false;
                FLUSHING.notify_all();
            }
            drop(queue);
            if exiting {
                return;
            }
            std::thread::park();
            continue;
        }
        buffer.clear();
        std::mem::swap(&mut buffer, &mut queue.encoder.buffer);
        if buffer.len() > max_size {
            max_size = buffer.len();
        }
        queue.sync_state = SyncState::Writing;
        drop(queue);
        let formatted = pretty_print_buffer(&mut logfmt, &mut parents, &buffer);
        let _ = std::io::stdout().write_all(&formatted.as_bytes());
    }
}
fn exit_logger() {
    let handle = {
        let mut queue = LOG_WRITER.lock().unwrap();
        queue.exiting = true;
        queue.handle.take()
    };
    if let Some(handle) = handle {
        handle.thread().unpark();
        if let Ok(_) = handle.join() {}
    }
}
/// Guard that keeps the log collector running.
///
/// When this guard is dropped, the log collector thread is flushed and terminated.
/// The guard must be held for the entire duration that logging is needed.
///
/// # Examples
///
/// ```no_run
/// fn main() {
///     let _guard = kvlog::collector::init_stdout_logger();
///
///     kvlog::info!("Application starting");
///     // ... application logic ...
///
///     // Guard dropped here, collector is flushed and stopped
/// }
/// ```
pub struct LoggerGuard {}

impl LoggerGuard {
    /// Flushes all pending log messages to their destination.
    ///
    /// Blocks until all buffered logs have been written, with a timeout of
    /// 2 seconds.
    pub fn flush(&self) {
        let mut queue = LOG_WRITER.lock().unwrap();
        if let Some(handle) = &queue.handle {
            handle.thread().unpark();
        } else {
            return;
        }
        queue.sync_state = SyncState::Flushing;
        FLUSHING
            .wait_timeout_while(queue, Duration::from_secs(2), |queue| {
                queue.sync_state != SyncState::Waiting
            })
            .ok();
    }
}

impl Drop for LoggerGuard {
    fn drop(&mut self) {
        exit_logger();
    }
}
const LOG_FILE_TARGET_SIZE: usize = 1024 * 1024 * 16; // 16 MB

fn directory_sync_thread(pathbuf: PathBuf) {
    let path = pathbuf.to_string_lossy().into_owned();
    let mut pathbuf = pathbuf;
    if let Err(err) = std::fs::create_dir_all(&pathbuf) {
        panic!("kvlog: {} {:?}", path, err);
    }
    let active_path = pathbuf.join("active");
    pathbuf.push(&"archive");
    if let Err(err) = std::fs::create_dir_all(&pathbuf) {
        panic!("kvlog: {} {:?}", path, err);
    }
    let (mut written, mut file) = if let Ok(meta) = std::fs::metadata(&active_path) {
        (
            meta.len() as usize,
            std::fs::OpenOptions::new()
                .append(true)
                .open(&active_path)
                .expect("kvlog SYNC THREAD: failed to open log"),
        )
    } else {
        (
            0,
            File::create(&active_path).expect("kvlog SYNC THREAD: failed to open log"),
        )
    };
    let mut max_size: usize = 0;
    let mut buffer = Vec::<u8>::with_capacity(BUFFER_CAPACITY);
    loop {
        if written >= LOG_FILE_TARGET_SIZE {
            pathbuf.push(&format!("{:?}.bin", now()));
            let _ = std::fs::rename(&active_path, &pathbuf);
            pathbuf.pop();
            file = File::create(&active_path).expect("kvlog SYNC THREAD: failed to open log");
            written = 0;
        }

        let mut queue = LOG_WRITER.lock().unwrap();
        if queue.encoder.buffer.is_empty() {
            queue.sync_state = SyncState::Waiting;
            let exiting = queue.exiting;
            drop(queue);
            if exiting {
                return;
            }
            std::thread::park();
            continue;
        }
        buffer.clear();
        std::mem::swap(&mut buffer, &mut queue.encoder.buffer);
        if buffer.len() > max_size {
            max_size = buffer.len();
        }
        queue.sync_state = SyncState::Writing;
        drop(queue);
        file.write_all(&mut buffer).unwrap();
        written += buffer.len();
    }
}

fn file_sync_thread(mut file: std::fs::File) {
    let mut max_size: usize = 0;
    let mut buffer = Vec::<u8>::with_capacity(BUFFER_CAPACITY);
    let mut wait_flushers = false;
    loop {
        let mut queue = LOG_WRITER.lock().unwrap();
        if queue.sync_state == SyncState::Flushing {
            wait_flushers = true;
        }
        if queue.encoder.buffer.is_empty() {
            queue.sync_state = SyncState::Waiting;
            let exiting = queue.exiting;
            if wait_flushers {
                wait_flushers = false;
                FLUSHING.notify_all();
            }
            drop(queue);
            if exiting {
                return;
            }
            std::thread::park();
            continue;
        }
        buffer.clear();
        std::mem::swap(&mut buffer, &mut queue.encoder.buffer);
        if buffer.len() > max_size {
            max_size = buffer.len();
        }
        queue.sync_state = SyncState::Writing;
        drop(queue);
        file.write_all(&buffer).ok();
    }
}

/// Initializes the log collector to write to stdout.
///
/// When stdout is a terminal, logs are formatted with colors and human-readable
/// timestamps. Otherwise, a plain text format is used.
///
/// # Examples
///
/// ```no_run
/// let _guard = kvlog::collector::init_stdout_logger();
/// kvlog::info!("Hello, world!");
/// ```
#[must_use]
pub fn init_stdout_logger() -> LoggerGuard {
    if std::io::stdout().is_terminal() {
        init_logger(stdout_sync_thread_pretty)
    } else {
        init_logger(stdout_sync_thread)
    }
}

/// Initializes the log collector to append binary logs to a file.
///
/// Logs are written in the kvlog binary format. The file is created if it
/// does not exist.
///
/// # Panics
///
/// Panics if the file cannot be opened or created.
///
/// # Examples
///
/// ```no_run
/// let _guard = kvlog::collector::init_file_logger("/var/log/myapp.kvlog");
/// ```
#[must_use]
pub fn init_file_logger(file: &str) -> LoggerGuard {
    let path = file.to_string();
    init_logger(move || {
        file_sync_thread(
            std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(&path)
                .expect("FAILED to open log file"),
        )
    })
}

/// Initializes the log collector to write to a directory with log rotation.
///
/// Logs are written to an `active` file in the directory. When the file
/// reaches 16 MB, it is rotated to the `archive` subdirectory with a
/// timestamp-based filename.
///
/// # Panics
///
/// Panics if the directory cannot be created.
///
/// # Examples
///
/// ```no_run
/// let _guard = kvlog::collector::init_directory_logger("/var/log/myapp/");
/// ```
#[must_use]
pub fn init_directory_logger(dir: &str) -> LoggerGuard {
    let dir: PathBuf = Path::new(dir).into();
    init_logger(move || directory_sync_thread(dir))
}

/// Initializes the log collector with socket fallback to stdout.
///
/// Attempts to connect to a Unix socket at the default path. If the socket
/// is unavailable, falls back to writing formatted logs to stdout.
#[must_use]
pub fn init_smart_logger() -> LoggerGuard {
    init_logger(|| hybrid_local_thread(DEFAULT_COLLECTOR_SOCKET_PATH.into(), "".into()))
}

/// Initializes the global logging collector with a custom closure for processing logs.
///
/// The closure receives a [`LogBuffer`] which wraps the raw encoded log bytes and provides
/// helpful methods for:
/// - Viewing the raw bytes via [`LogBuffer::as_bytes`]
/// - Iterating over decoded logs via [`LogBuffer::iter`]
/// - Swapping out the buffer via [`LogBuffer::swap`], [`LogBuffer::take`], or [`LogBuffer::replace`]
///
/// The closure is called each time there are logs to process. The buffer should be
/// cleared or swapped before returning to avoid processing the same logs again.
///
/// # Examples
///
/// ```no_run
/// use kvlog::collector::{init_closure_logger, LogBuffer};
///
/// let _guard = init_closure_logger(|buffer: &mut LogBuffer| {
///     // Process raw bytes
///     println!("Got {} bytes of logs", buffer.len());
///
///     // Or iterate over decoded logs
///     for log in buffer.iter() {
///         let (timestamp, level, span_info, fields) = log.unwrap();
///         println!("{:?} at {}", level, timestamp);
///     }
///
///     // Clear the buffer when done
///     buffer.clear();
/// });
/// ```
#[must_use]
pub fn init_closure_logger<F>(handler: F) -> LoggerGuard
where
    F: FnMut(&mut LogBuffer) + Send + 'static,
{
    init_logger(move || closure_sync_thread(handler))
}

fn closure_sync_thread<F>(mut handler: F)
where
    F: FnMut(&mut LogBuffer),
{
    let mut buffer = Vec::<u8>::with_capacity(BUFFER_CAPACITY);
    let mut wait_flushers = false;
    loop {
        let mut queue = LOG_WRITER.lock().unwrap();
        if queue.sync_state == SyncState::Flushing {
            wait_flushers = true;
        }
        if queue.encoder.buffer.is_empty() {
            queue.sync_state = SyncState::Waiting;
            let exiting = queue.exiting;
            if wait_flushers {
                wait_flushers = false;
                FLUSHING.notify_all();
            }
            drop(queue);
            if exiting {
                return;
            }
            std::thread::park();
            continue;
        }
        buffer.clear();
        std::mem::swap(&mut buffer, &mut queue.encoder.buffer);
        queue.sync_state = SyncState::Writing;
        drop(queue);

        let mut log_buffer = LogBuffer::new(&mut buffer);
        handler(&mut log_buffer);
    }
}

fn init_logger(syncfn: impl FnOnce() + Send + 'static) -> LoggerGuard {
    let _ = SpanID::next();
    LOG_WRITER.lock().unwrap().handle = Some(std::thread::spawn(syncfn));
    LoggerGuard {}
}
impl CollectorConfig {
    pub fn spawn(self) -> LoggerGuard {
        match self {
            CollectorConfig::Stdout => init_stdout_logger(),
            CollectorConfig::Directory { path } => init_logger(move || directory_sync_thread(path)),
            CollectorConfig::SingleFile { path } => init_logger(move || {
                file_sync_thread(
                    std::fs::OpenOptions::new()
                        .append(true)
                        .create(true)
                        .open(&path)
                        .expect("FAILED to open log file"),
                )
            }),
            CollectorConfig::SocketOrStdout {
                service_name,
                socket,
            } => init_logger(move || hybrid_local_thread(socket, service_name)),
        }
    }
}

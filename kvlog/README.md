# kvlog

High-performance structured binary logging for Rust.

kvlog provides macros for emitting structured log messages with key-value pairs, optimized for high throughput and fast compile times. Log messages are encoded in a compact binary format with nanosecond-resolution timestamps.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
kvlog = "0.1"
```

### Basic Logging

```rust
// Initialize the collector
let _guard = kvlog::spawn_collector_from_env(Some("my-service"), false);

// Simple message
kvlog::info!("Request completed");

// With key-value pairs
let status = 200u16;
let object_id = 42u64;
kvlog::info!("Request completed", status, object_id);

// Explicit key names
kvlog::info!("Request completed", status = 200, id = object_id);
```

### Log Levels

- `debug!` - Development-only logging (requires `debug` feature)
- `info!` - Useful context about application state
- `warn!` - Expected but potentially problematic conditions
- `error!` - Unexpected failures requiring intervention

### Format Specifiers

```rust
let err = "connection refused";
let config = vec![1, 2, 3];

// %expr uses Display formatting
kvlog::error!("Failed to connect", %err);

// ?expr uses Debug formatting
kvlog::warn!("Unexpected config", ?config);
```

### Conditional Fields

```rust
let response_length: Option<usize> = Some(1024);
kvlog::info!(
    "Response sent",
    if let Some(len) = response_length { length = len }
);
```

### Timer

```rust
let timer = kvlog::Timer::start();
// ... do some work ...
kvlog::info!("Operation completed", elapsed = timer);
```

### Spans

kvlog supports hierarchical spans for distributed tracing. Span markers:

- `span.start = id` - Mark the beginning of a span
- `span.current = id` - Log within an active span
- `span.end = id` - Mark the end of a span
- `span.parent = Some(parent_id)` - Establish parent-child relationship (use with `span.start`)

Basic span usage:

```rust
use kvlog::SpanID;

let span = SpanID::next();
kvlog::info!("Starting operation", span.start = span);
kvlog::info!("Processing", span.current = span);
kvlog::info!("Operation complete", span.end = span);
```

Nested spans with parent-child relationships:

```rust
use kvlog::SpanID;

let request_span = SpanID::next();
kvlog::info!("Request received", span.start = request_span);

// Create a child span for database work
let db_span = SpanID::next();
kvlog::info!("Querying database", span.start = db_span, span.parent = Some(request_span));
kvlog::info!("Query complete", span.end = db_span);

kvlog::info!("Request complete", span.end = request_span);
```

Using `SpanID::enter()` to set thread-local span context:

```rust
use kvlog::SpanID;

fn do_database_work(parent: SpanID) {
    let span = SpanID::next();
    kvlog::info!("DB query", span.start = span, span.parent = Some(parent));
    kvlog::info!("DB complete", span.end = span);
}

let span = SpanID::next();
let _guard = span.enter();  // Set as current thread-local span
kvlog::info!("Request received", span.start = span);
do_database_work(span);  // Can access current span via SpanID::current()
kvlog::info!("Request complete", span.end = span);
// Guard dropped here, previous span context restored
```

## Default Test Logger

When no collector is explicitly initialized, kvlog automatically emits logs to stdout in a human-readable format. This works seamlessly with Rust's test runner - use `cargo test -- --nocapture` to see log output during tests:

```rust
#[test]
fn my_test() {
    // No setup needed - logs are automatically captured
    kvlog::info!("Test starting", test_id = 42);

    // ... test logic ...

    kvlog::info!("Test complete");
}
```

```bash
cargo test -- --nocapture
```

This zero-configuration behavior is ideal for development and debugging. For production use, initialize a collector for better performance and output control.

## Collector Configuration

For production or when you need specific output behavior, configure a collector via the `KVLOG_COLLECTOR_CONFIG` environment variable.

Format: `[SERVICE_NAME@]KIND[:PATH]`

| Kind | Description |
|------|-------------|
| `Stdout` | Log to standard output |
| `Directory:/path` | Log to timestamped files in directory |
| `SingleFile:/path` | Log to a single file |
| `SocketOrStdout` | Log to Unix socket, fallback to stdout |

Examples:
- `Stdout`
- `Directory:/var/log/my_app`
- `SingleFile:/tmp/app.log`
- `my-service@SocketOrStdout:/tmp/collector.sock`

## Features

- `debug` - Enable the `debug!` macro (compiled out by default)
- `jiff-02` - Integration with the jiff datetime library

## License

MIT

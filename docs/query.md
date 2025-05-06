## KVLog Query Language

The KVLog query language is used to write queries for kvlog-index. The syntax closely follows Rust.

### Table of Contents

1.  [Overview](#overview)
2.  [Basic Syntax](#basic-syntax)
    - [Field Queries & Existence](#field-queries--existence)
    - [Literals and Values](#literals-and-values)
    - [Logical Operators](#logical-operators)
    - [Grouping](#grouping)
    - [Negation](#negation)
3.  [Data Types](#data-types)
4.  [Operators and Functions](#operators-and-functions)
    - [Equality and Comparison](#equality-and-comparison)
    - [Text Operations](#text-operations)
    - [Membership (`in`)](#membership-in)
    - [Type Checking (`is`)](#type-checking-is)
    - [Range Queries](#range-queries)
    - [Constant Folding](#constant-folding)
5.  [Special Field: Duration](#special-field-duration)
6.  [Special Field: Timestamp](#special-field-timestamp)
    - [Timestamp Queries](#timestamp-queries)
    - [Time Range Function](#time-range-function)
7.  [Meta-Fields](#meta-fields)
8.  [Operator Precedence](#operator-precedence)
9.  [Examples](#examples)

---

### 1. Overview

The query language allows you to filter logs based on the values of their structured fields. Queries are composed of one or more conditions (predicates) that can be combined using logical operators.

### 2. Basic Syntax

#### Field Queries & Existence

The fundamental building block of a query is a condition applied to a field. The general structures are:

- `field_name operator value`
- `field_name.function(arguments)`
- `field_name` (for existence check)

- `field_name`: The name of the field in the log entry (e.g., `status_code`, `message`, `user_id`). Field names are case-sensitive.
- `operator`: The comparison or test to perform (e.g., `=`, `!=`, `>`, `.contains()`).
- `value`: The value to compare against. The type of the value should be compatible with the field's type.

**Field Existence**:
A field name by itself acts as an explicit existence predicate:

- `field_name`: Matches log entries that contain `field_name`, even if its value is `null`.
- `!field_name`: Matches log entries that _do not_ contain `field_name`.

**Examples:**

```rust
http_method = "GET"
response_time > 100
error_message.contains("timeout")
user_id // True if 'user_id' field exists
!optional_config // True if 'optional_config' field does not exist
```

#### Literals and Values

Values in queries can be specified using various literal syntaxes, closely matching Rust:

- **Strings**:
  - Standard strings: Enclosed in double quotes (e.g., `"error message"`).
  - Raw strings: `r#"..."#`, `r##"..."##`, etc. These allow for literal content without escaping `"` or `\`.
    - Example: `json = r#"{"key": "value"}"#`
- **Numbers**:
  - Integers (e.g., `42`, `-100`) and Floats (e.g., `3.14`, `-0.5`).
  - Underscores `_` can be used for readability: `1_000_000`, `0.123_456`.
  - Alternative numeric bases for integers:
    - Hexadecimal: `0xDEADBEEF`, `0xff`
    - Binary: `0b1110_0001`, `0b0000_1111`
    - Octal: `0o777`, `0o644`
- **Booleans**: `true`, `false`.
- **Null**: `null`.
- **Durations and Timestamps**: See dedicated sections ([Duration](#special-field-duration), [Timestamp](#special-field-timestamp)).

#### Logical Operators

Multiple conditions can be combined using logical operators:

- **AND**: `&&`
- **OR**: `||`

**Examples:**

```rust
level = "error" && status_code = 500
user = "admin" || service = "auth-service"
```

#### Grouping

Parentheses `()` can be used to group conditions and control the order of evaluation, similar to mathematical expressions.

**Example:**
`(level = "error" || level = "warn") && service = "payment-service"`

#### Negation

Conditions can be negated using the `!` operator (or `not` prefix).

- For field existence: `!my_field`
- For equality/comparison: `!field_name = value` (equivalent to `field_name != value`)
- For functions/methods: `!field_name.function(arguments)`
- For grouped expressions: `!(condition1 && condition2)`

**Examples:**

```rust
`!status_code = 200`
`!message.contains("success")`
`!error_details` // Matches if 'error_details' field does not exist
`user = "test_user" && !(role = "admin" || role = "editor")`
```

You can also prefix `not` before a field check or condition, but `!` is generally preferred for consistency with boolean logic.
`not error.code = 404`
`not experimental_feature_enabled`

### 3. Data Types

Log fields can have various data types. The query language supports operations tailored to these types:

- **Null**: Represents the absence of a value.
- **String**: A sequence of characters (e.g., `"hello world"`, `r#"raw\string"#`).
- **Bytes**: A sequence of raw bytes.
- **Integer**: Whole numbers (e.g., `123`, `-42`, `0xCAFE`, `0b1010_0101`).
- **Float**: Floating-point numbers (e.g., `3.14`, `-0.001`).
- **Uuid**: Universally Unique Identifiers (e.g., `"123e4567-e89b-12d3-a456-426614174000"`).
- **Boolean**: `true` or `false`.
- **Duration**: A length of time (see [Duration](#special-field-duration)).
- **Timestamp**: A specific point in time (see [Timestamp](#special-field-timestamp)).

### 4. Operators and Functions

#### Equality and Comparison

- `=` : Equal to.
  - Example: `status = 200`, `message = "Request successful"`
- `!=` : Not equal to.
  - Example: `user != "guest"`
- `<` : Less than.
  - Example: `response_time < 50`
- `<=` : Less than or equal to.
  - Example: `cpu_usage <= 0.75`
- `>` : Greater than.
  - Example: `retries > 3`
- `>=` : Greater than or equal to.
  - Example: `queue_length >= 100`

These operators are generally applicable to `Integer`, `Float`, `Duration`, and `Timestamp` types.

#### Text Operations (for String fields)

- `field.starts_with("prefix")`: Checks if the string field starts with the given prefix.
  - Example: `filename.starts_with("IMG_")`
- `field.ends_with("suffix")`: Checks if the string field ends with the given suffix.
  - Example: `hostname.ends_with(".internal")`
- `field.contains("substring")`: Checks if the string field contains the given substring.
  - Example: `$message.contains("Exception")`
- **UNIMPLEMENTED** `field.matches_regex("pattern", "flags")`: Checks if the string field matches the given regular expression.
  - `pattern`: A string representing the regular expression.
  - `flags` (optional): A string for flags. Currently, only `"i"` for case-insensitive matching is supported.
  - Example: `$message.matches_regex("^ERROR:.*failed")`
  - Example (case-insensitive): `user_agent.matches_regex("firefox", "i")`
  - KVLog uses the Rust `regex` crate. For detailed regular expression syntax, see [https://docs.rs/regex/latest/regex/index.html#syntax](https://docs.rs/regex/latest/regex/index.html#syntax).

#### Membership (`in`)

The `in` operator checks if a field's value is present in a list of specified values.

- `field in [value1, value2, ...]`

**Examples:**

```rust
http_status_code in [400, 401, 403, 404]
$level in ["error", "critical"]
```

Note: `in` is not a reserved keyword and can be used as a field name if necessary.

#### Type Checking (`is`)

The `is` operator checks the type of a field.

- `field is Type`
- Supported `Type` identifiers:
  - `String`
  - `Bytes`
  - `Integer`
  - `Float`
  - `Number` (matches `Integer` or `Float`)
  - `Uuid` (or `UUID`)
  - `Boolean` (or `Bool`)
  - `Duration`
  - `Timestamp`
  - `Null` (or `None`)

**Examples:**

```rust
payload is String
value is Number
optional_field is Null
(user_id is String) || (user_id is Integer)
```

Note: `is` is not a reserved keyword and can be used as a field name if necessary.

#### Range Queries

Range queries are implicitly supported by `<, <=, >, >=` operators. You can combine them for a specific range:

**Example:**

```rust
response_time >= 100 && response_time < 500
temperature > 10.5 && temperature < 25.0
```

For `Timestamp` and `Duration` fields, specific range functions might be available or constructed.

#### Constant Folding

KVLog performs constant folding for expressions involving only literal values. These expressions are evaluated at query parse time, not per log entry. This allows for more readable or pre-calculated literal values within the query itself.

Supported operations include:

- **Numeric Arithmetic**:
  - Operators: `+` (addition), `-` (subtraction/negation), `*` (multiplication), `/` (division), `%` (remainder).
  - Example: `threshold = 1024 * 4` (evaluates to `4096`), `value = (100 - 50) / 2` (evaluates to `25`), `offset = -10`.
- **Bitwise Operations** (on integers):
  - Operators: `&` (Bitwise AND), `|` (Bitwise OR), `^` (Bitwise XOR), `<<` (Shift Left), `>>` (Shift Right).
  - Example: `mask = 0x0F & 0b1010`, `flags = 1 << 3` (evaluates to `8`).
- **String Concatenation**:
  - Operator: `+`.
  - Example: `prefix = "user_" + "id"` (evaluates to `"user_id"`).
- **Duration Arithmetic**:
  - Operators: `+` (addition), `-` (subtraction).
  - Example: `total_time = 1h + 30m`, `remaining = 1d - 1s`.

**Numeric Overflow**: If any numeric operation during constant folding results in an overflow (e.g., an integer exceeding its maximum representable value), the query will produce an error.

**Examples of Constant Folding in Queries**:

- `event_code = 0xBAD + 0xC0DE`
- `message_key = "log_" + "entry_" + "type"`
- `backup_interval = 24h - 1m`
- `max_connections = 2 * (10 + 5_000)` // Evaluates to 10020
- `default_port = 0o700 + 77` // Evaluates to 448 + 77 = 525

### 5. Special Field: Duration

Duration fields represent a span of time. Durations can be specified in queries using a number followed by a unit:

- `s`: seconds
- `ms`: milliseconds
- `µs` or `us`: microseconds
- `ns`: nanoseconds
- `m`: minutes
- `h`: hours
- `d`: days

**Example:**

```rust
elapsed_time > 500ms
job_runtime < 2h
timeout = 5s
timeout_precise = 1s + 500ms
```

**Duration Range Example (using provided test structure):**
A direct `duration_range` function is not explicitly shown in user queries but can be constructed:
`execution_time >= 100ms && execution_time <= 1s`

### 6. Special Field: Timestamp

Timestamp fields represent a specific point in time. They are typically stored with nanosecond precision.

#### Time Range Function

A special `time_range()` function is available for querying timestamps within a specific range or at a certain granularity.

- `field_name in time_range("YYYY-MM-DDTHH:MM:SSZ")`: Matches events at this exact second.
- `field_name in time_range("YYYY-MM-DDTHH:MMZ")`: Matches events within this minute.
- `field_name in time_range("YYYY-MM-DDTHHZ")`: Matches events within this hour.
- `field_name in time_range("YYYY-MM-DD")`: Matches events on this day.
- `field_name in time_range("YYYY-MM")`: Matches events in this month.
- `field_name in time_range("YYYY")`: Matches events in this year.

**Examples:**

```rust
$timestamp in time_range("2025-05-24T01:04:46Z") // Events exactly at this second
created_at in time_range("2025-05-24") // Events on May 24, 2025
modified_on in time_range("2025-05") // Events in May 2025
```

### 7. Meta-Fields

Meta-fields are special, system-provided fields prefixed with a `$` sign. They provide access to common log metadata.

- `$timestamp`: The primary timestamp of the log entry.
  - Example: `$timestamp in time_range("2023-01-01T00:00:00Z")`
- `$level`: The log level (e.g., "error", "warn", "info", "debug").
  - Example: `$level = "error"`
- `$message`: Often used for the primary, human-readable message of a log entry.
  - Example: `$message.contains("database connection failed")`
- `$span`: Identifier for a trace span, if logs are part of a distributed trace.
- `$parent_span`: Identifier for the parent span in a trace.
- `$span_duration`: The duration of the span. Can be queried like other duration fields.
  - Example: `$span_duration > 1s`
- `$target`: The origin of the log entry, often a module path (e.g., `myapp::payment::processor`).
  - Example: `$target = "myapp::api::v1"`

**Example combining meta-fields and regular fields:**

```rust
$level = "error" || err || $message.contains("fail")
```

Here, `err` is a field. As explained in [Field Queries & Existence](#field-queries--existence), `err` as a standalone predicate checks for the existence of an `err` field in the log entry. `$level` and `$message` are meta-fields.

### 8. Operator Precedence

The language follows a standard order of operations, similar to Rust or BEDMAS/PEMDAS:

1.  Parentheses `()`
2.  Function calls (e.g., `.contains()`, `time_range()`, `.matches_regex()`)
3.  Unary `!` (NOT), Unary `-` (Negation)
4.  `*`, `/`, `%` (Multiplication, Division, Remainder)
5.  `+`, `-` (Addition, Subtraction, String Concatenation)
6.  `<<`, `>>` (Bitwise Shifts)
7.  `&` (Bitwise AND)
8.  `^` (Bitwise XOR)
9.  `|` (Bitwise OR)
10. Comparison operators (`=`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `is`)
11. Logical `&&` (AND)
12. Logical `||` (OR)

_Note: Constant folding applies to arithmetic, bitwise, and string concatenation operators on literals before other evaluations._
It's always recommended to use parentheses `()` to clarify intent for complex queries.

### Examples

1. Find all error logs from the payment service:

```rust
$level = "error" && service_name = "payment-service"
```

2. Find logs where a request took longer than 500 milliseconds or resulted in a 5xx status code:

```rust
$span_duration > 500ms || status_code >= 500
```

3. Find logs from user "john.doe" that are not informational messages:

```rust
user = "john.doe" && $level != "info"
```

4. Find logs containing "timeout" or "connection refused" in the message field, specifically for the API module:

```rust
$target = "api_module" && (message.contains("timeout") || message.contains("connection refused"))
```

5. Find logs where the `item_count` field is an integer and greater than 10:

```rust
item_count is Integer && item_count > 10
```

6. Find logs from May 2025 where the message starts with "CRITICAL:":

```rust
$timestamp in time_range("2025-05") && $message.starts_with("CRITICAL:")
```

7. Find spans that lasted longer than 1 second (using constant folding for clarity if desired):

```rust
$span_duration > 1s
```

or

```rust
$span_duration > (1000ms)
```

8. Find logs where `mega` field is a string OR is null:

```rust
(mega is String) || (mega is Null)
```

9. Find logs where `$message` is "alpha", "beta", or "canary":

```rust
$message in ["alpha", "beta", "canary"]
```

10. Find error-ish logs (complex example from tests, `err` checks for field existence):

```rust
$level = "error" || err || msg.contains("fail")
```

11. Find logs where a debug flag `0x08` is set in `feature_flags` field:

```rust
feature_flags & 0x08 = 0x08
```

12. Find logs with a raw string path:

```rust
log_file_path = r#"C:\logs\app.log"#
```

13. Find logs where `event_id` matches a regex pattern, case-insensitively:

```rust
event_id.matches_regex("evt-[a-f0-9]{8}", "i")
```

14. Find logs where a field `user_data` does not exist:

```rust
!user_data
```

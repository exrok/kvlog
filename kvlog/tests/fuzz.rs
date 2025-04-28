use kvlog::encoding::{Encoder, MunchError, SpanInfo, StaticKey, Value};
use kvlog::{Encode, LogLevel, SpanID};
use rand::distr::{Alphanumeric, SampleString, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

type Rand = SmallRng;
struct ValueSampler {
    ascii_subbuffer: String,
    bytes_subbuffer: Vec<u8>,
    utf8_strings: Vec<Box<str>>,
}
impl ValueSampler {
    fn new(rng: &mut Rand) -> ValueSampler {
        let ascii_subbuffer = Alphanumeric.sample_string(rng, 256);
        let bytes_subbuffer: Vec<u8> = (0..2048).map(|_| rng.random()).collect();
        let utf8_strings: Vec<Box<str>> = (0..64)
            .map(|_| {
                let len = rng.random_range(0..512);
                StandardUniform.sample_string(rng, len).into()
            })
            .collect();
        ValueSampler {
            ascii_subbuffer,
            bytes_subbuffer,
            utf8_strings,
        }
    }
    fn key<'a>(&'a self, rng: &mut Rand) -> &'a str {
        if let Some(value) = StaticKey::u8_to_string(rng.random()) {
            return value;
        }
        let start = rng.random_range(0..self.ascii_subbuffer.len());
        let max_len = if rng.random_bool(0.85) {
            (start + 10).min(self.ascii_subbuffer.len() + 1)
        } else {
            (start + 127).min(self.ascii_subbuffer.len() + 1)
        };
        let end = rng.random_range(start..max_len);
        &self.ascii_subbuffer[start..end]
    }
    fn value<'a>(&'a self, rng: &mut Rand) -> Value<'a> {
        match rng.random_range(0..12) {
            0 => {
                if rng.random_bool(0.75) {
                    let start = rng.random_range(0..self.ascii_subbuffer.len());
                    let max_len = if rng.random_bool(0.85) {
                        (start + 32).min(self.ascii_subbuffer.len() + 1)
                    } else {
                        self.ascii_subbuffer.len() + 1
                    };
                    let end = rng.random_range(start..max_len);
                    Value::String(self.ascii_subbuffer[start..end].as_bytes())
                } else {
                    let sample_from_utf8_strings = rng.random_range(0..self.utf8_strings.len());
                    Value::String(self.utf8_strings[sample_from_utf8_strings].as_bytes())
                }
            }
            1 => {
                let start = rng.random_range(0..self.bytes_subbuffer.len());
                let max_len = if rng.random_bool(0.85) {
                    (start + 32).min(self.bytes_subbuffer.len() + 1)
                } else {
                    self.bytes_subbuffer.len() + 1
                };
                let end = rng.random_range(start..max_len);
                Value::Bytes(&self.bytes_subbuffer[start..end])
            }
            2 => Value::I32(rng.random()),
            3 => Value::U32(rng.random()),
            4 => Value::I64(rng.random()),
            5 => Value::U64(rng.random()),
            6 => Value::F32(f32::from_ne_bytes(rng.random())),
            7 => Value::F64(f64::from_ne_bytes(rng.random())),
            8 => Value::Bool(rng.random()),
            9 => Value::UUID(uuid::Uuid::from_bytes(rng.random())),
            10 => Value::Timestamp({
                kvlog::Timestamp::from_millisecond(rng.random())
                // jiff::Timestamp::from_nanosecond(rng.random_range(
                //     jiff::Timestamp::MIN.as_nanosecond()..=jiff::Timestamp::MAX.as_nanosecond(),
                // ))
                // .expect("This should be all based on min and max")
            }),
            _ => Value::None,
        }
    }
}

fn random_log_level(rng: &mut Rand) -> LogLevel {
    match rng.random_range(0..4) {
        0 => LogLevel::Debug,
        1 => LogLevel::Info,
        2 => LogLevel::Warn,
        _ => LogLevel::Error,
    }
}
#[derive(Debug)]
struct ExpectedLog<'a> {
    level: LogLevel,
    timestamp: u64,
    fields: Vec<(&'a str, Value<'a>)>,
    span_info: SpanInfo,
}

#[track_caller]
fn assert_values_eq(value: Value<'_>, expected_value: Value<'_>) {
    if let (Value::F32(a), Value::F32(b)) = (&value, expected_value) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{value:?} != {expected_value:?}: F32 Value mismatch"
        );
    } else if let (Value::F64(a), Value::F64(b)) = (&value, expected_value) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{value:?} != {expected_value:?}: f64 Value mismatch"
        );
    } else {
        assert_eq!(value, expected_value, "Value mismatch for log");
    }
}

fn random_span_info(rng: &mut Rand) -> SpanInfo {
    match rng.random_range(0..5) {
        0 => SpanInfo::Start {
            span: SpanID::next(),
            parent: Some(SpanID::next()),
        },
        1 => SpanInfo::Start {
            span: SpanID::next(),
            parent: None,
        },
        2 => SpanInfo::Current {
            span: SpanID::next(),
        },
        3 => SpanInfo::End {
            span: SpanID::next(),
        },
        _ => SpanInfo::None,
    }
}

#[test]
fn test() {
    let rng = &mut SmallRng::from_os_rng();
    let sample = ValueSampler::new(rng);
    let mut logs: Vec<ExpectedLog> = Vec::new();
    let mut encoder = Encoder::with_capacity(4096 * 32);
    for _ in 0..100 {
        logs.clear();
        for _ in 0..1000 {
            logs.push(ExpectedLog {
                level: random_log_level(rng),
                timestamp: rng.random(),
                fields: (0..rng.random_range(0..256u32))
                    .map(|_| (sample.key(rng), sample.value(rng)))
                    .collect(),
                span_info: random_span_info(rng),
            });
        }
        encoder.clear();
        let time = std::time::Instant::now();
        for log in &logs {
            let mut add = encoder.append(log.level.into(), log.timestamp);
            for (key, value) in &log.fields {
                value.encode_log_value_into(add.key(key))
            }
            add.apply_span_info(log.span_info.clone());
        }
        println!("ENCODE: {:?}", time.elapsed());
        let time = std::time::Instant::now();
        let mut count = 0;
        let mut field_count = 0;
        for (decoded_log, expected_log) in kvlog::encoding::decode(encoder.bytes()).zip(&logs) {
            count += 1;
            let (timestamp, log_level, span_info, fields) = match decoded_log {
                Ok(v) => v,
                Err(err) => {
                    panic!("Failed decoding {count} Log, {err:?}, {expected_log:#?}");
                }
            };
            assert_eq!(span_info, expected_log.span_info);
            assert_eq!(timestamp, expected_log.timestamp);
            assert_eq!(log_level, expected_log.level);
            for (i, (decoded_field, (expected_key, expected_value))) in
                fields.zip(expected_log.fields.iter().rev()).enumerate()
            {
                field_count += 1;
                // let (key, value) = decoded_field.expect("decoding field");
                let (key, value) = match decoded_field {
                    Ok(v) => v,
                    Err(err) => {
                        panic!(
                            "Failed decoding log {count}, Field {i}, {err:?}, expected: {:?}={:?}",
                            expected_key, expected_value
                        );
                    }
                };
                assert_eq!(
                    key, *expected_key,
                    "Key mismatch for log {count}, Field {i}"
                );
                assert_values_eq(value, *expected_value)
            }
        }
        let elapsed = time.elapsed();
        let klogs_per_second = (logs.len() as f64 / 1000.0) / elapsed.as_secs_f64();
        let mb_per_second = (encoder.bytes().len() as f64 / 1000_000.0) / elapsed.as_secs_f64();
        println!(
            "FORWARD_DECODE: total: {:?}, {:?} / log, {:?} / field, {:.1?}k logs/second, {:.1}mb/s",
            time.elapsed(),
            elapsed / (logs.len() as u32),
            elapsed / (field_count as u32),
            klogs_per_second,
            mb_per_second
        );
        assert_eq!(count, logs.len());
        let mut decoder = kvlog::encoding::ReverseDecoder {
            bytes: encoder.bytes(),
        };

        let time = std::time::Instant::now();
        for expected_log in logs.iter().rev() {
            let mut fields = expected_log.fields.iter().rev();
            let start_length = decoder.bytes.len();

            let span_info = decoder.pop_span_info().unwrap();
            assert_eq!(span_info, expected_log.span_info);
            for i in 0.. {
                match decoder.pop_key_value() {
                    Ok((key, value)) => {
                        let (expected_key, expected_value) = fields.next().unwrap();
                        assert_eq!(
                            key, *expected_key,
                            "Key mismatch for log {count}, Field {i}"
                        );
                        assert_values_eq(value, *expected_value)
                    }
                    Err(MunchError::InvalidValueKind) => {
                        break;
                    }
                    Err(err) => {
                        panic!("{err:?}");
                    }
                }
            }
            assert_eq!(fields.count(), 0, "some fields not parsed");
            let (length, timestamp, log_level) = decoder.pop_header().unwrap();
            let end_length = decoder.bytes.len();
            assert_eq!(timestamp, expected_log.timestamp);
            assert_eq!(log_level, expected_log.level);
            assert_eq!(start_length - end_length, length + 4);
        }
        let elapsed = time.elapsed();
        let klogs_per_second = (logs.len() as f64 / 1000.0) / elapsed.as_secs_f64();
        let mb_per_second = (encoder.bytes().len() as f64 / 1000_000.0) / elapsed.as_secs_f64();
        println!(
            "REVERSE_DECODE: total: {:?}, {:?} / log, {:?} / field, {:.1?}k logs/second, {:.1}mb/s",
            time.elapsed(),
            elapsed / (logs.len() as u32),
            elapsed / (field_count as u32),
            (klogs_per_second),
            mb_per_second
        );
        assert!(decoder.bytes.is_empty());
    }
}

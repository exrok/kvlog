#![allow(warnings, unsued)]

use std::path::Path;

use kvlog::{
    encoding::{munch_log_with_span, Key, LogFields, MunchError, StaticKey, Value},
    LogLevel, SpanID,
};

use kvlog_index::{
    accel::{slow_bloom_query, MicroBloom},
    index::{FieldKind, GeneralFilter, Index, Query, TimeFilter},
};

struct SyndromeDefintion {
    level: kvlog::LogLevel,
    fields: &'static [(Key<'static>, FieldGen)],
    in_span: bool,
}

enum FieldGen {
    Const(Value<'static>),
    OneOf(&'static [Value<'static>]),
    U32(std::ops::Range<u32>),
    Func(fn(&mut oorandom::Rand32) -> Value<'static>),
    UUID,
}

impl FieldGen {
    fn generate(&self, rng: &mut oorandom::Rand32) -> Value<'static> {
        match self {
            FieldGen::Const(value) => value.clone(),
            FieldGen::OneOf(value) => {
                let idx = rng.rand_u32() as usize % value.len();
                value[idx as usize].clone()
            }
            FieldGen::Func(func) => func(rng),
            FieldGen::U32(range) => Value::I64(rng.rand_range(range.clone()).into()),
            FieldGen::UUID => {
                let raw = [rng.rand_u32(), rng.rand_u32(), rng.rand_u32(), rng.rand_u32()];
                unsafe { Value::UUID(uuid::Uuid::from_bytes(*raw.as_ptr().cast())) }
            }
        }
    }
}

// let mut rng = oorandom::Rand32::new(0xdeafbeaf);
#[derive(Clone)]
struct RandomFields {
    rng: oorandom::Rand32,
    syndrome: &'static SyndromeDefintion,
    stage: usize,
}

// impl RandomFields {
//     fn new(mx: &mut oorandom::Rand32) -> RandomFields {}
// }

impl Iterator for RandomFields {
    type Item = Result<(Key<'static>, Value<'static>), MunchError>;

    fn next(&mut self) -> Option<Self::Item> {
        let (key, value) = self.syndrome.fields.get(self.stage)?;
        self.stage += 1;
        Some(Ok((*key, value.generate(&mut self.rng))))
    }
}

// #[bitmatch]
//     _pdep_u32(a, mask)
//     #[bitmatch]
//     let "xxxx_yyyy" = n;
//     bitpack!("xyxy_xyxy")

// fn interleave(n: u32) -> u32 {
//     use std::arch::x86_64;
//     unsafe { x86_64::_pdep_u32(n, 0xAAAA_AAAA) | x86_64::_pdep_u32(n & 0xFFFF_0000, 0x3333_3333) }
// }

macro_rules! fielddef_key {
    ($key: literal ) => {
        Key::Dynamic($key)
    };
    ($key: tt ) => {
        Key::Static(StaticKey::$key)
    };
}
macro_rules! fielddef {
    ($($key: tt : $value: expr),* $(,)*) => {
        &[
            $((fielddef_key!($key), $value)),*
        ]
    };
}

use uuid::Uuid;
use FieldGen::*;
use Value::String as S;

fn parse_check() {
    for token in ra_ap_rustc_lexer::tokenize("234.234ms") {
        println!("{:?}", token);
    }
}
fn ingest_single_file(path: &Path, index: &mut Index) {
    let timer = kvlog::Timer::start();
    let contents = std::fs::read(path).unwrap();
    let mut cursor: &[u8] = &contents;
    let mut count = 0;
    while !cursor.is_empty() {
        count += 1;
        match kvlog::encoding::munch_log_with_span(&mut cursor) {
            Ok((timestamp, level, span_info, fields)) => {
                if let Err(err) = index.write(timestamp, level, span_info, None, fields) {
                    panic!("Failed to write log to index: {err:?}")
                }
            }
            Err(MunchError::EofOnHeader | MunchError::EofOnFields | MunchError::Eof) => {
                panic!("Unexpected EOF")
            }
            Err(err) => {
                panic!("Error munching logs: {err:?}");
            }
        }
    }

    let memory_usage = index.current_bucket_memory_used();

    println!(
        "{:.1} MB {:.2} bytes / log, Stats [{}] {:#?}",
        memory_usage.total() as f64 / (1024.0 * 1024.0),
        memory_usage.bytes_per_log(),
        index.generation(),
        memory_usage
    );
    kvlog::info!("Logs read from file", ?path, count, elapsed = timer);
}
fn main() {
    // println!("{:X}", interleave(0xff));
    // return;
    let mut index = Index::new();
    let http_request = &[
        SyndromeDefintion {
            level: LogLevel::Info,
            fields: fielddef! {
                msg: Const(S(b"HTTP Request")),
                target: Const(S(b"web")),
                handler: OneOf(&[
                    S(b"webserver::endpoints::user::login_route"),
                    S(b"webserver::endpoints::system::system_events_stream"),
                    S(b"webserver::endpoints::user::google_login_endpoint"),
                ]),
                path: OneOf(&[
                    S(b"/user/login"),
                    S(b"/user/login/google"),
                ]),
                method: OneOf(&[
                    S(b"POST"),
                    S(b"GET"),
                    S(b"PATCH"),
                    S(b"DELETE"),
                ]),
            },
            in_span: true,
        },
        SyndromeDefintion {
            level: LogLevel::Info,
            fields: fielddef! {
                msg: Const(S(b"HTTP Response")),
                target: Const(S(b"web")),
                length: FieldGen::U32(0..30000),
                content_type: OneOf(&[
                    S(b"application/json"),
                    S(b"text"),
                ]),
                status: OneOf(&[
                    Value::U32(200),
                    Value::U32(201),
                    Value::U32(204),
                    Value::U32(404),
                    Value::U32(400),
                    Value::U32(500),
                    Value::U32(424),
                    Value::U32(304),
                ]),
                b"length": FieldGen::UUID,
            },
            in_span: true,
        },
    ];
    let syndromes = &[
        (
            10,
            SyndromeDefintion {
                level: LogLevel::Info,
                fields: fielddef! {
                    msg: Const(S(b"Archive events")),
                    target: Const(S(b"database::postgres::object_event")),
                    b"amount": FieldGen::U32(0..30),
                },
                in_span: false,
            },
        ),
        (
            1,
            SyndromeDefintion {
                level: LogLevel::Warn,
                fields: fielddef! {
                    msg: OneOf(&[
                        S(b"User Updated"),
                    ]),
                    target: Const(S(b"database::users")),
                    b"amount": Const(Value::I32(0)),
                    user_id: Func(|rng| Value::UUID(Uuid::from_u64_pair(0, (rng.rand_u32()&0xff) as u64))),
                },
                in_span: false,
            },
        ),
        (
            2,
            SyndromeDefintion {
                level: LogLevel::Info,
                fields: fielddef! {
                    msg: OneOf(&[
                        S(b"Object Archived"),
                        S(b"Object Deleted"),
                    ]),
                    target: Const(S(b"database::archiver")),
                    object_id: FieldGen::UUID,
                },
                in_span: false,
            },
        ),
    ];
    let mut map: Vec<u8> = Vec::new();
    for (i, (w, _)) in syndromes.iter().enumerate() {
        map.extend(std::iter::repeat(i as u8).take(*w))
    }
    let mut rng = oorandom::Rand32::new(0xdeadbeaf);
    let mut timestamp = 1000;
    let mut start = std::time::Instant::now();
    let mut logs = 500_000;
    for _ in 0..logs {
        timestamp += 1000 + (rng.rand_u32() as u64 & 0xfffff);
        if rng.rand_u32() & 0b11 == 0 {
            let span = SpanID::next();
            index
                .write(
                    timestamp,
                    http_request[0].level,
                    kvlog::SpanInfo::Start { span, parent: None },
                    None,
                    RandomFields { rng: rng.clone(), syndrome: &http_request[0], stage: 0 },
                )
                .unwrap();
            for _ in 0..(rng.rand_u32() & 0b11) {
                timestamp += 100 + (rng.rand_u32() as u64 & 0xfffff);
                let i = &map[rng.rand_u32() as usize % map.len()];
                let (_, syndrome) = &syndromes[*i as usize];
                index
                    .write(
                        timestamp,
                        syndrome.level,
                        kvlog::SpanInfo::Current { span },
                        None,
                        RandomFields { rng: rng.clone(), syndrome, stage: 0 },
                    )
                    .unwrap();
            }
            timestamp += 100 + (rng.rand_u32() as u64 & 0xfffff);
            index
                .write(
                    timestamp,
                    http_request[1].level,
                    kvlog::SpanInfo::End { span },
                    None,
                    RandomFields { rng: rng.clone(), syndrome: &http_request[1], stage: 0 },
                )
                .unwrap();
            continue;
        }
        let i = &map[rng.rand_u32() as usize % map.len()];
        let (_, syndrome) = &syndromes[*i as usize];
        index
            .write(
                timestamp,
                syndrome.level,
                kvlog::SpanInfo::None,
                None,
                RandomFields { rng: rng.clone(), syndrome, stage: 0 },
            )
            .unwrap();
    }
    let elapsed = start.elapsed();
    println!("Insertion: {:?}, {:?} / 1000 logs", elapsed, elapsed / ((logs / 1000) as u32));

    let query_start = std::time::Instant::now();
    let mut string = String::with_capacity(512);
    let mut count = 0;
    let mut queries = 100;
    for i in 0..queries {
        use std::fmt::Write;
        string.clear();
        let uuid = Uuid::from_u64_pair(0, (i) as u64);
        write!(string, "  msg.contains(\"pdated\")");
        // write!(string, "msg.contains(\"User\")");
        let query = Query::expr(&string).unwrap();
        for entry in index.reverse_query(&query.filters) {
            // println!("{}", entry.message().escape_ascii());
            // println!("{} {:?}", entry.message().escape_ascii(), entry.fields());
            // println!("__");
            count += 1;
        }
    }
    let mut e = query_start.elapsed();
    let mut total_logs = 0;
    for bucket in index.reader().read_newest_buckets() {
        println!("{}", bucket.entry_count());
        total_logs += bucket.entry_count();
    }
    println!(
        "{} of {} ({:.3}%)",
        count,
        total_logs * queries,
        count as f64 * 100.0 / (total_logs as f64 * queries as f64)
    );
    println!(
        "Query rate: {:.1?}M logs / second",
        ((((total_logs) as f64) * queries as f64) / 1000_000.0) / e.as_secs_f64()
    );

    let x = 2;
}

use crate::index::filter::archetype_filter::{build_pred_addr_map, ArchetypePrefilter, Match};
use crate::index::filter::{QueryFilter, QueryVm};
use crate::index::test::{test_index, TestIndexWriter};
use crate::index::{Bucket, GeneralFilter, Index, WeakLogEntry};
use crate::query::query_parts::parser::Parser;
use crate::query::query_parts::{self, OptimizationInfo};
use crate::query::QueryParseError;
use crate::{log, ServiceId};
use bumpalo::Bump;
use hashbrown::{HashMap, HashSet};
use kvlog::encoding::{FieldBuffer, Seconds};
use kvlog::{Encode, LogLevel};
struct QueryTester<'a> {
    pub(crate) time: u64,
    pub(crate) buf: FieldBuffer,
    pub(crate) index: &'a mut Index,
    bump: bumpalo::Bump,
}
#[derive(Default, Debug, Clone, Copy)]
struct QueryAssertions {
    instructions: usize,
}

impl<'a> QueryTester<'a> {
    pub fn new(index: &'_ mut Index) -> QueryTester<'_> {
        QueryTester { time: 1, buf: FieldBuffer::default(), index, bump: Bump::new() }
    }
    pub fn set_time(&mut self, ts: &str) {
        let jiffed: jiff::Timestamp = ts.parse().unwrap();
        self.time = jiffed.as_nanosecond().try_into().unwrap();
    }
    #[track_caller]
    fn assert_eval_multi(&mut self, inputs: &[(WeakLogEntry, u64)], queries: &[(u64, &str, &str, QueryAssertions)]) {
        for (mask, name, query, assumptions) in queries {
            self.assert_eval(name, *mask, inputs, *query, *assumptions);
        }
    }
    #[track_caller]
    fn assert_eval(
        &mut self,
        name: &str,
        mask: u64,
        inputs: &[(WeakLogEntry, u64)],
        query: &str,
        props: QueryAssertions,
    ) {
        self.bump.reset();

        let mut ogr = {
            let input = query.trim();
            if input.is_empty() {
                panic!();
            }
            match Parser::init(&self.bump, input).and_then(|mut p| {
                let (expr, span) = p.parse_expr(0)?;
                p.expr_to_predicates(expr)
            }) {
                Ok(predicates) => Ok(predicates),
                Err(err) => Err(QueryParseError::from_internal(err, input)),
            }
        }
        .unwrap();
        for negate in [false, true] {
            if negate {
                ogr.negate();
            }
            let mut bucket = self.index.reader().newest_bucket().unwrap();
            println!("PREDICATE: {:?}", ogr);
            let rules = match ogr.build_with_opt(&self.bump, &bucket, &*bucket.maps()) {
                query_parts::PredBuildResult::AlwaysTrue => {
                    for (entry, set_mask) in inputs {
                        let expected = (set_mask & mask) != 0;
                        let entry = bucket.upgrade(*entry).unwrap();
                        assert_eq!(
                            true ^ negate,
                            expected,
                            "Query(AlwaysTrue): {query} on entry: {:?}",
                            entry.fields()
                        );
                    }
                    continue;
                }
                query_parts::PredBuildResult::AlwaysFalse => {
                    for (entry, set_mask) in inputs {
                        let expected = (set_mask & mask) != 0;
                        let entry = bucket.upgrade(*entry).unwrap();
                        assert_eq!(
                            false ^ negate,
                            expected,
                            "{} Query(AlwaysFalse): {query} on entry: {:?}",
                            if negate { "Negated" } else { "" },
                            entry.fields()
                        );
                    }
                    continue;
                }
                query_parts::PredBuildResult::Ok(pred) => pred,
            };
            let mut vm = QueryVm::compile(&self.bump, rules, &bucket, self.index.reader().targets.mapper()).unwrap();
            if props.instructions > 0 && vm.reachable_instruction_count() != props.instructions {
                println!("------------------------------------- PRED --------------------------------------");
                println!("{:?}", rules);
                println!("-------------------------------------- VM ---------------------------------------");
                vm.print();
                println!("---------------------------------------------------------------------------------");
                panic!(
                    "{} number of instructions did not match expected:\n    query: {} \n expected: {}\n      got: {}",
                    if negate { "Negated" } else { "" },
                    query,
                    props.instructions,
                    vm.reachable_instruction_count()
                )
            }
            for (entry, set_mask) in inputs {
                let entry = bucket.upgrade(*entry).unwrap();
                let expected = (set_mask & mask) != 0;
                let got = vm.matches(entry);
                if got != expected ^ negate {
                    println!("------------------------------------- PRED --------------------------------------");
                    println!("{:?}", rules);
                    println!("-------------------------------------- VM ---------------------------------------");
                    vm.print();
                    println!("---------------------------------------------------------------------------------");
                    panic!(
                        "{} Query unexpectedly {}: \n QUERY[{}]: {}\n LOG: {{\n  fields: {:?}\n  msg: `{}`, \n  service: {:?}\n  level: {:?}\n}}",
                        if negate { "Negated" } else { "" },
                        if got { "matched" } else { "didn't match" },
                        name,
                        query,
                        entry.fields(),
                        entry.message().escape_ascii(),
                        entry.archetype().service(),
                        entry.level()
                    );
                    println!("}}");
                }

                // assert_eq!(got, *expected, "Query: {query} on entry: {:?}", entry.fields());
            }
        }
        let mut found_positive: HashSet<WeakLogEntry> = HashSet::new();
        let mut found_negative: HashSet<WeakLogEntry> = HashSet::new();
        let reader = self.index.reader();
        let filters = [GeneralFilter::Query(QueryFilter::from_str(query).unwrap())];
        let mut walker = reader.reverse_query(&filters);
        while let Some(mut items) = walker.next() {
            for entry in &items {
                found_positive.insert(entry.weak());
            }
        }
        {
            let filters = [GeneralFilter::Query(QueryFilter::from_str(&format!("!( {query} )")).unwrap())];
            let mut walker = reader.reverse_query(&filters);
            while let Some(mut items) = walker.next() {
                for entry in &items {
                    found_negative.insert(entry.weak());
                }
            }
        }
        for (entry, set_mask) in inputs {
            let expected = (set_mask & mask) != 0;
            if expected != found_positive.contains(entry) {
                let bucket = reader.newest_bucket().unwrap();
                let entry = bucket.upgrade(*entry).unwrap();
                panic!(
                    "Query unexpectedly {}: \n QUERY[{}]: {}\n LOG: {{\n  fields: {:?}\n  msg: `{}`, \n  service: {:?}\n  level: {:?}\n}}",
                    if expected { "FAILED" } else { "SUCCEEDED" },
                    name,
                    query,
                    entry.fields(),
                    entry.message().escape_ascii(),
                    entry.archetype().service(),
                    entry.level()
                );
            }

            if !expected != found_negative.contains(entry) {
                let bucket = reader.newest_bucket().unwrap();
                let entry = bucket.upgrade(*entry).unwrap();
                panic!(
                    "NEGATIVE Query unexpectedly {}: \n QUERY[{}]: {}\n LOG: {{\n  fields: {:?}\n  msg: `{}`, \n  service: {:?}\n  level: {:?}\n}}",
                    if expected { "FAILED" } else { "SUCCEEDED" },
                    name,
                    query,
                    entry.fields(),
                    entry.message().escape_ascii(),
                    entry.archetype().service(),
                    entry.level()
                );
            }
            // assert_eq!(got, *expected, "Query: {query} on entry: {:?}", entry.fields());
        }
    }
}
macro_rules! assert_query_results {
    ($tester:ident [$($query:tt)*] $({$($prop:tt)*})? $( ( $($log:tt)* ): $eval:literal ),* $(,)?) => {{
        unsafe {
            $tester.index.clear_unchecked();
        }
        let expects = [$(
            ( log!($tester; $($log)*), $eval as u64 )
            ),*];
        $tester.assert_eval("", 1, &expects, stringify!($($query)*), QueryAssertions{$($($prop)*,)? ..Default::default()});
    }}
}

macro_rules! assert_multi_query_results {
    (
        $tester:ident {
            $(
                $query_label: ident: [$($query:tt)*] $({$($per_query_assertions:tt)*})?
            ),* $(,)?
        }
        $( {$($base_assertions:tt)*} )?
        $(
           $([$($log_cmd_prefix:tt)*])? (  $($log:tt)* ): $($true_queries:ident)|*
        ),* $(,)?
    ) => {{
        let mut mask = 1u64;
        $(
            let $query_label = mask;
            mask <<= 1;
        )*
        let logs = [$( ( { $({$($log_cmd_prefix)*};)? log!($tester; $($log)*)}, 0 $(| $true_queries)* )),*];
        let base_assertions = QueryAssertions{$($($base_assertions)*,)? ..Default::default()};
        let queries = [$(
            (
                $query_label,
                stringify!($query_label),
                stringify!($($query)*),
                QueryAssertions{$($($per_query_assertions)*,)? ..base_assertions}
            )
        ),*];
        $tester.assert_eval_multi(&logs, &queries);
    }};
}

#[test]
fn in_operator() {
    let mut index = test_index();
    let mut test_query = QueryTester::new(&mut index);

    assert_query_results! {
        test_query [ !(method in ["GET", "POST"]) || method = "PATCH" ]
        {instructions: 2}
        (method = "GET"): false,
        (method = "POST"): false,
        (method = "PATCH"): true,
        (worst = "POST"): true,
    }

    assert_query_results! {
        test_query [ method in ["GET", "POST"] ]
        {instructions: 1}
        (method = "GET"): true,
        (method = "POST"): true,
        (method = "PATCH"): false,
        (worst = "POST"): false,
    }

    assert_query_results! {
        test_query [ method in ["GET", "POST"] || method == 22 ]
        {instructions: 1}
        (method = "GET"): true,
        (method = "POST"): true,
        (method = "PATCH"): false,
        (method = 22): true
    }
}

#[test]
fn msg_query() {
    let mut index = test_index();
    let mut test_query = QueryTester::new(&mut index);
    assert_multi_query_results! {
        test_query {
            starts_with: [$message.starts_with("alp")],
            ends_with: [$message.ends_with("pha")],
            contains: [$message.contains("lph")],
            eq: [$message = "alpha"],
            not_eq: [$message != "alpha"],
            any: [$message in ["alpha", "beta", "canary"]],
        }
        {instructions: 1}
        (arb = 22, msg = "alpha"): starts_with | ends_with | contains | eq | any,
        (arb = 23, msg = "beta"): any | not_eq,
        (arb = 24): not_eq,
    }
}

#[test]
fn level_query() {
    let mut index = test_index();
    let mut test_query = QueryTester::new(&mut index);
    assert_multi_query_results! {
        test_query {
            all: [$level in [ "debug", "info", "warn", "error"]],
            just_debug: [$level = "DEBUG"],
            info_or_warn: [$level = "INFO" || $level == "WARN"],
            just_error: [$level = "error"],
        }
        {instructions: 1}
        ({level: LogLevel::Debug} msg = "tracing"): all | just_debug,
        ({level: LogLevel::Info} msg = "normal"): all | info_or_warn ,
        ({level: LogLevel::Warn} msg = "kinda bad but expected"): all | info_or_warn,
        ({level: LogLevel::Error} msg = "really bad"): all | just_error
    }
}

#[test]
fn target_query() {
    let mut index = test_index();
    let mut test_query = QueryTester::new(&mut index);
    assert_multi_query_results! {
        test_query {
            starts_with: [$target.starts_with("alp")],
            ends_with: [$target.ends_with("pha")],
            contains: [$target.contains("lph")],
            eq: [$target = "alpha"],
            not_eq: [$target != "alpha"],
            any: [$target in ["alpha", "beta", "canary"]],
        }
        {instructions: 1}
        (arb = 22, target = "alpha"): starts_with | ends_with | contains | eq | any,
        (arb = 23, target = "beta"): any | not_eq,
        (arb = 24): not_eq,
    }
}

#[test]
fn service_query() {
    let mut index = test_index();
    let mut test_query = QueryTester::new(&mut index);
    assert_multi_query_results! {
        test_query {
            starts_with: [$service.starts_with("alp")],
            ends_with: [$service.ends_with("pha")],
            contains: [$service.contains("lph")],
            eq: [$service = "alpha"],
            not_eq: [$service != "alpha"],
            any: [$service in ["alpha", "beta", "canary"]],
        }
        {instructions: 1}
        ({service: Some(ServiceId::intern("alpha"))} arb = 22): starts_with | ends_with | contains | eq | any,
        ({service: Some(ServiceId::intern("beta"))} arb = 23): any | not_eq,
        ({service: None} arb = 24): not_eq,
    }
}

const EMPTY: u64 = 0;

#[test]
fn never_seen_field() {
    let mut index = test_index();
    let mut test_query = QueryTester::new(&mut index);
    assert_multi_query_results! {
        test_query{
            never_seen: [ never_seen_field ],
            never_seen_starts_with: [ never_seen_field.starts_with("hello") ],
            never_seen_in: [ never_seen_field in ["hello", 1]],
            not_never_seen_exists: [ !never_seen_field.exists()],
            never_seen_or_seen: [ never_seen_field || arb ],
            never_seen_and_seen: [ never_seen_field && arb ],
        }
        {instructions: 1}
        (arb = 21): never_seen_or_seen | not_never_seen_exists,
        (count = 100): not_never_seen_exists
    }
}

#[test]
fn duration_ranges() {
    let mut index = test_index();
    let mut test_query = QueryTester::new(&mut index);
    assert_multi_query_results! {
        // Note Duration ranges are always floats so exactly equality is kinds sketchy
        // and not too useful, however we do support it and it works for certain values
        test_query{
            exclusive_range: [ elapsed > 2s && elapsed < 4s ],
            min_and_range: [ elapsed >= 2s && elapsed < 4s ],
            max_and_range: [ elapsed > 2s && elapsed <= 4s ],
            inclusive_range: [ elapsed >= 2s && elapsed <= 4s ],
        }
        {instructions: 1}
        (elapsed = Seconds(1.0)): EMPTY,
        (elapsed = Seconds(2.0)): inclusive_range | min_and_range,
        (elapsed = Seconds(3.0)): exclusive_range | inclusive_range | min_and_range | max_and_range,
        (elapsed = Seconds(4.0)): inclusive_range | max_and_range,
        (elapsed = Seconds(5.0)): EMPTY,
    }

    assert_multi_query_results! {
        test_query{
            lt:    [ elapsed < -2s],
            lt_eq: [ elapsed <= -2s],
            gt_eq: [ elapsed >= -2s],
            gt:    [ elapsed > -2s],
        }
        {instructions: 1}
        (elapsed = Seconds(-3.0)):    lt | lt_eq,
        (elapsed = Seconds(-2.0)): lt_eq | gt_eq,
        (elapsed = Seconds(-1.0)):    gt | gt_eq,
        (elapsed = Seconds(0.0)):     gt | gt_eq
    }
}

#[test]
fn comparisons() {
    let mut index = test_index();
    let mut test_query = QueryTester::new(&mut index);
    assert_multi_query_results! {
        test_query {
            lt:    [ only_int < -10 ],
            lt_eq: [ only_int <= -10 ],
            gt_eq: [ only_int >= -10 ],
            gt:    [ only_int > -10 ],
        }
        {instructions: 1}
        (only_int = -11):    lt | lt_eq,
        (only_int = -10): gt_eq | lt_eq,
        (only_int = -9):     gt | gt_eq,
    }

    assert_query_results! {
        test_query [ only_int > -10 && only_int < 100 ]
        {instructions: 1}
        (only_int = -32i32): false,
        (only_int = -5i32): true,
        (only_int = 10i32): true,
        (only_int = 23u32): true,
        (only_int = 2342101u32): false,
    }
    assert_query_results! {
        test_query [only_float >= -10.0 && only_float < 100.0 ]
        {instructions: 1}
        (only_float = -11f32): false,
        (only_float = -10.0): true,
        (only_float = -9.8f32): true,
        (only_float = 23.5f32): true,
        (only_float = 101f32): false,
    }
    assert_query_results! {
        test_query [only_float > -10 && only_float < 100 ]
        {instructions: 1}
        (only_float = -11f32): false,
        (only_float = -9.8f32): true,
        (only_float = 23.5f32): true,
        (only_float = 101f32): false,
    }
    assert_query_results! {
        test_query [ mixed_num > -10 && mixed_num < 100 ]
        {instructions: 1}
        (mixed_num = -11f32): false,
        (mixed_num = -9.8f32): true,
        (mixed_num = 23.5f32): true,
        (mixed_num = 101f32): false,
        (mixed_num = -10i32): false,
        (mixed_num = -32i32): false,
        (mixed_num = -5i32): true,
        (mixed_num = 10i32): true,
        (mixed_num = 23u32): true,
        (mixed_num = 2342101u32): false,
        (mixed_num = u64::MAX): false,
        (mixed_num = i64::MAX): false,
        (mixed_num = f64::MAX): false,
        (mixed_num = f64::MIN): false,
        (mixed_num = i64::MIN): false,
    }
    assert_multi_query_results! {
        test_query {
            lt:    [ mixed_num < -10 ],
            lt_eq: [ mixed_num <= -10 ],
            gt_eq: [ mixed_num >= -10 ],
            gt:    [ mixed_num > -10 ],
        }
        {instructions: 1}
        (mixed_num = f64::MIN):    lt | lt_eq,
        (mixed_num = i64::MIN):    lt | lt_eq,
        (mixed_num = -11f32):      lt | lt_eq,
        (mixed_num = -10.0001f32): lt | lt_eq,
        (mixed_num = -10i32):   lt_eq | gt_eq,
        (mixed_num = -9.8f32):     gt | gt_eq,
        (mixed_num = i64::MAX):    gt | gt_eq,
        (mixed_num = f64::MAX):    gt | gt_eq,
    }
}

#[test]
fn is_type() {
    let mut index = test_index();
    let mut test_query = QueryTester::new(&mut index);
    assert_multi_query_results! {
        test_query {
            is_string: [mega is String],
            is_string_or_none: [(mega is String) || (mega is None)],
            is_number: [mega is Number],
            is_int: [mega is Integer],
            is_float: [mega is Float],
            is_none: [mega is Null],
            is_duration: [mega is Duration],
            is_timestamp: [mega is Timestamp],
            is_uuid: [mega is UUID],
            is_bool: [mega is bool],
        }
        {instructions: 1}
        (mega = "hello"): is_string | is_string_or_none,
        (mega = 3): is_int | is_number,
        (mega = 3.14): is_float | is_number,
        (mega = 3.14): is_float | is_number,
        (mega = None::<u8>): is_string_or_none | is_none,
        (mega = Seconds(32.0)): is_duration,
        (mega = kvlog::Timestamp::from_millisecond(222)): is_timestamp,
        (mega = uuid::Uuid::from_u128(222)): is_uuid,
        (mega = false): is_bool,
        (mega = true): is_bool,
    }
}

fn ts(timestamp: &str) -> kvlog::Timestamp {
    let jiffed: jiff::Timestamp = timestamp.parse().unwrap();
    kvlog::Timestamp::from_millisecond(jiffed.as_millisecond())
}

#[test]
fn time_range() {
    let mut index = test_index();
    let mut test_query = QueryTester::new(&mut index);
    assert_multi_query_results! {
        test_query {
            sec: [time in time_range("2025-05-24T01:04:46Z")],
            day: [time in time_range("2025-05-24")],
            month: [time in time_range("2025-05")],
            year: [time in time_range("2025")],
        }
        {instructions: 1}
        (time = ts("2024-12-21T01:04:46Z")): EMPTY,
        (time = ts("2025-05-24T01:04:46Z")): sec | day | month | year,
        (time = ts("2025-05-24T01:04:46.999Z")): sec | day | month | year,
        (time = ts("2025-05-24T01:04:47Z")): day | month | year,
        (time = ts("2025-05-27T01:04:47Z")): month | year,
        (time = ts("2025-02-22T01:04:47Z")): year,
        (time = ts("2026-02-22T01:04:47Z")): EMPTY,
    }
}

#[test]
fn timestamp_meta_field() {
    let mut index = test_index();
    let mut tester = QueryTester::new(&mut index);
    assert_multi_query_results! {
        tester {
            y2024: [$timestamp in time_range("2024")],
            exact: [$timestamp in time_range("2026-12-21T01:04:46.123123123Z")],
        }
        {instructions: 1}
        [tester.set_time("2023-12-21T01:04:46Z")] (arb = 0): EMPTY,
        [tester.set_time("2024-12-21T01:04:46Z")] (arb = 10): y2024,
        [tester.set_time("2026-12-21T01:04:46.123123122Z")] (arb = 40): EMPTY,
        [tester.set_time("2026-12-21T01:04:46.123123123Z")] (arb = 20): exact,
        [tester.set_time("2026-12-21T01:04:46.123123124Z")] (arb = 30): EMPTY,
    }
}

#[test]
fn combo() {
    let mut index = test_index();
    let mut tester = QueryTester::new(&mut index);
    assert_multi_query_results! {
        tester {
            error_ish: [$level == "error" || err || msg.contains("fail")],
        }
        (msg = "Happy", status = 200): EMPTY,
        (msg = "failure", status = 500): error_ish,
        ({level: LogLevel::Error} arb = 31): error_ish,
        ({level: LogLevel::Warn} msg = "fail", arb = 31, err="badness"): error_ish,
        ({level: LogLevel::Info} msg = "not so bad", arb = 31, err="badness"): error_ish,
    }
}

use std::u16;

use ahash::HashSet;
use hashbrown::HashMap;
use kvlog::encoding::Key;
use libc::posix_spawnattr_t;
use uuid::Uuid;

use super::*;
use crate::{
    index::{self, BucketGuard, FieldKind, FieldKindSet, IntermentMaps},
    query::{
        parse_query,
        query_parts::{FieldPredicate, FieldTest, Pred, PredBuilder},
        QueryExpr, QueryParseError,
    },
    ServiceId,
};

// let im = bucket.maps();
// // im.field_uuid(bucket, key, value)

fn compile_field_tests<'b>(
    Compiler { asm, maps, bucket, bump, .. }: &mut Compiler<'_, 'b>,
    parent: BlockIdx,
    mut and: bool, /* if false then tests are OR'd together */
    mut fail: Ret,
    mut succ: Ret,
    key: KeyID,
    tests: &[FieldTest<'b>],
) -> Result<(), VmCompileError> {
    if and {
        std::mem::swap(&mut fail, &mut succ);
    }

    let types = maps.keys.get(key).map(|info| info.kinds).unwrap_or_default();
    let mut field_eqs: Vec<Field> = Vec::new();
    let bb = asm.block(Some(parent), fail);
    for test in tests {
        let mut fail_t = Ret::Next(bb);
        let mut succ_t = succ;
        if and ^ test.is_negated() {
            std::mem::swap(&mut fail_t, &mut succ_t);
        }

        match test {
            FieldTest::AnyRaw(negated, fields) => {
                let mut start = field_eqs.len();
                if *negated != and {
                    match fields {
                        [] => (),
                        [single] => {
                            asm.field_eq(bb, fail_t, succ_t, key, *single);
                        }
                        many => {
                            let bb = asm.block(Some(bb), fail_t);
                            fail_t = Ret::Next(bb);
                            succ_t = succ_t;
                            asm.field_eq_or(bb, fail_t, succ_t, key, many);
                        }
                    }
                } else {
                    field_eqs.extend_from_slice(fields);
                };
            }
            FieldTest::EqRaw(negated, field) => {
                if and == *negated {
                    field_eqs.push(*field);
                } else {
                    asm.field_eq(bb, fail_t, succ_t, key, *field);
                }
            }
            FieldTest::TextStartsWith(_, value) => {
                asm.field_starts_with(bb, fail_t, succ_t, key, value.as_bytes());
            }
            FieldTest::TextEndWith(_, value) => {
                asm.field_ends_with(bb, fail_t, succ_t, key, value.as_bytes());
            }
            FieldTest::TextContains(_, value) => {
                asm.field_contains_text(bb, fail_t, succ_t, key, value.as_bytes(), bump);
            }
            FieldTest::Defined(_) => {
                asm.field_exists(bb, fail_t, succ_t, key);
            }
            FieldTest::RangeRaw { min, max, .. } => {
                asm.field_raw_range(bb, fail_t, succ_t, key, *min, *max);
            }
            FieldTest::FloatRange { min, max, .. } => {
                asm.field_float_range(bb, fail_t, succ_t, key, *min, *max);
            }
            FieldTest::Type(_, set) => {
                asm.field_type_in_set(bb, fail_t, succ_t, key, *set);
            }
            FieldTest::BytesEqual(_, items) => {
                return Err(VmCompileError::UnsupportedFieldTest(FieldTestKind::BytesEqual))
            }
            FieldTest::IsTrue(_) => return Err(VmCompileError::UnsupportedFieldTest(FieldTestKind::IsTrue)),
            FieldTest::NullValue(_) => return Err(VmCompileError::UnsupportedFieldTest(FieldTestKind::NullValue)),
            FieldTest::FiniteFloat(_) => return Err(VmCompileError::UnsupportedFieldTest(FieldTestKind::FiniteFloat)),
            FieldTest::U64Eq(_, _) => return Err(VmCompileError::UnsupportedFieldTest(FieldTestKind::U64Eq)),
            FieldTest::DurationRange { negated, min_seconds, max_seconds } => {
                return Err(VmCompileError::UnsupportedFieldTest(FieldTestKind::DurationRange))
            }
            FieldTest::TextAny(..) => return Err(VmCompileError::UnsupportedFieldTest(FieldTestKind::TextAny)),
            FieldTest::TextEqual(negated, value) => {
                return Err(VmCompileError::UnsupportedFieldTest(FieldTestKind::TextEqual))
            }
            FieldTest::I64Eq(negated, value) => return Err(VmCompileError::UnsupportedFieldTest(FieldTestKind::I64Eq)),
            FieldTest::TimeRange { negated, min_ns, max_ns } => {
                return Err(VmCompileError::UnsupportedFieldTest(FieldTestKind::TimeRange))
            }
        }
    }
    if !field_eqs.is_empty() {
        asm.field_eq_or(bb, fail, succ, key, &field_eqs);
    }
    Ok(())
}

fn compile_pred<'b>(
    comp: &mut Compiler<'_, 'b>,
    bb: BlockIdx,
    mut fail: Ret,
    mut succ: Ret,
    pred: &Pred<'b>,
) -> Result<(), VmCompileError> {
    match pred {
        Pred::Field(key_id, field_test) => {
            compile_field_tests(comp, bb, false, fail, succ, *key_id, std::slice::from_ref(field_test));
        }
        Pred::FieldOr(key_id, field_tests) => {
            compile_field_tests(comp, bb, false, fail, succ, *key_id, field_tests);
        }
        Pred::FieldAnd(key_id, field_tests) => {
            compile_field_tests(comp, bb, true, fail, succ, *key_id, field_tests);
        }
        Pred::And(preds) => {
            let inner_block = comp.asm.block(Some(bb), succ);
            for pred in *preds {
                compile_pred(comp, inner_block, fail, Ret::Next(inner_block), pred);
            }
        }
        Pred::Or(preds) => {
            let inner_block = comp.asm.block(Some(bb), fail);
            for pred in *preds {
                compile_pred(comp, inner_block, Ret::Next(inner_block), succ, pred);
            }
        }
        Pred::HasParentSpan(_) => return Err(VmCompileError::UnsupportedPredicate(PredKind::HasParentSpan)),
        Pred::SpanIs(_, _) => return Err(VmCompileError::UnsupportedPredicate(PredKind::SpanIs)),
        Pred::ParentSpanIs(_, _) => return Err(VmCompileError::UnsupportedPredicate(PredKind::ParentSpanIs)),
        Pred::HasSpan(negated) => {
            if *negated {
                std::mem::swap(&mut fail, &mut succ);
            }
            comp.asm.any_flag(bb, fail, succ, 1 << 4);
        }
        Pred::SpanDurationRange { negated, min_ns, max_ns } => {
            if *negated {
                std::mem::swap(&mut fail, &mut succ);
            }
            comp.asm.span_duration_range(bb, fail, succ, *min_ns, *max_ns);
        }
        Pred::TimestampRange { negated, min_ns, max_ns } => {
            if *negated {
                std::mem::swap(&mut fail, &mut succ);
            }
            comp.asm.timestamp_range(bb, fail, succ, *min_ns, *max_ns);
        }
        Pred::LevelMask(negated, mask) => {
            if *negated {
                std::mem::swap(&mut fail, &mut succ);
            }
            comp.asm.any_flag(bb, fail, succ, *mask as u16);
        }
        Pred::Target(field_test) => {
            // todo could optimize this a lot, but probably won't get used much
            let len = comp.targets.len();
            let buffer = comp.bump.alloc_slice_fill_copy((len + 63) / 64, 0u64);
            for (id, value) in comp.targets.iter() {
                if field_test.matches_text(value.as_bytes()) {
                    buffer[(id >> 6) as usize] |= 1u64 << (id & 0b111111);
                }
            }
            comp.asm.push_inst(bb, Op::InTargetSet, 0, len as u16, fail, succ);
            comp.asm.code.push(DataRepr { ptr: buffer.as_ptr() as *const u8 })
        }
        Pred::Service(field_test) => {
            let mut service_set = [0u64; 4];
            let negated = field_test.is_negated();
            for service in ServiceId::known() {
                // we xor negated because we want to negate use swap
                // which will handle unknown/None services better.
                if field_test.matches_text(service.as_str().as_bytes()) ^ negated {
                    service_set_insert(service.as_u8(), &mut service_set);
                }
            }
            if negated {
                std::mem::swap(&mut fail, &mut succ);
            }
            comp.asm.push_inst(bb, Op::InServiceSet, 0, 0, fail, succ);
            for s in service_set {
                comp.asm.code.push(DataRepr { raw: s })
            }
        }
        Pred::Message(field_test) => {
            if field_test.is_negated() {
                std::mem::swap(&mut fail, &mut succ);
            }
            match field_test {
                FieldTest::BytesEqual(_, text) => comp.asm.msg_eq(bb, fail, succ, text),
                FieldTest::TextEqual(_, text) => comp.asm.msg_eq(bb, fail, succ, text.as_bytes()),
                FieldTest::TextAny(_, texts) => comp.asm.msg_any(bb, fail, succ, texts),
                FieldTest::TextContains(_, needle) => {
                    comp.asm.msg_contains(bb, fail, succ, needle.as_bytes(), comp.bump)
                }
                FieldTest::TextStartsWith(_, needle) => comp.asm.msg_starts_with(bb, fail, succ, needle.as_bytes()),
                FieldTest::TextEndWith(_, needle) => comp.asm.msg_ends_with(bb, fail, succ, needle.as_bytes()),
                _ => return Err(VmCompileError::UnsupportedMessageTest),
            };
        }
    }
    Ok(())
}

struct Compiler<'a, 'b> {
    bucket: &'a BucketGuard<'a>,
    maps: &'a IntermentMaps,
    asm: Assembler<'b>,
    bump: &'b Bump,
    targets: Mapper<'a>,
}

pub fn compile<'a, 'b>(
    bump: &'b Bump,
    pred: Pred<'b>,
    bucket: &BucketGuard<'a>,
    targets: Mapper<'a>,
) -> Result<QueryVm<'b>, VmCompileError> {
    let maps = bucket.maps();
    let mut compiler = Compiler { bucket, maps: &*maps, asm: Assembler::default(), bump, targets };
    let bb = compiler.asm.block(None, Ret::False);
    compile_pred(&mut compiler, bb, Ret::False, Ret::True, &pred)?;
    compiler.asm.build()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::index::filter::archetype_filter::{build_pred_addr_map, ArchetypePrefilter, Match};
    use crate::index::test::{test_index, TestIndexWriter};
    use crate::index::{Bucket, Index, WeakLogEntry};
    use crate::log;
    use crate::query::query_parts::parser::Parser;
    use crate::query::query_parts::OptimizationInfo;
    use hashbrown::HashMap;
    use kvlog::encoding::{FieldBuffer, Seconds};
    use kvlog::Encode;

    fn key_mask(key: &str) -> u64 {
        let key = KeyID::intern(key);
        let field = Field::new(key.raw(), crate::index::FieldKind::Bool, 0);
        field.raw & !INV_KEY_MASK
    }
    #[test]
    fn simple_compliation() {
        let mut index = test_index();
        KeyID::intern("k1");
        KeyID::intern("k2");
        KeyID::intern("k3");
        KeyID::intern("k4");
        KeyID::intern("k5");
        let reader = index.reader().clone();
        let mut writer = TestIndexWriter::new(&mut index);
        let _ = log!(writer; msg="Initialize bucket");
        let mut bucket = reader.newest_bucket().unwrap();
        macro_rules! entry {
            ($($tt:tt)*) => {
                {
                    let w0 = log!(writer; $($tt)*);
                    bucket.renew();
                    bucket.upgrade(w0).unwrap()
                }
            };
        }

        let uuid: Uuid = "2c9e0493-5f31-4ee3-9326-5f05ca83d639".parse().unwrap();
        let e1 = log!(writer; k5 = "hello");
        let e1 = log!(writer; k1 = "hello", k2 = 48, k1 =32);
        let e2 = log!(writer; k1 = uuid);
        let bump = bumpalo::Bump::new();
        bucket.renew();
        let ogr = parse_query(&bump, "k1 in [23,43,53]").unwrap();
        let rules = ogr.build_with_opt(&bump, &bucket, &*bucket.maps());
        // let rules = parse_query(&bump, stringify!(k1 = 32 && k2 = 22 && k1 == 43)).unwrap();
        let rules_un_opt = ogr.build(&bump).pred().unwrap();
        let rules = rules.pred().unwrap();
        let thing = build_pred_addr_map(&rules);

        let mut filter = ArchetypePrefilter {
            pred_addr_map: build_pred_addr_map(&rules),
            always_mask: [0; 2],
            bucket: &bucket,
            targets: index.reader().targets.mapper(),
        };

        let archetypes = unsafe { filter.bucket.archetypes_bucket_lifetime() };
        let mut maxi = [u64::MAX; 2];
        for (i, arch) in archetypes.iter().enumerate() {
            filter.always_mask = [0; 2];
            let value = filter.implicate(&rules, arch);
            unsafe {
                arch.print(&filter.bucket);
            }
            if value != Match::AlwaysFalse {
                maxi[0] &= filter.always_mask[0];
                maxi[1] &= filter.always_mask[1];
            }
        }
        println!("{:?}", maxi);
        let mut opt = OptimizationInfo { pred_addr_trival_map: Default::default() };
        for (&key, &mask) in &filter.pred_addr_map {
            if mask & maxi[0] != 0 {
                opt.pred_addr_trival_map.insert(key, false);
            } else if mask & maxi[1] != 0 {
                opt.pred_addr_trival_map.insert(key, true);
            }
        }

        explore(&filter.pred_addr_map, &rules, &mut |entry, mask| {
            if mask & maxi[0] != 0 {
                println!("ALWAYS_FALSE: {:?}", entry);
            }
            if mask & maxi[1] != 0 {
                println!("ALWAYS_TRUE: {:?}", entry);
            }
        });

        let final_rules = rules.reduce(&bump, &opt);
        println!("{:#?}", final_rules);

        // println!("{}", thing.len());
        // println!("{:#?}", rules);
        // println!("-----------");
        // vm.print();
    }

    #[derive(Debug)]
    enum Entry<'a> {
        FieldTest(&'a FieldTest<'a>),
        Predicate(&'a Pred<'a>),
    }

    pub fn explore(map: &HashMap<usize, u64>, pred: &Pred<'_>, mut func: &mut dyn FnMut(Entry, u64)) {
        if let Some(mask) = map.get(&<*const Pred>::addr(pred)) {
            func(Entry::Predicate(pred), *mask);
        }
        match pred {
            Pred::Field(key_id, field_test) => {
                return;
            }
            Pred::FieldOr(_, field_tests) | Pred::FieldAnd(_, field_tests) => {
                for field_test in *field_tests {
                    if let Some(mask) = map.get(&<*const Pred>::addr(pred)) {
                        func(Entry::Predicate(pred), *mask);
                    }
                }
            }
            Pred::And(preds) | Pred::Or(preds) => {
                for pred in *preds {
                    explore(map, pred, func);
                }
            }
            _ => (),
        }
    }
}

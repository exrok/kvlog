use bumpalo::Bump;
use hashbrown::HashMap;
use jsony::Jsony;

use crate::{
    accel::U16Set,
    field_table,
    index::{
        archetype::{self, Archetype},
        Bucket, BucketGuard,
    },
    query::query_parts::{FieldKey, FieldTest, OptimizationInfo, Pred, PredBuilder},
    shared_interner::Mapper,
};

use super::LevelFilter;

fn has(field_key: &FieldKey, arch: &archetype::Archetype) -> bool {
    match field_key {
        FieldKey::Seen(key_id) => arch.contains_key(*key_id),
        FieldKey::New(_) => false,
    }
}

// values chosen to allow direct mapping from bool
#[derive(Debug, Jsony, PartialEq, Eq, Clone, Copy)]
#[jsony(ToStr)]
pub enum Match {
    AlwaysTrue = 1,
    Inconclusive = 2,
    AlwaysFalse = 0,
}
impl From<bool> for Match {
    fn from(value: bool) -> Self {
        if value {
            Match::AlwaysTrue
        } else {
            Match::AlwaysFalse
        }
    }
}

impl Match {
    fn negate(self) -> Match {
        match self {
            Match::AlwaysTrue => Match::AlwaysFalse,
            Match::Inconclusive => Match::Inconclusive,
            Match::AlwaysFalse => Match::AlwaysTrue,
        }
    }
}

pub fn build_pred_addr_map_rec(map: &mut HashMap<usize, u64>, pred: &Pred<'_>) {
    macro_rules! record {
        ($ty: ident, $expr: expr) => {{
            let len = map.len();
            if len > 62 {
                return;
            }
            let addr = <*const $ty>::addr($expr);
            map.insert(addr, 1 << len);
        }};
    }
    record!(Pred, pred);
    match pred {
        Pred::FieldOr(_, field_tests) | Pred::FieldAnd(_, field_tests) => {
            for field_test in *field_tests {
                record!(FieldTest, field_test);
            }
        }
        Pred::And(preds) | Pred::Or(preds) => {
            for pred in *preds {
                build_pred_addr_map_rec(map, pred);
            }
        }
        _ => (),
    }
}

pub fn build_pred_addr_map(pred: &Pred<'_>) -> HashMap<usize, u64> {
    let mut map = HashMap::new();
    build_pred_addr_map_rec(&mut map, pred);
    map
}

pub fn specialize_on_archetypes<'b>(
    bump: &'b Bump,
    bucket: BucketGuard<'_>,
    mut pred: &Pred<'b>,
    targets: Mapper<'_>,
) -> (Box<U16Set>, Result<Pred<'b>, bool>) {
    let mut possible_archetypes: Box<U16Set> = Default::default();

    let mut filter =
        ArchetypePrefilter { pred_addr_map: build_pred_addr_map(pred), always_mask: [0; 2], bucket: &bucket, targets };
    let archetypes = unsafe { filter.bucket.archetypes_bucket_lifetime() };

    let mut always_mask = [u64::MAX; 2];
    for (archetype_id, archetype) in archetypes.iter().enumerate() {
        filter.always_mask = [0; 2];
        if filter.implicate(&pred, archetype) != Match::AlwaysFalse {
            possible_archetypes.insert(archetype_id as u16);
            always_mask[0] &= filter.always_mask[0];
            always_mask[1] &= filter.always_mask[1];
        }
    }

    if always_mask != [u64::MAX; 2] {
        let mut opt = OptimizationInfo { pred_addr_trival_map: Default::default() };
        for (&key, &mask) in &filter.pred_addr_map {
            if mask & always_mask[0] != 0 {
                opt.pred_addr_trival_map.insert(key, false);
            } else if mask & always_mask[1] != 0 {
                opt.pred_addr_trival_map.insert(key, true);
            }
        }
        match pred.reduce(bump, &opt) {
            crate::query::query_parts::PredBuildResult::AlwaysTrue => return (possible_archetypes, Err(true)),
            crate::query::query_parts::PredBuildResult::AlwaysFalse => return (possible_archetypes, Err(false)),
            crate::query::query_parts::PredBuildResult::Ok(pred) => return (possible_archetypes, Ok(pred)),
        }
    }

    (possible_archetypes, Ok(*pred))
}

pub struct ArchetypePrefilter<'a> {
    pub pred_addr_map: HashMap<usize, u64>,
    pub always_mask: [u64; 2],
    pub bucket: &'a BucketGuard<'a>,
    pub targets: Mapper<'a>,
}

impl<'a> ArchetypePrefilter<'a> {
    pub fn implicate_field_test(&mut self, key_contained: bool, test: &FieldTest<'_>) -> Match {
        let result = match test {
            FieldTest::Defined(negated) => Match::from(key_contained ^ *negated),
            _ if key_contained => return Match::Inconclusive,
            _ => Match::from(test.is_negated()),
        };
        let always = match result {
            Match::AlwaysTrue => true,
            Match::Inconclusive => return result,
            Match::AlwaysFalse => false,
        };

        if let Some(mask) = self.pred_addr_map.get(&(<*const FieldTest>::addr(test))) {
            self.always_mask[always as usize] |= *mask;
        }

        result
    }
    pub fn implicate(&mut self, pred: &Pred<'_>, arch: &'a Archetype) -> Match {
        let result = match pred {
            Pred::Field(key, test) => {
                // we inline the single case we don't want to bother checking for addr
                let key_contained = arch.contains_key(*key);
                match test {
                    FieldTest::Defined(negated) => Match::from(key_contained ^ *negated),
                    _ if key_contained => return Match::Inconclusive,
                    _ => Match::from(test.is_negated()),
                }
            }
            Pred::FieldOr(key, tests) => 'check: {
                let mut overall = Match::AlwaysFalse;
                let key_contained = arch.contains_key(*key);
                let mut tests = tests.iter();
                while let Some(test) = tests.next() {
                    match self.implicate_field_test(key_contained, test) {
                        Match::AlwaysTrue => {
                            for skipped_test in tests {
                                let _ = self.implicate_field_test(key_contained, skipped_test);
                            }

                            break 'check Match::AlwaysTrue;
                        }
                        Match::Inconclusive => {
                            overall = Match::Inconclusive;
                        }
                        Match::AlwaysFalse => continue,
                    }
                }

                overall
            }
            Pred::FieldAnd(key, tests) => 'check: {
                let mut overall = Match::AlwaysTrue;
                let key_contained = arch.contains_key(*key);
                let mut tests = tests.iter();
                while let Some(test) = tests.next() {
                    match self.implicate_field_test(key_contained, test) {
                        Match::AlwaysFalse => {
                            println!("FAILED >> {:?}", test);
                            // We do this record the
                            for skipped_test in tests {
                                let _ = self.implicate_field_test(key_contained, skipped_test);
                            }

                            break 'check Match::AlwaysFalse;
                        }
                        Match::Inconclusive => {
                            overall = Match::Inconclusive;
                        }
                        Match::AlwaysTrue => continue,
                    }
                }

                overall
            }
            Pred::And(preds) => 'check: {
                let mut overall = Match::AlwaysTrue;
                let mut preds = preds.iter();
                while let Some(pred) = preds.next() {
                    match self.implicate(pred, arch) {
                        Match::AlwaysFalse => {
                            // We do this record the
                            for skipped_pred in preds {
                                let _ = self.implicate(skipped_pred, arch);
                            }
                            break 'check Match::AlwaysFalse;
                        }
                        Match::Inconclusive => {
                            overall = Match::Inconclusive;
                        }
                        Match::AlwaysTrue => continue,
                    }
                }

                overall
            }
            Pred::Or(preds) => 'check: {
                let mut overall = Match::AlwaysFalse;
                let mut preds = preds.iter();
                while let Some(pred) = preds.next() {
                    match self.implicate(pred, arch) {
                        Match::AlwaysTrue => {
                            // We do this record the
                            for skipped_pred in preds {
                                let _ = self.implicate(skipped_pred, arch);
                            }

                            break 'check Match::AlwaysTrue;
                        }
                        Match::Inconclusive => {
                            overall = Match::Inconclusive;
                        }
                        Match::AlwaysFalse => continue,
                    }
                }

                overall
            }
            Pred::SpanIs(negated, _)
            | Pred::ParentSpanIs(negated, _)
            | Pred::HasParentSpan(negated)
            | Pred::SpanDurationRange { negated, .. } => {
                if arch.in_span() {
                    return Match::Inconclusive;
                } else if *negated {
                    Match::AlwaysTrue
                } else {
                    Match::AlwaysFalse
                }
            }
            Pred::HasSpan(negated) => {
                if arch.in_span() {
                    Match::AlwaysTrue
                } else if *negated {
                    Match::AlwaysTrue
                } else {
                    Match::AlwaysFalse
                }
            }
            Pred::LevelMask(negated, mask) => {
                if arch.level_in(LevelFilter { mask: *mask }) {
                    if *negated {
                        Match::AlwaysFalse
                    } else {
                        Match::AlwaysTrue
                    }
                } else {
                    if *negated {
                        Match::AlwaysTrue
                    } else {
                        Match::AlwaysFalse
                    }
                }
            }
            Pred::Target(field_test) => {
                if let Some(target) = self.targets.get(arch.target_id) {
                    eval_text_field_test(target.as_bytes(), &field_test)
                } else {
                    if field_test.is_negated() {
                        Match::AlwaysTrue
                    } else {
                        Match::AlwaysFalse
                    }
                }
            }
            Pred::Message(field_test) => {
                let msg = unsafe { arch.message(&self.bucket) };
                let res = eval_text_field_test(msg, &field_test);
                res
            }
            Pred::Service(field_test) => {
                if let Some(service) = arch.service {
                    eval_text_field_test(service.as_str().as_bytes(), &field_test)
                } else {
                    if field_test.is_negated() {
                        Match::AlwaysTrue
                    } else {
                        Match::AlwaysFalse
                    }
                }
            }
            Pred::TimestampRange { .. } => {
                return Match::Inconclusive;
            }
        };
        let always = match result {
            Match::AlwaysTrue => true,
            Match::Inconclusive => return result,
            Match::AlwaysFalse => false,
        };

        if let Some(mask) = self.pred_addr_map.get(&(<*const Pred>::addr(pred))) {
            self.always_mask[always as usize] |= *mask;
        }

        return result;
    }
}

fn eval_text_field_test(bytes: &[u8], field_test: &FieldTest) -> Match {
    Match::from(field_test.matches_text(bytes))
}

use crate::index::f48::f64_to_f48;

use super::*;
use bumpalo::collections::Vec as BumpVec;
use kvlog::LogLevel;

pub struct Parser<'a, 'bump> {
    pub bump: &'bump Bump,
    pub tokens: TokenStream<'bump>,
    pub span_info: SpanInfo<'bump>,
    pub source: &'a str,
}

macro_rules! throw {
    ($error_name:ident $({ $($fields:tt)* })? $(( $($tuple:tt)* ))? $(@ $ctx: expr)?) => {
        return Err(InternalQueryError::from((
            ErrorKind::$error_name $({$($fields)*})?  $(($($tuple)*))?
            $(, $ctx)?
        )))
    };
}

enum MergeResult {
    Merged,
    AlwaysTrue(usize),
    AlwaysFalse(usize),
    Unmerged,
}

fn try_field_test_merge<'b>(tests: &mut [FieldTest<'b>], merge_candidate: FieldTest<'b>, is_and: bool) -> MergeResult {
    match merge_candidate {
        FieldTest::Type(false, set) if !is_and => {
            for (i, test) in tests.iter_mut().enumerate() {
                if let FieldTest::Type(false, other) = test {
                    *other |= set;
                    return MergeResult::Merged;
                }
            }
        }
        FieldTest::FloatRange { negated: a_neg, min: a_min, max: a_max } => {
            let a = range_eval::RangePredicate::<f64> { negated: a_neg, min: a_min, max: a_max };
            for (i, test) in tests.iter_mut().enumerate() {
                if let FieldTest::FloatRange { negated: b_neg, min: b_min, max: b_max } = *test {
                    let b = range_eval::RangePredicate::<f64> { negated: b_neg, min: b_min, max: b_max };
                    return match range_eval::range_merge(a, b, f64::MIN, f64::MAX, true) {
                        range_eval::RangeMerge::None => continue,
                        range_eval::RangeMerge::Merged(range_predicate) => {
                            *test = FieldTest::FloatRange {
                                negated: range_predicate.negated,
                                min: range_predicate.min,
                                max: range_predicate.max,
                            };
                            return MergeResult::Merged;
                        }
                        range_eval::RangeMerge::AlwaysTrue => MergeResult::AlwaysTrue(i),
                        range_eval::RangeMerge::AlwaysFalse => MergeResult::AlwaysFalse(i),
                    };
                }
            }
        }
        FieldTest::DurationRange { negated: a_neg, min_seconds: a_min, max_seconds: a_max } => {
            let a = range_eval::RangePredicate::<u64> { negated: a_neg, min: a_min, max: a_max };
            for (i, test) in tests.iter_mut().enumerate() {
                if let FieldTest::DurationRange { negated: b_neg, min_seconds: b_min, max_seconds: b_max } = *test {
                    let b = range_eval::RangePredicate::<u64> { negated: b_neg, min: b_min, max: b_max };
                    return match range_eval::range_merge(a, b, 0, (1 << 48) - 1, true) {
                        range_eval::RangeMerge::None => continue,
                        range_eval::RangeMerge::Merged(range_predicate) => {
                            *test = FieldTest::DurationRange {
                                negated: range_predicate.negated,
                                min_seconds: range_predicate.min,
                                max_seconds: range_predicate.max,
                            };
                            return MergeResult::Merged;
                        }
                        range_eval::RangeMerge::AlwaysTrue => MergeResult::AlwaysTrue(i),
                        range_eval::RangeMerge::AlwaysFalse => MergeResult::AlwaysFalse(i),
                    };
                }
            }
        }
        _ => {}
    }

    MergeResult::Unmerged
}

fn level_text_to_mask(text: &str) -> Option<u8> {
    let level = match text {
        "warn" | "WARN" | "warning" => LogLevel::Warn,
        "err" | "ERR" | "ERROR" | "error" => LogLevel::Error,
        "info" | "INFO" | "information" => LogLevel::Info,
        "debug" | "DEBUG" => LogLevel::Debug,
        _ => return None,
    };
    Some(1 << level as u8)
}

fn type_text_to_mask(text: &str) -> Option<FieldKindSet> {
    let mut buf = [0u8; 16];
    if text.len() > 16 {
        return None;
    }
    for (i, ch) in text.as_bytes().iter().enumerate() {
        buf[i] = ch.to_ascii_lowercase();
    }

    let text = text.to_ascii_lowercase();

    let set = match &buf[..text.len()] {
        b"number" => FieldKindSet { raw: ANY_NUMBER_TYPE },
        b"integer" | b"int" => FieldKindSet { raw: ANY_INTEGER_TYPE },
        b"float" => FieldKindSet { raw: ANY_FLOAT_TYPE },
        b"bool" | b"boolean" => FieldKind::Bool.into(),
        b"string" => FieldKind::String.into(),
        b"duration" => FieldKind::Seconds.into(),
        b"timestamp" => FieldKind::Timestamp.into(),
        b"none" | b"null" => FieldKind::None.into(),
        b"uuid" => FieldKind::UUID.into(),

        _ => return None,
    };
    return Some(set);
}

#[derive(Clone, Copy)]
enum MetaField {
    Span,
    ParentSpan,
    SpanDuration,
    Message,
    Target,
    Level,
    Service,
    Timestamp,
}

fn finialize_field_key<'b>(key: &mut FieldKey<'b>, parser: &Parser) -> Result<Option<MetaField>, InternalQueryError> {
    let FieldKey::New(text) = key else {
        return Ok(None);
    };
    if let Some(metavar) = text.strip_prefix("$") {
        match metavar {
            "span" => return Ok(Some(MetaField::Span)),
            "span_duration" => return Ok(Some(MetaField::SpanDuration)),
            "parent_span" => return Ok(Some(MetaField::ParentSpan)),
            "message" => return Ok(Some(MetaField::Message)),
            "service" => return Ok(Some(MetaField::Service)),
            "target" => return Ok(Some(MetaField::Target)),
            "level" => return Ok(Some(MetaField::Level)),
            "timestamp" => return Ok(Some(MetaField::Timestamp)),
            _ => throw!(UnknownMetaVar @ parser.span_info.text_span(metavar)),
        }
    }
    match *text {
        "msg" => return Ok(Some(MetaField::Message)),
        "target" => return Ok(Some(MetaField::Target)),
        _ => (),
    }
    *key = FieldKey::from(*text);
    Ok(None)
}

fn field_to_level_mask(test: &FieldTest, parser: &Parser) -> Result<u8, InternalQueryError> {
    match test {
        FieldTest::TextEqual(negated, text) => {
            if let Some(mask) = level_text_to_mask(text) {
                Ok(mask ^ if *negated { 0b1111 } else { 0 })
            } else {
                throw!(UnknownLogLevel @ parser.span_info.text_span(text))
            }
        }
        FieldTest::TextAny(negated, items) => {
            let mut mask = 0;
            for text in *items {
                if let Some(item_mask) = level_text_to_mask(text) {
                    mask |= item_mask;
                } else {
                    throw!(UnknownLogLevel @ parser.span_info.text_span(text))
                }
            }
            Ok(mask ^ if *negated { 0b1111 } else { 0 })
        }
        _ => throw!(ExpectedLevelFilter),
    }
}

fn field_test_supports_text_field(test: &FieldTest) -> bool {
    match test {
        FieldTest::Defined(_)
        | FieldTest::TextEqual(_, _)
        | FieldTest::TextAny(_, _)
        | FieldTest::TextContains(_, _)
        | FieldTest::TextStartsWith(_, _)
        | FieldTest::TextEndWith(_, _) => true,
        _ => false,
    }
}

fn f48_seconds_to_u64_nanos_clamped(sec: u64) -> u64 {
    let seconds = f48_to_f64(sec);
    let nanos = seconds.abs() * 1_000_000_000.0;
    if nanos > u64::MAX as f64 || sec >= (1u64 << 48) - 1 {
        return u64::MAX;
    }
    if nanos < 0.0 || sec == 0 {
        return 0;
    }
    nanos as u64
}

fn meta_field_test_to_pred<'b>(
    key: &FieldKey,
    meta: MetaField,
    test: &FieldTest<'b>,
    parser: &Parser<'_, 'b>,
) -> Result<PredBuilder<'b>, InternalQueryError> {
    let pred = match meta {
        MetaField::Span => match *test {
            FieldTest::Defined(negated) => PredBuilder::HasSpan(negated),
            FieldTest::I64Eq(negated, value) => PredBuilder::SpanIs(negated, value as u64),
            FieldTest::U64Eq(negated, value) => PredBuilder::SpanIs(negated, value as u64),
            _ => throw!(UnsupportedOperationOnSpan @ parser.span_info.text_span(key.as_str())),
        },
        MetaField::SpanDuration => match *test {
            FieldTest::DurationRange { negated, min_seconds, max_seconds } => PredBuilder::SpanDurationRange {
                negated,
                min_ns: f48_seconds_to_u64_nanos_clamped(min_seconds),
                max_ns: f48_seconds_to_u64_nanos_clamped(max_seconds),
            },
            _ => throw!(UnsupportedOperationOnSpan @ parser.span_info.text_span(key.as_str())),
        },
        MetaField::ParentSpan => match *test {
            FieldTest::Defined(negated) => PredBuilder::HasParentSpan(negated),
            FieldTest::I64Eq(negated, value) => PredBuilder::ParentSpanIs(negated, value as u64),
            FieldTest::U64Eq(negated, value) => PredBuilder::ParentSpanIs(negated, value as u64),
            _ => throw!(UnsupportedOperationOnSpan @ parser.span_info.text_span(key.as_str())),
        },
        MetaField::Level => {
            let mask = field_to_level_mask(test, parser)?;
            PredBuilder::LevelMask(false, mask)
        }
        MetaField::Timestamp => match *test {
            FieldTest::TimeRange { negated, min_ns, max_ns } => {
                PredBuilder::TimestampRange { negated, min_ns: min_ns.unsigned_abs(), max_ns: max_ns.unsigned_abs() }
            }
            FieldTest::I64Eq(negated, value) => {
                PredBuilder::TimestampRange { negated, min_ns: value.unsigned_abs(), max_ns: value.unsigned_abs() }
            }
            FieldTest::U64Eq(negated, value) => PredBuilder::TimestampRange { negated, min_ns: value, max_ns: value },
            _ => throw!(UnsupportedOperationOnTimestamp @ parser.span_info.text_span(key.as_str())),
        },
        MetaField::Message => {
            if !field_test_supports_text_field(test) {
                throw!(UnsupportedOperationOnMessage @ parser.span_info.text_span(key.as_str()))
            }
            PredBuilder::Message(*test)
        }
        MetaField::Target => {
            if !field_test_supports_text_field(test) {
                throw!(UnsupportedOperationOnTarget @ parser.span_info.text_span(key.as_str()))
            }
            PredBuilder::Target(*test)
        }
        MetaField::Service => {
            if !field_test_supports_text_field(test) {
                throw!(UnsupportedOperationOnService @ parser.span_info.text_span(key.as_str()))
            }
            PredBuilder::Service(*test)
        }
    };
    Ok(pred)
}

impl<'b> PredBuilder<'b> {
    pub fn finalize<'a>(&mut self, parser: &Parser<'_, 'b>) -> Result<(), InternalQueryError> {
        match self {
            PredBuilder::Field(key, test) => {
                let Some(meta) = finialize_field_key(key, parser)? else { return Ok(()) };
                *self = meta_field_test_to_pred(key, meta, test, parser)?;
            }
            PredBuilder::FieldOr(key, tests) => {
                let Some(meta) = finialize_field_key(key, parser)? else { return Ok(()) };
                match meta {
                    MetaField::Level => {
                        let mut mask = 0;
                        for test in tests.iter() {
                            mask |= field_to_level_mask(test, parser)?;
                        }
                        *self = PredBuilder::LevelMask(false, mask);
                        return Ok(());
                    }
                    _ => (),
                }
                let mut preds = ManuallyDrop::new(BumpVec::with_capacity_in(tests.len(), parser.bump));
                for test in tests.iter() {
                    preds.push(meta_field_test_to_pred(key, meta, test, parser)?);
                }
                *self = PredBuilder::Or(preds);
            }
            PredBuilder::FieldAnd(key, tests) => {
                let Some(meta) = finialize_field_key(key, parser)? else { return Ok(()) };
                match meta {
                    MetaField::Level => {
                        let mut mask = 0b1111;
                        for test in tests.iter() {
                            mask &= field_to_level_mask(test, parser)?;
                        }
                        *self = PredBuilder::LevelMask(false, mask);
                        return Ok(());
                    }
                    _ => (),
                }
                let mut preds = ManuallyDrop::new(BumpVec::with_capacity_in(tests.len(), parser.bump));
                for test in tests.iter() {
                    preds.push(meta_field_test_to_pred(key, meta, test, parser)?);
                }
                *self = PredBuilder::And(preds);
            }
            PredBuilder::And(preds) => {
                for pred in preds.iter_mut() {
                    pred.finalize(parser)?
                }
            }
            PredBuilder::Or(preds) => {
                for pred in preds.iter_mut() {
                    pred.finalize(parser)?
                }
            }
            _ => (),
        }
        Ok(())
    }
    pub fn and_field_predicate<'a>(&mut self, bump: &'b Bump, field_key_name: &'b str, test: FieldTest<'b>) {
        let field = FieldKey::New(field_key_name);
        use PredBuilder as P;
        match self {
            P::False => (),
            P::True => {
                *self = P::Field(field, test);
            }
            P::Field(name, field_test) if *name == field => {
                match try_field_test_merge(std::slice::from_mut(field_test), test, true) {
                    MergeResult::Merged => return,
                    MergeResult::AlwaysTrue(_) => {
                        *self = PredBuilder::True;
                        return;
                    }
                    MergeResult::AlwaysFalse(_) => {
                        *self = PredBuilder::False;
                        return;
                    }
                    MergeResult::Unmerged => (),
                }

                let mut vec = ManuallyDrop::new(BumpVec::<FieldTest>::with_capacity_in(4, bump));
                vec.push(*field_test);
                vec.push(test);
                *self = P::FieldAnd(*name, vec);
            }
            P::FieldAnd(name, predicates) if *name == field => {
                predicates.push(test);
            }
            P::And(predicates) => {
                predicates.push(P::Field(field, test));
            }
            _ => {
                let mut preds = ManuallyDrop::new(BumpVec::<PredBuilder<'b>>::with_capacity_in(4, bump));
                preds.push(std::mem::replace(self, P::True));
                preds.push(P::Field(field, test));
                *self = P::And(preds);
            }
        }
    }
    fn or_maybe_field_predicate<'a>(
        &mut self,
        parser: &Parser,
        bump: &'b Bump,
        field_key_name: &'b str,
        test: FieldTest<'b>,
    ) -> Result<(), InternalQueryError> {
        self.or_field_predicate(bump, field_key_name, test);
        Ok(())
    }
    fn or_field_predicate<'a>(&mut self, bump: &'b Bump, field_key_name: &'b str, test: FieldTest<'b>) {
        let field = FieldKey::New(field_key_name);

        use PredBuilder as P;
        match self {
            P::True => (),
            P::False => {
                *self = P::Field(field, test);
            }
            P::Field(name, field_test) if *name == field => {
                match try_field_test_merge(std::slice::from_mut(field_test), test, false) {
                    MergeResult::Merged => return,
                    MergeResult::AlwaysTrue(_) => {
                        *self = PredBuilder::True;
                        return;
                    }
                    MergeResult::AlwaysFalse(_) => {
                        *self = PredBuilder::False;
                        return;
                    }
                    MergeResult::Unmerged => (),
                }

                let mut vec = ManuallyDrop::new(BumpVec::<FieldTest>::with_capacity_in(4, bump));
                vec.push(*field_test);
                vec.push(test);
                *self = P::FieldOr(*name, vec);
            }
            P::FieldOr(name, predicates) if *name == field => {
                predicates.push(test);
            }
            P::Or(predicates) => {
                predicates.push(P::Field(field, test));
            }
            _ => {
                let mut preds = ManuallyDrop::new(BumpVec::<PredBuilder<'b>>::with_capacity_in(4, bump));
                preds.push(std::mem::replace(self, P::True));
                preds.push(P::Field(field, test));
                *self = P::Or(preds);
            }
        }
    }
    pub fn or<'a>(&mut self, bump: &'b Bump, pred: PredBuilder<'b>) {
        use PredBuilder as P;
        if let P::Field(a, c) = pred {
            return self.or_field_predicate(bump, a.as_str(), c);
        }
        match self {
            P::True => (),
            P::False => {
                *self = pred;
            }
            // P::FieldOr(name, predicates) if *name == field => {
            //     predicates.push((require, test));
            // }
            P::Or(predicates) => {
                if let P::Or(mut preds) = pred {
                    predicates.append(&mut *preds);
                } else {
                    predicates.push(pred);
                }
            }
            _ => {
                let mut preds = ManuallyDrop::new(BumpVec::<PredBuilder<'b>>::with_capacity_in(4, bump));
                preds.push(std::mem::replace(self, P::True));
                if let P::Or(mut other) = pred {
                    preds.append(&mut *other);
                } else {
                    preds.push(pred);
                }
                *self = P::Or(preds);
            }
        }
    }
    pub fn and<'a>(&mut self, bump: &'b Bump, pred: PredBuilder<'b>) {
        use PredBuilder as P;
        if let P::Field(a, c) = pred {
            return self.and_field_predicate(bump, a.as_str(), c);
        }
        match self {
            P::True => *self = pred,
            P::False => (),
            // P::Field(name, field_require, field_test) if *name == field => {
            //     let mut vec = ManuallyDrop::new(BumpVec::<FieldPredicate>::with_capacity_in(4, &parser.bump));
            //     vec.push((*field_require, *field_test));
            //     vec.push((require, test));
            //     *self = P::FieldOr(name, vec);
            // }
            // P::FieldOr(name, predicates) if *name == field => {
            //     predicates.push((require, test));
            // }
            P::And(predicates) => {
                if let P::And(mut other) = pred {
                    predicates.append(&mut *other);
                } else {
                    predicates.push(pred);
                }
            }
            _ => {
                let mut preds = ManuallyDrop::new(BumpVec::<PredBuilder<'b>>::with_capacity_in(4, bump));
                preds.push(std::mem::replace(self, P::True));
                if let P::And(mut other) = pred {
                    preds.append(&mut *other);
                } else {
                    preds.push(pred);
                }
                *self = P::And(preds);
            }
        }
    }
}

enum Dest<'b> {
    Empty,
    Single(&'b str, FieldPredicate<'b>),
    Any(&'b str, ManuallyDrop<BumpVec<'b, FieldPredicate<'b>>>),
}

impl<'b> Dest<'b> {
    fn push<'a>(
        &mut self,
        parser: &Parser<'a, 'b>,
        field: &'b str,
        predicate: FieldPredicate<'b>,
    ) -> Result<(), InternalQueryError> {
        match self {
            Dest::Empty => {
                *self = Dest::Single(field, predicate);
            }
            Dest::Single(name, field_predicate) => {
                if *name != field {
                    throw!(DisjuctionNotSupportedAcrossFields @ parser.span_info.text_span(field));
                }
                let mut vec = ManuallyDrop::new(BumpVec::<FieldPredicate>::with_capacity_in(4, &parser.bump));
                vec.push(*field_predicate);
                vec.push(predicate);
                *self = Dest::Any(name, vec);
            }
            Dest::Any(name, predicates) => {
                if *name != field {
                    throw!(DisjuctionNotSupportedAcrossFields @ parser.span_info.text_span(field));
                }
                predicates.push(predicate);
            }
        }
        Ok(())
    }
}
macro_rules! bitset {
    ($ident:ident {$($value: ident),* $(,)?} as $ty: ty) => {
        const {
            $(((1 as $ty) << ($ident::$value as u8)))|*
        }
    };
}

fn expr_eq_field_test<'b>(expr: Expr<'b>) -> Option<FieldTest<'b>> {
    use Expr::*;
    let test = match expr {
        String(value) => FieldTest::TextEqual(false, value),
        Null => FieldTest::NullValue(false),
        Bytes(value) => FieldTest::BytesEqual(false, value),
        Float(value) => FieldTest::FloatRange { negated: false, min: value, max: value },
        Duration(value) => {
            FieldTest::DurationRange { negated: false, min_seconds: f64_to_f48(value), max_seconds: f64_to_f48(value) }
        }
        Signed(value) => FieldTest::I64Eq(false, value),
        Unsigned(value) => FieldTest::U64Eq(false, value),
        Bool(value) => FieldTest::IsTrue(value ^ true),
        _ => return None,
    };
    Some(test)
}

const EQ_LITERAL_TYPES: LiteralKindBitSet = bitset! {
    LiteralKind {
        String,
        Null,
        Bytes,
        Float,
        Duration,
        Integer,
        Bool
    } as u16
};
impl<'a, 'b> Parser<'a, 'b> {
    fn build_predicate_or_expression(
        &self,
        dest: &mut PredBuilder<'b>,
        expr: &Expr<'b>,
    ) -> Result<(), InternalQueryError> {
        use Expr::*;
        match expr {
            Ident(field) => {
                dest.or_field_predicate(&self.bump, field, FieldTest::Defined(false));
                return Ok(());
            }
            UniOp(op, expr) => return self.field_uniop(dest, *op, expr),
            BinOp(op, args) => return self.field_binop(dest, *op, args),
            _ => throw!(NonPredicateExpression @ self.span_info.expr_span(expr)),
        }
    }
    fn field_uniop(&self, dest: &mut PredBuilder<'b>, op: Op1, expr: &'b Expr<'b>) -> Result<(), InternalQueryError> {
        use Expr::*;
        use Op1::*;
        let predicate = match (op, expr) {
            (LogicalNot, arg) => {
                let mut temp_dest = PredBuilder::False;
                self.build_predicate_or_expression(&mut temp_dest, arg)?;
                temp_dest.negate();
                dest.or(&self.bump, temp_dest);
                return Ok(());
            }
            (Exists, arg) => match arg {
                Ident(value) => {
                    dest.or_field_predicate(&self.bump, value, FieldTest::Defined(false));
                    return Ok(());
                }
                _ => throw!(MethodDoesNotExistOnType @ self.span_info.expr_span(arg)),
            },
            (IsFinite, arg) => match arg {
                Ident(value) => {
                    dest.or_field_predicate(&self.bump, value, FieldTest::FiniteFloat(false));
                    return Ok(());
                }
                _ => throw!(MethodDoesNotExistOnType @ self.span_info.expr_span(arg)),
            },
            (Negate, _) => throw!(NonPredicateExpression @ self.span_info.expr_span(expr)),
        };
    }
    #[track_caller]
    fn expected_literal_set(&self, kind: LiteralKindBitSet, expr: &'b Expr<'b>) -> Result<(), InternalQueryError> {
        throw!(ExpectedLiteralOfKind {
            allowed: kind,
            found: expr.kind()
        } @ self.span_info.expr_span(expr))
    }
    fn expected_literal(&self, kind: LiteralKind, expr: &'b Expr<'b>) -> Result<(), InternalQueryError> {
        throw!(ExpectedLiteralOfKind {
            allowed: 1u16 << (kind as u8),
            found: expr.kind()
        } @ self.span_info.expr_span(expr))
    }
    fn any_eq(
        &self,
        dest: &mut PredBuilder<'b>,
        field: &'b str,
        entries: &[Expr<'b>],
    ) -> Result<(), InternalQueryError> {
        if let [single_expr] = entries {
            if let Some(test) = expr_eq_field_test(*single_expr) {
                dest.or_maybe_field_predicate(self, self.bump, field, test)?;
                return Ok(());
            } else {
                return self.expected_literal_set(EQ_LITERAL_TYPES, single_expr);
            }
        }
        let mut strings: BumpVec<'b, &'b str> = BumpVec::new_in(self.bump);

        for expr in entries {
            if let Expr::String(text) = expr {
                strings.push(text);
                continue;
            }
            if let Some(test) = expr_eq_field_test(*expr) {
                dest.or_maybe_field_predicate(self, self.bump, field, test)?;
            } else {
                return self.expected_literal_set(EQ_LITERAL_TYPES, expr);
            }
        }
        match strings.into_bump_slice() {
            [] => Ok(()),
            [single_value] => {
                dest.or_maybe_field_predicate(self, self.bump, field, FieldTest::TextEqual(false, single_value))
            }
            many => dest.or_maybe_field_predicate(self, self.bump, field, FieldTest::TextAny(false, many)),
        }
    }

    fn field_binop(
        &self,
        dest: &mut PredBuilder<'b>,
        op: Op2,
        expr: &'b [Expr<'b>; 2],
    ) -> Result<(), InternalQueryError> {
        use Expr::*;
        use Op2::*;
        let Ident(field_name) = expr[0] else {
            if Op2::LogicalOr == op {
                self.build_predicate_or_expression(dest, &expr[0])?;
                self.build_predicate_or_expression(dest, &expr[1])?;
                return Ok(());
            }
            if Op2::LogicalAnd == op {
                let mut temp_dest = PredBuilder::True;
                self.build_predicate_and_expression(&mut temp_dest, &expr[0])?;
                self.build_predicate_and_expression(&mut temp_dest, &expr[1])?;
                dest.or(self.bump, temp_dest);
                return Ok(());
            }
            if let (Op2::Contains, Expr::List(items), Expr::Ident(field_name)) = (op, expr[0], expr[1]) {
                return self.any_eq(dest, field_name, items);
            }

            throw!(NonPredicateExpression @ self.span_info.expr_span(&expr[0]));
        };

        const COMPARE_LITERAL_TYPES: LiteralKindBitSet = bitset! {
            LiteralKind {
                Float,
                Duration,
                Integer,
            } as u16
        };
        let field_test = match op {
            Is => match expr[1] {
                Null => FieldTest::Type(false, FieldKindSet::from(FieldKind::None).raw),
                Ident(value) => {
                    if let Some(test) = type_text_to_mask(value) {
                        FieldTest::Type(false, test.raw)
                    } else {
                        throw!(UnknownType @ self.span_info.expr_span(&expr[1]))
                    }
                }
                _ => throw!(ExpectedTypeClass @ self.span_info.expr_span(&expr[1])),
            },
            Eq => match expr_eq_field_test(expr[1]) {
                Some(test) => test,
                None => return self.expected_literal_set(EQ_LITERAL_TYPES, &expr[1]),
            },
            NotEq => match expr_eq_field_test(expr[1]) {
                Some(mut test) => {
                    test.negate();
                    test
                }
                None => return self.expected_literal_set(EQ_LITERAL_TYPES, &expr[1]),
            },
            Lt => match expr[1] {
                Float(value) => FieldTest::FloatRange { negated: false, min: f64::MIN, max: value.next_down() },
                Duration(value) => FieldTest::DurationRange {
                    negated: false,
                    min_seconds: 0,
                    max_seconds: f64_to_f48(value).saturating_sub(1),
                },
                Signed(value) => {
                    FieldTest::FloatRange { negated: false, min: f64::MIN, max: (value as f64).next_down() }
                }
                Unsigned(value) => {
                    FieldTest::FloatRange { negated: false, min: f64::MIN, max: (value as f64).next_down() }
                }
                _ => return self.expected_literal_set(COMPARE_LITERAL_TYPES, &expr[1]),
            },
            LtEq => match expr[1] {
                Float(value) => FieldTest::FloatRange { negated: false, min: f64::MIN, max: value },
                Duration(value) => {
                    FieldTest::DurationRange { negated: false, min_seconds: 0, max_seconds: f64_to_f48(value) }
                }
                Signed(value) => FieldTest::FloatRange { negated: false, min: f64::MIN, max: (value as f64) },
                Unsigned(value) => FieldTest::FloatRange { negated: false, min: f64::MIN, max: (value as f64) },
                _ => return self.expected_literal_set(COMPARE_LITERAL_TYPES, &expr[1]),
            },
            Gt => match expr[1] {
                Float(value) => FieldTest::FloatRange { negated: false, min: value.next_up(), max: f64::MAX },
                Duration(value) => FieldTest::DurationRange {
                    negated: false,
                    min_seconds: (f64_to_f48(value) + 1).min((1 << 48) - 1),
                    max_seconds: u64::MAX,
                },
                Signed(value) => FieldTest::FloatRange { negated: false, min: (value as f64).next_up(), max: f64::MAX },
                Unsigned(value) => {
                    FieldTest::FloatRange { negated: false, min: (value as f64).next_up(), max: f64::MAX }
                }
                _ => return self.expected_literal_set(COMPARE_LITERAL_TYPES, &expr[1]),
            },
            GtEq => match expr[1] {
                Float(value) => FieldTest::FloatRange { negated: false, min: value, max: f64::MAX },
                Duration(value) => {
                    FieldTest::DurationRange { negated: false, min_seconds: f64_to_f48(value), max_seconds: u64::MAX }
                }
                Signed(value) => FieldTest::FloatRange { negated: false, min: value as f64, max: f64::MAX },
                Unsigned(value) => FieldTest::FloatRange { negated: false, min: value as f64, max: f64::MAX },
                _ => return self.expected_literal_set(COMPARE_LITERAL_TYPES, &expr[1]),
            },
            In => match expr[1] {
                List(entries) => {
                    return self.any_eq(dest, field_name, entries);
                }
                TimeRange { min_ns, max_ns } => FieldTest::TimeRange { negated: false, min_ns, max_ns },
                _ => return self.expected_literal(LiteralKind::List, &expr[1]),
            },
            LogicalAnd => {
                let mut temp_dest = PredBuilder::True;
                temp_dest.and_field_predicate(self.bump, field_name, FieldTest::Defined(false));
                self.build_predicate_and_expression(&mut temp_dest, &expr[1]);
                dest.or(self.bump, temp_dest);
                return Ok(());
            }
            LogicalOr => {
                dest.or_field_predicate(self.bump, field_name, FieldTest::Defined(false));
                return self.build_predicate_or_expression(dest, &expr[1]);
            }
            Add | Sub | Mul | Div | Rem | BitwiseAnd | BitwiseOr | Xor | ShiftLeft | ShiftRight => {
                throw!(NonPredicateExpression @ self.span_info.expr_span(&expr[0]))
            }
            Contains => match expr[1] {
                String(value) => FieldTest::TextContains(false, value),
                _ => return self.expected_literal(LiteralKind::String, &expr[1]),
            },
            StartsWith => match expr[1] {
                String(value) => FieldTest::TextStartsWith(false, value),
                _ => return self.expected_literal(LiteralKind::String, &expr[1]),
            },
            EndsWith => match expr[1] {
                String(value) => FieldTest::TextEndWith(false, value),
                _ => return self.expected_literal(LiteralKind::String, &expr[1]),
            },
        };

        dest.or_maybe_field_predicate(self, self.bump, field_name, field_test)?;
        Ok(())
    }
    fn build_predicate_and_expression(
        &self,
        output: &mut PredBuilder<'b>,
        expr: &Expr<'b>,
    ) -> Result<(), InternalQueryError> {
        use Expr::*;
        use Op2::*;

        if let BinOp(LogicalAnd, args) = *expr {
            self.build_predicate_and_expression(output, &args[0])?;
            return self.build_predicate_and_expression(output, &args[1]);
        }
        let mut dest = PredBuilder::False;
        self.build_predicate_or_expression(&mut dest, expr)?;
        output.and(self.bump, dest);

        Ok(())
    }

    pub fn expr_to_predicates(&self, expr: Expr<'b>) -> Result<PredBuilder<'b>, InternalQueryError> {
        let mut dest = PredBuilder::False;
        self.build_predicate_or_expression(&mut dest, &expr)?;
        dest.finalize(self)?;
        Ok(dest)
    }
}

pub type QueryPredicates<'b> = &'b [(&'b str, &'b [FieldPredicate<'b>])];

struct FieldFilter<'a> {
    name: &'a str,
    any: &'a [FieldPredicate<'a>],
}

pub struct TokenStream<'a> {
    next: usize,
    tokens: &'a [Token],
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    fn extend(&self, after: Span) -> Span {
        Span { start: self.start, end: after.end }
    }
    fn merge(&self, other: Span) -> Span {
        Span { start: self.start.min(other.start), end: self.end.max(other.end) }
    }
    pub fn text<'a>(&self, source: &'a str) -> &'a str {
        &source[self.start as usize..self.end as usize]
    }
}
impl Span {
    const EMPTY: Span = Span { start: 0, end: 0 };
}

impl std::fmt::Debug for QueryParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as std::fmt::Display>::fmt(self, f)
    }
}
impl std::fmt::Display for QueryParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QueryParseError: {:?}", self.kind)?;
        if let Some(span) = self.span {
            write!(f, ", span: [{}, {}]", span.start, span.end)?;
        }
        if let Some(token_kind) = self.token_kind {
            write!(f, ", token_kind: {:?}", token_kind)?;
        }
        Ok(())
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Token {
    pub of: Of,
    pub span: Span,
}

impl Token {
    fn text<'a>(&self, source: &'a str) -> &'a str {
        &source[self.span.start as usize..self.span.end as usize]
    }
}

impl<'a> TokenStream<'a> {
    fn is_empty(&self) -> bool {
        self.next >= self.tokens.len()
    }
    fn peek(&self) -> Token {
        self.tokens.get(self.next).copied().unwrap_or(Token { of: Of::Eof, span: Span::EMPTY })
    }
    fn peek_2(&self) -> Token {
        self.tokens.get(self.next + 1).copied().unwrap_or(Token { of: Of::Eof, span: Span::EMPTY })
    }
    fn advance(&mut self, amount: usize) {
        self.next += amount;
    }
    fn previous_index(&self) -> usize {
        self.next.saturating_sub(1)
    }
    fn next(&mut self) -> Token {
        let res = self.peek();
        self.next += 1;
        res
    }
    fn peek_slice(&mut self) -> &'a [Token] {
        &self.tokens[self.next..]
    }
}

type TokenIndex = u32;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Of {
    Ident,
    Literal {
        kind: RaLiteralKind,
        suffix_start: u32,
    },

    /// `;`
    Semi,
    /// `,`
    Comma,
    /// `.`
    Dot,
    /// `(`
    OpenParen {
        end: TokenIndex,
    },
    /// `)`
    CloseParen {
        start: TokenIndex,
    },
    /// `{`
    OpenBrace {
        end: TokenIndex,
    },
    /// `}`
    CloseBrace {
        start: TokenIndex,
    },
    /// `[`
    OpenBracket {
        end: TokenIndex,
    },
    /// `]`
    CloseBracket {
        start: TokenIndex,
    },
    /// `@`
    At,
    /// `#`
    Pound,
    /// `~`
    Tilde,
    /// `?`
    Question,
    /// `:`
    Colon,
    /// `$`
    Dollar,
    /// `=`
    SingleEq,
    /// `==`
    DoubleEq,
    /// `!`
    Bang,
    /// `<`
    Lt,
    /// `<=`
    LtEq,
    /// `>`
    Gt,
    /// `>=`
    GtEq,
    /// `!=`
    BangEq,
    /// `-`
    Minus,
    /// `&`
    SingleAnd,
    /// `|`
    SingleOr,
    /// `&&`
    DoubleAnd,
    /// `||`
    DoubleOr,
    /// `+`
    Plus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `^`
    Caret,
    /// `%`
    Percent,
    /// `<<`
    ShiftLeft,
    /// `>>`
    ShiftRight,
    /// End of input.
    Eof,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op1 {
    LogicalNot, // ! (prefix)
    Negate,     // - (prefix)
    Exists,     // .exists()
    IsFinite,   // .is_finite()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op2 {
    Eq,         // == (or =)
    NotEq,      // !=
    Lt,         // <
    LtEq,       // <=
    Gt,         // >
    GtEq,       // >=
    Is,         // is
    In,         // in
    LogicalAnd, // &&
    LogicalOr,  // ||
    Add,        // +
    Sub,        // - (infix)
    Mul,        // *
    Div,        // /
    Rem,        // %
    BitwiseAnd, // &
    BitwiseOr,  // |
    Xor,        // ^
    ShiftLeft,  // <<
    ShiftRight, // >>
    Contains,   // .contains(arg)
    StartsWith, // .starts_with(arg)
    EndsWith,   // .ends_with(arg)
}

#[derive(Debug)]
pub enum ErrorContext {
    Token(Token),
    Range(Span),
    None,
}

#[derive(Debug)]
pub struct InternalQueryError {
    pub kind: ErrorKind,
    pub context: ErrorContext,
}

impl From<(ErrorKind, Option<Span>)> for InternalQueryError {
    #[track_caller]
    fn from((kind, span): (ErrorKind, Option<Span>)) -> Self {
        #[cfg(test)]
        {
            let loc = std::panic::Location::caller();
            println!("throw[{:?}] from {}:{}", kind, loc.file(), loc.line());
        }
        InternalQueryError {
            kind,
            context: if let Some(span) = span { ErrorContext::Range(span) } else { ErrorContext::None },
        }
    }
}

impl From<(ErrorKind, Token)> for InternalQueryError {
    #[track_caller]
    fn from((kind, token): (ErrorKind, Token)) -> Self {
        #[cfg(test)]
        {
            let loc = std::panic::Location::caller();
            println!("throw[{:?} @ {:?}] from {}:{}", kind, token.of, loc.file(), loc.line());
        }
        InternalQueryError { kind, context: ErrorContext::Token(token) }
    }
}

impl From<(ErrorKind, Span)> for InternalQueryError {
    #[track_caller]
    fn from((kind, span): (ErrorKind, Span)) -> Self {
        #[cfg(test)]
        {
            let loc = std::panic::Location::caller();
            println!("throw[{:?}] from {}:{}", kind, loc.file(), loc.line());
        }
        InternalQueryError { kind, context: ErrorContext::Range(span) }
    }
}

impl From<(ErrorKind)> for InternalQueryError {
    #[track_caller]
    fn from(kind: ErrorKind) -> Self {
        #[cfg(test)]
        {
            let loc = std::panic::Location::caller();
            println!("throw[{:?}] from {}:{}", kind, loc.file(), loc.line());
        }
        InternalQueryError { kind, context: ErrorContext::None }
    }
}
type Res<T> = Result<T, InternalQueryError>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Expr<'b> {
    Null,
    String(&'b str),
    Bytes(&'b [u8]),
    List(&'b [Expr<'b>]),
    Float(f64),
    Duration(f64),
    Signed(i64),
    Unsigned(u64),
    Bool(bool),
    Ident(&'b str),
    TimeRange { min_ns: i64, max_ns: i64 },
    UniOp(Op1, &'b Expr<'b>),
    BinOp(Op2, &'b [Expr<'b>; 2]),
}

#[derive(Debug, PartialEq)]
pub enum ExprKind {
    Literal(LiteralKind),
    Ident,
    Operation,
}

impl<'b> Expr<'b> {
    fn kind(&self) -> ExprKind {
        match self {
            Expr::Null => ExprKind::Literal(LiteralKind::Null),
            Expr::String(_) => ExprKind::Literal(LiteralKind::String),
            Expr::Bytes(_) => ExprKind::Literal(LiteralKind::Bytes),
            Expr::List(_) => ExprKind::Literal(LiteralKind::List),
            Expr::Float(_) => ExprKind::Literal(LiteralKind::Float),
            Expr::Duration(_) => ExprKind::Literal(LiteralKind::Duration),
            Expr::Signed(_) => ExprKind::Literal(LiteralKind::Integer),
            Expr::Unsigned(_) => ExprKind::Literal(LiteralKind::Integer),
            Expr::Bool(_) => ExprKind::Literal(LiteralKind::Bool),
            Expr::Ident(_) => ExprKind::Ident,
            _ => ExprKind::Operation,
        }
    }
}

// used for error printing
use std::mem::ManuallyDrop;

const HIGH_BIT: usize = 1 << (std::mem::size_of::<usize>() * 8 - 1);
struct LazySpanExprList {
    start: usize,
    rewrite_atom: usize,
}

struct SpanInfo<'b> {
    table: ManuallyDrop<BumpVec<'b, (usize, Span)>>,
    rewrite_atom: usize,
}

impl<'b> SpanInfo<'b> {
    fn new_in(bump: &'b Bump) -> SpanInfo<'b> {
        SpanInfo { table: ManuallyDrop::new(BumpVec::with_capacity_in(16, bump)), rewrite_atom: 0 }
    }

    fn new_lazy_list(&mut self) -> LazySpanExprList {
        let start = self.table.len();
        let rewrite_atom = self.rewrite_atom;
        self.rewrite_atom += 1;
        LazySpanExprList { start, rewrite_atom: rewrite_atom | HIGH_BIT }
    }

    fn insert_lazy(&mut self, lazy: &LazySpanExprList, span: Span) {
        self.table.push((lazy.rewrite_atom, span));
    }

    fn finish_lazy(&mut self, lazy: LazySpanExprList, args: &'b [Expr]) {
        let mut offset = <*const [Expr]>::addr(args);
        for (addr, _) in self.table[lazy.start..].iter_mut() {
            if *addr == lazy.rewrite_atom {
                *addr = offset;
                offset += std::mem::size_of::<Expr>();
            }
        }
    }

    fn insert(&mut self, addr: usize, span: Span) {
        self.table.push((addr, span));
    }
    // gets the span for the tokens defining the operation given the slice of args
    fn span_from_addr(&self, addr: usize) -> Option<Span> {
        for (addr2, span) in self.table.as_slice() {
            if *addr2 == addr {
                return Some(*span);
            }
        }
        None
    }

    // gets the span for the tokens defining the operation given the slice of args
    fn insert_op_span(&mut self, args: &'b [Expr], span: Span) {
        let addr = <*const [Expr]>::addr(args);
        self.table.push((addr + 1, span));
    }
    // gets the span for the tokens defining the operation given the slice of args
    fn uni_op_span(&self, args: &'b Expr) -> Option<Span> {
        self.span_from_addr(<*const Expr>::addr(args) + 1)
    }

    // gets the span for the tokens defining the operation given the slice of args
    fn bin_op_span(&self, args: &[Expr; 2]) -> Option<Span> {
        self.span_from_addr(<*const [Expr; 2]>::addr(args) + 1)
    }

    fn text_span(&self, value: &'b str) -> Option<Span> {
        self.span_from_addr(<*const str>::addr(value))
    }
    // gets the span for the tokens defining the operation given the slice of args
    fn expr_span(&self, value: &'b Expr<'b>) -> Option<Span> {
        self.span_from_addr(<*const Expr<'b>>::addr(value))
    }
    // gets the span for the tokens defining the operation given the slice of args
    fn get<T: ?Sized>(&self, value: &'b T) -> Option<Span> {
        self.span_from_addr(<*const T>::addr(value))
    }
}

#[derive(Debug, PartialEq)]
pub enum ErrorKind {
    ExpectedKind { kind: TokenKind },
    ExpectedString,
    ExpectedLiteral,
    ExpectedFieldName,
    ExpectedMethodName,
    ExpectedIdent,
    ExpectedOperator,
    ExpectedExpression,
    ExpectedArgument,
    ExpectedOpenParen,
    UnknownMethod,
    ExpectedCloseParen,
    ExpectedCloseBracket,
    ExpectedCommaOrCloseBracket,
    ExpectedCommaOrCloseParen,
    UnclosedGroup,
    RepeatedComma,
    InvalidNegation,
    CannotCompareArrayToField,
    FieldsNotAllowedOnRight,
    InvalidInOperand,
    InvalidIsOperand,
    InvalidNumberOfArguments { expected: u8, got: u8 },
    MismatchedGroup,
    UnknownSymbol,
    ExpectedLiteralOfKind { allowed: LiteralKindBitSet, found: ExprKind },
    DisjuctionNotSupportedAcrossFields,
    UnknownMethodName,
    UnknownFunction,
    InvalidLiteral(LiteralParseError),
    LiteralEvalFailure(EvalError),
    UnexpectedEof,
    UnknownMetaVar,
    UnknownLogLevel,
    UnknownType,
    ExpectedTypeClass,
    ExpectedLevelFilter,
    ExpectedFunctionName,
    UnsupportedOperationOnSpan,
    UnsupportedOperationOnMessage,
    UnsupportedOperationOnService,
    UnsupportedOperationOnTarget,
    UnsupportedOperationOnTimestamp,
    NonPredicateExpression,
    MethodDoesNotExistOnType,
}

const BP_NONE: u8 = 0;
// BP_TERM removed, atoms have implicit highest precedence start
const BP_LOGICAL_OR: u8 = 4; // ||
const BP_LOGICAL_AND: u8 = 6; // &&
const BP_IN: u8 = 9; // in (Similar to comparison)
const BP_COMPARISON: u8 = 11; // == != < <= > >= (Placeholder if added to parse_expr)
const BP_BITWISE_OR: u8 = 13; // |
const BP_BITWISE_XOR: u8 = 15; // ^
const BP_BITWISE_AND: u8 = 17; // &
const BP_ADDITIVE: u8 = 19; // + -
const BP_MULTIPLICATIVE: u8 = 21; // * / %
const BP_UNARY: u8 = 23; // unary -, !
const BP_FUNCTION_CALL: u8 = 25; // name(func args)
const BP_CALL_MEMBER: u8 = 27; // . (method call)

impl<'a, 'bump> Parser<'a, 'bump> {
    fn infix_binding_power(&self, token: Token) -> Option<(u8, u8)> {
        let bp = match token.of {
            // Logical
            Of::DoubleOr => (BP_LOGICAL_OR, BP_LOGICAL_OR + 1),
            Of::DoubleAnd => (BP_LOGICAL_AND, BP_LOGICAL_AND + 1),

            Of::Ident => {
                if token.text(self.source) == "in" || token.text(self.source) == "is" {
                    (BP_IN, BP_IN + 1)
                } else {
                    return None;
                }
            }

            Of::SingleEq | Of::DoubleEq | Of::BangEq | Of::Lt | Of::LtEq | Of::Gt | Of::GtEq => {
                (BP_COMPARISON, BP_COMPARISON + 1)
            }

            // Bitwise
            Of::SingleOr => (BP_BITWISE_OR, BP_BITWISE_OR + 1),
            Of::Caret => (BP_BITWISE_XOR, BP_BITWISE_XOR + 1),
            Of::SingleAnd => (BP_BITWISE_AND, BP_BITWISE_AND + 1),
            Of::ShiftLeft | Of::ShiftRight => (BP_BITWISE_AND, BP_BITWISE_AND + 1),

            // Arithmetic
            Of::Plus | Of::Minus => (BP_ADDITIVE, BP_ADDITIVE + 1),
            Of::Star | Of::Slash | Of::Percent => (BP_MULTIPLICATIVE, BP_MULTIPLICATIVE + 1),

            Of::Dot => (BP_CALL_MEMBER, BP_CALL_MEMBER + 1),
            Of::OpenParen { .. } => (BP_FUNCTION_CALL, BP_FUNCTION_CALL + 1),

            _ => return None,
        };
        Some(bp)
    }

    fn prefix_binding_power(&self, token: Token) -> Option<u8> {
        match token.of {
            Of::Minus | Of::Bang => Some(BP_UNARY),
            _ => None,
        }
    }
    fn spanned_text(&mut self, text: &str, span: Span) -> &'bump str {
        let arg = self.bump.alloc_str(text);
        self.span_info.insert(<*const str>::addr(arg), span);
        arg
    }
    fn spanned_binop_op(&mut self, op: Op2, op_span: Span, args: [Expr<'bump>; 2], spans: [Span; 2]) -> Expr<'bump> {
        let arg = self.bump.alloc([args[0], args[1]]);
        self.span_info.insert(<*const Expr>::addr(&arg[0]) + 1, op_span);
        self.span_info.insert(<*const Expr>::addr(&arg[0]), spans[0]);
        self.span_info.insert(<*const Expr>::addr(&arg[1]), spans[1]);
        Expr::BinOp(op, arg)
    }
    fn spanned_unary_op(&mut self, op: Op1, op_span: Span, operand: Expr<'bump>, operand_span: Span) -> Expr<'bump> {
        let arg = self.bump.alloc(operand);
        self.span_info.insert(<*const Expr>::addr(arg), operand_span);
        self.span_info.insert(<*const Expr>::addr(arg) + 1, op_span);
        Expr::UniOp(op, arg)
    }
    fn parse_atom(&mut self) -> Res<(Expr<'bump>, Span)> {
        let token = self.tokens.next();
        match token.of {
            Of::Minus | Of::Bang => {
                let op = if token.of == Of::Minus { Op1::Negate } else { Op1::LogicalNot };
                let rbp = self
                    .prefix_binding_power(token)
                    .ok_or_else(|| InternalQueryError::from((ErrorKind::UnknownSymbol, token)))?;

                let (operand, operand_span) = self.parse_expr(rbp)?;
                let span = token.span.extend(operand_span);

                if op == Op1::Negate {
                    match operand {
                        Expr::Duration(value) => {
                            return Ok((Expr::Duration(-value), span));
                        }
                        Expr::Float(value) => {
                            return Ok((Expr::Float(-value), span));
                        }
                        Expr::Signed(value) => {
                            if value == i64::MAX {
                                return Ok((Expr::Unsigned(9223372036854775808), span));
                            } else {
                                return Ok((Expr::Signed(-value), span));
                            }
                        }
                        Expr::Unsigned(value) => {
                            if value > 9223372036854775808 {
                                throw!(LiteralEvalFailure(EvalError::Overflow) @ token);
                            } else {
                                return Ok((Expr::Signed(value.wrapping_neg() as i64), span));
                            }
                        }
                        _ => (),
                    }
                }
                Ok((self.spanned_unary_op(op, token.span, operand, operand_span), span))
            }

            Of::OpenParen { end: _ } => {
                let expected_start = self.tokens.previous_index();
                let expr = self.parse_expr(BP_NONE)?;
                let next_token = self.tokens.peek();
                if matches!(next_token.of, Of::CloseParen { start } if start == expected_start as u32) {
                    self.tokens.advance(1);
                    Ok((expr.0, token.span.extend(next_token.span)))
                } else {
                    throw!(ExpectedCloseParen @ token)
                }
            }

            Of::Literal { kind, suffix_start } => Ok((
                Expr::parse_literal_kind(self.bump, token.text(self.source), kind, suffix_start)
                    .map_err(|e| InternalQueryError::from((ErrorKind::InvalidLiteral(e), token)))?,
                token.span,
            )),

            Of::Ident => match token.text(self.source) {
                "None" | "null" => Ok((Expr::Null, token.span)),
                "true" => Ok((Expr::Bool(true), token.span)),
                "false" => Ok((Expr::Bool(false), token.span)),
                ident_text => Ok((Expr::Ident(self.spanned_text(ident_text, token.span)), token.span)),
            },

            Of::OpenBracket { end: _ } => {
                let expected_start = self.tokens.previous_index();

                if matches!(self.tokens.peek().of, Of::CloseBracket { start } if start == expected_start as u32) {
                    let end_token = self.tokens.next();
                    return Ok((Expr::List(&[]), token.span.extend(end_token.span)));
                }

                let mut list = BumpVec::new_in(self.bump);
                let lazy_span = self.span_info.new_lazy_list();

                let mut allow_comma = false;
                let end_token = 'done: {
                    loop {
                        let next_token = self.tokens.peek();
                        if let Of::CloseBracket { start } = next_token.of {
                            if start != expected_start as u32 {
                                throw!(MismatchedGroup @ next_token);
                            }
                            break 'done self.tokens.next();
                        }

                        let (expr, span) = self.parse_expr(BP_NONE)?;
                        self.span_info.insert_lazy(&lazy_span, span);
                        list.push(expr);

                        let next_token = self.tokens.peek();
                        match next_token.of {
                            Of::Comma => self.tokens.advance(1),
                            Of::CloseBracket { start } => {
                                if start != expected_start as u32 {
                                    throw!(MismatchedGroup @ next_token);
                                }
                                break 'done self.tokens.next();
                            }
                            Of::Eof => throw!(UnexpectedEof @ token.span),
                            _ => {
                                throw!(ExpectedCommaOrCloseBracket @ next_token);
                            }
                        }
                    }
                };
                let list = list.into_bump_slice();
                self.span_info.finish_lazy(lazy_span, list);
                Ok((Expr::List(list), token.span.extend(end_token.span)))
            }

            Of::Eof => throw!(UnexpectedEof @ token),
            _ => throw!(ExpectedExpression @ token),
        }
    }

    pub fn parse_expr(&mut self, min_bp: u8) -> Res<(Expr<'bump>, Span)> {
        let mut left = self.parse_atom()?;

        loop {
            let op_token = self.tokens.peek();

            match op_token.of {
                Of::Eof
                | Of::CloseParen { .. }
                | Of::CloseBracket { .. }
                | Of::CloseBrace { .. }
                | Of::Comma
                | Of::Semi => break,
                _ => {}
            }

            let (lbp, rbp) = match self.infix_binding_power(op_token) {
                Some((l, r)) if l >= min_bp => (l, r),
                _ => break,
            };

            self.tokens.advance(1);

            match op_token.of {
                Of::OpenParen { .. } => {
                    let function_name = match left.0 {
                        Expr::Ident(function_name) => function_name,
                        _ => throw!(ExpectedFunctionName @ left.1),
                    };
                    let expected_start = self.tokens.previous_index();

                    let mut args = BumpVec::new_in(self.bump);
                    let lazy_span = self.span_info.new_lazy_list();

                    let end_token = 'done: {
                        loop {
                            let next_token = self.tokens.peek();
                            if let Of::CloseParen { start } = next_token.of {
                                if start != expected_start as u32 {
                                    throw!(MismatchedGroup @ next_token);
                                }
                                break 'done self.tokens.next();
                            }
                            let (expr, span) = self.parse_expr(BP_NONE)?;
                            self.span_info.insert_lazy(&lazy_span, span);
                            args.push(expr);

                            let next_token = self.tokens.peek();
                            match next_token.of {
                                Of::Comma => self.tokens.advance(1),
                                Of::CloseParen { start } => {
                                    if start != expected_start as u32 {
                                        throw!(MismatchedGroup @ next_token);
                                    }
                                    break 'done self.tokens.next();
                                }
                                Of::Eof => throw!(UnexpectedEof @ op_token),
                                _ => {
                                    throw!(ExpectedCommaOrCloseParen @ next_token);
                                }
                            }
                        }
                    };
                    fn saturate_into_i64(value: i128) -> i64 {
                        if value < i64::MIN as i128 {
                            i64::MIN
                        } else if value > i64::MAX as i128 {
                            i64::MAX
                        } else {
                            value as i64
                        }
                    }

                    match function_name {
                        "time_range" => match args.as_slice() {
                            [arg] => {
                                let Expr::String(text) = arg else {
                                    throw!(ExpectedLiteralOfKind {
                                        allowed: literal_type_set(&[FieldKind::String]),
                                        found: arg.kind()
                                    } @ self.span_info.expr_span(arg))
                                };
                                match timestamp_eval::parse_timestamp_to_nanosecond_range(text) {
                                    Ok((min, max)) => {
                                        left = (
                                            Expr::TimeRange {
                                                min_ns: saturate_into_i64(min),
                                                max_ns: saturate_into_i64(max),
                                            },
                                            left.1.extend(end_token.span),
                                        );
                                    }
                                    Err(e) => {
                                        throw!(LiteralEvalFailure(EvalError::InvalidTimeString) @ self.span_info.expr_span(arg))
                                    }
                                }
                            }
                            _ => {
                                throw!(InvalidNumberOfArguments { expected: 1, got: args.len().min(u8::MAX as usize) as u8 } @ end_token)
                            }
                        },
                        _ => throw!(UnknownFunction @ left.1),
                    }
                }
                Of::Dot => {
                    let method_name_token = self.tokens.next();
                    let method_name = match method_name_token.of {
                        Of::Ident => method_name_token.text(self.source),
                        _ => throw!(ExpectedMethodName @ method_name_token),
                    };

                    let open_paren_token = self.tokens.next();
                    if !matches!(open_paren_token.of, Of::OpenParen { .. }) {
                        throw!(ExpectedOpenParen @ open_paren_token);
                    }
                    let expected_start = self.tokens.previous_index();

                    let mut args = BumpVec::new_in(self.bump);
                    let lazy_span = self.span_info.new_lazy_list();
                    self.span_info.insert_lazy(&lazy_span, left.1);
                    args.push(left.0);

                    let end_token = 'done: {
                        loop {
                            let next_token = self.tokens.peek();
                            if let Of::CloseParen { start } = next_token.of {
                                if start != expected_start as u32 {
                                    throw!(MismatchedGroup @ next_token);
                                }
                                break 'done self.tokens.next();
                            }
                            let (expr, span) = self.parse_expr(BP_NONE)?;
                            self.span_info.insert_lazy(&lazy_span, span);
                            args.push(expr);

                            let next_token = self.tokens.peek();
                            match next_token.of {
                                Of::Comma => self.tokens.advance(1),
                                Of::CloseParen { start } => {
                                    if start != expected_start as u32 {
                                        throw!(MismatchedGroup @ next_token);
                                    }
                                    break 'done self.tokens.next();
                                }
                                Of::Eof => throw!(UnexpectedEof @ op_token),
                                _ => {
                                    throw!(ExpectedCommaOrCloseParen @ next_token);
                                }
                            }
                        }
                    };

                    let (expr, span) = match method_name {
                        "starts_with" => {
                            if args.len() != 2 {
                                throw!(ExpectedArgument @ end_token);
                            }
                            let arr = self.bump.alloc([args[0], args[1]]);
                            (Expr::BinOp(Op2::StartsWith, arr), op_token.span.extend(end_token.span))
                        }
                        "ends_with" => {
                            if args.len() != 2 {
                                throw!(ExpectedArgument @ end_token);
                            }
                            let arr = self.bump.alloc([args[0], args[1]]);
                            (Expr::BinOp(Op2::EndsWith, arr), op_token.span.extend(end_token.span))
                        }
                        "contains" => {
                            if args.len() != 2 {
                                throw!(ExpectedArgument @ end_token);
                            }
                            let arr = self.bump.alloc([args[0], args[1]]);
                            (Expr::BinOp(Op2::Contains, arr), op_token.span.extend(end_token.span))
                        }
                        "exists" => {
                            if args.len() != 1 {
                                throw!(ExpectedArgument @ end_token);
                            }
                            let arg = self.bump.alloc(args[0]);
                            (Expr::UniOp(Op1::Exists, arg), op_token.span.extend(end_token.span))
                        }
                        "is_finite" => {
                            if args.len() != 1 {
                                throw!(ExpectedArgument @ end_token);
                            }
                            let arg = self.bump.alloc(args[0]);
                            (Expr::UniOp(Op1::IsFinite, arg), op_token.span.extend(end_token.span))
                        }
                        unknown => {
                            throw!(UnknownMethod @ method_name_token)
                        }
                    };

                    let args_slice = args.into_bump_slice();
                    self.span_info.insert_op_span(args_slice, op_token.span.extend(method_name_token.span));
                    self.span_info.finish_lazy(lazy_span, args_slice);
                    left = (expr, span);
                }

                Of::Ident if op_token.text(self.source) == "is" => {
                    let right = self.parse_expr(rbp)?;
                    match right.0 {
                        Expr::Ident(_) | Expr::Null => {
                            left = (
                                self.spanned_binop_op(Op2::Is, op_token.span, [left.0, right.0], [left.1, right.1]),
                                left.1.merge(right.1),
                            );
                        }
                        _ => {
                            throw!(InvalidIsOperand @ op_token)
                        }
                    }
                }
                Of::Ident if op_token.text(self.source) == "in" => {
                    let right = self.parse_expr(rbp)?;
                    match right.0 {
                        Expr::List(_) | Expr::TimeRange { .. } => {
                            left = (
                                self.spanned_binop_op(Op2::In, op_token.span, [left.0, right.0], [left.1, right.1]),
                                left.1.merge(right.1),
                            );
                        }
                        _ => throw!(InvalidInOperand @ op_token),
                    }
                }

                _ => {
                    let op = match op_token.of {
                        Of::DoubleOr => Op2::LogicalOr,
                        Of::DoubleAnd => Op2::LogicalAnd,
                        Of::SingleOr => Op2::BitwiseOr,
                        Of::Caret => Op2::Xor,
                        Of::SingleAnd => Op2::BitwiseAnd,
                        Of::Plus => Op2::Add,
                        Of::Minus => Op2::Sub,
                        Of::Star => Op2::Mul,
                        Of::Slash => Op2::Div,
                        Of::Percent => Op2::Rem,
                        Of::ShiftLeft => Op2::ShiftLeft,
                        Of::ShiftRight => Op2::ShiftRight,
                        Of::GtEq => Op2::GtEq,
                        Of::SingleEq => Op2::Eq,
                        Of::DoubleEq => Op2::Eq,
                        Of::BangEq => Op2::NotEq,
                        Of::LtEq => Op2::LtEq,
                        Of::Lt => Op2::Lt,
                        Of::Gt => Op2::Gt,
                        _ => unreachable!("Operator token {:?} passed check but has no BinOp mapping", op_token.of),
                    };

                    let right = self.parse_expr(rbp)?;
                    let fold_op = match op {
                        Op2::Add => Some(LiteralBinOp::Add),
                        Op2::Sub => Some(LiteralBinOp::Sub),
                        Op2::Mul => Some(LiteralBinOp::Mul),
                        Op2::Div => Some(LiteralBinOp::Div),
                        Op2::Rem => Some(LiteralBinOp::Rem),
                        Op2::BitwiseAnd => Some(LiteralBinOp::BitwiseAnd),
                        Op2::BitwiseOr => Some(LiteralBinOp::BitwiseOr),
                        Op2::Xor => Some(LiteralBinOp::Xor),
                        Op2::ShiftLeft => Some(LiteralBinOp::ShiftLeft),
                        Op2::ShiftRight => Some(LiteralBinOp::ShiftRight),
                        _ => None,
                    };
                    if let Some(op) = fold_op {
                        match Expr::eval_binop(self.bump, left.0, right.0, op) {
                            Ok(expr) => {
                                left = (expr, left.1.merge(right.1));
                            }
                            Err(error) => {
                                throw!(LiteralEvalFailure(error))
                            }
                        }
                    } else {
                        left = (
                            self.spanned_binop_op(op, op_token.span, [left.0, right.0], [left.1, right.1]),
                            left.1.merge(right.1),
                        );
                    }
                }
            }
        }
        Ok(left)
    }

    pub fn init(bump: &'bump Bump, source: &'a str) -> Res<Parser<'a, 'bump>> {
        let mut token_buffer = bumpalo::collections::Vec::new_in(bump);
        let mut offset = 0;
        let mut cursor = ra_ap_rustc_lexer::Cursor::new(source);
        let mut next_prev = TokenKind::Whitespace;
        let mut depth_buf: Vec<u32> = Vec::new();
        loop {
            let token = cursor.advance_token();
            let prev = next_prev;
            let mut span = Span { start: offset, end: offset + token.len };
            offset += token.len;
            next_prev = token.kind;
            macro_rules! join {
                ([$default: ident]$($prev: ident => $kind: ident),* $(,)?) => {
                    match prev {
                        $(TokenKind::$prev => {
                            token_buffer.pop();
                            span.start = span.start.saturating_sub(1);
                            Of::$kind
                        },)*
                        _ => Of::$default
                    }
                };
            }
            macro_rules! finish_group {
                ($open_token: ident, $close_token: ident) => {{
                    let Some(start) = depth_buf.pop() else { throw!(UnclosedGroup @ span); };
                    let end_index = token_buffer.len() as u32;
                    match token_buffer.get_mut(start as usize) {
                        Some(Token { of: Of::$open_token { end }, .. }) => *end = end_index,
                        Some(tok) => throw!(MismatchedGroup @ *tok),
                        _ => throw!(MismatchedGroup @ span),
                    }
                    Of::$close_token { start }
                }};
            }
            use TokenKind::*;
            let of = match token.kind {
                LineComment { .. } | TokenKind::BlockComment { .. } | TokenKind::Whitespace => continue,
                Ident => {
                    if prev == TokenKind::Dollar {
                        token_buffer.pop();
                        span.start = span.start.saturating_sub(1);
                    }
                    Of::Ident
                }
                InvalidIdent
                | RawIdent
                | RawLifetime { .. }
                | UnknownPrefix
                | UnknownPrefixLifetime
                | GuardedStrPrefix
                | Semi
                | Unknown
                | Lifetime { .. } => {
                    throw!(UnknownSymbol @ span);
                }
                Literal { kind, suffix_start } => Of::Literal { kind, suffix_start },
                Comma => Of::Comma,
                Dot => Of::Dot,
                OpenParen => {
                    depth_buf.push(token_buffer.len() as u32);
                    Of::OpenParen { end: 0 }
                }
                CloseParen => finish_group!(OpenParen, CloseParen),
                OpenBrace => {
                    depth_buf.push(token_buffer.len() as u32);
                    Of::OpenBrace { end: 0 }
                }
                CloseBrace => finish_group!(OpenBrace, CloseBrace),
                OpenBracket => {
                    depth_buf.push(token_buffer.len() as u32);
                    Of::OpenBracket { end: 0 }
                }
                CloseBracket => finish_group!(OpenBracket, CloseBracket),
                At => Of::At,
                Pound => Of::Pound,
                Tilde => Of::Tilde,
                Question => Of::Question,
                Colon => Of::Colon,
                Dollar => Of::Dollar,
                Eq => join! { [SingleEq]
                    Bang => BangEq,
                    Eq => DoubleEq,
                    Gt => GtEq,
                    Lt => LtEq,
                },
                Bang => Of::Bang,
                Lt => join! { [Lt]
                    Lt => ShiftLeft,
                    Eq => LtEq,
                },
                Gt => join! { [Gt]
                    Gt => ShiftRight,
                    Eq => GtEq,
                },
                Minus => Of::Minus,
                And => join! { [SingleAnd] And => DoubleAnd },
                Or => join! { [SingleOr] Or => DoubleOr },
                Plus => Of::Plus,
                Star => Of::Star,
                Slash => Of::Slash,
                Caret => Of::Caret,
                Percent => Of::Percent,
                Eof => {
                    token_buffer.push(Token { of: Of::Eof, span });
                    break;
                }
            };
            token_buffer.push(Token { of, span });
        }
        token_buffer.shrink_to_fit();
        Ok(Parser {
            span_info: SpanInfo::new_in(bump),
            bump,
            tokens: TokenStream { next: 0, tokens: token_buffer.into_bump_slice() },
            source,
        })
    }
}

struct BumpyPred {
    bump: Bump,
    pred: PredBuilder<'static>,
}
impl BumpyPred {
    fn pred<'a>(&'a self) -> &'a PredBuilder<'a> {
        unsafe { std::mem::transmute::<&PredBuilder<'static>, &'a PredBuilder<'a>>(&self.pred) }
    }
}

impl Drop for BumpyPred {
    fn drop(&mut self) {
        // sanity just case we accidently have drop on Pred although we shouldn't
        // we don't want dropping pred to potentially access bump which may have already been dropped
        self.pred = PredBuilder::False;
    }
}

#[cfg(test)]
mod test {
    use std::fmt::Debug;

    use super::*;
    use bumpalo::Bump;

    #[derive(Default)]
    struct Tester {
        bump: Bump,
    }
    impl Tester {
        #[track_caller]
        fn expr_eq(&mut self, input: &str, expected: Expr) {
            self.bump.reset();
            let mut parser = Parser::init(&self.bump, input).unwrap();
            match parser.parse_expr(BP_NONE) {
                Ok((got, _)) => {
                    if got != expected {
                        panic!(
                            "{}:\n input: {:?}\n expected: {:?}\n got: {:?}",
                            "Expression parse did not match expected", input, expected, got
                        )
                    }
                    if !parser.tokens.is_empty() && parser.tokens.peek().of != Of::Eof {
                        panic!(
                            "{}:\n input: {:?}\n expected: {:?}\n Unconsumed tokens: {:?}",
                            "Expression parse succeeded but did not consume all input",
                            input,
                            expected,
                            parser.tokens.peek_slice()
                        )
                    }
                }
                Err(err) => {
                    panic!(
                        "{}: \n input: {:?}\n expected: {:?}\n err: {:?}",
                        "Expression failed to parse", input, expected, err,
                    )
                }
            }
        }

        #[track_caller]
        fn expr_err(&mut self, input: &str, expected_kind: ErrorKind) {
            self.bump.reset();
            let mut parser = Parser::init(&self.bump, input).unwrap();
            match parser.parse_expr(BP_NONE) {
                Ok(got) => {
                    panic!(
                        "{}:\n input: {:?}\n expected_err: {:?}\n got: {:?}",
                        "Expression parse succeeded unexpectedly", input, expected_kind, got
                    )
                }
                Err(err) => {
                    if err.kind != expected_kind {
                        panic!(
                            "{}:\n input: {:?}\n expected_err: {:?}\n got_err: {:?}",
                            "Expression parse failed with wrong error kind", input, expected_kind, err
                        )
                    }
                }
            }
        }
    }
    #[derive(Clone, Copy, Default)]
    struct Indent(u32);
    impl std::fmt::Display for Indent {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            for _ in 0..self.0 {
                f.write_str("    ");
            }
            Ok(())
        }
    }
    impl Indent {
        fn more(self) -> Indent {
            Indent(self.0 + 1)
        }
    }

    fn out(indent: Indent, kind: &str, thing: &dyn Debug, text: &str) {
        println!("{indent}{kind:>5} ┬ {thing:?}\n{indent}      └ `{text}`");
    }

    fn recursive_span_print(indent: Indent, info: &SpanInfo, source: &str, expr: &Expr) {
        if let Some(span) = info.get(expr) {
            out(indent, "EXPR", expr, span.text(source));
        }

        let indent = indent.more();
        match expr {
            Expr::Ident(text) => {
                if let Some(span) = info.get(*text) {
                    out(indent, "IDENT", text, span.text(source));
                }
            }
            Expr::String(text) => {
                if let Some(span) = info.get(*text) {
                    out(indent, "TEXT", text, span.text(source));
                }
            }
            Expr::Bytes(items) => (),
            Expr::List(exprs) => {
                for child in *exprs {
                    recursive_span_print(indent, info, source, child);
                }
            }
            Expr::BinOp(operator, exprs) => {
                if let Some(span) = info.bin_op_span(exprs) {
                    out(indent, "BINOP", operator, span.text(source));
                }
                for child in *exprs {
                    recursive_span_print(indent, info, source, child);
                }
            }
            Expr::UniOp(operator, expr) => {
                if let Some(span) = info.uni_op_span(expr) {
                    out(indent, "UNIOP", operator, span.text(source));
                }
                recursive_span_print(indent, info, source, *expr);
            }
            _ => (),
        }
    }

    #[test]
    fn expr_span_test() {
        let source = r#"!foo.starts_with("text") && ($span_duration > 23ms + 42ms || x in [1, "hello", false])"#;
        let bump = Bump::new();
        let mut parser = Parser::init(&bump, &source).unwrap();
        let (expr, span) = parser.parse_expr(0).unwrap();
        let indent = Indent::default();
        out(indent, "BASE", &expr, span.text(source));
        recursive_span_print(indent.more(), &parser.span_info, source, &expr);
    }

    #[test]
    fn infix_collection() {
        use Expr::*;
        use Op1::*;
        use Op2::*;
        let mut must = Tester::default();
        let a = Ident("a");
        must.expr_eq(stringify!(a != 0), BinOp(NotEq, &[a, Signed(0)]));
    }

    #[test]
    fn in_operator() {
        let mut must = Tester::default();
        must.expr_eq(stringify!(in), Expr::Ident("in"));
        must.expr_eq(
            stringify!(field in ["a", "b"]),
            Expr::BinOp(Op2::In, &[Expr::Ident("field"), Expr::List(&[Expr::String("a"), Expr::String("b")])]),
        );
    }

    #[test]
    fn method_expr() {
        let mut must = Tester::default();
        must.expr_eq(
            stringify!(field.starts_with("hello")),
            Expr::BinOp(Op2::StartsWith, &[Expr::Ident("field"), Expr::String("hello")]),
        );
        must.expr_eq(
            stringify!(field.ends_with("hello")),
            Expr::BinOp(Op2::EndsWith, &[Expr::Ident("field"), Expr::String("hello")]),
        );
        must.expr_eq(
            stringify!(field.ends_with("hello",)),
            Expr::BinOp(Op2::EndsWith, &[Expr::Ident("field"), Expr::String("hello")]),
        );
        must.expr_eq(
            stringify!(field.contains("hello")),
            Expr::BinOp(Op2::Contains, &[Expr::Ident("field"), Expr::String("hello")]),
        );
        must.expr_eq(stringify!(field.exists()), Expr::UniOp(Op1::Exists, &Expr::Ident("field")));
    }

    #[test]
    fn method_not() {
        use Expr::*;
        use Op1::*;
        use Op2::*;
        let mut must = Tester::default();
        must.expr_eq(
            stringify!(!field.starts_with("hello")),
            UniOp(LogicalNot, &BinOp(StartsWith, &[Ident("field"), String("hello")])),
        );
    }

    #[test]
    fn builtin() {
        let mut must = Tester::default();
        must.expr_eq("$span_parent", Expr::Ident("$span_parent"));
        must.expr_eq("$span", Expr::Ident("$span"));
    }

    #[test]
    fn list_literal() {
        use Expr::*;
        let mut must = Tester::default();
        must.expr_eq("[true, \"hello\"]", List(&[Bool(true), String("hello")]));
        must.expr_eq("[]", List(&[]));
        must.expr_eq("[1]", List(&[Signed(1)]));
        must.expr_eq("[1, 2]", List(&[Signed(1), Signed(2)]));
        must.expr_eq("[1, 2,]", List(&[Signed(1), Signed(2)]));
        must.expr_eq("[1 + 2, 3 * 4]", List(&[Signed(3), Signed(12)]));
    }

    #[test]
    fn expression_parsing_and_folding() {
        use Expr as Lit;
        let mut must = Tester::default();
        must.expr_eq("1", Lit::Signed(1));
        must.expr_eq("1 + 1", Lit::Signed(2));
        must.expr_eq("1024 * 1024", Lit::Signed(1024 * 1024));
        must.expr_eq("100 / 5", Lit::Signed(20));

        must.expr_eq(stringify!("hello" + "world"), Lit::String("helloworld"));
        must.expr_eq(stringify!("a" + "b" + "c"), Lit::String("abc"));

        must.expr_eq(stringify!(10 + 30 / 2), Lit::Signed(10 + (30 / 2)));
        must.expr_eq(stringify!(10 * 3 + 2), Lit::Signed((10 * 3) + 2));
        must.expr_eq(stringify!(10 + 3 * 2), Lit::Signed(10 + (3 * 2)));
        must.expr_eq(stringify!(10 / 2 * 5), Lit::Signed((10 / 2) * 5));

        must.expr_eq(stringify!(10 - 3 - 2), Lit::Signed((10 - 3) - 2));
        must.expr_eq(stringify!(10 / 5 / 2), Lit::Signed((10 / 5) / 2));

        must.expr_eq(stringify!((10 + 30) / 2), Lit::Signed((10 + 30) / 2));
        must.expr_eq(stringify!(10 * (3 + 2)), Lit::Signed(10 * (3 + 2)));

        must.expr_eq(stringify!(-10), Lit::Signed(-10));
        must.expr_eq(stringify!(5 + -10), Lit::Signed(5 + (-10)));
        must.expr_eq(stringify!(-5 + 10), Lit::Signed((-5) + 10));
        must.expr_eq(stringify!(-5 * -10), Lit::Signed((-5) * (-10)));
        must.expr_eq(stringify!(-(5 + 10)), Lit::Signed(-(5 + 10)));
        must.expr_eq("-1.5", Lit::Float(-1.5));
        must.expr_eq("-10s", Lit::Duration(-10.0));

        must.expr_eq(stringify!(10 - 30), Lit::Signed(-20));
        must.expr_eq("10 - -30", Lit::Signed(40));

        must.expr_eq(stringify!(1.0 + 1.0), Lit::Float(2.0));
        must.expr_eq(stringify!(1.0 + 2 * 3.5), Lit::Float(1.0 + (2.0 * 3.5)));
    }

    #[test]
    fn logical_conjuctions() {
        use Expr::*;
        use Op1::*;
        use Op2::*;
        let mut must = Tester::default();
        must.expr_eq("a && b", BinOp(LogicalAnd, &[Ident("a"), Ident("b")]));
        must.expr_eq("a || b", BinOp(LogicalOr, &[Ident("a"), Ident("b")]));
        must.expr_eq("(a || b) && c", BinOp(LogicalAnd, &[BinOp(LogicalOr, &[Ident("a"), Ident("b")]), Ident("c")]));
    }

    #[test]
    fn simple_literal() {
        use Expr::*;
        let mut must = Tester::default();
        must.expr_eq("10 - -30", Signed(40));
        must.expr_eq("1", Signed(1));
        must.expr_eq("1 + 1", Signed(2));
        must.expr_eq("1024 * 1024", Signed(1024 * 1024));
        must.expr_eq("100 / 5", Signed(20));
        must.expr_eq(stringify!("hello" + "world"), String("helloworld"));
        must.expr_eq(stringify!("a" + "b" + "c"), String("abc"));
        must.expr_eq(stringify!(10 - 30), Signed(-20));
        must.expr_eq(stringify!(1.0 + 1.0), Float(2.0));
    }
}

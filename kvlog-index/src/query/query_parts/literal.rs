use std::alloc::Layout;
use std::mem::MaybeUninit;
use std::num::{ParseFloatError, ParseIntError};

use bumpalo::Bump;
use ra_ap_rustc_lexer::unescape::{
    byte_from_char, unescape_byte, unescape_char, unescape_mixed, unescape_unicode, MixedUnit, Mode,
};
use ra_ap_rustc_lexer::LiteralKind;

use super::parser::Expr;

impl<'a> Expr<'a> {
    fn as_float(&self) -> Option<f64> {
        match self {
            Expr::Float(value) => Some(*value),
            Expr::Signed(value) => Some(*value as f64),
            Expr::Unsigned(value) => Some(*value as f64),
            _ => None,
        }
    }
}
impl From<u64> for Expr<'_> {
    fn from(value: u64) -> Self {
        if value > (i64::MAX as u64) {
            Expr::Unsigned(value)
        } else {
            Expr::Signed(value as i64)
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum LiteralParseError {
    Unterminated,
    UnknownIntegerSuffix,
    NumericLiteralTooLong,
    UnknownPrefix,
    FloatError(ParseFloatError),
    IntError(ParseIntError),
    EscapeError(ra_ap_rustc_lexer::unescape::EscapeError), // todo fill in Varients
}

#[derive(Debug, PartialEq)]
pub enum EvalError {
    MismatchedTypes,
    UnsupportedOperation,
    DivideByZero,
    Overflow,
    InvalidTimeString,
}

fn bump_join<'a>(bump: &'a Bump, a: &[u8], b: &[u8]) -> &'a [u8] {
    unsafe {
        let mut buf = bump.alloc_layout(Layout::from_size_align(a.len() + b.len(), 1).unwrap());
        std::ptr::copy_nonoverlapping(a.as_ptr(), buf.as_ptr(), a.len());
        std::ptr::copy_nonoverlapping(b.as_ptr(), buf.as_ptr().add(a.len()), b.len());
        unsafe { std::slice::from_raw_parts(buf.as_ptr(), a.len() + b.len()) }
    }
}

enum Unified<'a, 'b> {
    String(&'a str, &'b str),
    Bytes(&'a [u8], &'b [u8]),
    Float(f64, f64),
    Duration(f64, f64),
    Integer(i128, i128),
    FailedToUnify,
}

fn unify_binop<'a, 'b>(lhs: Expr<'a>, rhs: Expr<'b>) -> Unified<'a, 'b> {
    use Expr::*;
    match lhs {
        Duration(lhs) => {
            let Duration(rhs) = rhs else {
                return Unified::FailedToUnify;
            };
            Unified::Duration(lhs, rhs)
        }
        String(lhs) => {
            let String(rhs) = rhs else {
                return Unified::FailedToUnify;
            };
            Unified::String(lhs, rhs)
        }
        Bytes(lhs) => {
            let Bytes(rhs) = rhs else {
                return Unified::FailedToUnify;
            };
            Unified::Bytes(lhs, rhs)
        }
        Float(lhs) => {
            let rhs = match rhs {
                Float(rhs) => rhs,
                Signed(rhs) => rhs as f64,
                Unsigned(rhs) => rhs as f64,
                _ => return Unified::FailedToUnify,
            };
            Unified::Float(lhs, rhs)
        }
        Signed(value) => {
            let rhs = match rhs {
                Float(rhs) => return Unified::Float(value as f64, rhs),
                Signed(rhs) => rhs as i128,
                Unsigned(rhs) => rhs as i128,
                _ => return Unified::FailedToUnify,
            };
            Unified::Integer(value as i128, rhs)
        }
        Unsigned(value) => {
            let rhs = match rhs {
                Float(rhs) => rhs as i128,
                Signed(rhs) => rhs as i128,
                Unsigned(rhs) => rhs as i128,
                _ => return Unified::FailedToUnify,
            };
            Unified::Integer(value as i128, rhs)
        }
        _ => return Unified::FailedToUnify,
    }
}

#[derive(Debug)]
pub enum LiteralBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitwiseAnd,
    BitwiseOr,
    Xor,
    ShiftLeft,
    ShiftRight,
}
impl Expr<'_> {
    fn eval_i128<'a>(value: i128) -> Result<Expr<'a>, EvalError> {
        if let Ok(signed) = value.try_into() {
            return Ok(Expr::Signed(signed));
        } else if let Ok(unsigned) = value.try_into() {
            return Ok(Expr::Unsigned(unsigned));
        } else {
            return Err(EvalError::Overflow);
        }
    }
    pub fn eval_binop<'a>(
        bump: &'a Bump,
        lhs: Expr<'_>,
        rhs: Expr<'_>,
        op: LiteralBinOp,
    ) -> Result<Expr<'a>, EvalError> {
        match op {
            LiteralBinOp::Add => Self::add(bump, lhs, rhs),
            LiteralBinOp::Sub => Self::sub(lhs, rhs),
            LiteralBinOp::Mul => Self::mul(lhs, rhs),
            LiteralBinOp::Div => Self::div(lhs, rhs),
            LiteralBinOp::Rem => Self::rem(lhs, rhs),
            LiteralBinOp::BitwiseAnd => Self::bitwise_and(lhs, rhs),
            LiteralBinOp::BitwiseOr => Self::bitwise_or(lhs, rhs),
            LiteralBinOp::Xor => Self::bitwise_xor(lhs, rhs),
            LiteralBinOp::ShiftLeft => Self::shift_left(lhs, rhs),
            LiteralBinOp::ShiftRight => Self::shift_right(lhs, rhs),
        }
    }
    pub fn div<'a>(lhs: Expr<'_>, rhs: Expr<'_>) -> Result<Expr<'a>, EvalError> {
        use Expr::*;
        if let Expr::Duration(lhs) = lhs {
            if let Some(rhs) = rhs.as_float() {
                if rhs == 0.0 {
                    return Err(EvalError::DivideByZero);
                } else {
                    return Ok(Duration(lhs / rhs));
                }
            }
        }
        match unify_binop(lhs, rhs) {
            Unified::Duration(lhs, rhs) => {
                if rhs == 0.0 {
                    Err(EvalError::DivideByZero)
                } else {
                    Ok(Duration(lhs / rhs))
                }
            }
            Unified::Float(lhs, rhs) => {
                if rhs == 0.0 {
                    Err(EvalError::DivideByZero)
                } else {
                    Ok(Float(lhs / rhs))
                }
            }
            Unified::Integer(lhs, rhs) => {
                if rhs == 0 {
                    return Err(EvalError::DivideByZero);
                } else {
                    Expr::eval_i128(lhs / rhs)
                }
            }
            Unified::FailedToUnify => {
                return Err(EvalError::MismatchedTypes);
            }
            _ => return Err(EvalError::UnsupportedOperation),
        }
    }
    pub fn mul<'a>(lhs: Expr<'_>, rhs: Expr<'_>) -> Result<Expr<'a>, EvalError> {
        use Expr::*;
        if let Expr::Duration(lhs) = lhs {
            if let Some(rhs) = rhs.as_float() {
                return Ok(Duration(lhs * rhs));
            }
        }
        if let Expr::Duration(rhs) = rhs {
            if let Some(lhs) = lhs.as_float() {
                return Ok(Duration(lhs * rhs));
            }
        }
        match unify_binop(lhs, rhs) {
            Unified::Float(lhs, rhs) => Ok(Float(lhs * rhs)),
            Unified::Integer(lhs, rhs) => Expr::eval_i128(lhs * rhs),
            Unified::Duration(lhs, rhs) => Ok(Duration(lhs * rhs)),
            Unified::FailedToUnify => {
                return Err(EvalError::MismatchedTypes);
            }
            _ => return Err(EvalError::UnsupportedOperation),
        }
    }
    pub fn sub<'a>(lhs: Expr<'_>, rhs: Expr<'_>) -> Result<Expr<'a>, EvalError> {
        use Expr::*;
        match unify_binop(lhs, rhs) {
            Unified::Float(lhs, rhs) => Ok(Float(lhs - rhs)),
            Unified::Integer(lhs, rhs) => Expr::eval_i128(lhs - rhs),
            Unified::Duration(lhs, rhs) => Ok(Duration(lhs - rhs)),
            Unified::FailedToUnify => {
                return Err(EvalError::MismatchedTypes);
            }
            _ => return Err(EvalError::UnsupportedOperation),
        }
    }
    pub fn add<'a>(bump: &'a Bump, lhs: Expr<'_>, rhs: Expr<'_>) -> Result<Expr<'a>, EvalError> {
        use Expr::*;
        match unify_binop(lhs, rhs) {
            Unified::String(lhs, rhs) => {
                let bytes = bump_join(bump, lhs.as_bytes(), rhs.as_bytes());
                unsafe { Ok(String(std::str::from_utf8_unchecked(bytes))) }
            }
            Unified::Bytes(lhs, rhs) => Ok(Bytes(bump_join(bump, lhs, rhs))),
            Unified::Duration(lhs, rhs) => Ok(Duration(lhs + rhs)),
            Unified::Float(lhs, rhs) => Ok(Float(lhs + rhs)),
            Unified::Integer(lhs, rhs) => Expr::eval_i128(lhs + rhs),
            Unified::FailedToUnify => {
                return Err(EvalError::MismatchedTypes);
            }
        }
    }
    pub fn rem<'a>(lhs: Expr<'_>, rhs: Expr<'_>) -> Result<Expr<'a>, EvalError> {
        match unify_binop(lhs, rhs) {
            Unified::Float(lhs, rhs) => {
                if rhs == 0.0 {
                    Err(EvalError::DivideByZero)
                } else {
                    Ok(Expr::Float(lhs % rhs))
                }
            }
            Unified::Integer(lhs, rhs) => {
                if rhs == 0 {
                    return Err(EvalError::DivideByZero);
                } else {
                    Expr::eval_i128(lhs % rhs)
                }
            }
            Unified::FailedToUnify => {
                return Err(EvalError::MismatchedTypes);
            }
            _ => return Err(EvalError::UnsupportedOperation),
        }
    }

    pub fn bitwise_and<'a>(lhs: Expr<'_>, rhs: Expr<'_>) -> Result<Expr<'a>, EvalError> {
        match unify_binop(lhs, rhs) {
            Unified::Float(lhs, rhs) => {
                // Cast floats to integers for bitwise operations
                Expr::eval_i128((lhs as i64 & rhs as i64) as i128)
            }
            Unified::Integer(lhs, rhs) => Expr::eval_i128(lhs & rhs),
            Unified::FailedToUnify => {
                return Err(EvalError::MismatchedTypes);
            }
            _ => return Err(EvalError::UnsupportedOperation),
        }
    }

    pub fn bitwise_or<'a>(lhs: Expr<'_>, rhs: Expr<'_>) -> Result<Expr<'a>, EvalError> {
        match unify_binop(lhs, rhs) {
            Unified::Float(lhs, rhs) => {
                // Cast floats to integers for bitwise operations
                Expr::eval_i128((lhs as i64 | rhs as i64) as i128)
            }
            Unified::Integer(lhs, rhs) => Expr::eval_i128(lhs | rhs),
            Unified::FailedToUnify => {
                return Err(EvalError::MismatchedTypes);
            }
            _ => return Err(EvalError::UnsupportedOperation),
        }
    }

    pub fn bitwise_xor<'a>(lhs: Expr<'_>, rhs: Expr<'_>) -> Result<Expr<'a>, EvalError> {
        match unify_binop(lhs, rhs) {
            Unified::Float(lhs, rhs) => {
                // Cast floats to integers for bitwise operations
                Expr::eval_i128((lhs as i64 ^ rhs as i64) as i128)
            }
            Unified::Integer(lhs, rhs) => Expr::eval_i128(lhs ^ rhs),
            Unified::FailedToUnify => {
                return Err(EvalError::MismatchedTypes);
            }
            _ => return Err(EvalError::UnsupportedOperation),
        }
    }
    pub fn shift_left<'a>(lhs: Expr<'_>, rhs: Expr<'_>) -> Result<Expr<'a>, EvalError> {
        match unify_binop(lhs, rhs) {
            Unified::Integer(lhs, rhs) => {
                if rhs < 0 || rhs > 63 {
                    return Err(EvalError::UnsupportedOperation);
                }
                Expr::eval_i128(lhs << rhs)
            }
            Unified::FailedToUnify => Err(EvalError::MismatchedTypes),
            _ => Err(EvalError::UnsupportedOperation),
        }
    }
    pub fn shift_right<'a>(lhs: Expr<'_>, rhs: Expr<'_>) -> Result<Expr<'a>, EvalError> {
        match unify_binop(lhs, rhs) {
            Unified::Integer(lhs, rhs) => {
                if rhs < 0 || rhs > 63 {
                    return Err(EvalError::UnsupportedOperation);
                }
                Expr::eval_i128(lhs >> rhs)
            }
            Unified::FailedToUnify => Err(EvalError::MismatchedTypes),
            _ => Err(EvalError::UnsupportedOperation),
        }
    }
    pub fn is_bool(&self) -> bool {
        matches!(self, Expr::Bool(_))
    }
    pub fn is_signed(&self) -> bool {
        matches!(self, Expr::Signed(_))
    }
    pub fn is_unsigned(&self) -> bool {
        matches!(self, Expr::Unsigned(_))
    }
}

const fn duration_unit_to_seconds(suffix: &[u8]) -> Option<f64> {
    const MICRO: &[u8] = "μs".as_bytes();
    let value = match suffix {
        b"ns" => 1.0e-9,
        MICRO | b"us" => 1.0e-6,
        b"ms" => 1.0e-3,
        b"s" => 1.0,
        b"m" => 60.0,
        b"h" => 3600.0,
        b"d" => 3600.0 * 24.0,
        b"y" => 3600.0 * 24.0 * 365.25,
        _ => return None,
    };
    Some(value)
}

const STACK_BUF_CAPACITY: usize = 48;
struct StackBuf {
    buf: MaybeUninit<[u8; STACK_BUF_CAPACITY]>,
    len: usize,
}
impl StackBuf {
    fn new() -> StackBuf {
        StackBuf { buf: MaybeUninit::uninit(), len: 0 }
    }
    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.buf.as_ptr() as *const u8, self.len) }
    }
    fn extend(&mut self, value: &[u8]) -> Result<(), LiteralParseError> {
        if self.len + value.len() > 48 {
            return Err(LiteralParseError::NumericLiteralTooLong);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                value.as_ptr(),
                (self.buf.as_mut_ptr() as *mut u8).add(self.len),
                value.len(),
            );
            self.len += value.len();
        }
        Ok(())
    }
}

impl<'b> Expr<'b> {
    pub fn parse_literal_kind(
        arena: &'b Bump,
        raw: &str,
        kind: LiteralKind,
        suffix_start: u32,
    ) -> Result<Self, LiteralParseError> {
        let raw = raw;
        use LiteralKind as Lit;
        let (mode, range) = match kind {
            Lit::Str { terminated: true } => (Mode::Str, 1..raw.len() - 1),
            Lit::ByteStr { terminated: true } => (Mode::ByteStr, 2..raw.len() - 1),
            Lit::RawStr { n_hashes: Some(n) } => (Mode::RawStr, 2 + n as usize..raw.len() - n as usize - 1),
            Lit::RawByteStr { n_hashes: Some(n) } => (Mode::RawByteStr, 3 + n as usize..raw.len() - n as usize - 1),
            Lit::Char { terminated: true } => (Mode::Char, 1..raw.len() - 1),
            Lit::Byte { terminated: true } => (Mode::Byte, 2..raw.len() - 1),
            Lit::Int { base, empty_int } => {
                let (offset, radix) = match base {
                    ra_ap_rustc_lexer::Base::Binary => (2, 2),
                    ra_ap_rustc_lexer::Base::Octal => (2, 8),
                    ra_ap_rustc_lexer::Base::Decimal => (0, 10),
                    ra_ap_rustc_lexer::Base::Hexadecimal => (2, 16),
                };
                let mut buf = StackBuf::new();
                let mut consumed = 0;
                let mut inner = &raw.as_bytes()[offset..suffix_start as usize];
                let mut i = 0;
                while let Some(ch) = inner.get(i) {
                    if *ch == b'_' {
                        buf.extend(&inner[consumed..i])?;
                        consumed = i + 1;
                    }
                    i += 1;
                }
                let bytes = if consumed == 0 {
                    inner
                } else {
                    buf.extend(&inner[consumed..])?;
                    buf.as_slice()
                };
                let contents = unsafe { std::str::from_utf8_unchecked(bytes) };

                let value = u64::from_str_radix(contents, radix).map_err(LiteralParseError::IntError)?;
                let suffix = &raw.as_bytes()[suffix_start as usize..];
                if suffix == b"" {
                    return Ok(Expr::from(value));
                }
                let Some(duration_offset) = duration_unit_to_seconds(suffix) else {
                    return Err(LiteralParseError::UnknownIntegerSuffix);
                };
                let duration = value as f64 * duration_offset;
                return Ok(Expr::Duration(duration));
            }
            Lit::Float { base, empty_exponent } => {
                let unsuffixed = &raw[..suffix_start as usize];
                let float: f64 = unsuffixed.parse().map_err(LiteralParseError::FloatError)?;
                let suffix = &raw.as_bytes()[suffix_start as usize..];
                if suffix == b"" {
                    return Ok(Expr::Float(float));
                }
                let Some(duration_offset) = duration_unit_to_seconds(suffix) else {
                    return Err(LiteralParseError::UnknownIntegerSuffix);
                };
                let duration = float * duration_offset;
                return Ok(Expr::Duration(duration));
            }
            // CStr not support in query syntax
            Lit::CStr { .. } | Lit::RawCStr { .. } => return Err(LiteralParseError::UnknownPrefix),
            // The only other cases are when Lit is unterminated
            _ => {
                return Err(LiteralParseError::Unterminated);
            }
        };
        match mode {
            Mode::Char => match unescape_char(&raw[range]) {
                Ok(value) => {
                    let mut buf = [0u8; 4];
                    let text = value.encode_utf8(&mut buf);
                    return Ok(Expr::String(arena.alloc_str(text)));
                }
                Err(err) => return Err(LiteralParseError::EscapeError(err)),
            },
            Mode::Byte => {
                let value = unescape_byte(&raw[range]).map_err(LiteralParseError::EscapeError)?;
                return Ok(Expr::Signed(value as i64));
            }
            mode @ (Mode::Str | Mode::RawStr) => {
                let inner = &raw[range];
                if mode == Mode::RawStr || inner.contains('\\') {
                    let mut buf = bumpalo::collections::Vec::with_capacity_in(inner.len(), arena);
                    let mut found_err = None;
                    unescape_unicode(inner, mode, &mut |v, chr| match chr {
                        Ok(ch) => {
                            buf.extend_from_slice(ch.encode_utf8(&mut [0u8; 4]).as_bytes());
                        }
                        Err(err) => {
                            found_err = Some(err);
                            return;
                        }
                    });
                    if let Some(err) = found_err {
                        return Err(LiteralParseError::EscapeError(err));
                    }
                    buf.shrink_to_fit();
                    return Ok(Expr::String(unsafe { std::str::from_utf8_unchecked(buf.into_bump_slice()) }));
                } else {
                    return Ok(Expr::String(arena.alloc_str(inner)));
                }
            }
            mode @ (Mode::ByteStr | Mode::RawByteStr) => {
                let inner = &raw[range];
                if mode == Mode::RawByteStr || inner.contains('\\') {
                    let mut buf = bumpalo::collections::Vec::with_capacity_in(inner.len(), arena);
                    let mut found_err = None;
                    unescape_unicode(inner, mode, &mut |v, chr| match chr {
                        Ok(ch) => buf.push(byte_from_char(ch)),
                        Err(err) => {
                            found_err = Some(err);
                            return;
                        }
                    });
                    if let Some(err) = found_err {
                        return Err(LiteralParseError::EscapeError(err));
                    }
                    return Ok(Expr::Bytes(buf.into_bump_slice()));
                } else {
                    return Ok(Expr::Bytes(arena.alloc_str(inner).as_bytes()));
                }
            }
            Mode::CStr | Mode::RawCStr => unreachable!("Due to UnknownPrefix error above"),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ra_ap_rustc_lexer::TokenKind;
    use Expr::*;
    use LiteralParseError::*;
    #[derive(Default)]
    struct Tester {
        buf: Bump,
    }
    impl Tester {
        #[track_caller]
        fn parse_error(&mut self, input: &str, expected_err: LiteralParseError) {
            let token = ra_ap_rustc_lexer::tokenize(input).next().unwrap();
            let TokenKind::Literal { kind, suffix_start } = token.kind else {
                panic!("Invalid test input must tokenize to a literal")
            };
            if token.len != input.len() as u32 {
                panic!("Invalid test input must tokenize to a single literal")
            }

            self.buf.reset();
            match Expr::parse_literal_kind(&self.buf, input, kind, suffix_start) {
                Ok(got) => {
                    panic!(
                        "{}: \n input: `{}`\n expected_err: {:?}\n got: {:?}",
                        "Expected error but literal parsed successfully", input, expected_err, got
                    )
                }
                Err(err) => {
                    if err != expected_err {
                        panic!(
                            "{}: \n input: `{}`\n expected_err: {:?}\n got_err: {:?}",
                            "Expected one error but got another while parsing literal", input, expected_err, err
                        )
                    }
                }
            }
        }
        #[track_caller]
        fn parse_eq(&mut self, input: &str, expected: Expr) {
            let token = ra_ap_rustc_lexer::tokenize(input).next().unwrap();
            let TokenKind::Literal { kind, suffix_start } = token.kind else {
                panic!("Invalid test input must tokenize to a literal")
            };
            if token.len != input.len() as u32 {
                panic!("Invalid test input must tokenize to a single literal")
            }

            self.buf.reset();
            match Expr::parse_literal_kind(&mut self.buf, input, kind, suffix_start) {
                Ok(got) => {
                    if got != expected {
                        panic!(
                            "{}: \n input: `{}`\n expected: {:?}\n got: {:?}",
                            "Literal parse did not match expected", input, expected, got
                        )
                    }
                }
                Err(err) => {
                    panic!(
                        "{}: \n input: `{}`\n expected: {:?}\n err: {:?}",
                        "Unexpected literal parse failure", input, expected, err,
                    )
                }
            }
        }
    }

    #[test]
    fn chars() {
        let mut must = Tester::default();
        must.parse_eq("'a'", String("a"));
        must.parse_eq("'\\n'", String("\n"));
        must.parse_eq("'\\u{1F600}'", String("😀"));
        must.parse_error("'\\xFF'", EscapeError(ra_ap_rustc_lexer::unescape::EscapeError::OutOfRangeHexEscape));

        must.parse_eq("b'a'", Signed(b'a' as i64));
        must.parse_eq("b'\\xFF'", Signed(255));
    }
    #[test]
    fn strings() {
        let mut must = Tester::default();
        must.parse_eq(stringify!("a"), String("a"));
        must.parse_eq(stringify!("a\na"), String("a\na"));
        must.parse_eq(stringify!("a\"a"), String("a\"a"));
        must.parse_eq(stringify!(r##"a"##), String("a"));
        must.parse_eq(stringify!(r##"abc"##), String("abc"));
    }
    #[test]
    fn duration() {
        let mut must = Tester::default();
        must.parse_eq(stringify!(2ns), Duration(2e-9));
        must.parse_eq(stringify!(12.9us), Duration(12.9e-6));
        must.parse_eq(stringify!(32ms), Duration(32e-3));
        must.parse_eq(stringify!(142.5s), Duration(142.5));
        must.parse_eq(stringify!(142.5s), Duration(142.5));
    }
    #[test]
    fn integer() {
        let mut must = Tester::default();
        must.parse_eq(stringify!(123), Signed(123));
        must.parse_eq(stringify!(0xff_ff__ff), Signed(0xff_ffff));
        must.parse_eq(stringify!(0xFF_FF__FF), Signed(0xFF_FFFF));
        must.parse_eq(stringify!(0b101_01111_01011), Signed(0b101_01111_01011));
        must.parse_eq(stringify!(0xffff_ffff_ffff_ffff), Unsigned(u64::MAX));
    }
    #[test]
    fn bytes() {
        let mut must = Tester::default();
        must.parse_eq(stringify!(b"a"), Bytes(b"a"));
        must.parse_eq(stringify!(br##"a"##), Bytes(b"a"));
        must.parse_eq(stringify!(br##"abc"##), Bytes(b"abc"));
        must.parse_eq(stringify!(b"a\nbc"), Bytes(b"a\nbc"));
        must.parse_eq(stringify!(b"a\xFFbc"), Bytes(b"a\xFFbc"));
    }
}

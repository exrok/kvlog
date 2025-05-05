use kvlog::encoding::{Key, Value};
use kvlog::LogLevel;
use ra_ap_rustc_lexer::TokenKind as Tok;
use ra_ap_rustc_lexer::{Cursor, LiteralKind};
use std::fs;
use std::hash::Hasher;
use std::io;
use std::path::Path;
use std::process::Command;
use std::str::FromStr;
use std::{borrow::Cow, fmt::write};

// mod query_parts;
// mod ai;

enum SimplePredicate {
    Debug,
    Info,
    Warn,
    Err,
    HasSpan,
}
#[derive(Default, Debug)]
pub struct PredicateSet {
    require: u32,
    mask: u32,
}
impl PredicateSet {
    pub fn matches_raw(&self, bits: u8) -> bool {
        (bits | (self.require as u8)) + (bits & (self.mask as u8)) > 127
    }
    pub fn apply_level_mask(&mut self, mask: u8) {
        self.require |= 0b1111111;
        self.mask = mask as u32;
    }
}

#[derive(Debug)]
pub struct QueryExpr {
    pub simple: PredicateSet,
    pub fields: Vec<(Box<str>, FieldPredicate)>,
}
impl QueryExpr {
    pub fn new(text: &str) -> Result<QueryExpr, ParseError> {
        let mut fields = Vec::default();
        let mut ts = TokenStream::new(text);
        while let Some((key, predicate)) = munch(&mut ts)? {
            fields.push((key.into(), predicate));
        }
        Ok(QueryExpr { simple: PredicateSet::default(), fields })
    }
}

// white space is ignored
// quotes around strings are optional unless they contain char outside '-_a-zA-Z0-9.'
#[derive(PartialEq, Clone)]
pub enum FieldPredicate {
    Eq(Box<[u8]>),                     // field = value
    EqUUID(uuid::Uuid),                // field: uuid = value
    Contains(Box<[u8]>),               // field.contains(value)
    StartsWith(Box<[u8]>),             // field.starts_with(value)
    EndsWith(Box<[u8]>),               // field.ends_with(value)
    IsI64(i64),                        // field: int = 34234
    IsU64(u64),                        // field: int = 34234
    IsBool(bool),                      // field: bool = true
    RangeNum { min: i64, max: u64 },   // field: int > 492234
    RangeFloat { min: f64, max: f64 }, // field: float > 23423.0
    Exists,
    NotExists,
    // Not(Box<FieldPredicate>),
    // And(Box<[FieldPredicate]>),
    // Or(Box<[FieldPredicate]>),
}

impl FieldPredicate {
    pub fn matches(&self, value: Value) -> bool {
        match value {
            Value::String(text) => self.matches_text(text),
            Value::UUID(uuid) => self.matches_uuid(uuid),
            Value::U64(int) => {
                if let FieldPredicate::IsU64(num) = self {
                    return *num == int;
                }
                return false;
            }
            Value::I64(int) => {
                if let FieldPredicate::IsI64(num) = self {
                    return *num == int;
                }
                return false;
            }
            _ => false,
        }
    }
    pub fn matches_text(&self, text: &[u8]) -> bool {
        match self {
            Self::Eq(value) => text == &value[..],
            Self::Contains(value) => memchr::memmem::find(text, value).is_some(),
            Self::StartsWith(value) => text.starts_with(value),
            Self::EndsWith(value) => text.ends_with(value),
            Self::Exists => true,
            _ => false,
        }
    }
    pub fn matches_uuid(&self, id: uuid::Uuid) -> bool {
        match self {
            Self::EqUUID(value) => *value == id,
            Self::Exists => true,
            _ => false,
        }
    }
    pub fn matches_missing(&self) -> bool {
        match self {
            FieldPredicate::NotExists => true,
            _ => false,
        }
    }
}

impl std::fmt::Debug for FieldPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Eq(arg0) => write!(f, "Eq(\"{}\")", arg0.escape_ascii()),
            Self::EqUUID(arg0) => write!(f, "EqUUID({:?})", arg0),
            Self::Contains(arg0) => write!(f, "Contains(\"{}\")", arg0.escape_ascii()),
            Self::StartsWith(arg0) => write!(f, "StartsWith(\"{}\")", arg0.escape_ascii()),
            Self::EndsWith(arg0) => write!(f, "EndsWith(\"{}\")", arg0.escape_ascii()),
            Self::IsI64(arg0) => f.debug_tuple("IsI64").field(arg0).finish(),
            Self::IsU64(arg0) => f.debug_tuple("IsU64").field(arg0).finish(),
            Self::IsBool(arg0) => f.debug_tuple("IsBool").field(arg0).finish(),
            Self::RangeNum { min, max } => f.debug_struct("RangeNum").field("min", min).field("max", max).finish(),
            Self::RangeFloat { min, max } => f.debug_struct("RangeFloat").field("min", min).field("max", max).finish(),
            Self::Exists => f.write_str("Exists"),
            Self::NotExists => f.write_str("NotExists"),
        }
    }
}

struct ParsedQuery {
    level_mask: u8,
    fields: Vec<(String, FieldPredicate)>,
    before_ns: Option<u64>,
    after_ns: Option<u64>,
}
type Str<'a> = beef::lean::Cow<'a, str>;

fn extra_str_literal<'a>(raw: &'a str, kind: LiteralKind) -> Option<Str<'a>> {
    use ra_ap_rustc_lexer::unescape::Mode;
    let (mode, inner) = match kind {
        LiteralKind::Str { .. } => (Mode::Str, raw.get(1..raw.len() - 1)?),
        LiteralKind::ByteStr { .. } => (Mode::ByteStr, raw.get(2..raw.len() - 1)?),
        LiteralKind::RawStr { n_hashes } => {
            let n_hashes = n_hashes.unwrap_or(0);
            (Mode::RawStr, raw.get(2 + n_hashes as usize..raw.len() - n_hashes as usize - 1)?)
        }
        LiteralKind::RawByteStr { n_hashes } => {
            let n_hashes = n_hashes.unwrap_or(0);
            (Mode::RawByteStr, raw.get(3 + n_hashes as usize..raw.len() - n_hashes as usize - 1)?)
        }
        LiteralKind::RawCStr { n_hashes } => {
            let n_hashes = n_hashes.unwrap_or(0);
            (Mode::RawCStr, raw.get(2 + n_hashes as usize..raw.len() - n_hashes as usize - 1)?)
        }
        _ => {
            return None;
        }
    };
    for ch in inner.as_bytes() {
        if *ch != b'\\' {
            continue;
        }
        let mut data = String::with_capacity(inner.len());
        ra_ap_rustc_lexer::unescape::unescape_unicode(inner, mode, &mut |_, res| {
            if let Ok(s) = res {
                data.push(s);
            }
        });
        return Some(data.into());
    }
    return Some(Str::borrowed(inner));
}

struct Muncher<'a> {
    data: &'a [u8],
    off: usize,
}

impl<'a> Muncher<'a> {
    fn new(data: &'a [u8]) -> Self {
        Muncher { data, off: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.data.get(self.off).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let got = self.peek()?;
        self.off += 1;
        Some(got)
    }
    fn wext(&mut self) -> Option<u8> {
        while let Some(&ch) = self.data.get(self.off) {
            self.off += 1;
            if ch == b' ' || ch == b'\t' || ch == b'\n' {
                continue;
            } else {
                return Some(ch);
            }
        }
        return None;
    }
    fn skip_ws(&mut self) {
        while let Some(&ch) = self.data.get(self.off) {
            self.off += 1;
            if ch == b' ' || ch == b'\t' || ch == b'\n' {
                continue;
            } else {
                return;
            }
        }
    }
}

struct TokenStream<'a> {
    text: &'a str,
    cursor: Cursor<'a>,
    offset: u32,
    depth: u32,
}

#[derive(Clone, Copy)]
struct Span {
    start: u32,
    end: u32,
}

impl<'a> std::ops::Index<Span> for TokenStream<'a> {
    type Output = str;

    fn index(&self, index: Span) -> &Self::Output {
        &self.text[index.start as usize..index.end as usize]
    }
}
struct Token {
    kind: Tok,
    span: Span,
}

#[derive(Debug)]
pub enum ParseError {
    ExpectedIdent,
    UnexpectedEof,
    InvalidValue,
    Expected(Tok),
    ExpectedLiteral,
    Unknown,
    UnknownMethod(Box<str>),
}

impl<'a> TokenStream<'a> {
    fn get(&self, span: Span) -> &'a str {
        &self.text[span.start as usize..span.end as usize]
    }
    fn new(content: &'a str) -> TokenStream<'a> {
        return TokenStream { text: content, cursor: Cursor::new(content), depth: 0, offset: 0 };
    }
    fn expect(&mut self, kind: Tok) -> Result<Token, ParseError> {
        let next = self.next();
        if next.kind != kind {
            Err(ParseError::Expected(kind))
        } else {
            Ok(next)
        }
    }
    fn expect_ident(&mut self) -> Result<Token, ParseError> {
        let next = self.next();
        if next.kind != Tok::Ident {
            Err(ParseError::ExpectedIdent)
        } else {
            Ok(next)
        }
    }
    fn next(&mut self) -> Token {
        loop {
            let tok = self.cursor.advance_token();
            let start = self.offset;
            self.offset += tok.len;
            match tok.kind {
                Tok::CloseBrace | Tok::CloseBracket | Tok::CloseParen => {
                    self.depth -= 1;
                }
                Tok::OpenBrace | Tok::OpenBracket | Tok::OpenParen => {
                    self.depth += 1;
                }
                Tok::Whitespace => {
                    continue;
                }
                _ => (),
            }
            return Token { kind: tok.kind, span: Span { start, end: self.offset } };
        }
    }
    fn expect_stringly_value(&mut self) -> Result<Box<[u8]>, ParseError> {
        let value = self.next();
        match value.kind {
            Tok::Ident => {
                return Ok(self[value.span].as_bytes().into());
            }
            Tok::Literal { kind, .. } => {
                let value: Box<[u8]> = if let Some(value) = extra_str_literal(&self[value.span], kind) {
                    value.as_bytes().into()
                } else {
                    self[value.span].as_bytes().into()
                };
                return Ok(value);
            }
            _ => return Err(ParseError::Unknown),
        }
    }
}

fn munch<'a>(ts: &mut TokenStream<'a>) -> Result<Option<(&'a str, FieldPredicate)>, ParseError> {
    let field = {
        let this = &mut *ts;
        let next = this.next();
        if next.kind != Tok::Ident {
            if next.kind == Tok::Eof {
                return Ok(None);
            }

            Err(ParseError::ExpectedIdent)
        } else {
            Ok(next)
        }
    }?;
    let next = ts.next();
    match next.kind {
        Tok::Dot => {
            let method = ts.expect_ident()?;
            let method_name = ts.get(method.span);
            ts.expect(Tok::OpenParen)?;
            if method_name == "exists" {
                ts.expect(Tok::CloseParen)?;
                return Ok(Some((ts.get(field.span), FieldPredicate::Exists)));
            }
            let value = ts.expect_stringly_value()?;
            ts.expect(Tok::CloseParen)?;
            match method_name {
                "contains" => {
                    return Ok(Some((ts.get(field.span), FieldPredicate::Contains(value))));
                }
                "starts_with" => {
                    return Ok(Some((ts.get(field.span), FieldPredicate::StartsWith(value))));
                }
                "ends_with" => {
                    return Ok(Some((ts.get(field.span), FieldPredicate::EndsWith(value))));
                }
                method => return Err(ParseError::UnknownMethod(method.into())),
            }
        }
        Tok::Colon => {
            let ty = ts.expect_ident()?;
            ts.expect(Tok::Eq)?;
            let value = ts.expect_stringly_value()?;
            let text = std::str::from_utf8(&value).unwrap();
            let num = text.parse::<i64>().map_err(|_err| ParseError::InvalidValue)?;
            return Ok(Some((ts.get(field.span), FieldPredicate::IsI64(num))));
        }
        Tok::Eq => {
            let name = ts.get(field.span);
            let value = ts.next();
            let value: Box<[u8]> = match value.kind {
                Tok::Literal { kind, suffix_start } => {
                    if let Some(value) = extra_str_literal(&ts[value.span], kind) {
                        value.as_bytes().into()
                    } else {
                        let bytes = ts[value.span].as_bytes();
                        let text = std::str::from_utf8(&bytes).unwrap();
                        let num = text.parse::<i64>().map_err(|_err| ParseError::InvalidValue)?;
                        return Ok(Some((ts.get(field.span), FieldPredicate::IsI64(num))));
                    }
                }
                _ => {
                    return Err(ParseError::ExpectedLiteral);
                }
            };
            return Ok(Some((name, FieldPredicate::Eq(value))));

            // string value equality probably.
        }
        Tok::Eof => return Ok(Some((ts.get(field.span), FieldPredicate::Exists))),
        _ => return Err(ParseError::Unknown),
    }
}
/// given sorted parsed data with byte offsets, converts
/// the byte offsets to the line number
fn lineify(source: &[u8], data: &mut [(i32, Vec<String>)]) {
    let mut data = data.iter_mut();
    let Some(mut current) = data.next() else {
        return;
    };
    for (line, offset) in memchr::memchr_iter(b'\n', source).enumerate() {
        let offset = offset as i32;
        while current.0 < offset {
            current.0 = line as i32;
            if let Some(next) = data.next() {
                current = next;
            } else {
                return;
            }
        }
    }
}
/// using the same hash as the rust compiler, this function
fn hash_ignoring_white_space(bytes: &[u8]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for &ch in bytes {
        if ch == b' ' || ch == b'\t' || ch == b'\n' {
            continue;
        } else {
            hasher.write_u8(ch);
        }
    }
    hasher.finish()
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn tokenizer() {
        let mut t = TokenStream::new("field.starts_with(\"hello\") nice = \"hello may dariling\"");
        let got = munch(&mut t).unwrap().unwrap();
        println!("{:?}", got);
        let got = munch(&mut t).unwrap().unwrap();
        println!("{:?}", got);
        // let token = t.next();
        // assert_eq!(token.kind, Tok::Ident);
        // assert_eq!(&t[token.span], "field");
        // assert_eq!(t.next().unwrap(), Token::RawLiteral(b"1234"));
    }
}

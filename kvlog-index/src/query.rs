use bumpalo::Bump;
use kvlog::encoding::{Key, Value};
use kvlog::LogLevel;
use query_parts::PredBuilder;
use ra_ap_rustc_lexer::TokenKind as Tok;
use ra_ap_rustc_lexer::{Cursor, LiteralKind};
use std::fs;
use std::hash::Hasher;
use std::io;
use std::path::Path;
use std::process::Command;
use std::str::FromStr;
use std::sync::Arc;
use std::{borrow::Cow, fmt::write};

pub mod query_parts;
pub use query_parts::parse_query;
pub use query_parts::QueryParseError;

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
    pub(crate) bump: Bump,
    pred: PredBuilder<'static>,
}

unsafe impl Send for QueryExpr {}
unsafe impl Sync for QueryExpr {}

impl Default for QueryExpr {
    fn default() -> Self {
        Self { bump: Default::default(), pred: PredBuilder::True }
    }
}

impl Drop for QueryExpr {
    fn drop(&mut self) {
        // just being extract careful
        self.pred = PredBuilder::False;
    }
}

impl QueryExpr {
    pub fn pred<'a>(&'a self) -> &'a PredBuilder<'a> {
        // SAFETY: we are only using the bump allocator to parse the query, so
        // the lifetime of the pred is tied to the lifetime of the bump allocator.
        unsafe { std::mem::transmute::<&'a PredBuilder<'static>, &'a PredBuilder<'a>>(&self.pred) }
    }
    pub fn new_arc(text: &str) -> Result<Arc<QueryExpr>, query_parts::QueryParseError> {
        let mut query_expr = Arc::new(QueryExpr::default());
        let pred = query_parts::parse_query(&query_expr.bump, text)?;

        Arc::get_mut(&mut query_expr).unwrap().pred =
            unsafe { std::mem::transmute::<PredBuilder<'_>, PredBuilder<'static>>(pred) };
        // Safety: We tie the lifetime to the bumpallocator, since the the underlying allocations of dropped
        // or pin this is safe as along we don't reset or inspect the checks on the bumpalo.
        // This is tested by miri.
        Ok(query_expr)
    }
    pub fn new(text: &str) -> Result<QueryExpr, query_parts::QueryParseError> {
        let bump = bumpalo::Bump::new();
        let pred = query_parts::parse_query(&bump, text)?;
        // Safety: We tie the lifetime to the bumpallocator, since the the underlying allocations of dropped
        // or pin this is safe as along we don't reset or inspect the checks on the bumpalo.
        // This is tested by miri.
        Ok(QueryExpr { pred: unsafe { std::mem::transmute::<PredBuilder, PredBuilder<'static>>(pred) }, bump })
    }
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

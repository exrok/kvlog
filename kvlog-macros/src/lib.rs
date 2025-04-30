#![allow(dead_code)]
extern crate proc_macro;
mod lit;

macro_rules! builtin_keys {
    ($($key: ident),* $(,)?) => {
        #[allow(non_camel_case_types)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(u8)]
        enum StaticKey{
            $($key,)*
        }
        fn get_static_key(key: &str) -> Option<StaticKey> {
            match key {
                $(stringify!($key) => Some(StaticKey::$key),)*
                _ => None
            }
        }
    };
}

builtin_keys! {
    msg,
    err,
    error,
    cause,
    method,
    status,
    size,
    time,
    count,
    total,
    ms,
    id,
    user_id,
    object_id,
    caller,
    target,
    duration,
    timezone,
    content_type,
    conn_id,
    path,
    length,
    on,
    kind,
    sensor_id,
    handler,
    timestamp,
    elapsed,
    camera_id,
    system_id,
    next
}

use lit::{literal_inline, InlineKind};
use proc_macro::{
    token_stream::IntoIter, Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream,
    TokenTree,
};

fn chr(ch: char) -> TokenTree {
    TokenTree::Punct(Punct::new(ch, Spacing::Alone))
}
fn j(ch: char) -> TokenTree {
    TokenTree::Punct(Punct::new(ch, Spacing::Joint))
}

fn braced(ts: TokenStream) -> TokenTree {
    TokenTree::Group(Group::new(Delimiter::Brace, ts))
}

macro_rules! tok {
    ($ident:ident) => {
        TokenTree::Ident(Ident::new(stringify!($ident), Span::call_site()))
    };
    (_) => {
        TokenTree::Ident(Ident::new("_", Span::call_site()))
    };
    (()) => {
        TokenTree::Group(Group::new(Delimiter::Parenthesis, TokenStream::default()))
    };
    (( $($tt:tt)*) ) => {
        TokenTree::Group(Group::new(Delimiter::Parenthesis, TokenStream::from_iter(toks!($($tt)*))))
    };
    ({$($tt:tt)*}) => { $($tt)* };
    ([$com:literal]) => {
        TokenTree::Literal(Literal::string($com))
    };
    ([$com:literal; $ident:ident]) => {
        TokenTree::Literal(Literal::$ident($com))
    };
    ([_, $span: expr]) => {
        TokenTree::Ident(Ident::new("_", $span))
    };
    ([$ident:ident, $span: expr]) => {
        TokenTree::Ident(Ident::new(stringify!($ident), $span))
    };
    (<) => { chr('<') };
    (%) => { j(':') };
    (:) => { chr(':') };
    (>) => { chr('>') };
    (.) => { chr('.') };
    (;) => { chr(';') };
    (&) => { chr('&') };
    (=) => { chr('=') };
    (,) => { chr(',') };
    ($com:ident; $tt:tt) => {
        TokenTree::from($com.$tt.clone())
    };
}

macro_rules! toks {
    ($($tt:tt)*) => {
        [$(tok!($tt)),*]
    }
}

fn is_char(tt: &TokenTree, ch: char) -> bool {
    if let TokenTree::Punct(p) = tt {
        if p.as_char() == ch {
            return true;
        }
    }
    false
}

// could optize this but won't worry about it for now.
#[derive(Default)]
struct SpanAttributes {
    current: Option<Vec<TokenTree>>,
    start: Option<Vec<TokenTree>>,
    end: Option<Vec<TokenTree>>,
    parent: Option<Vec<TokenTree>>,
}

struct Codegen {
    out: TokenStream,
    collect: Vec<TokenTree>,
    builder: Ident,
    error: Option<(Span, Box<str>)>,
    message: Option<Box<str>>,
    span: SpanAttributes,
}

enum ExprKind {
    Debug,
    Display,
    Normal,
}

fn munch_expr(input: &mut IntoIter, output: &mut Vec<TokenTree>) -> ExprKind {
    output.clear();
    let mut kind = ExprKind::Normal;
    let Some(first) = input.next() else {
        return kind;
    };
    'lemmy: {
        if let TokenTree::Punct(tt) = &first {
            match tt.as_char() {
                '%' => {
                    kind = ExprKind::Display;
                    break 'lemmy;
                }
                '?' => {
                    kind = ExprKind::Debug;
                    break 'lemmy;
                }
                _ => (),
            }
        }
        output.push(first);
    }
    while let Some(tok) = input.next() {
        if is_char(&tok, ',') {
            break;
        }
        output.push(tok);
    }
    return kind;
}

impl Codegen {
    fn new(level: &str, builder: Ident) -> Codegen {
        let mut codegen = Codegen {
            out: TokenStream::new(),
            collect: Vec::with_capacity(64),
            builder,
            error: None,
            message: None,
            span: SpanAttributes::default(),
        };
        let log_level = TokenTree::Ident(Ident::new(level, Span::call_site()));
        codegen.out.extend(toks![
            use kvlog%:encoding%:Encode;
            let mut log = kvlog%:global_logger();
            let mut {codegen.fields()} = log.encoder.append_now(
                kvlog%:LogLevel%:{log_level}
            );
        ]);
        codegen
    }
    fn fields(&self) -> TokenTree {
        self.builder.clone().into()
    }
    fn runtime_field_from_key(&self, field: Ident) -> TokenTree {
        let field_text = field.to_string();
        if let Some(static_key) = get_static_key(&field_text) {
            let key = TokenTree::Literal(Literal::u8_unsuffixed(static_key as u8));
            TokenTree::Group(Group::new(
                Delimiter::Parenthesis,
                TokenStream::from_iter(toks![{ self.fields() }.raw_key({ key })]),
            ))
        } else {
            let key = TokenTree::Literal(Literal::string(&field_text));
            TokenTree::Group(Group::new(
                Delimiter::Parenthesis,
                TokenStream::from_iter(toks![{ self.fields() }.dynamic_key({ key })]),
            ))
        }
    }
    fn emit_field_value(&mut self, expr: TokenTree, field: Ident) {
        let method = TokenTree::Ident(Ident::new("encode_log_value_into", field.span()));
        self.collect.extend_from_slice(&toks![
            { expr }.{method}{self.runtime_field_from_key(field)};
        ]);
    }
    fn emit_display_field_value(&mut self, expr: TokenTree, field: Ident) {
        self.collect.extend_from_slice(&toks![
            { self.runtime_field_from_key(field) }.value_via_display(&{ expr });
        ]);
    }
    fn emit_debug_field_value(&mut self, expr: TokenTree, field: Ident) {
        self.collect.extend_from_slice(&toks![
            { self.runtime_field_from_key(field) }.value_via_debug(&{ expr });
        ]);
    }
    fn error(&mut self, span: Span, msg: &str) {
        if self.error.is_none() {
            self.error = Some((span, msg.into()))
        }
    }
    fn error_fmt(&mut self, span: Span, msg: Box<str>) {
        if self.error.is_none() {
            self.error = Some((span, msg))
        }
    }
    fn finish_creating(mut self) -> TokenStream {
        let target_key = TokenTree::Literal(Literal::u8_unsuffixed(StaticKey::target as u8));
        let message_key = TokenTree::Literal(Literal::u8_unsuffixed(StaticKey::msg as u8));
        self.collect.extend_from_slice(&toks![
            (module_path {chr('!')} ()).encode_log_value_into(
                {self.fields()}.raw_key({target_key})
            );

        ]);
        let fields = self.fields();
        self.out.extend(std::mem::take(&mut self.collect));

        if let Some(message) = &self.message {
            self.out.extend(toks![
                ({TokenTree::Literal(Literal::string(message))}).encode_log_value_into(
                    {fields.clone()}.raw_key({message_key})
                );
            ]);
        };
        let base_set = (self.span.current.is_some() as u8)
            + (self.span.start.is_some() as u8)
            + (self.span.end.is_some() as u8);

        if base_set > 1 {
            self.error(
                Span::call_site(),
                "More then one 'start', 'end', 'current' field for span specified.".into(),
            )
        }
        if self.span.parent.is_some() && self.span.start.is_none() {
            self.error(
                Span::call_site(),
                "`span.parent` can only be specified with span.start".into(),
            )
        }

        if let Some(current) = self.span.current {
            let tok = TokenTree::Group(Group::new(
                Delimiter::Parenthesis,
                TokenStream::from_iter(current),
            ));
            self.out.extend(toks![ {fields}.apply_span {tok}; ]);
        } else if let Some(start) = self.span.start {
            if let Some(parent) = self.span.parent {
                let mut args = TokenStream::from_iter(start);
                args.extend([TokenTree::Punct(Punct::new(',', Spacing::Alone))]);
                args.extend(parent);
                let tok = TokenTree::Group(Group::new(Delimiter::Parenthesis, args));
                self.out
                    .extend(toks![ {fields}.start_span_with_parent {tok}; ]);
            } else {
                let tok = TokenTree::Group(Group::new(
                    Delimiter::Parenthesis,
                    TokenStream::from_iter(start),
                ));
                self.out.extend(toks![ {fields}.start_span {tok}; ]);
            }
        } else if let Some(end) = self.span.end {
            let tok = TokenTree::Group(Group::new(
                Delimiter::Parenthesis,
                TokenStream::from_iter(end),
            ));
            self.out.extend(toks![ {fields}.end_span {tok}; ]);
        } else {
            self.out.extend(toks![ {fields}.apply_current_span(); ]);
        }

        self.out.extend(toks![ log.poke(); ]);

        if let Some((span, msg)) = self.error {
            let mut group = TokenTree::Group(Group::new(
                Delimiter::Parenthesis,
                TokenStream::from_iter([TokenTree::Literal(Literal::string(&msg))]),
            ));
            let mut punc = TokenTree::Punct(Punct::new('!', Spacing::Alone));
            punc.set_span(span);
            group.set_span(span);

            self.out.extend([
                TokenTree::Ident(Ident::new("compile_error", span)),
                punc,
                group,
                TokenTree::Punct(Punct::new(';', Spacing::Alone)),
            ]);
            //     }
        }
        TokenStream::from_iter([braced(self.out)])
    }
    fn parse_if(&mut self, ident: Ident, tokens: &mut IntoIter) {
        let span = ident.span();
        self.collect.push(ident.into());
        loop {
            let Some(tok) = tokens.next() else {
                self.error(span, "Expected Block for if statement".into());
                return;
            };
            if let TokenTree::Group(group) = &tok {
                if group.delimiter() == Delimiter::Brace {
                    let mut tmp = std::mem::take(&mut self.collect);
                    self.parse_values(group.stream().into_iter());
                    std::mem::swap(&mut tmp, &mut self.collect);
                    self.collect.push(TokenTree::Group(Group::new(
                        Delimiter::Brace,
                        TokenStream::from_iter(tmp),
                    )));
                    return;
                }
            }
            self.collect.push(tok);
        }
    }
    fn parse_span_attrib(&mut self, span: Ident, tokens: &mut IntoIter) {
        let Some(tok) = tokens.next() else {
            self.error(
                span.span(),
                "Eof will parsing span attribute expected".into(),
            );
            return;
        };
        let TokenTree::Ident(ident) = tok else {
            self.error(Span::call_site(), "Expected span field".into());
            return;
        };
        let Some(eq_tok) = tokens.next() else {
            self.error(ident.span(), "Eof after span expected `=`".into());
            return;
        };
        if !is_char(&eq_tok, '=') {
            self.error(eq_tok.span(), "expected `=` after span field".into());
            return;
        }
        let field = ident.to_string();
        let mut output = Vec::default();
        let _expr = munch_expr(tokens, &mut output);
        let attrib = match field.as_str() {
            "current" => &mut self.span.current,
            "start" => &mut self.span.start,
            "end" => &mut self.span.end,
            "parent" => &mut self.span.parent,
            other => {
                self.error_fmt(ident.span(), format!("unknown span field {}", other).into());
                return;
            }
        };
        if attrib.is_some() {
            self.error_fmt(
                ident.span(),
                format!("Span `{}` specified more than once", field.as_str()).into(),
            );
            return;
        }
        *attrib = Some(output);
    }

    fn parse_values(&mut self, mut tokens: IntoIter) {
        let mut tmp: Vec<TokenTree> = Vec::with_capacity(8);
        let mut last_was_block_set = false;
        while let Some(tok) = tokens.next() {
            let last_was_block = last_was_block_set;
            last_was_block_set = false;
            let key = match tok {
                TokenTree::Group(value) => {
                    self.error(value.span(), "unexpected span");
                    continue;
                }
                TokenTree::Ident(ident) => ident,
                TokenTree::Punct(punct) => {
                    let ch = punct.as_char();
                    if ch == ',' && last_was_block {
                        continue;
                    }
                    if ch != '?' && ch != '%' {
                        self.error(punct.span(), "Unexpected punct");
                        continue;
                    }
                    let Some(tt) = tokens.next() else {
                        self.error(punct.span(), "Unexpected EOF");
                        return;
                    };
                    let TokenTree::Ident(key_expr) = tt else {
                        self.error(tt.span(), "Expected name");
                        continue;
                    };

                    if ch == '?' {
                        self.emit_debug_field_value(TokenTree::Ident(key_expr.clone()), key_expr);
                    } else {
                        self.emit_display_field_value(TokenTree::Ident(key_expr.clone()), key_expr);
                    }

                    if let Some(tt) = tokens.next() {
                        if !is_char(&tt, ',') {
                            self.error(tt.span(), "Expected ,");
                            return;
                        }
                    }
                    continue;
                }
                TokenTree::Literal(lit) => {
                    if let InlineKind::String(lit) = literal_inline(lit.to_string()) {
                        self.message = Some(lit);
                    } else {
                        self.error(lit.span(), "Expected string message ");
                        return;
                    }
                    if let Some(tt) = tokens.next() {
                        if !is_char(&tt, ',') {
                            self.error(tt.span(), "Expected ,");
                            continue;
                        }
                    }
                    continue;
                }
            };
            let key_name = key.to_string();
            if key_name == "if" {
                self.parse_if(key, &mut tokens);
                last_was_block_set = true;
                continue;
            }
            let Some(tt) = tokens.next() else {
                self.emit_field_value(TokenTree::Ident(key.clone()), key);
                return;
            };
            match &tt {
                TokenTree::Punct(punct) => {
                    let ch = punct.as_char();
                    if ch == ',' {
                        self.emit_field_value(TokenTree::Ident(key.clone()), key);
                        continue;
                    }
                    if ch == '.' && key_name == "span" {
                        self.parse_span_attrib(key, &mut tokens);
                        continue;
                    }
                    if ch != '=' {
                        self.error(punct.span(), "Expected =");
                        continue;
                    }
                    tmp.clear();
                    let kind = munch_expr(&mut tokens, &mut tmp);
                    if tmp.is_empty() {
                        self.error(punct.span(), "Expected Expression following =");
                        continue;
                    }
                    let expr = TokenStream::from_iter(tmp.drain(..));
                    let expr = TokenTree::Group(Group::new(Delimiter::Parenthesis, expr));
                    match kind {
                        ExprKind::Debug => {
                            self.emit_debug_field_value(expr, key);
                        }
                        ExprKind::Display => {
                            self.emit_display_field_value(expr, key);
                        }
                        ExprKind::Normal => {
                            self.emit_field_value(expr, key);
                        }
                    }
                }
                _ => {
                    self.error(tt.span(), "Expected =");
                    continue;
                }
            }
        }
    }
}

fn emit_log(input: TokenStream, level: &str) -> TokenStream {
    let toks = input.into_iter();
    let mut codegen = Codegen::new(level, Ident::new("fields", Span::call_site()));
    codegen.parse_values(toks);
    codegen.finish_creating()
}

#[proc_macro]
pub fn info(input: TokenStream) -> TokenStream {
    emit_log(input, "Info")
}

#[proc_macro]
pub fn debug(input: TokenStream) -> TokenStream {
    emit_log(input, "Debug")
}

#[proc_macro]
pub fn error(input: TokenStream) -> TokenStream {
    emit_log(input, "Error")
}

#[proc_macro]
pub fn warn(input: TokenStream) -> TokenStream {
    emit_log(input, "Warn")
}

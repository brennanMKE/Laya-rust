//! JSON read the way Python's `json.loads` does and written the way `json.dumps` does
//! (laya-go's `internal/pyjson`).
//!
//! Laya serializes structured states and criteria with `json.dumps` before tokenizing them,
//! so the text must match Python byte for byte: key order kept, `", "` and `": "` separators,
//! Python's float `repr`, and escaped or literal non-ASCII.
//!
//! Parsing mirrors `json.loads`: object members keep their order, a repeated key keeps its
//! first position but takes its last value (dict semantics), numbers with a fraction or
//! exponent become floats and others arbitrary-precision integers (kept as digits), and `NaN`,
//! `Infinity` and `-Infinity` are accepted.

use std::fmt::Write as _;

/// A decoded JSON value, typed as Python would see it.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `None`
    Null,
    Bool(bool),
    /// An `int` of any size: its decimal digits with a leading `-` if negative (`-0` is `0`).
    Int(String),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    /// A `dict`: members in order, keys unique.
    Object(Vec<(String, Value)>),
}

impl Value {
    /// The value of `key` in an object.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Python's truthiness, inverted.
    pub fn is_falsy(&self) -> bool {
        match self {
            Value::Null => true,
            Value::Bool(b) => !b,
            Value::Int(s) => s == "0",
            Value::Float(f) => *f == 0.0,
            Value::Str(s) => s.is_empty(),
            Value::Array(a) => a.is_empty(),
            Value::Object(o) => o.is_empty(),
        }
    }

    /// Python's `v is None or v == ""`.
    pub fn is_none_or_empty(&self) -> bool {
        matches!(self, Value::Null) || matches!(self, Value::Str(s) if s.is_empty())
    }

    /// The Python type name: `NoneType`, `bool`, `int`, `float`, `str`, `list` or `dict`.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "NoneType",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "str",
            Value::Array(_) => "list",
            Value::Object(_) => "dict",
        }
    }

    /// `json.dumps(self, ensure_ascii=ascii)` with the default separators.
    pub fn dumps(&self, ascii: bool) -> String {
        let mut s = String::new();
        write_value(&mut s, self, ascii);
        s
    }
}

/// A parse failure, with the byte offset where it happened.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("pyjson: {msg} at offset {offset}")]
pub struct ParseError {
    pub msg: String,
    pub offset: usize,
}

/// Bounds nesting, as Python's recursion limit does.
const MAX_DEPTH: usize = 900;

/// Decodes one JSON document.
pub fn parse(data: &[u8]) -> Result<Value, ParseError> {
    let s = std::str::from_utf8(data).map_err(|e| ParseError { msg: "invalid UTF-8".into(), offset: e.valid_up_to() })?;
    let mut p = Parser { s, b: s.as_bytes(), i: 0 };
    p.space();
    let v = p.value(0)?;
    p.space();
    if p.i != p.b.len() {
        return Err(p.err("extra data"));
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a str,
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn err(&self, msg: &str) -> ParseError {
        ParseError { msg: msg.to_string(), offset: self.i }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    /// Skips the four whitespace characters `json.loads` allows.
    fn space(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn lit(&mut self, word: &str) -> bool {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, ParseError> {
        if depth > MAX_DEPTH {
            return Err(self.err("nesting too deep"));
        }
        let Some(c) = self.peek() else {
            return Err(self.err("expecting value"));
        };
        match c {
            b'{' => self.object(depth),
            b'[' => self.array(depth),
            b'"' => self.string().map(Value::Str),
            _ if self.lit("null") => Ok(Value::Null),
            _ if self.lit("true") => Ok(Value::Bool(true)),
            _ if self.lit("false") => Ok(Value::Bool(false)),
            _ if self.lit("NaN") => Ok(Value::Float(f64::NAN)),
            _ if self.lit("Infinity") => Ok(Value::Float(f64::INFINITY)),
            _ if self.lit("-Infinity") => Ok(Value::Float(f64::NEG_INFINITY)),
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(self.err("expecting value")),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value, ParseError> {
        self.i += 1;
        let mut members: Vec<(String, Value)> = Vec::new();
        let mut index = std::collections::HashMap::new();
        self.space();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Value::Object(members));
        }
        loop {
            self.space();
            if self.peek() != Some(b'"') {
                return Err(self.err("expecting property name enclosed in double quotes"));
            }
            let key = self.string()?;
            self.space();
            if self.peek() != Some(b':') {
                return Err(self.err("expecting ':' delimiter"));
            }
            self.i += 1;
            self.space();
            let val = self.value(depth + 1)?;
            if let Some(&j) = index.get(&key) {
                members[j] = (key, val); // dict(pairs): first position, last value
            } else {
                index.insert(key.clone(), members.len());
                members.push((key, val));
            }
            self.space();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Value::Object(members));
                }
                _ => return Err(self.err("expecting ',' delimiter")),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value, ParseError> {
        self.i += 1;
        let mut items = Vec::new();
        self.space();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.space();
            items.push(self.value(depth + 1)?);
            self.space();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(self.err("expecting ',' delimiter")),
            }
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.i;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.i += 1;
        }
        self.i - start
    }

    /// `json.loads`' NUMBER_RE: `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?`
    fn number(&mut self) -> Result<Value, ParseError> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        if self.peek() == Some(b'0') {
            self.i += 1;
        } else if self.digits() == 0 {
            return Err(self.err("expecting value"));
        }
        let int_end = self.i;
        let mut is_float = false;
        if self.peek() == Some(b'.') && matches!(self.b.get(self.i + 1), Some(b'0'..=b'9')) {
            self.i += 1;
            self.digits();
            is_float = true;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            let save = self.i;
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if self.digits() == 0 {
                self.i = save; // not an exponent; json.loads stops here too
            } else {
                is_float = true;
            }
        }
        if !is_float {
            let n = &self.s[start..int_end];
            return Ok(Value::Int(if n == "-0" { "0".into() } else { n.into() }));
        }
        let text = &self.s[start..self.i];
        // Out of range gives ±inf or ±0, as float() does.
        text.parse::<f64>().map(Value::Float).map_err(|_| self.err("bad number"))
    }

    /// Decodes a string at `self.i` (a `"`). Lone surrogate escapes, which Python keeps as
    /// lone code points, become U+FFFD.
    fn string(&mut self) -> Result<String, ParseError> {
        self.i += 1;
        let mut out = String::new();
        loop {
            // Copy the run of plain characters in one go.
            let run = self.i;
            while let Some(c) = self.peek() {
                if c == b'"' || c == b'\\' || c < 0x20 {
                    break;
                }
                self.i += 1;
            }
            out.push_str(&self.s[run..self.i]);
            let Some(c) = self.peek() else {
                return Err(self.err("unterminated string"));
            };
            match c {
                b'"' => {
                    self.i += 1;
                    return Ok(out);
                }
                b'\\' => {
                    let Some(e) = self.b.get(self.i + 1).copied() else {
                        return Err(self.err("unterminated string"));
                    };
                    self.i += 2;
                    match e {
                        b'"' | b'\\' | b'/' => out.push(e as char),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut r = self.hex4()?;
                            if (0xd800..0xdc00).contains(&r) && self.b[self.i..].starts_with(b"\\u") {
                                let save = self.i;
                                self.i += 2;
                                match self.hex4() {
                                    Ok(r2) if (0xdc00..=0xdfff).contains(&r2) => {
                                        r = 0x10000 + ((r - 0xd800) << 10) + (r2 - 0xdc00);
                                    }
                                    _ => self.i = save,
                                }
                            }
                            out.push(char::from_u32(r).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err(self.err("invalid \\escape")),
                    }
                }
                _ => return Err(self.err("invalid control character")),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, ParseError> {
        let h = self.b.get(self.i..self.i + 4).ok_or_else(|| self.err("invalid \\uXXXX escape"))?;
        let h = std::str::from_utf8(h).map_err(|_| self.err("invalid \\uXXXX escape"))?;
        if !h.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err(self.err("invalid \\uXXXX escape"));
        }
        self.i += 4;
        Ok(u32::from_str_radix(h, 16).unwrap())
    }
}

fn write_value(out: &mut String, v: &Value, ascii: bool) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Int(s) => out.push_str(s),
        Value::Float(f) => out.push_str(&float_repr(*f)),
        Value::Str(s) => quote(out, s, ascii),
        Value::Array(a) => {
            out.push('[');
            for (i, e) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value(out, e, ascii);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, e)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                quote(out, k, ascii);
                out.push_str(": ");
                write_value(out, e, ascii);
            }
            out.push('}');
        }
    }
}

/// Formats `f` as `json.dumps` does: Python's `repr(float)` (the shortest string that
/// round-trips), with NaN and the infinities spelled `NaN`, `Infinity` and `-Infinity`.
///
/// `repr` uses fixed notation when the decimal exponent is in [-4, 16) and scientific notation
/// otherwise (`1e-05`, `1e+16`, `1.5e+300`); fixed notation always has a fractional part
/// (`100.0`).
pub fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    // Rust's `{:e}` gives the shortest round-trip digits as d.ddde<exp>.
    let e = format!("{f:e}");
    let (neg, e) = match e.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, e.as_str()),
    };
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let digits: String = mant.chars().filter(|&c| c != '.').collect();

    let mut out = String::with_capacity(24);
    if neg {
        out.push('-');
    }
    if (-4..16).contains(&exp) {
        let point = exp + 1; // digits before the decimal point
        if point <= 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-point) as usize));
            out.push_str(&digits);
        } else if point as usize >= digits.len() {
            out.push_str(&digits);
            out.extend(std::iter::repeat_n('0', point as usize - digits.len()));
            out.push_str(".0");
        } else {
            out.push_str(&digits[..point as usize]);
            out.push('.');
            out.push_str(&digits[point as usize..]);
        }
        return out;
    }
    out.push_str(&digits[..1]);
    if digits.len() > 1 {
        out.push('.');
        out.push_str(&digits[1..]);
    }
    let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.unsigned_abs());
    out
}

/// Writes `s` as a JSON string literal the way `json.dumps` does. With `ascii`
/// (`ensure_ascii=True`) everything outside printable ASCII is escaped as `\uXXXX`, using
/// surrogate pairs above U+FFFF; without it only `"`, `\` and control characters below U+0020
/// are escaped.
pub fn quote(out: &mut String, s: &str, ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (ascii && c as u32 > 0x7e) => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{u:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_errors() {
        for s in ["", "{", "[1,]", "{\"a\" 1}", "01", "1.", "1e", "\"a", "\"\u{1}\"", "tru", "{} x", "{'a': 1}"] {
            assert!(parse(s.as_bytes()).is_err(), "{s:?} parsed");
        }
    }

    #[test]
    fn duplicate_keys() {
        let v = parse(br#"{"a": 1, "b": 2, "a": 3}"#).unwrap();
        assert_eq!(v.dumps(false), r#"{"a": 3, "b": 2}"#);
    }

    #[test]
    fn floats() {
        for (f, want) in [
            (1e-5, "1e-05"),
            (1e16, "1e+16"),
            (123.0, "123.0"),
            (-0.0, "-0.0"),
            (0.1, "0.1"),
            (1e22, "1e+22"),
            (5e-324, "5e-324"),
            (1.5e300, "1.5e+300"),
            (0.0001, "0.0001"),
            (123456789012345.6, "123456789012345.6"),
        ] {
            assert_eq!(float_repr(f), want);
        }
    }
}

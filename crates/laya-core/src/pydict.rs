//! What `laya_server.py` and `laya_batch.py` do to the dicts `json.loads` gives them:
//! indexing, `.items()`, `.get()` and `laya_batch.cache_key`, failing with the messages
//! Python's exceptions carry (`str(e)`), since the server returns them as `{"error": str(e)}`
//! (laya-go's `internal/pydict`).

use std::fmt::Write as _;

use crate::pyjson::Value;

/// A Python exception: `str(e)` is `msg`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{msg}")]
pub struct PyError {
    /// `KeyError`, `TypeError`, `AttributeError`.
    pub kind: &'static str,
    pub msg: String,
}

fn err(kind: &'static str, msg: String) -> PyError {
    PyError { kind, msg }
}

/// `v[key]` for a string key.
pub fn index<'a>(v: &'a Value, key: &str) -> Result<&'a Value, PyError> {
    match v {
        Value::Object(_) => v.get(key).ok_or_else(|| err("KeyError", repr(key))),
        Value::Str(_) => Err(err("TypeError", "string indices must be integers, not 'str'".into())),
        Value::Array(_) => Err(err("TypeError", "list indices must be integers or slices, not str".into())),
        v => Err(err("TypeError", format!("'{}' object is not subscriptable", v.type_name()))),
    }
}

/// `v.items()`.
pub fn items(v: &Value) -> Result<&[(String, Value)], PyError> {
    match v {
        Value::Object(m) => Ok(m),
        v => Err(err("AttributeError", format!("'{}' object has no attribute 'items'", v.type_name()))),
    }
}

/// `v.get(key)`: `None` when missing.
pub fn get<'a>(v: &'a Value, key: &str) -> Result<&'a Value, PyError> {
    match v {
        Value::Object(_) => Ok(v.get(key).unwrap_or(&Value::Null)),
        v => Err(err("AttributeError", format!("'{}' object has no attribute 'get'", v.type_name()))),
    }
}

/// Python's `key in v`.
pub fn contains(v: &Value, key: &str) -> Result<bool, PyError> {
    match v {
        Value::Object(_) => Ok(v.get(key).is_some()),
        Value::Array(a) => Ok(a.iter().any(|e| matches!(e, Value::Str(s) if s == key))),
        Value::Str(s) => Ok(s.contains(key)),
        v => Err(err("TypeError", format!("argument of type '{}' is not a container or iterable", v.type_name()))),
    }
}

/// `laya_batch._text`: strings as is, anything else `json.dumps`'d with `ensure_ascii=False`.
fn text(v: &Value) -> String {
    match v {
        Value::Str(s) => s.clone(),
        v => v.dumps(false),
    }
}

/// `laya_batch.cache_key(state, qdef)`:
///
/// ```text
/// json.dumps([_text(state), qdef["type"], _text(qdef["instructions"]), crit], ensure_ascii=False)
/// ```
///
/// where `crit` is `qdef.get("criteria")`, a dict turned into `[[k, v], …]`. The question's
/// name is not part of it; the criteria keep their order.
pub fn cache_key(state: &Value, qdef: &Value) -> Result<String, PyError> {
    let crit = match get(qdef, "criteria")? {
        Value::Object(m) => Value::Array(m.iter().map(|(k, v)| Value::Array(vec![Value::Str(k.clone()), v.clone()])).collect()),
        c => c.clone(),
    };
    let st = text(state);
    let typ = index(qdef, "type")?.clone();
    let ins = text(index(qdef, "instructions")?);
    Ok(Value::Array(vec![Value::Str(st), typ, Value::Str(ins), crit]).dumps(false))
}

/// Python's `repr()` of a str: single quotes unless the string holds a single quote and no
/// double quote; backslash escapes for the quote, `\\`, `\t`, `\n`, `\r` and non-printable
/// characters.
pub fn repr(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut b = String::with_capacity(s.len() + 2);
    b.push(q);
    for c in s.chars() {
        let u = c as u32;
        match c {
            _ if c == q || c == '\\' => {
                b.push('\\');
                b.push(c);
            }
            '\t' => b.push_str("\\t"),
            '\n' => b.push_str("\\n"),
            '\r' => b.push_str("\\r"),
            _ if u < 0x20 || u == 0x7f => {
                let _ = write!(b, "\\x{u:02x}");
            }
            _ if u < 0x80 || is_printable(c) => b.push(c),
            _ if u <= 0xff => {
                let _ = write!(b, "\\x{u:02x}");
            }
            _ if u <= 0xffff => {
                let _ = write!(b, "\\u{u:04x}");
            }
            _ => {
                let _ = write!(b, "\\U{u:08x}");
            }
        }
    }
    b.push(q);
    b
}

/// Approximates `str.isprintable` for one non-ASCII character without Unicode tables: C1
/// controls, separators (Python treats even U+00A0 as non-printable), common format
/// characters, surrogates, private use and noncharacters are not printable; everything else
/// is. Unassigned code points, which Python also escapes, are treated as printable.
fn is_printable(c: char) -> bool {
    let u = c as u32;
    !matches!(u,
        0x80..=0xa0 | 0xad | 0x600..=0x605 | 0x61c | 0x6dd | 0x70f | 0x1680 | 0x180e
        | 0x2000..=0x200f | 0x2028..=0x202f | 0x205f..=0x2064 | 0x2066..=0x206f | 0x3000
        | 0xd800..=0xdfff | 0xe000..=0xf8ff | 0xfdd0..=0xfdef | 0xfeff | 0xfff9..=0xfffb
        | 0xfffe | 0xffff | 0x110bd | 0x1bca0..=0x1bca3 | 0x1d173..=0x1d17a | 0xe0001 | 0xe0020..=0xe007f
        | 0xf0000..)
        && (u & 0xfffe) != 0xfffe
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyjson::parse;

    #[test]
    fn reprs() {
        assert_eq!(repr("a"), "'a'");
        assert_eq!(repr("it's"), "\"it's\"");
        assert_eq!(repr("it's \"x\""), "'it\\'s \"x\"'");
        assert_eq!(repr("a\tb\u{1}\u{a0}é😀"), "'a\\tb\\x01\\xa0é😀'");
    }

    #[test]
    fn cache_keys() {
        let st = parse(r#"{"b": 1, "a": "é"}"#.as_bytes()).unwrap();
        let q = parse(br#"{"type": "choice", "instructions": "go?", "criteria": {"up": "u", "down": null}}"#).unwrap();
        assert_eq!(cache_key(&st, &q).unwrap(), r#"["{\"b\": 1, \"a\": \"é\"}", "choice", "go?", [["up", "u"], ["down", null]]]"#);
        let bad = parse(br#"{"type": "choice"}"#).unwrap();
        assert_eq!(cache_key(&st, &bad).unwrap_err().msg, "'instructions'");
    }
}

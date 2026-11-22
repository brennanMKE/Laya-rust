//! Shared helpers for the golden fixture tests (`tests/golden`, synced from laya-go).
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use laya_core::checkpoint;
use laya_core::pyjson::{self, Value};
use laya_core::{Batch, Engine, EngineError, Output};

/// The repository's `tests/golden`.
pub fn golden_dir() -> PathBuf {
    std::env::var_os("LAYA_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden"))
}

/// Every line of a golden JSONL file, parsed with Python's semantics (key order kept).
pub fn read_jsonl(name: &str) -> Vec<Value> {
    let path = golden_dir().join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e} (run `cargo xtask sync-fixtures`)", path.display()));
    text.lines().filter(|l| !l.is_empty()).map(|l| pyjson::parse(l.as_bytes()).unwrap()).collect()
}

/// The checkpoint directory: `$LAYA_CHECKPOINT` or the pinned revision in the HF cache.
pub fn checkpoint_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("LAYA_CHECKPOINT").filter(|d| !d.is_empty()) {
        return Some(d.into());
    }
    checkpoint::find(None, "", "", "").ok()
}

/// A fixture float32, written as its shortest decimal: the f64 the parser produced prints
/// back as that decimal, which then parses to the exact float32.
pub fn f32_of(v: &Value) -> f32 {
    match v {
        Value::Float(f) => format!("{f}").parse().unwrap(),
        Value::Int(s) => s.parse().unwrap(),
        v => panic!("not a number: {v:?}"),
    }
}

pub fn f32s(v: &Value) -> Vec<f32> {
    arr(v).iter().map(f32_of).collect()
}

pub fn u32s(v: &Value) -> Vec<u32> {
    arr(v).iter().map(|x| int(x) as u32).collect()
}

pub fn arr(v: &Value) -> &[Value] {
    match v {
        Value::Array(a) => a,
        v => panic!("not an array: {v:?}"),
    }
}

pub fn int(v: &Value) -> i64 {
    match v {
        Value::Int(s) => s.parse().unwrap(),
        v => panic!("not an int: {v:?}"),
    }
}

pub fn str_of(v: &Value) -> &str {
    match v {
        Value::Str(s) => s,
        v => panic!("not a string: {v:?}"),
    }
}

/// `v[a][b]…`, panicking on a missing key.
pub fn at<'a>(v: &'a Value, path: &[&str]) -> &'a Value {
    path.iter().fold(v, |v, k| v.get(k).unwrap_or_else(|| panic!("missing {k:?}")))
}

/// An engine that returns logits queued by the test, so the text side and calibration can be
/// checked against Python's own logits.
#[derive(Default)]
pub struct Replay {
    pub queued: Mutex<Option<Output>>,
}

impl Replay {
    pub fn queue(&self, out: Output) {
        *self.queued.lock().unwrap() = Some(out);
    }
}

/// The engine side of a shared [`Replay`].
pub struct ReplayEngine(pub std::sync::Arc<Replay>);

impl Engine for ReplayEngine {
    fn forward(&self, b: &Batch) -> Result<Output, EngineError> {
        let out = self.0.queued.lock().unwrap().take().ok_or_else(|| EngineError::new("replay: nothing queued"))?;
        if out.logits.len() != b.len() {
            return Err(EngineError::new(format!("replay: batch of {}, {} queued", b.len(), out.logits.len())));
        }
        Ok(out)
    }

    fn name(&self) -> String {
        "replay".into()
    }
}

/// Counts checks and reports the first few failures.
#[derive(Default)]
pub struct Tally {
    pub n: usize,
    pub bad: usize,
}

impl Tally {
    pub fn check(&mut self, ok: bool, msg: impl FnOnce() -> String) {
        self.n += 1;
        if !ok {
            self.bad += 1;
            if self.bad <= 15 {
                eprintln!("MISMATCH {}", msg());
            }
        }
    }

    pub fn summary(&self) -> String {
        format!("{}/{}", self.n - self.bad, self.n)
    }
}

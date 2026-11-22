//! Shared helpers for the model tests: find the checkpoint (skip when missing, never
//! download), load each kernel set once, and read the golden fixtures.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use laya_cpu::{CpuEngine, KernelSet, Options};

const REVISION: &str = "1c5edc17a7acd8701df6fc341c0d179f1c62c982";

/// `$LAYA_CHECKPOINT`, else the pinned snapshot in the Hugging Face cache; `None` when absent.
pub fn checkpoint() -> Option<PathBuf> {
    let dir = match std::env::var_os("LAYA_CHECKPOINT") {
        Some(d) => PathBuf::from(d),
        None => {
            let hub = if let Some(h) = std::env::var_os("HF_HUB_CACHE") {
                PathBuf::from(h)
            } else if let Some(h) = std::env::var_os("HF_HOME") {
                PathBuf::from(h).join("hub")
            } else {
                PathBuf::from(std::env::var_os("HOME")?).join(".cache/huggingface/hub")
            };
            hub.join("models--convaiinnovations--laya/snapshots").join(REVISION)
        }
    };
    if dir.join("model.safetensors").is_file() {
        Some(dir)
    } else {
        eprintln!("skipping: Laya checkpoint not found at {} (set LAYA_CHECKPOINT)", dir.display());
        None
    }
}

/// The engine for `kernels`, loaded once per test binary.
pub fn engine(kernels: KernelSet) -> Option<Arc<CpuEngine>> {
    static CACHE: OnceLock<Mutex<HashMap<&'static str, Arc<CpuEngine>>>> = OnceLock::new();
    let dir = checkpoint()?;
    let key = match kernels {
        KernelSet::Gemm => "gemm",
        KernelSet::Portable => "portable",
    };
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    let e = cache.entry(key).or_insert_with(|| Arc::new(CpuEngine::load(&dir, Options { threads: 0, kernels }).expect("load checkpoint")));
    Some(e.clone())
}

/// `tests/golden` at the workspace root.
pub fn golden_dir() -> PathBuf {
    std::env::var_os("LAYA_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden"))
}

/// A JSON number as the f32 it was written from.
pub fn f32s(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap() as f32).collect()).unwrap_or_default()
}

/// A JSON array of integers.
pub fn u32s(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// The gap between the two largest values (infinite for one value).
pub fn top_gap(v: &[f32]) -> f32 {
    if v.len() < 2 {
        return f32::INFINITY;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| b.total_cmp(a));
    s[0] - s[1]
}

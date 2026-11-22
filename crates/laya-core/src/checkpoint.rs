//! Finding a Laya checkpoint in the local Hugging Face cache and reading its
//! `rl_agent_config.json` (laya-go's `internal/checkpoint`). Downloading lives in `laya-hub`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The checkpoint laya-rust is tested against.
pub const REPO: &str = "convaiinnovations/laya";
pub const REVISION: &str = "1c5edc17a7acd8701df6fc341c0d179f1c62c982";

/// Files every checkpoint directory must contain.
pub const FILES: [&str; 4] = ["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json"];

/// A checkpoint that can't be found or read.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("{dir} is missing {missing}")]
    Missing { dir: PathBuf, missing: String },
    #[error("checkpoint {repo}@{revision} not in the local cache: {source}")]
    NotCached { repo: String, revision: String, source: Box<CheckpointError> },
    #[error("rl_agent_config.json: {0}")]
    Config(String),
}

/// The Hugging Face hub cache: `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else
/// `~/.cache/huggingface/hub`.
pub fn cache_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("HF_HUB_CACHE").filter(|d| !d.is_empty()) {
        return d.into();
    }
    if let Some(d) = std::env::var_os("HF_HOME").filter(|d| !d.is_empty()) {
        return Path::new(&d).join("hub");
    }
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).unwrap_or_default();
    Path::new(&home).join(".cache/huggingface/hub")
}

/// The snapshot directory of `repo` at `revision` (a commit or a ref such as `main`) in the
/// cache, checked to hold every file in [`FILES`]. Empty arguments take the defaults.
pub fn find(cache: Option<&Path>, repo: &str, revision: &str, subfolder: &str) -> Result<PathBuf, CheckpointError> {
    let cache = cache.map(Path::to_path_buf).unwrap_or_else(cache_dir);
    let repo = if repo.is_empty() { REPO } else { repo };
    let mut revision = if revision.is_empty() { REVISION } else { revision }.to_string();
    let base = cache.join(format!("models--{}", repo.replace('/', "--")));
    if let Ok(r) = std::fs::read_to_string(base.join("refs").join(&revision)) {
        revision = r.trim().to_string();
    }
    let dir = base.join("snapshots").join(&revision).join(subfolder);
    check(&dir).map_err(|e| CheckpointError::NotCached { repo: repo.into(), revision, source: Box::new(e) })?;
    Ok(dir)
}

/// Whether `dir` holds every file in [`FILES`].
pub fn check(dir: &Path) -> Result<(), CheckpointError> {
    let missing: Vec<&str> = FILES.iter().copied().filter(|f| !dir.join(f).exists()).collect();
    if missing.is_empty() { Ok(()) } else { Err(CheckpointError::Missing { dir: dir.into(), missing: missing.join(", ") }) }
}

/// The part of `rl_agent_config.json` the text side reads.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub max_len: usize,
    pub head_max_len: usize,
    /// Per question type (choice, score, noul), as fitted; not yet clamped.
    pub temperature: Vec<f64>,
    /// `"<type>:<size>"` buckets, as fitted; not yet clamped.
    pub temperature_by_options: BTreeMap<String, f64>,
}

impl Config {
    /// Reads `dir/rl_agent_config.json`, filling Python's defaults (max_len 512, head_max_len
    /// 192, temperatures 1.0). Non-numeric temperatures become NaN, which the clamp turns into
    /// 1.0, as Python's `float()` failure does.
    pub fn read(dir: &Path) -> Result<Config, CheckpointError> {
        let text = std::fs::read_to_string(dir.join("rl_agent_config.json")).map_err(|e| CheckpointError::Config(e.to_string()))?;
        Config::parse(&text)
    }

    /// Parses the text of `rl_agent_config.json`.
    pub fn parse(text: &str) -> Result<Config, CheckpointError> {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| CheckpointError::Config(e.to_string()))?;
        let num = |v: &serde_json::Value| v.as_f64().unwrap_or(f64::NAN);
        let size = |key: &str, default: usize| v.get(key).and_then(|x| x.as_u64()).map_or(default, |x| x as usize);
        let temperature = match v.get("temperature").and_then(|t| t.as_array()) {
            Some(a) => a.iter().map(num).collect(),
            None => vec![1.0; 3],
        };
        if temperature.len() < 3 {
            return Err(CheckpointError::Config("temperature needs one value per question type".into()));
        }
        let temperature_by_options = v
            .get("temperature_by_options")
            .and_then(|t| t.as_object())
            .map(|m| m.iter().map(|(k, x)| (k.clone(), num(x))).collect())
            .unwrap_or_default();
        Ok(Config { max_len: size("max_len", 512), head_max_len: size("head_max_len", 192), temperature, temperature_by_options })
    }
}

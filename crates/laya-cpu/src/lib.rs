//! Laya's CPU inference engine: a port of laya-go's native engine
//! (`internal/modernbert`), a ModernBERT-large encoder plus Laya's decision head, run on packed,
//! unpadded sequences in fp32.
//!
//! ```no_run
//! use laya_core::engine::{Batch, Engine};
//! use laya_cpu::{CpuEngine, Options};
//!
//! let engine = CpuEngine::load("/path/to/checkpoint".as_ref(), Options::default())?;
//! let mut batch = Batch::default();
//! batch.push(&[50281, 50284, 50282], vec![1], 0);
//! let out = engine.forward(&batch)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod attention;
pub mod config;
pub mod kernels;
mod model;
pub mod ops;

use std::path::Path;
use std::str::FromStr;
use std::sync::Mutex;

use laya_core::engine::{Batch, Engine, EngineError, Output};

pub use config::Config;
pub use kernels::{GemmKernels, Kernels, PortableKernels, Weight};

/// A trace hook: receives each named intermediate as a flat row-major tensor.
///
/// Names, in order: `emb`, `layer.0` … `layer.27`, `final_norm`, `head_in`, `head.0`, `head.1`
/// (each `[T, hidden]` over the packed tokens), `logits` (every marker's logit, sequences
/// concatenated) and `act_logits` (`[n, 2]`). The slices are valid only during the call.
pub type TraceFn<'a> = dyn FnMut(&str, &[f32]) + Send + 'a;

/// Which kernel set runs the matrix multiplies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KernelSet {
    /// The `gemm` crate (pure Rust, SIMD dispatched at run time). The default.
    #[default]
    Gemm,
    /// Plain loops: the correctness baseline.
    Portable,
}

impl FromStr for KernelSet {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "gemm" | "" => Ok(KernelSet::Gemm),
            "portable" => Ok(KernelSet::Portable),
            _ => Err(format!("unknown kernel set {s:?} (want gemm or portable)")),
        }
    }
}

/// Options for [`CpuEngine::load`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Options {
    /// Worker threads for the engine's rayon pool (0 = all cores).
    pub threads: usize,
    /// The kernel set.
    pub kernels: KernelSet,
}

/// Errors from [`CpuEngine::load`].
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// Reading a file failed.
    #[error("laya-cpu: {0}")]
    Io(#[from] std::io::Error),
    /// A config file is malformed or describes an unsupported model.
    #[error("laya-cpu: config: {0}")]
    Config(String),
    /// The checkpoint doesn't match the config.
    #[error("laya-cpu: checkpoint: {0}")]
    Checkpoint(String),
    /// The thread pool could not be built.
    #[error("laya-cpu: thread pool: {0}")]
    Pool(String),
}

/// The CPU engine. Safe for concurrent use; passes run one at a time, since each uses every
/// core.
pub struct CpuEngine {
    model: model::Model,
    pool: rayon::ThreadPool,
    ws: Mutex<model::Workspace>,
}

impl CpuEngine {
    /// Loads a Laya checkpoint directory (`model.safetensors`, `encoder/config.json`,
    /// `rl_agent_config.json`), widening the fp16 linears to fp32.
    pub fn load(dir: &Path, opts: Options) -> Result<CpuEngine, LoadError> {
        let cfg = Config::load(dir)?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(opts.threads)
            .thread_name(|i| format!("laya-cpu-{i}"))
            .build()
            .map_err(|e| LoadError::Pool(e.to_string()))?;
        let k: Box<dyn Kernels> = match opts.kernels {
            KernelSet::Gemm => Box::new(GemmKernels::new(pool.current_num_threads())),
            KernelSet::Portable => Box::new(PortableKernels),
        };
        let model = pool.install(|| model::Model::load(dir, cfg, k))?;
        Ok(CpuEngine { model, pool, ws: Mutex::new(model::Workspace::default()) })
    }

    /// The model's configuration.
    pub fn config(&self) -> &Config {
        &self.model.cfg
    }

    /// The number of worker threads.
    pub fn threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Runs the network, passing every named intermediate to `trace` (see [`TraceFn`]).
    pub fn forward_traced(&self, batch: &Batch, trace: Option<&mut TraceFn<'_>>) -> Result<Output, EngineError> {
        batch.validate(self.model.cfg.vocab, 0)?;
        let mut guard = self.ws.lock().unwrap_or_else(|e| e.into_inner());
        let ws: &mut model::Workspace = &mut guard;
        Ok(self.pool.install(|| self.model.forward(ws, batch, trace)))
    }
}

impl Engine for CpuEngine {
    fn forward(&self, batch: &Batch) -> Result<Output, EngineError> {
        self.forward_traced(batch, None)
    }

    fn name(&self) -> String {
        format!("cpu ({}, {} threads)", self.model.k.name(), self.threads())
    }
}

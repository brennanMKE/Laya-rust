//! Embeddable CPU inference for Laya.
//!
//! ```no_run
//! use laya::{Model, Named, Options, Prompt, Question, State};
//!
//! let model = Model::load("my-checkpoint", Options::default())?;
//! let prompt = Prompt {
//!     key: "scene".into(),
//!     state: State("The road is empty.".into()),
//!     questions: vec![Named { name: "safe".into(), question: Question::noul("Is it safe to cross?") }],
//! };
//! let result = model.predict(&prompt)?;
//! println!("{:?}", result.answers);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::ops::Deref;
use std::path::Path;

pub use laya_core::answercache::AnswerCache;
pub use laya_core::question::{Answer, LayaError, Named, Opt, Prompt, QType, Question, State, WireStyle};
pub use laya_core::{Batch, Engine, EngineError, Info, Output, PromptResult, answercache, calibrate, checkpoint, pydict, pyjson, question};
pub use laya_cpu::{KernelSet, Options};
pub use laya_hub as hub;

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error(transparent)]
    Hub(#[from] hub::Error),
    #[error(transparent)]
    Cpu(#[from] laya_cpu::LoadError),
    #[error(transparent)]
    Model(#[from] laya_core::LoadError),
}

/// A Laya model ready for prediction.
pub struct Model(laya_core::Model);

impl Model {
    /// Load the model from a local checkpoint directory.
    pub fn load(dir: impl AsRef<Path>, options: Options) -> Result<Self, LoadError> {
        let dir = dir.as_ref();
        let engine = laya_cpu::CpuEngine::load(dir, options)?;
        Ok(Self::new(dir, Box::new(engine))?)
    }

    /// Resolve the checkpoint through the Hub and open it with the CPU engine.
    pub fn load_checkpoint(checkpoint: hub::Options<'_>, options: Options) -> Result<Self, LoadError> {
        Self::load(hub::download(checkpoint)?, options)
    }

    /// Use a local checkpoint's tokenizer and configuration with a custom inference engine.
    pub fn new(dir: &Path, engine: Box<dyn Engine>) -> Result<Self, laya_core::LoadError> {
        laya_core::Model::new(dir, engine).map(Self)
    }

    pub fn predict(&self, prompt: &Prompt) -> Result<PromptResult, LayaError> {
        self.0.predict(prompt)
    }

    pub fn predict_many(&self, prompts: &[Prompt]) -> Result<Vec<PromptResult>, LayaError> {
        self.0.predict_many(prompts)
    }
}

impl Deref for Model {
    type Target = laya_core::Model;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

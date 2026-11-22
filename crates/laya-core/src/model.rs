//! [`Model`]: the text side around an [`Engine`] (laya-go's `model.go`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::calibrate::{self, Temperatures};
use crate::checkpoint::{CheckpointError, Config};
use crate::engine::{Batch, Engine, Output};
use crate::prompt::Builder;
use crate::question::{Answer, LayaError, Prompt, QType, Question};
use crate::tokenizer::{Tokenizer, TokenizerError};

/// Failure to open a model.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
}

/// What a loaded model is running with.
#[derive(Debug, Clone)]
pub struct Info {
    pub dir: PathBuf,
    pub max_len: usize,
    pub head_max_len: usize,
    /// The temperatures applied (clamped).
    pub temperatures: Temperatures,
    /// Fitted temperatures the clamp changed, such as `"choice:11+=0.1006"`.
    pub clamped: Vec<String>,
    /// The engine's description.
    pub engine: String,
}

/// One built model input: the tokens and option markers of a (state, question) pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sequence {
    pub key: String,
    pub name: String,
    pub qtype: QType,
    pub tokens: Vec<u32>,
    pub markers: Vec<u32>,
}

/// Answers about one state.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptResult {
    pub key: String,
    /// `(question name, answer)` in question order.
    pub answers: Vec<(String, Answer)>,
    /// Real (unpadded) tokens run through the model for this state.
    pub input_tokens: usize,
}

/// Answers questions. Safe to share between threads; engine calls are serialized, since one
/// pass already uses every core.
pub struct Model {
    tok: Tokenizer,
    max_len: usize,
    head_max_len: usize,
    temps: Temperatures,
    info: Info,
    engine: Mutex<Box<dyn Engine>>,
}

impl Model {
    /// A model from a checkpoint directory's tokenizer and `rl_agent_config.json`, running the
    /// network on `engine`.
    pub fn new(dir: &Path, engine: Box<dyn Engine>) -> Result<Model, LoadError> {
        let tok = Tokenizer::from_file(dir.join("tokenizer/tokenizer.json"))?;
        let cfg = Config::read(dir)?;
        Ok(Model::with_parts(dir, tok, cfg, engine))
    }

    /// A model from already loaded parts.
    pub fn with_parts(dir: &Path, tok: Tokenizer, cfg: Config, engine: Box<dyn Engine>) -> Model {
        let (temps, clamped) = Temperatures::new(&cfg.temperature, &cfg.temperature_by_options);
        let info = Info {
            dir: dir.to_path_buf(),
            max_len: cfg.max_len,
            head_max_len: cfg.head_max_len,
            temperatures: temps.clone(),
            clamped,
            engine: engine.name(),
        };
        Model { tok, max_len: cfg.max_len, head_max_len: cfg.head_max_len, temps, info, engine: Mutex::new(engine) }
    }

    pub fn info(&self) -> &Info {
        &self.info
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// The sequence builder for this model's tokenizer and limits.
    pub fn builder(&self) -> Builder<'_> {
        Builder { tok: &self.tok, max_len: self.max_len, head_max_len: self.head_max_len }
    }

    /// Builds the sequence of every question of every prompt, in order, exactly as
    /// `build_sequence` does, failing as Python would (options too long, no options, …).
    pub fn encode(&self, prompts: &[Prompt]) -> Result<Vec<Sequence>, LayaError> {
        let b = self.builder();
        let mut out = Vec::new();
        for p in prompts {
            for q in &p.questions {
                let wrap = |err| LayaError::Question { key: p.key.clone(), name: q.name.clone(), err: Box::new(err) };
                q.question.validate().map_err(wrap)?;
                let s = b.build(p.state.as_str(), &q.question.render());
                if !s.is_complete() {
                    return Err(wrap(LayaError::OptionsTooLong(self.head_max_len)));
                }
                out.push(Sequence { key: p.key.clone(), name: q.name.clone(), qtype: q.question.qtype, tokens: s.ids, markers: s.markers });
            }
        }
        Ok(out)
    }

    /// Answers many prompts in one forward pass, like `laya_batch.predict_many`.
    pub fn predict_many(&self, prompts: &[Prompt]) -> Result<Vec<PromptResult>, LayaError> {
        let seqs = self.encode(prompts)?;
        let out = if seqs.is_empty() { Output::default() } else { self.forward(seqs.iter())? };
        Ok(self.results(prompts, &seqs, &out, 0..seqs.len()))
    }

    /// Answers independent requests in one pass. Invalid requests retain their own errors.
    /// With deduplication enabled, identical model inputs share one row of logits.
    pub fn predict_grouped(&self, requests: &[Vec<Prompt>], deduplicate: bool) -> Vec<Result<Vec<PromptResult>, LayaError>> {
        let encoded: Vec<_> = requests.iter().map(|request| self.encode(request)).collect();
        let mut unique = Vec::new();
        let mut seen = HashMap::new();
        let indices: Vec<Vec<usize>> = encoded
            .iter()
            .map(|request| {
                request
                    .iter()
                    .flatten()
                    .map(|sequence| {
                        if deduplicate {
                            let key = (&sequence.tokens, &sequence.markers, sequence.qtype);
                            *seen.entry(key).or_insert_with(|| {
                                unique.push(sequence);
                                unique.len() - 1
                            })
                        } else {
                            unique.push(sequence);
                            unique.len() - 1
                        }
                    })
                    .collect()
            })
            .collect();
        let output = if unique.is_empty() { Ok(Output::default()) } else { self.forward(unique.iter().copied()) };
        requests
            .iter()
            .zip(&encoded)
            .zip(indices)
            .map(|((request, encoded), indices)| {
                let sequences = encoded.as_ref().map_err(Clone::clone)?;
                match &output {
                    Ok(output) => Ok(self.results(request, sequences, output, indices.into_iter())),
                    Err(_) if requests.len() > 1 => self.predict_many(request),
                    Err(error) => Err(error.clone()),
                }
            })
            .collect()
    }

    fn results(&self, prompts: &[Prompt], seqs: &[Sequence], output: &Output, rows: impl Iterator<Item = usize>) -> Vec<PromptResult> {
        let mut inputs = seqs.iter().zip(rows);
        prompts
            .iter()
            .map(|p| {
                let mut result = PromptResult { key: p.key.clone(), answers: Vec::with_capacity(p.questions.len()), input_tokens: 0 };
                for q in &p.questions {
                    let (sequence, row) = inputs.next().expect("one output row per question");
                    result.answers.push((q.name.clone(), self.answer(&q.question, &output.logits[row], output.act_logits[row])));
                    result.input_tokens += sequence.tokens.len();
                }
                result
            })
            .collect()
    }

    /// Answers every question about one state in one pass, like `Agent.predict`.
    pub fn predict(&self, prompt: &Prompt) -> Result<PromptResult, LayaError> {
        Ok(self.predict_many(std::slice::from_ref(prompt))?.remove(0))
    }

    fn forward<'a>(&self, seqs: impl Iterator<Item = &'a Sequence>) -> Result<Output, LayaError> {
        let mut b = Batch::default();
        for s in seqs {
            b.push(&s.tokens, s.markers.clone(), s.qtype as u8);
        }
        let out = self.engine.lock().unwrap_or_else(|e| e.into_inner()).forward(&b)?;
        let bad = |m: String| Err(LayaError::Engine(crate::engine::EngineError(m)));
        if out.logits.len() != b.len() || out.act_logits.len() != b.len() {
            return bad(format!("engine returned {} logit rows for {} sequences", out.logits.len(), b.len()));
        }
        for (l, markers) in out.logits.iter().zip(&b.markers) {
            if l.len() != markers.len() {
                return bad(format!("engine returned {} logits for {} options", l.len(), markers.len()));
            }
        }
        Ok(out)
    }

    /// Calibrates one question's raw logits, as `Agent.system_one` does.
    pub fn answer(&self, q: &Question, logits: &[f32], act_logits: [f32; 2]) -> Answer {
        calibrate::answer(&self.temps, q, logits, act_logits)
    }
}

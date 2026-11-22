//! The boundary between the text side and the network (laya-go's `engine` package and
//! CONTRACT.md).
//!
//! The text side packs every sequence of a call into one [`Batch`]; an [`Engine`] returns the
//! raw scorer logits at each marker and the two raw act-head logits per sequence. Temperatures,
//! softmax, confidence and rounding stay on the text side.

use std::fmt;

/// Every sequence of a call, packed back to back with no padding.
#[derive(Debug, Clone, Default)]
pub struct Batch {
    /// All sequences, concatenated.
    pub tokens: Vec<u32>,
    /// `n + 1` offsets: sequence `i` is `tokens[starts[i]..starts[i + 1]]`.
    pub starts: Vec<usize>,
    /// Option `[MASK]` positions within each sequence, strictly increasing.
    pub markers: Vec<Vec<u32>>,
    /// Question type per sequence: 0 choice, 1 score, 2 noul (the `type_emb` row).
    pub qtypes: Vec<u8>,
}

impl Batch {
    /// The number of sequences.
    pub fn len(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    /// Whether the batch holds no sequences.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sequence `i`'s tokens.
    pub fn seq(&self, i: usize) -> &[u32] {
        &self.tokens[self.starts[i]..self.starts[i + 1]]
    }

    /// Appends one sequence.
    pub fn push(&mut self, tokens: &[u32], markers: Vec<u32>, qtype: u8) {
        if self.starts.is_empty() {
            self.starts.push(0);
        }
        self.tokens.extend_from_slice(tokens);
        self.starts.push(self.tokens.len());
        self.markers.push(markers);
        self.qtypes.push(qtype);
    }

    /// Checks the batch's structure against a vocabulary size and a maximum sequence length
    /// (0 means no limit).
    pub fn validate(&self, vocab: usize, max_len: usize) -> Result<(), EngineError> {
        let n = self.len();
        let bad = |m: String| Err(EngineError::new(m));
        if n == 0 {
            return bad("empty batch".into());
        }
        if self.starts[0] != 0 || self.starts[n] != self.tokens.len() {
            return bad("starts must run from 0 to tokens.len()".into());
        }
        if self.markers.len() != n || self.qtypes.len() != n {
            return bad(format!("{n} sequences but {} marker lists and {} qtypes", self.markers.len(), self.qtypes.len()));
        }
        for i in 0..n {
            if self.starts[i + 1] <= self.starts[i] {
                return bad(format!("sequence {i} is empty"));
            }
            let l = self.starts[i + 1] - self.starts[i];
            if max_len > 0 && l > max_len {
                return bad(format!("sequence {i} has {l} tokens, more than {max_len}"));
            }
            if self.qtypes[i] > 2 {
                return bad(format!("sequence {i} has unknown question type {}", self.qtypes[i]));
            }
            if self.markers[i].is_empty() {
                return bad(format!("sequence {i} has no option markers"));
            }
            if let Some(&m) = self.markers[i].iter().find(|&&m| m as usize >= l) {
                return bad(format!("sequence {i} marker {m} outside its {l} tokens"));
            }
        }
        if let Some((i, &t)) = self.tokens.iter().enumerate().find(|&(_, &t)| t as usize >= vocab) {
            return bad(format!("token {t} at {i} outside the {vocab}-token vocabulary"));
        }
        Ok(())
    }
}

/// The result of a forward pass.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Output {
    /// Per sequence, the raw scorer logit at each marker, in marker order.
    pub logits: Vec<Vec<f32>>,
    /// Per sequence, the raw act-head logits `[act, escalate]`.
    pub act_logits: Vec<[f32; 2]>,
}

/// An engine failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineError(pub String);

impl EngineError {
    pub fn new(msg: impl Into<String>) -> Self {
        EngineError(msg.into())
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "engine: {}", self.0)
    }
}

impl std::error::Error for EngineError {}

/// Runs the whole network. Output `i` depends only on sequence `i`: attention never crosses
/// sequences and position ids restart at 0 for each one.
pub trait Engine: Send + Sync {
    fn forward(&self, batch: &Batch) -> Result<Output, EngineError>;

    /// A short description, such as `"cpu (gemm, 4 threads)"`.
    fn name(&self) -> String {
        std::any::type_name::<Self>().to_string()
    }
}

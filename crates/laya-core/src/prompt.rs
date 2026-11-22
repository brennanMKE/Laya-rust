//! Laya's token sequences: `render_options` and `build_sequence` from laya 0.3.6's
//! `common.py` (laya-go's `internal/prompt`).
//!
//! One sequence per (state, question) pair:
//!
//! ```text
//! [CLS] "<type> question: <instructions>" [SEP] [MASK] " opt0" [MASK] " opt1" … [SEP] <state> [SEP]
//! ```
//!
//! Every tokenizer call is made without special tokens.

use crate::tokenizer::Tokenizer;

/// Special token ids of Laya's ModernBERT tokenizer.
pub const CLS: u32 = 50281;
pub const SEP: u32 = 50282;
pub const MASK: u32 = 50284;

/// The tokenizer's mask token; Laya replaces it with a space in instructions, options and state.
const MASK_TEXT: &str = "[MASK]";

/// Tokens kept of each rendered option (after its MASK).
pub const MAX_OPTION_TOKENS: usize = 48;
/// Floor of the per-option cut when options overflow.
pub const MIN_OPTION_TOKENS: usize = 4;
/// Options are squeezed when fewer tokens than this are left for the head.
pub const MIN_HEAD_BUDGET: isize = 16;
/// Instructions always keep at least this many tokens.
pub const MIN_INSTR_TOKENS: usize = 8;

/// Default noul descriptions.
pub const NOUL_FALSE: &str = "no, the statement does not hold";
pub const NOUL_TRUE: &str = "yes, the statement holds";

/// One question as `build_sequence` sees it.
#[derive(Debug, Clone)]
pub struct PromptQuestion<'a> {
    /// `"choice"`, `"score"` or `"noul"`; it starts the instructions text.
    pub type_name: &'a str,
    /// The question text (non-string instructions are `json.dumps`'d by the caller).
    pub instructions: &'a str,
    /// Rendered option texts in label order.
    pub options: Vec<String>,
}

/// A choice option: `name`, or `name: description` when the description is non-empty.
pub fn choice_option(name: &str, description: &str) -> String {
    if description.is_empty() { name.to_string() } else { format!("{name}: {description}") }
}

/// Score level `i`: `level i: text`.
pub fn score_option(i: usize, text: &str) -> String {
    format!("level {i}: {text}")
}

/// A noul question's two options, false then true; an empty description takes the default.
pub fn noul_options(false_desc: &str, true_desc: &str) -> Vec<String> {
    let f = if false_desc.is_empty() { NOUL_FALSE } else { false_desc };
    let t = if true_desc.is_empty() { NOUL_TRUE } else { true_desc };
    vec![format!("false: {f}"), format!("true: {t}")]
}

/// A built sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sequence {
    pub ids: Vec<u32>,
    /// Positions of the options' MASK tokens that fit within `max_len`. When fewer survive
    /// than there are options, Python raises "options exceed head_max_len".
    pub markers: Vec<u32>,
    /// The number of options the question has.
    pub options: usize,
}

impl Sequence {
    /// Whether every option kept its marker.
    pub fn is_complete(&self) -> bool {
        self.markers.len() == self.options
    }
}

/// Builds sequences with one tokenizer and the checkpoint's limits.
pub struct Builder<'t> {
    pub tok: &'t Tokenizer,
    /// `rl_agent_config` `max_len` (512).
    pub max_len: usize,
    /// `rl_agent_config` `head_max_len` (192).
    pub head_max_len: usize,
}

impl Builder<'_> {
    /// `build_sequence(tok, state, q, max_len, head_max_len)` with the default option order and
    /// right truncation, as `Agent.predict` calls it.
    pub fn build(&self, state: &str, q: &PromptQuestion<'_>) -> Sequence {
        let tok = self.tok;
        let mut head = tok.encode(&format!("{} question: {}", q.type_name, q.instructions.replace(MASK_TEXT, " ")));

        let mut opts: Vec<Vec<u32>> = q
            .options
            .iter()
            .map(|o| {
                let mut ids = vec![MASK];
                tok.append_encode(&mut ids, &format!(" {}", o.replace(MASK_TEXT, " ")));
                ids.truncate(1 + MAX_OPTION_TOKENS); // the cut applies to the option's own tokens
                ids
            })
            .collect();
        let head_max = self.head_max_len as isize;
        let mut total: usize = opts.iter().map(Vec::len).sum();
        let mut budget = head_max - total as isize;
        if budget < MIN_HEAD_BUDGET {
            let per = MIN_OPTION_TOKENS.max(((head_max - MIN_HEAD_BUDGET) / opts.len().max(1) as isize).max(0) as usize);
            total = 0;
            for o in &mut opts {
                o.truncate(per);
                total += o.len();
            }
            budget = head_max - total as isize;
        }
        head.truncate((MIN_INSTR_TOKENS as isize).max(budget) as usize);

        let mut ids = Vec::with_capacity(self.max_len);
        ids.push(CLS);
        ids.extend_from_slice(&head);
        ids.push(SEP);
        let mut markers = Vec::with_capacity(opts.len());
        for o in &opts {
            markers.push(ids.len() as u32);
            ids.extend_from_slice(o);
        }
        ids.push(SEP);
        let room = self.max_len.saturating_sub(ids.len() + 1);
        let st = tok.encode(&state.replace(MASK_TEXT, " "));
        ids.extend_from_slice(&st[..st.len().min(room)]);
        ids.push(SEP);

        ids.truncate(self.max_len);
        markers.retain(|&m| (m as usize) < self.max_len);
        Sequence { ids, markers, options: q.options.len() }
    }
}

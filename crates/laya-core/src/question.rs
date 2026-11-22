//! Questions, requests and answers: decoding a `/predict` body the way Laya's `Agent.predict`
//! and `laya_batch` read it, and writing answers exactly as `laya_server.py` does (laya-go's
//! `laya.go` and `wire.go`).

use std::fmt;

use crate::calibrate::round4;
use crate::prompt::{self, PromptQuestion};
use crate::pyjson::{self, Value};

/// A question type. Its value is the checkpoint's qtype id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum QType {
    /// Pick one named option.
    Choice = 0,
    /// An expected level on an ordered scale.
    Score = 1,
    /// The probability a statement is true.
    Noul = 2,
}

impl QType {
    /// `"choice"`, `"score"` or `"noul"`.
    pub fn as_str(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }

    pub fn parse(s: &str) -> Result<QType, LayaError> {
        match s {
            "choice" => Ok(QType::Choice),
            "score" => Ok(QType::Score),
            "noul" => Ok(QType::Noul),
            _ => Err(LayaError::UnknownType(format!("{s:?}"))),
        }
    }
}

impl fmt::Display for QType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Errors for requests and questions Laya can't answer.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum LayaError {
    /// Some option marker fell past `max_len` after squeezing. Python raises
    /// "question … options exceed head_max_len=192".
    #[error("laya: options exceed head_max_len={0}")]
    OptionsTooLong(usize),
    #[error("laya: question has no options")]
    NoOptions,
    #[error("laya: duplicate option name {0:?}")]
    DuplicateOption(String),
    #[error("laya: unknown question type {0}")]
    UnknownType(String),
    #[error("laya: {0}")]
    BadRequest(String),
    #[error(transparent)]
    Json(#[from] pyjson::ParseError),
    /// An error in one question of a request.
    #[error("question {}: {err}", qualified(.key, .name))]
    Question { key: String, name: String, err: Box<LayaError> },
    #[error(transparent)]
    Engine(#[from] crate::engine::EngineError),
}

impl LayaError {
    /// The underlying error, seen through [`LayaError::Question`].
    pub fn root(&self) -> &LayaError {
        match self {
            LayaError::Question { err, .. } => err.root(),
            e => e,
        }
    }
}

/// `key.name`, or `name` for a single-state request.
pub fn qualified(key: &str, name: &str) -> String {
    if key.is_empty() { name.to_string() } else { format!("{key}.{name}") }
}

/// A choice option. An empty description means "no description".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Opt {
    pub name: String,
    pub description: String,
}

/// One typed question. The order of options and levels is part of the prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub qtype: QType,
    pub instructions: String,
    /// Choice: options in order.
    pub options: Vec<Opt>,
    /// Score: levels, lowest first.
    pub levels: Vec<String>,
    /// Noul: optional descriptions (empty takes Laya's default).
    pub true_desc: String,
    pub false_desc: String,
    /// Structured score levels as decoded from JSON, echoed in the legend as Python does.
    legend: Option<Vec<Value>>,
}

impl Question {
    fn new(qtype: QType, instructions: impl Into<String>) -> Question {
        Question {
            qtype,
            instructions: instructions.into(),
            options: Vec::new(),
            levels: Vec::new(),
            true_desc: String::new(),
            false_desc: String::new(),
            legend: None,
        }
    }

    /// A choice question with options in the given order.
    pub fn choice(instructions: impl Into<String>, options: Vec<Opt>) -> Question {
        Question { options, ..Question::new(QType::Choice, instructions) }
    }

    /// A score question with levels from lowest to highest.
    pub fn score(instructions: impl Into<String>, levels: Vec<String>) -> Question {
        Question { levels, ..Question::new(QType::Score, instructions) }
    }

    /// A noul question: the probability that a statement holds.
    pub fn noul(instructions: impl Into<String>) -> Question {
        Question::new(QType::Noul, instructions)
    }

    /// The answer labels in option order: option names, `"0"`…`"k-1"`, or `"false"`, `"true"`.
    pub fn labels(&self) -> Vec<String> {
        match self.qtype {
            QType::Choice => self.options.iter().map(|o| o.name.clone()).collect(),
            QType::Score => (0..self.levels.len()).map(|i| i.to_string()).collect(),
            QType::Noul => vec!["false".into(), "true".into()],
        }
    }

    /// Rejects questions Python would reject (or, for duplicate names, couldn't express).
    pub fn validate(&self) -> Result<(), LayaError> {
        match self.qtype {
            QType::Choice => {
                if self.options.is_empty() {
                    return Err(LayaError::NoOptions);
                }
                let mut seen = std::collections::HashSet::new();
                for o in &self.options {
                    if !seen.insert(o.name.as_str()) {
                        return Err(LayaError::DuplicateOption(o.name.clone()));
                    }
                }
            }
            QType::Score if self.levels.is_empty() => return Err(LayaError::NoOptions),
            _ => {}
        }
        Ok(())
    }

    /// `render_options` plus the instructions: what `build_sequence` reads.
    pub fn render(&self) -> PromptQuestion<'_> {
        let options = match self.qtype {
            QType::Choice => self.options.iter().map(|o| prompt::choice_option(&o.name, &o.description)).collect(),
            QType::Score => self.levels.iter().enumerate().map(|(i, l)| prompt::score_option(i, l)).collect(),
            QType::Noul => prompt::noul_options(&self.false_desc, &self.true_desc),
        };
        PromptQuestion { type_name: self.qtype.as_str(), instructions: &self.instructions, options }
    }
}

/// A question with the name its answer is reported under.
#[derive(Debug, Clone, PartialEq)]
pub struct Named {
    pub name: String,
    pub question: Question,
}

/// The text a question is asked about: a string, or JSON serialized as Python's
/// `json.dumps(state, ensure_ascii=False)` does.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct State(pub String);

impl State {
    /// A JSON string state is that string; anything else is dumped.
    pub fn from_value(v: &Value) -> State {
        match v {
            Value::Str(s) => State(s.clone()),
            v => State(v.dumps(false)),
        }
    }

    /// Parses a JSON document into a state.
    pub fn from_json(raw: &[u8]) -> Result<State, LayaError> {
        Ok(State::from_value(&pyjson::parse(raw)?))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One state with its questions: the unit of `predict_many`.
#[derive(Debug, Clone, PartialEq)]
pub struct Prompt {
    /// Batch key, such as `"up"` or `"p12s3"`; empty for a single-state request.
    pub key: String,
    pub state: State,
    pub questions: Vec<Named>,
}

/// Decodes a `/predict` request body, keeping key order:
///
/// ```text
/// {"state": …, "questions": {name: question, …}}          one prompt, key ""
/// {"batch": {key: {"state": …, "questions": {…}}, …}}    one prompt per key
/// ```
pub fn decode_request(body: &[u8]) -> Result<Vec<Prompt>, LayaError> {
    decode_request_value(&pyjson::parse(body)?)
}

/// [`decode_request`] on an already parsed body.
pub fn decode_request_value(v: &Value) -> Result<Vec<Prompt>, LayaError> {
    if !matches!(v, Value::Object(_)) {
        return Err(LayaError::BadRequest("request is not a JSON object".into()));
    }
    if let Some(batch) = v.get("batch") {
        let Value::Object(members) = batch else {
            return Err(LayaError::BadRequest("batch is not an object".into()));
        };
        return members.iter().map(|(k, p)| decode_prompt(k, p)).collect();
    }
    Ok(vec![decode_prompt("", v)?])
}

fn decode_prompt(key: &str, v: &Value) -> Result<Prompt, LayaError> {
    if !matches!(v, Value::Object(_)) {
        return Err(LayaError::BadRequest(format!("prompt {key:?} is not an object")));
    }
    let state = v.get("state").ok_or_else(|| LayaError::BadRequest(format!("prompt {key:?} has no state")))?;
    let Some(Value::Object(qs)) = v.get("questions") else {
        return Err(LayaError::BadRequest(format!("prompt {key:?} has no questions object")));
    };
    let questions = qs
        .iter()
        .map(|(name, q)| {
            decode_question(q)
                .map(|question| Named { name: name.clone(), question })
                .map_err(|e| LayaError::BadRequest(format!("question {}: {e}", qualified(key, name))))
        })
        .collect::<Result<_, _>>()?;
    Ok(Prompt { key: key.to_string(), state: State::from_value(state), questions })
}

/// `render_criterion`: strings pass through, anything else is
/// `json.dumps(value, ensure_ascii=False)`.
fn render_criterion(v: &Value) -> String {
    match v {
        Value::Str(s) => s.clone(),
        v => v.dumps(false),
    }
}

/// Decodes one question as `Agent._to_internal` and `render_options` read it.
pub fn decode_question(v: &Value) -> Result<Question, String> {
    if !matches!(v, Value::Object(_)) {
        return Err("not an object".into());
    }
    let qtype = match v.get("type") {
        Some(Value::Str(s)) => QType::parse(s).map_err(|e| e.to_string())?,
        _ => return Err(LayaError::UnknownType("(missing)".into()).to_string()),
    };
    let ins = v.get("instructions").ok_or("no instructions")?;
    let mut q = Question::new(
        qtype,
        match ins {
            Value::Str(s) => s.clone(),
            ins => ins.dumps(true), // _to_internal: json.dumps(ins)
        },
    );
    let crit = v.get("criteria").unwrap_or(&Value::Null);
    match qtype {
        QType::Choice => match crit {
            Value::Array(names) => {
                // A list of names: {name: None for name in list}, one entry per name.
                for e in names {
                    let Value::Str(name) = e else {
                        return Err("choice criteria list holds a non-string".into());
                    };
                    if !q.options.iter().any(|o| &o.name == name) {
                        q.options.push(Opt { name: name.clone(), description: String::new() });
                    }
                }
            }
            Value::Object(m) => {
                for (name, d) in m {
                    let description = if d.is_none_or_empty() { String::new() } else { render_criterion(d) };
                    q.options.push(Opt { name: name.clone(), description });
                }
            }
            _ => return Err("choice criteria must be an object or a list".into()),
        },
        QType::Score => {
            let levels: Vec<Value> = match crit {
                Value::Array(a) => a.clone(),
                Value::Object(m) => m.iter().map(|(k, _)| Value::Str(k.clone())).collect(), // a dict's keys
                _ => return Err("score criteria must be a list".into()),
            };
            q.levels = levels.iter().map(render_criterion).collect();
            if levels.iter().any(|e| !matches!(e, Value::Str(_))) {
                q.legend = Some(levels);
            }
        }
        QType::Noul => {
            if crit.is_falsy() {
                // `crit or {}`: the default descriptions.
            } else if let Value::Object(_) = crit {
                if let Some(f) = crit.get("false").filter(|f| !f.is_none_or_empty()) {
                    q.false_desc = render_criterion(f);
                }
                if let Some(t) = crit.get("true").filter(|t| !t.is_none_or_empty()) {
                    q.true_desc = render_criterion(t);
                }
            } else {
                return Err("noul criteria must be an object".into());
            }
        }
    }
    Ok(q)
}

/// The calibrated answer to one question, at full precision (rounded on output).
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    pub qtype: QType,
    /// Choice: the most probable option.
    pub choice: String,
    /// Score: the expected level Σ i·pᵢ.
    pub score: f64,
    /// Noul: P(true).
    pub noul: f64,
    /// Per label in option order (option names, or `"0"`… for a score); empty for a noul.
    pub probabilities: Vec<(String, f64)>,
    /// Score levels, keyed `"0"`… on the wire.
    pub legend: Vec<String>,
    /// 1 − H(p)/log k for choice and score; max(p, 1 − p) for noul.
    pub confidence: f64,
    /// The act head's probability of acting rather than escalating.
    pub act_probability: f64,
    structured_legend: Option<Vec<Value>>,
}

/// Which Python function's answer shape to write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireStyle {
    /// `Agent.predict` (single-state replies): score answers carry legend and confidence, and
    /// every answer has `action.act_probability`.
    Predict,
    /// `laya_batch.predict_many` (batch replies): no action; score answers have no legend or
    /// confidence.
    Batch,
}

impl Answer {
    pub(crate) fn new(q: &Question, p: &[f32], act_probability: f64) -> Answer {
        use crate::calibrate as c;
        let mut a = Answer {
            qtype: q.qtype,
            choice: String::new(),
            score: 0.0,
            noul: 0.0,
            probabilities: Vec::new(),
            legend: Vec::new(),
            confidence: 0.0,
            act_probability,
            structured_legend: None,
        };
        match q.qtype {
            QType::Choice | QType::Score => {
                let labels = q.labels();
                a.probabilities = labels.iter().zip(p).map(|(l, &v)| (l.clone(), v as f64)).collect();
                a.confidence = c::confidence(p);
                if q.qtype == QType::Choice {
                    a.choice = labels[c::argmax(p)].clone();
                } else {
                    a.score = c::expected_score(p);
                    a.legend = q.levels.clone();
                    a.structured_legend = q.legend.clone();
                }
            }
            QType::Noul => (a.noul, a.confidence) = c::noul_answer(p),
        }
        a
    }

    /// The answer as the Python code builds it before `json.dumps`: every number rounded to 4
    /// decimals, keys in Python's order.
    pub fn wire(&self, style: WireStyle) -> Value {
        let num = |x: f64| Value::Float(round4(x));
        let mut obj = vec![("type".to_string(), Value::Str(self.qtype.as_str().into()))];
        let probs = || Value::Object(self.probabilities.iter().map(|(l, p)| (l.clone(), num(*p))).collect());
        match self.qtype {
            QType::Choice => {
                obj.push(("choice".into(), Value::Str(self.choice.clone())));
                obj.push(("probabilities".into(), probs()));
                obj.push(("confidence".into(), num(self.confidence)));
            }
            QType::Score => {
                obj.push(("score".into(), num(self.score)));
                if style == WireStyle::Predict {
                    let legend = self
                        .legend
                        .iter()
                        .enumerate()
                        .map(|(i, l)| {
                            let v = match &self.structured_legend {
                                Some(s) => s[i].clone(),
                                None => Value::Str(l.clone()),
                            };
                            (i.to_string(), v)
                        })
                        .collect();
                    obj.push(("legend".into(), Value::Object(legend)));
                }
                obj.push(("probabilities".into(), probs()));
                if style == WireStyle::Predict {
                    obj.push(("confidence".into(), num(self.confidence)));
                }
            }
            QType::Noul => {
                obj.push(("noul".into(), num(self.noul)));
                obj.push(("confidence".into(), num(self.confidence)));
            }
        }
        if style == WireStyle::Predict {
            obj.push(("action".into(), Value::Object(vec![("act_probability".into(), num(self.act_probability))])));
        }
        Value::Object(obj)
    }

    /// The answer's JSON exactly as `laya_server.py` writes it (ASCII-only `json.dumps`).
    pub fn to_wire_json(&self, style: WireStyle) -> String {
        self.wire(style).dumps(true)
    }
}

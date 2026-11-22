//! The HTTP contract of System 1 Arcade's `agents/laya_server.py` and laya-go's
//! `cmd/laya-server`, independent of the transport:
//!
//! ```text
//! POST /predict  {"state": …, "questions": {…}}            -> {"answers": {name: answer}}
//! POST /predict  {"batch": {key: {"state", "questions"}}}  -> {"answers": {"key.name": answer}}
//! GET  /health   -> {"ok": true}
//! GET  /stats    -> answer cache stats
//! ```
//!
//! Bodies are written the way Python's `json.dumps` writes them, answers in the shape the
//! Python code builds: `Agent.predict`'s for a single state, `predict_many`'s for a batch.
//! Errors carry Python's `str(e)` where laya-go matches it.

mod decisions;
mod merge;
mod policy;

pub use merge::MergedModel;
pub use policy::RequestPolicy;

use laya::answercache::AnswerCache;
use laya::pydict::{self, repr};
use laya::pyjson::{self, Value};
use laya::question::{Answer, LayaError, Named, Prompt, QType, State, WireStyle, decode_request_value};
use laya::{Model, PromptResult};

/// Largest request body accepted.
pub const MAX_BODY: usize = 64 << 20;

/// What the server needs from a model.
pub trait Predictor: Send + Sync {
    fn predict_many(&self, prompts: &[Prompt]) -> Result<Vec<PromptResult>, LayaError>;

    /// Unpadded token count for a question, when available.
    fn sequence_length(&self, _state: &State, _question: &Named) -> Result<usize, LayaError> {
        Ok(0)
    }
}

impl Predictor for Model {
    fn predict_many(&self, prompts: &[Prompt]) -> Result<Vec<PromptResult>, LayaError> {
        Model::predict_many(self, prompts)
    }

    fn sequence_length(&self, state: &State, question: &Named) -> Result<usize, LayaError> {
        let prompt = Prompt { key: String::new(), state: state.clone(), questions: vec![question.clone()] };
        Ok(self.encode(&[prompt])?[0].tokens.len())
    }
}

/// A response: status code, content type and body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    pub status: u16,
    pub content_type: &'static str,
    pub body: String,
}

impl Response {
    fn json(status: u16, body: impl Into<String>) -> Response {
        Response { status, content_type: "application/json", body: body.into() }
    }

    fn error(status: u16, msg: &str) -> Response {
        let mut b = String::from(r#"{"error": "#);
        pyjson::quote(&mut b, msg, true);
        b.push('}');
        Response::json(status, b)
    }
}

/// The server state: a model and the answer cache.
pub struct Server<P> {
    pub model: P,
    pub cache: AnswerCache<Option<Answer>>,
}

impl<P: Predictor> Server<P> {
    pub fn new(model: P, cache_size: usize) -> Self {
        Server { model, cache: AnswerCache::new(cache_size) }
    }

    /// Answers one request. `target` is the raw request target: Python compares `self.path`,
    /// so a query string or a trailing slash is a different path.
    pub fn handle(&self, method: &str, target: &str, body: &[u8]) -> Response {
        match method {
            "GET" => match target {
                "/health" => Response::json(200, r#"{"ok": true}"#),
                "/stats" => Response::json(200, self.cache.stats().to_python_json()),
                _ => Response::json(404, r#"{"error": "not found"}"#),
            },
            "POST" if target == "/v1/decisions" => self.decisions(body),
            "POST" if target != "/predict" => Response::json(404, r#"{"error": "POST /predict"}"#),
            "POST" => match self.predict(body) {
                Ok(b) => Response::json(200, b),
                Err(msg) => Response::error(400, &msg),
            },
            m => {
                // BaseHTTPRequestHandler has no do_PUT, do_HEAD, …: 501.
                let msg = format!("Unsupported method ({})", repr(m));
                Response {
                    status: 501,
                    content_type: "text/html;charset=utf-8",
                    body: format!(
                        "<!DOCTYPE HTML>\n<html lang=\"en\">\n    <head>\n        <meta charset=\"utf-8\">\n        <title>Error response</title>\n    </head>\n    <body>\n        <h1>Error response</h1>\n        <p>Error code: 501</p>\n        <p>Message: {msg}.</p>\n        <p>Error code explanation: 501 - Server does not support this operation.</p>\n    </body>\n</html>\n"
                    ),
                }
            }
        }
    }

    /// Answers one `/predict` body: the response JSON, or the text `laya_server.py` puts in
    /// `{"error": str(e)}`.
    pub fn predict(&self, body: &[u8]) -> Result<String, String> {
        if body.iter().all(|c| matches!(c, b' ' | b'\t' | b'\n' | b'\r')) {
            // json.loads of nothing but whitespace.
            let pos = body.len();
            let line = body.iter().filter(|&&c| c == b'\n').count() + 1;
            let col = pos - body.iter().rposition(|&c| c == b'\n').map_or(0, |i| i + 1) + 1;
            return Err(format!("Expecting value: line {line} column {col} (char {pos})"));
        }
        let v = pyjson::parse(body).map_err(|e| e.to_string())?;
        let batch = pydict::contains(&v, "batch").map_err(|e| e.msg)?;
        // The entries are answer_cached's prompts: the batch, or {"": body}.
        let single = [(String::new(), v.clone())];
        let entries: &[(String, Value)] =
            if batch { pydict::items(pydict::index(&v, "batch").map_err(|e| e.msg)?).map_err(|e| e.msg)? } else { &single };

        // Cache keys in answer_cached's order, which is also where Python's KeyErrors for a
        // missing state, questions, type or instructions arise (computed with the cache off
        // too, for those errors).
        let mut keys: Vec<Vec<String>> = vec![Vec::new(); entries.len()];
        let mut names: Vec<Vec<String>> = vec![Vec::new(); entries.len()];
        let mut types = vec![Vec::new(); entries.len()];
        for (i, (_, e)) in entries.iter().enumerate() {
            let qs = pydict::items(pydict::index(e, "questions").map_err(|e| e.msg)?).map_err(|e| e.msg)?;
            for (name, q) in qs {
                names[i].push(name.clone());
                let state = pydict::index(e, "state").map_err(|e| e.msg)?;
                keys[i].push(pydict::cache_key(state, q).map_err(|e| e.msg)?);
                types[i].push(q.get("type").cloned());
            }
        }
        // Python only builds the questions it sends to the model, so an unanswerable question
        // fails there, after the cache has counted it.
        let prompts = decode(entries, &names);
        let style = if batch { WireStyle::Batch } else { WireStyle::Predict };
        let answers = self.cache.answer(&keys, |miss| match &prompts {
            Err(e) => Err(decode_error(e, &types, miss)),
            Ok(prompts) => self.run(prompts, miss, batch),
        })?;

        // {"answers": {…}}: a batch flattens to "key.name", as a dict would (a repeated flat
        // name keeps its first position and its last answer).
        let mut flat: Vec<(String, &Answer)> = Vec::new();
        let mut at: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for (i, (key, _)) in entries.iter().enumerate() {
            for (j, n) in names[i].iter().enumerate() {
                let k = if batch { format!("{key}.{n}") } else { n.clone() };
                let answer = answers[i][j].as_ref().ok_or("missing prediction answer")?;
                match at.get(&k) {
                    Some(&x) => flat[x].1 = answer,
                    None => {
                        at.insert(k.clone(), flat.len());
                        flat.push((k, answer));
                    }
                }
            }
        }
        let mut b = String::from(r#"{"answers": {"#);
        for (i, (k, v)) in flat.iter().enumerate() {
            if i > 0 {
                b.push_str(", ");
            }
            pyjson::quote(&mut b, k, true);
            b.push_str(": ");
            b.push_str(&v.to_wire_json(style));
        }
        b.push_str("}}");
        Ok(b)
    }

    /// Answers the masked questions in one forward pass.
    fn run(&self, prompts: &[Prompt], miss: &[Vec<bool>], batch: bool) -> Result<Vec<Vec<Option<Answer>>>, String> {
        let mut sub = Vec::new();
        let mut from = Vec::new();
        for (i, p) in prompts.iter().enumerate() {
            let qs: Vec<_> = p.questions.iter().zip(&miss[i]).filter(|(_, m)| **m).map(|(q, _)| q.clone()).collect();
            if !qs.is_empty() {
                sub.push(Prompt { key: p.key.clone(), state: p.state.clone(), questions: qs });
                from.push(i);
            }
        }
        let mut out: Vec<Vec<Option<Answer>>> = prompts.iter().map(|p| vec![None; p.questions.len()]).collect();
        if sub.is_empty() {
            return Ok(out);
        }
        let results = self.model.predict_many(&sub).map_err(|e| python_error(&e, batch))?;
        if results.len() != from.len() {
            return Err("unexpected number of model results".into());
        }
        for (r, i) in results.into_iter().zip(from) {
            let mut answers = r.answers.into_iter();
            for (j, _) in prompts[i].questions.iter().enumerate() {
                if miss[i][j] {
                    let (_, answer) = answers.next().ok_or("missing model answer")?;
                    out[i][j] = Some(answer);
                }
            }
        }
        Ok(out)
    }
}

/// Decodes the entries into prompts, one to one. Python never reads the state of a prompt
/// without questions, so those decode as empty prompts.
fn decode(entries: &[(String, Value)], names: &[Vec<String>]) -> Result<Vec<Prompt>, LayaError> {
    entries
        .iter()
        .zip(names)
        .map(|((key, e), names)| {
            if names.is_empty() {
                return Ok(Prompt { key: key.clone(), state: Default::default(), questions: Vec::new() });
            }
            let one = Value::Object(vec![("batch".into(), Value::Object(vec![(key.clone(), e.clone())]))]);
            Ok(decode_request_value(&one)?.remove(0))
        })
        .collect()
}

/// Words a question Python can't build the way Python fails: an unknown type is a KeyError on
/// `QTYPES`.
fn decode_error(err: &LayaError, types: &[Vec<Option<Value>>], miss: &[Vec<bool>]) -> String {
    if err.to_string().contains("unknown question type") {
        for (ts, ms) in types.iter().zip(miss) {
            for (t, &m) in ts.iter().zip(ms) {
                if let (true, Some(Value::Str(s))) = (m, t)
                    && QType::parse(s).is_err()
                {
                    return repr(s);
                }
            }
        }
    }
    err.to_string()
}

/// Rewrites errors Python reports in its own words.
fn python_error(err: &LayaError, batch: bool) -> String {
    if let LayaError::Question { key, name, err: inner } = err {
        match inner.as_ref() {
            LayaError::OptionsTooLong(limit) if batch => {
                return format!("question {key}.{name} options exceed head_max_len={limit}"); // laya_batch.predict_many
            }
            LayaError::OptionsTooLong(limit) => {
                return format!("question {} options exceed head_max_len={limit}", repr(name)); // Agent.system_one
            }
            // The forward pass fails on a question without options (torch.topk).
            LayaError::NoOptions => return "selected index k out of range".into(),
            _ => {}
        }
    }
    err.to_string()
}

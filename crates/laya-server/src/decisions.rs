//! The text-only Decisions endpoint, following laya-swift's LayaService and Decisions.

use std::collections::HashSet;

use laya::calibrate::round4;
use laya::pydict;
use laya::pyjson::Value as PyValue;
use laya::question::{Answer, LayaError, Named, Opt, Prompt, QType, Question, State};
use serde_json::{Value, json};

use crate::{Predictor, Response, Server};

struct Decision {
    name: Option<String>,
    question: Question,
    criteria: PyValue,
    choices: Vec<Value>,
    labels: Vec<String>,
}

impl<P: Predictor> Server<P> {
    pub(super) fn decisions(&self, body: &[u8]) -> Response {
        let (state, decisions) = match decode(body) {
            Ok(request) => request,
            Err(message) => return error(400, "invalid_request_error", &message),
        };
        match self.decide(state, &decisions) {
            Ok(value) => Response::json(200, value.to_string()),
            Err(message) => error(500, "server_error", &message),
        }
    }

    fn decide(&self, state: String, decisions: &[Decision]) -> Result<Value, String> {
        let mut prompt = Prompt { key: String::new(), state: State(state.clone()), questions: Vec::new() };
        let state = PyValue::Str(state);
        let mut keys = Vec::new();
        let mut positions = Vec::new();
        let mut lengths = Vec::new();

        // Validate and measure each question independently so one refusal leaves the others
        // available for a single forward pass. Only usable questions enter the shared cache.
        for (i, decision) in decisions.iter().enumerate() {
            let named = Named { name: format!("q{i}"), question: decision.question.clone() };
            match self.model.sequence_length(&prompt.state, &named) {
                Ok(tokens) => lengths.push(tokens),
                Err(LayaError::Question { .. }) => continue,
                Err(err) => return Err(err.to_string()),
            }
            let question = PyValue::Object(vec![
                ("type".into(), PyValue::Str(decision.question.qtype.as_str().into())),
                ("instructions".into(), PyValue::Str(decision.question.instructions.clone())),
                ("criteria".into(), decision.criteria.clone()),
            ]);
            keys.push(pydict::cache_key(&state, &question).map_err(|e| e.msg)?);
            positions.push(i);
            prompt.questions.push(named);
        }

        let mut output: Vec<_> = decisions.iter().map(|d| json!({"type": "refusal", "name": d.name})).collect();
        let mut cached_tokens = 0;
        let input_tokens: usize = lengths.iter().sum();
        if !keys.is_empty() {
            cached_tokens = input_tokens;
            let answers = self.cache.answer(&[keys], |miss| {
                cached_tokens -= lengths.iter().zip(&miss[0]).filter(|(_, asked)| **asked).map(|(length, _)| length).sum::<usize>();
                self.run(&[prompt], miss, false)
            })?;
            for (i, answer) in positions.into_iter().zip(&answers[0]) {
                output[i] = decisions[i].encode(answer.as_ref().ok_or("missing decision answer")?);
            }
        }
        Ok(json!({
            "model": "laya",
            "answers": output,
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_details": {"cached_tokens": cached_tokens, "cache_write_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": input_tokens
            }
        }))
    }
}

fn error(status: u16, kind: &str, message: &str) -> Response {
    Response::json(status, json!({"error": {"message": message, "type": kind, "param": null, "code": null}}).to_string())
}

fn decode(body: &[u8]) -> Result<(String, Vec<Decision>), String> {
    let root: Value = serde_json::from_slice(body).map_err(|_| "request body must be a JSON object")?;
    if !root.is_object() {
        return Err("request body must be a JSON object".into());
    }
    root["model"].as_str().ok_or("model: required string")?;
    let state = flatten(root.get("input").ok_or("input: required")?)?;
    let questions =
        root["questions"].as_array().filter(|q| (1..=400).contains(&q.len())).ok_or("questions: 1 to 400 questions required")?;
    let decisions = questions.iter().enumerate().map(|(i, q)| Decision::decode(q, i)).collect::<Result<_, _>>()?;
    Ok((state, decisions))
}

fn optional_string<'a>(value: &'a Value, path: &str) -> Result<Option<&'a str>, String> {
    match value {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(s)),
        _ => Err(format!("{path}: must be a string")),
    }
}

impl Decision {
    fn decode(q: &Value, i: usize) -> Result<Self, String> {
        let path = format!("questions[{i}]");
        if !q.is_object() {
            return Err(format!("{path}: must be an object"));
        }
        let kind = q["type"]
            .as_str()
            .filter(|kind| matches!(*kind, "predicate" | "choice" | "score"))
            .ok_or_else(|| format!("{path}.type: must be predicate, choice or score"))?;
        let name = optional_string(&q["name"], &format!("{path}.name"))?.map(str::to_owned);
        let instructions =
            parts(q.get("instructions").ok_or_else(|| format!("{path}.instructions: required"))?, &format!("{path}.instructions"))?;
        let mut choices = Vec::new();
        let mut labels = Vec::new();
        let (question, criteria) = match kind {
            "predicate" => (Question::noul(instructions), PyValue::Null),
            "choice" => {
                let items = q["choices"]
                    .as_array()
                    .filter(|items| (2..=255).contains(&items.len()))
                    .ok_or_else(|| format!("{path}.choices: 2 to 255 choices required"))?;
                let mut names = HashSet::new();
                let mut options = Vec::new();
                let mut criteria = Vec::new();
                for (j, item) in items.iter().enumerate() {
                    let item_path = format!("{path}.choices[{j}]");
                    if !item.is_object() {
                        return Err(format!("{item_path}: must be an object"));
                    }
                    let value = &item["value"];
                    let label = match value {
                        Value::String(s) => s.as_str(),
                        Value::Bool(true) => "true",
                        Value::Bool(false) => "false",
                        _ => return Err(format!("{item_path}.value: must be a string or boolean")),
                    };
                    if choices.contains(value) {
                        return Err(format!("{path}.choices: values must be unique"));
                    }
                    if !names.insert(label) {
                        return Err(format!(
                            "{path}.choices: \"{label}\" appears as both a string and a boolean, which Laya can't tell apart"
                        ));
                    }
                    let desc = optional_string(&item["description"], &format!("{item_path}.description"))?;
                    options.push(Opt { name: label.into(), description: desc.unwrap_or_default().into() });
                    criteria.push((label.into(), desc.map_or(PyValue::Null, |s| PyValue::Str(s.into()))));
                    choices.push(value.clone());
                }
                (Question::choice(instructions, options), PyValue::Object(criteria))
            }
            _ => {
                let items = q["levels"]
                    .as_array()
                    .filter(|items| (2..=10).contains(&items.len()))
                    .ok_or_else(|| format!("{path}.levels: 2 to 10 levels required"))?;
                let mut levels = Vec::new();
                for (j, item) in items.iter().enumerate() {
                    let item_path = format!("{path}.levels[{j}]");
                    let label = item["label"].as_str().ok_or_else(|| format!("{item_path}.label: required string"))?;
                    let desc = optional_string(&item["description"], &format!("{item_path}.description"))?;
                    labels.push(label.to_owned());
                    levels.push(desc.map_or_else(|| label.to_owned(), |d| format!("{label}: {d}")));
                }
                let criteria = PyValue::Array(levels.iter().cloned().map(PyValue::Str).collect());
                (Question::score(instructions, levels), criteria)
            }
        };
        Ok(Self { name, question, criteria, choices, labels })
    }

    fn encode(&self, answer: &Answer) -> Value {
        let confidence = round4(answer.confidence);
        match self.question.qtype {
            QType::Noul => json!({"type": "predicate", "name": self.name, "probability": round4(answer.noul)}),
            QType::Choice => {
                let mut best = 0;
                let probabilities: Vec<_> = self
                    .choices
                    .iter()
                    .zip(&answer.probabilities)
                    .enumerate()
                    .map(|(i, (value, (_, p)))| {
                        if round4(*p) > round4(answer.probabilities[best].1) {
                            best = i;
                        }
                        json!({"value": value, "probability": round4(*p)})
                    })
                    .collect();
                json!({"type": "choice", "name": self.name, "choice": self.choices[best], "confidence": confidence, "probabilities": probabilities})
            }
            QType::Score => {
                let probabilities: Vec<_> = self
                    .labels
                    .iter()
                    .zip(&answer.probabilities)
                    .enumerate()
                    .map(|(i, (label, (_, p)))| json!({"value": i, "label": label, "probability": round4(*p)}))
                    .collect();
                json!({"type": "score", "name": self.name, "score": round4(answer.score), "confidence": confidence, "probabilities": probabilities})
            }
        }
    }
}

fn flatten(input: &Value) -> Result<String, String> {
    if let Some(s) = input.as_str() {
        return Ok(s.into());
    }
    let items = input.as_array().ok_or("input: must be a string or an array")?;
    let mut lines = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let path = format!("input[{i}]");
        if !item.is_object() {
            return Err(format!("{path}: must be an object"));
        }
        let text = |field: &str| item[field].as_str().unwrap_or_default();
        let output = |prefix: &str| -> Result<String, String> {
            let value = match item.get("output") {
                Some(value) => parts(value, &format!("{path}.output"))?,
                None => String::new(),
            };
            Ok(format!("{prefix}: {value}"))
        };
        let line = match item["type"].as_str().unwrap_or("message") {
            "message" => {
                let role = text("role");
                if !matches!(role, "user" | "developer" | "system" | "assistant") {
                    return Err(format!("{path}.role: must be user, developer, system or assistant"));
                }
                let content = item.get("content").ok_or_else(|| format!("{path}.content: required"))?;
                format!("{role}: {}", parts(content, &format!("{path}.content"))?)
            }
            "function_call" => format!("function call {}: {}", text("name"), text("arguments")),
            "function_call_output" => output("function output")?,
            "custom_tool_call" => format!("tool call {}: {}", text("name"), text("input")),
            "custom_tool_call_output" => output("tool output")?,
            kind => return Err(format!("{path}.type: unsupported item type \"{kind}\"")),
        };
        lines.push(line);
    }
    Ok(lines.join("\n"))
}

fn parts(value: &Value, path: &str) -> Result<String, String> {
    if let Some(s) = value.as_str() {
        return Ok(s.into());
    }
    let items = value.as_array().ok_or_else(|| format!("{path}: must be a string or an array of parts"))?;
    let mut text = Vec::new();
    for (i, part) in items.iter().enumerate() {
        let path = format!("{path}[{i}]");
        let kind = part["type"].as_str().ok_or_else(|| format!("{path}: must be a content part"))?;
        text.push(match kind {
            "input_text" | "output_text" | "text" => part["text"].as_str().unwrap_or_default().into(),
            "refusal" => part["refusal"].as_str().unwrap_or_default().into(),
            "input_json" if part["data"].is_object() || part["data"].is_array() => part["data"].to_string(),
            "input_json" => return Err(format!("{path}.data: must be an object or array")),
            "input_image" => return Err(format!("{path}: images are not supported (Laya reads text)")),
            _ => return Err(format!("{path}.type: unsupported part type \"{kind}\"")),
        });
    }
    Ok(text.join("\n"))
}

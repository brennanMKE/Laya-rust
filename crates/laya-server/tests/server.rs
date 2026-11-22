//! laya-go's `cmd/laya-server` server tests, ported: response shapes, the answer cache, and
//! Python's error wording. They need the checkpoint's tokenizer and skip without it; the
//! network is a fake engine.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use laya::pyjson::{self, Value};
use laya::{Batch, Engine, EngineError, Model, Output, checkpoint};
use laya_server::Server;

/// Logits made from each sequence's tokens, so answers are deterministic and differ between
/// prompts; counts what reaches it.
#[derive(Default)]
struct Counts {
    calls: AtomicUsize,
    seqs: AtomicUsize,
}

struct Fake(Arc<Counts>);

impl Engine for Fake {
    fn forward(&self, b: &Batch) -> Result<Output, EngineError> {
        self.0.calls.fetch_add(1, Ordering::SeqCst);
        self.0.seqs.fetch_add(b.len(), Ordering::SeqCst);
        let mut out = Output::default();
        for i in 0..b.len() {
            let mut h: u32 = 2166136261;
            for &t in b.seq(i) {
                h = (h ^ t).wrapping_mul(16777619);
            }
            out.logits.push((0..b.markers[i].len()).map(|j| ((h >> (j * 3)) % 97) as f32 / 20.0).collect());
            out.act_logits.push([(h % 7) as f32 / 3.0, 0.5]);
        }
        Ok(out)
    }
}

fn checkpoint_dir() -> Option<PathBuf> {
    std::env::var_os("LAYA_CHECKPOINT").map(PathBuf::from).or_else(|| checkpoint::find(None, "", "", "").ok())
}

fn server(cache: usize) -> Option<(Server<Model>, Arc<Counts>)> {
    let Some(dir) = checkpoint_dir() else {
        eprintln!("skipped: no checkpoint (the tokenizer is needed)");
        return None;
    };
    let counts = Arc::new(Counts::default());
    let model = Model::new(&dir, Box::new(Fake(counts.clone()))).unwrap();
    Some((Server::new(model, cache), counts))
}

const CHOICE_Q: &str =
    r#"{"type": "choice", "instructions": "What kind of place is it?", "criteria": {"deadly": "a deadly place", "safe": "a safe place"}}"#;

fn answers(body: &str) -> Value {
    let v = pyjson::parse(body.as_bytes()).unwrap_or_else(|e| panic!("{e}: {body}"));
    v.get("answers").unwrap_or_else(|| panic!("no answers: {body}")).clone()
}

fn keys(v: &Value) -> Vec<String> {
    match v {
        Value::Object(m) => m.iter().map(|(k, _)| k.clone()).collect(),
        _ => panic!("not an object"),
    }
}

#[test]
fn single_and_batch_shapes() {
    let Some((s, _)) = server(100) else { return };
    let r =
        s.handle("POST", "/predict", format!(r#"{{"state": "a frog", "questions": {{"place": {CHOICE_Q}}}, "model": "laya"}}"#).as_bytes());
    assert_eq!(r.status, 200, "{}", r.body);
    let a = answers(&r.body);
    let place = a.get("place").unwrap();
    assert_eq!(keys(place), ["type", "choice", "probabilities", "confidence", "action"]);

    let body = format!(
        r#"{{"batch": {{"k1": {{"state": "x", "questions": {{"place": {CHOICE_Q}, "lvl": {{"type": "score", "instructions": "how?", "criteria": ["low", "high"]}}}}}}, "k2": {{"state": "y", "questions": {{"n": {{"type": "noul", "instructions": "?"}}}}}}}}}}"#
    );
    let r = s.handle("POST", "/predict", body.as_bytes());
    assert_eq!(r.status, 200, "{}", r.body);
    let a = answers(&r.body);
    assert_eq!(keys(&a), ["k1.place", "k1.lvl", "k2.n"]);
    assert_eq!(keys(a.get("k1.lvl").unwrap()), ["type", "score", "probabilities"]);
    assert_eq!(keys(a.get("k2.n").unwrap()), ["type", "noul", "confidence"]);
}

#[test]
fn cache_sends_only_misses() {
    let Some((s, counts)) = server(100) else { return };
    let q = |state: &str| format!(r#"{{"state": "{state}", "questions": {{"a": {CHOICE_Q}, "b": {CHOICE_Q}}}}}"#);
    let first = s.handle("POST", "/predict", q("s1").as_bytes());
    assert_eq!(counts.seqs.load(Ordering::SeqCst), 1, "the same question under two names is asked once");
    let again = s.handle("POST", "/predict", q("s1").as_bytes());
    assert_eq!(first, again);
    assert_eq!(counts.calls.load(Ordering::SeqCst), 1, "a fully cached request skips the model");
    s.handle("POST", "/predict", q("s2").as_bytes());
    assert_eq!(counts.calls.load(Ordering::SeqCst), 2);
    let stats = s.handle("GET", "/stats", b"");
    assert_eq!(
        stats.body,
        r#"{"enabled": true, "size": 2, "capacity": 100, "hits": 4, "misses": 2, "hit_rate": 0.6667, "model_calls": 2, "skipped_calls": 1}"#
    );
}

#[test]
fn cache_off() {
    let Some((s, counts)) = server(0) else { return };
    let body = format!(r#"{{"state": "s", "questions": {{"a": {CHOICE_Q}}}}}"#);
    s.handle("POST", "/predict", body.as_bytes());
    s.handle("POST", "/predict", body.as_bytes());
    assert_eq!(counts.calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        s.handle("GET", "/stats", b"").body,
        r#"{"enabled": false, "size": 0, "capacity": 0, "hits": 0, "misses": 0, "hit_rate": 0.0, "model_calls": 0, "skipped_calls": 0}"#
    );
}

#[test]
fn flat_key_collision() {
    let Some((s, _)) = server(100) else { return };
    // "a.b" + "c" and "a" + "b.c" flatten to the same name; like a dict, the first position
    // and the last answer win.
    let body = format!(
        r#"{{"batch": {{"a.b": {{"state": "x", "questions": {{"c": {CHOICE_Q}}}}}, "a": {{"state": "y", "questions": {{"b.c": {{"type": "noul", "instructions": "?"}}}}}}}}}}"#
    );
    let a = answers(&s.handle("POST", "/predict", body.as_bytes()).body);
    assert_eq!(keys(&a), ["a.b.c"]);
    assert_eq!(a.get("a.b.c").unwrap().get("type"), Some(&Value::Str("noul".into())));
}

#[test]
fn errors_and_paths() {
    let Some((s, _)) = server(100) else { return };
    let many: Vec<String> = (0..140).map(|i| format!(r#""option number {i}": "a rather long description of option {i}""#)).collect();
    let long = format!(r#"{{"type": "choice", "instructions": "Pick", "criteria": {{{}}}}}"#, many.join(", "));
    let cases: Vec<(&str, &str, String, u16, &str)> = vec![
        ("GET", "/health", "".into(), 200, r#"{"ok": true}"#),
        ("GET", "/health?x=1", "".into(), 404, r#"{"error": "not found"}"#),
        ("GET", "/predict", "".into(), 404, r#"{"error": "not found"}"#),
        ("POST", "/predict/", "{}".into(), 404, r#"{"error": "POST /predict"}"#),
        ("POST", "/predict", "".into(), 400, r#"{"error": "Expecting value: line 1 column 1 (char 0)"}"#),
        ("POST", "/predict", format!(r#"{{"questions": {{"a": {CHOICE_Q}}}}}"#), 400, r#"{"error": "'state'"}"#),
        ("POST", "/predict", r#"{"state": "s"}"#.into(), 400, r#"{"error": "'questions'"}"#),
        ("POST", "/predict", r#"{"state": "s", "questions": {"a": {"instructions": "i"}}}"#.into(), 400, r#"{"error": "'type'"}"#),
        ("POST", "/predict", r#"{"state": "s", "questions": []}"#.into(), 400, r#"{"error": "'list' object has no attribute 'items'"}"#),
        ("POST", "/predict", "5".into(), 400, r#"{"error": "argument of type 'int' is not a container or iterable"}"#),
        ("POST", "/predict", r#"{"batch": {"k": "s"}}"#.into(), 400, r#"{"error": "string indices must be integers, not 'str'"}"#),
        (
            "POST",
            "/predict",
            format!(r#"{{"state": "s", "questions": {{"it's": {long}}}}}"#),
            400,
            r#"{"error": "question \"it's\" options exceed head_max_len=192"}"#,
        ),
        (
            "POST",
            "/predict",
            format!(r#"{{"batch": {{"k": {{"state": "s", "questions": {{"q": {long}}}}}}}}}"#),
            400,
            r#"{"error": "question k.q options exceed head_max_len=192"}"#,
        ),
        ("POST", "/predict", r#"{"batch": {}}"#.into(), 200, r#"{"answers": {}}"#),
        ("POST", "/predict", r#"{"batch": {"k": {"questions": {}}}}"#.into(), 200, r#"{"answers": {}}"#),
        (
            "POST",
            "/predict",
            r#"{"state": "s", "questions": {"a": {"type": "foo", "instructions": "i"}}}"#.into(),
            400,
            r#"{"error": "'foo'"}"#,
        ),
        ("POST", "/predict", " \n ".into(), 400, r#"{"error": "Expecting value: line 2 column 2 (char 3)"}"#),
    ];
    let mut bad = 0;
    for (method, path, body, code, want) in &cases {
        let r = s.handle(method, path, body.as_bytes());
        if r.status != *code || r.body != *want {
            bad += 1;
            eprintln!("{method} {path} {:.60}: {} {}\n   want {code} {want}", body, r.status, r.body);
        }
    }
    assert_eq!(bad, 0);
    assert_eq!(s.handle("PUT", "/predict", b"{}").status, 501);
}

#[test]
fn cache_keeps_single_and_batch_wire_shapes() {
    let Some((s, counts)) = server(100) else { return };
    for (state, single_first) in [("first", true), ("second", false)] {
        let single = format!(r#"{{"state":"{state}","questions":{{"place":{CHOICE_Q}}}}}"#);
        let batch = format!(r#"{{"batch":{{"game":{single}}}}}"#);
        let mut requests = [(&single, "place", true), (&batch, "game.place", false)];
        if !single_first {
            requests.reverse();
        }
        for (body, key, has_action) in requests {
            let response = s.handle("POST", "/predict", body.as_bytes());
            assert_eq!(response.status, 200, "{}", response.body);
            assert_eq!(answers(&response.body).get(key).unwrap().get("action").is_some(), has_action);
        }
    }
    assert_eq!(counts.calls.load(Ordering::SeqCst), 2);
}

#[test]
fn decisions_returns_ordered_answers_refusals_and_usage() {
    let Some((s, counts)) = server(100) else { return };
    let mut body = serde_json::json!({
        "model": "requested",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "a frog"}]}],
        "questions": [
            {"type": "predicate", "name": "safe", "instructions": "Safe?"},
            {"type": "choice", "instructions": "Move?", "choices": [{"value": true}, {"value": "wait"}]},
            {"type": "score", "name": "rating", "instructions": "How safe?", "levels": [{"label": "low"}, {"label": "high", "description": "very safe"}]},
            {"type": "choice", "name": "too long", "instructions": "Pick", "choices": []},
        ]
    });
    body["questions"][3]["choices"] = serde_json::json!(
        (0..255)
            .map(|i| serde_json::json!({
                "value": format!("a very long decision option with several words and index {i}"), "description": "a rather long description of that option"
            }))
            .collect::<Vec<_>>()
    );
    let body = body.to_string();
    let request = || {
        let r = s.handle("POST", "/v1/decisions", body.as_bytes());
        assert_eq!(r.status, 200, "{}", r.body);
        serde_json::from_str::<serde_json::Value>(&r.body).unwrap()
    };
    let first = request();
    assert_eq!(first["model"], "laya");
    assert!(first["answers"][0]["probability"].is_number());
    assert_eq!(first["answers"][1]["probabilities"][0]["value"], true);
    assert_eq!(first["answers"][1]["name"], serde_json::Value::Null);
    assert_eq!(first["answers"][2]["probabilities"][1]["label"], "high");
    assert_eq!(first["answers"][2]["probabilities"][1]["value"], 1);
    assert_eq!(first["answers"][3], serde_json::json!({"type": "refusal", "name": "too long"}));
    assert_eq!(counts.seqs.load(Ordering::SeqCst), 3);
    assert_eq!(first["usage"]["input_tokens_details"]["cached_tokens"], 0);
    assert!(first["usage"]["input_tokens"].as_u64().unwrap() > 0);
    let again = request();
    assert_eq!(first["answers"], again["answers"]);
    assert_eq!(first["usage"]["input_tokens"], again["usage"]["input_tokens_details"]["cached_tokens"]);

    let predict = br#"{"state":"user: a frog","questions":{"safe":{"type":"noul","instructions":"Safe?"}}}"#;
    assert_eq!(s.handle("POST", "/predict", predict).status, 200);
    assert_eq!(counts.calls.load(Ordering::SeqCst), 1, "both endpoints share model answers");
}

#[test]
fn decisions_rejects_unsupported_content() {
    let Some((s, counts)) = server(0) else { return };
    let r = s.handle("POST", "/v1/decisions", br#"{"model":"laya","input":[{"role":"user","content":[{"type":"input_image"}]}]}"#);
    assert_eq!(r.status, 400);
    let body: serde_json::Value = serde_json::from_str(&r.body).unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["message"], "input[0].content[0]: images are not supported (Laya reads text)");
    assert_eq!(counts.calls.load(Ordering::SeqCst), 0);
}

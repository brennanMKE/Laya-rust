//! Phase 0 parity: the text side against laya-go's golden fixtures (made by the Python
//! reference). Token ids and markers must be identical, and given Python's own logits every
//! answer must be identical JSON text after 4-decimal rounding, in both answer shapes.
//!
//! Tests that need the checkpoint's tokenizer skip when it isn't in `$LAYA_CHECKPOINT` or the
//! Hugging Face cache; they never download.

mod common;

use std::sync::{Arc, OnceLock};

use common::*;
use laya_core::pyjson::{self, Value};
use laya_core::question::decode_request_value;
use laya_core::tokenizer::Tokenizer;
use laya_core::{LayaError, Model, Output, WireStyle};

struct Fixture {
    model: Model,
    replay: Arc<Replay>,
    /// Held while a test queues logits and predicts, since tests share the replay engine.
    turn: std::sync::Mutex<()>,
}

fn fixture() -> Option<&'static Fixture> {
    static F: OnceLock<Option<Fixture>> = OnceLock::new();
    F.get_or_init(|| {
        let dir = checkpoint_dir()?;
        let replay = Arc::new(Replay::default());
        let model = Model::new(&dir, Box::new(ReplayEngine(replay.clone()))).expect("load model");
        Some(Fixture { model, replay, turn: Default::default() })
    })
    .as_ref()
}

macro_rules! need_fixture {
    () => {
        match fixture() {
            Some(f) => f,
            None => {
                eprintln!("skipped: no checkpoint (set LAYA_CHECKPOINT)");
                return;
            }
        }
    };
}

/// A fixture answer rendered as Python's `json.dumps` would.
fn py_text(v: &Value) -> String {
    v.dumps(true)
}

fn queue(f: &Fixture, seqs: &[Value], single: bool) {
    let (l, a) = if single { ("logits_single", "act_logits_single") } else { ("logits", "act_logits") };
    let mut out = Output::default();
    for s in seqs {
        out.logits.push(f32s(at(s, &[l])));
        let act = f32s(at(s, &[a]));
        out.act_logits.push([act[0], act[1]]);
    }
    f.replay.queue(out);
}

fn check_records(file: &str) {
    let f = need_fixture!();
    let _turn = f.turn.lock().unwrap_or_else(|e| e.into_inner());
    let b = f.model.builder();
    let recs = read_jsonl(file);
    let (mut seqs, mut answers, mut preds, mut usage) = (Tally::default(), Tally::default(), Tally::default(), Tally::default());
    for rec in &recs {
        let id = str_of(at(rec, &["id"]));
        let prompts = decode_request_value(at(rec, &["request"])).unwrap_or_else(|e| panic!("{id}: {e}"));
        let want_seqs = arr(at(rec, &["sequences"]));

        // Token ids and markers, question by question.
        let mut i = 0;
        for p in &prompts {
            for q in &p.questions {
                let want = &want_seqs[i];
                let got = b.build(p.state.as_str(), &q.question.render());
                let (ids, markers) = (u32s(at(want, &["ids"])), u32s(at(want, &["markers"])));
                seqs.check(
                    got.ids == ids
                        && got.markers == markers
                        && got.options as i64 == int(at(want, &["n_options"]))
                        && q.question.qtype as i64 == int(at(want, &["qtype"])),
                    || {
                        format!(
                            "{id} {}.{}: sequence differs\n got  {:?} {:?}\n want {ids:?} {markers:?}",
                            p.key, q.name, got.ids, got.markers
                        )
                    },
                );
                i += 1;
            }
        }
        assert_eq!(i, want_seqs.len(), "{id}: questions decoded vs fixture");

        if let Some(err) = rec.get("error") {
            match f.model.encode(&prompts) {
                Err(e) if matches!(e.root(), LayaError::OptionsTooLong(_)) => {}
                other => panic!("{id}: encode gave {other:?}, want options too long (Python: {})", str_of(err)),
            }
            continue;
        }

        // predict_many: one pass over every prompt.
        queue(f, want_seqs, false);
        let results = f.model.predict_many(&prompts).unwrap_or_else(|e| panic!("{id}: {e}"));
        for r in &results {
            for (name, a) in &r.answers {
                let want = py_text(at(rec, &["predict_many", &r.key, name]));
                let got = a.to_wire_json(WireStyle::Batch);
                answers.check(got == want, || format!("{id} {}.{name} predict_many:\n got  {got}\n want {want}", r.key));
            }
        }

        // Agent.predict: one pass per state, with that pass's logits.
        let mut i = 0;
        for p in &prompts {
            let n = p.questions.len();
            queue(f, &want_seqs[i..i + n], true);
            i += n;
            let r = f.model.predict(p).unwrap_or_else(|e| panic!("{id}: {e}"));
            let single = at(rec, &["predict", &p.key]);
            for (name, a) in &r.answers {
                let want = py_text(at(single, &["answers", name]));
                let got = a.to_wire_json(WireStyle::Predict);
                preds.check(got == want, || format!("{id} {}.{name} predict:\n got  {got}\n want {want}", p.key));
            }
            let tokens = int(at(single, &["usage", "input_tokens"])) as usize;
            usage.check(r.input_tokens == tokens, || format!("{id} {}: input_tokens {}, want {tokens}", p.key, r.input_tokens));
        }
    }
    eprintln!(
        "{file}: {} records: sequences {}, predict_many answers {}, predict answers {}, usage {} identical",
        recs.len(),
        seqs.summary(),
        answers.summary(),
        preds.summary(),
        usage.summary()
    );
    assert!(seqs.bad + answers.bad + preds.bad + usage.bad == 0, "{file}: mismatches (see stderr)");
}

#[test]
fn golden_edge() {
    check_records("edge.jsonl");
}

#[test]
fn golden_games() {
    check_records("games.jsonl");
}

/// Synthetic logits (ties, huge gaps, every temperature bucket and option count) through both
/// answer shapes.
#[test]
fn golden_calibration() {
    let meta = std::fs::read_to_string(golden_dir().join("meta.json")).unwrap();
    let config = laya_core::checkpoint::Config::parse(&meta).unwrap();
    let (temperatures, _) = laya_core::calibrate::Temperatures::new(&config.temperature, &config.temperature_by_options);
    let mut c = Tally::default();
    for rec in read_jsonl("calibration.jsonl") {
        let id = str_of(at(&rec, &["id"]));
        let body = Value::Object(vec![
            ("state".into(), Value::Str("s".into())),
            ("questions".into(), Value::Object(vec![("q".into(), at(&rec, &["question"]).clone())])),
        ]);
        let prompts = decode_request_value(&body).unwrap();
        let q = &prompts[0].questions[0].question;
        let logits = f32s(at(&rec, &["logits"]));
        let act = f32s(at(&rec, &["act_logits"]));
        let a = laya_core::calibrate::answer(&temperatures, q, &logits, [act[0], act[1]]);
        for (style, key) in [(WireStyle::Batch, "predict_many"), (WireStyle::Predict, "predict")] {
            let (got, want) = (a.to_wire_json(style), py_text(at(&rec, &[key])));
            c.check(got == want, || format!("{id} {key}: logits {logits:?}\n got  {got}\n want {want}"));
        }
    }
    eprintln!("calibration answers {} identical", c.summary());
    assert_eq!(c.bad, 0);
}

/// Every string in `tokenizer.jsonl` tokenizes to Python's ids.
#[test]
fn golden_tokenizer() {
    let Some(dir) = checkpoint_dir() else {
        eprintln!("skipped: no checkpoint");
        return;
    };
    let tok = Tokenizer::from_file(dir.join("tokenizer/tokenizer.json")).unwrap();
    let mut c = Tally::default();
    for rec in read_jsonl("tokenizer.jsonl") {
        let text = str_of(at(&rec, &["text"]));
        let want = u32s(at(&rec, &["ids"]));
        let got = tok.encode(text);
        c.check(got == want, || format!("{text:?}\n got  {got:?}\n want {want:?}"));
    }
    eprintln!("tokenizer ids {} identical", c.summary());
    assert_eq!(c.bad, 0);
}

/// Python's `json.dumps` output for awkward documents, both `ensure_ascii` settings.
#[test]
fn pyjson_dumps_matches_python() {
    let text = include_str!("../testdata/pyjson/dumps.jsonl");
    let mut n = 0;
    for line in text.lines() {
        let c = pyjson::parse(line.as_bytes()).unwrap();
        let input = str_of(at(&c, &["in"]));
        let v = pyjson::parse(input.as_bytes()).unwrap_or_else(|e| panic!("parse {input:?}: {e}"));
        assert_eq!(v.dumps(true), str_of(at(&c, &["ascii"])), "ascii dumps of {input:?}");
        assert_eq!(v.dumps(false), str_of(at(&c, &["raw"])), "raw dumps of {input:?}");
        n += 1;
    }
    assert!(n > 0);
}

/// Python's `repr` of random floats.
#[test]
fn pyjson_float_repr() {
    let v = pyjson::parse(include_bytes!("../testdata/pyjson/floats.json")).unwrap();
    for r in arr(&v) {
        let want = str_of(r);
        let f: f64 = want.parse().unwrap();
        assert_eq!(pyjson::float_repr(f), want);
    }
}

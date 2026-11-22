//! Batched forward passes over `edge.jsonl` and `games.jsonl`, compared with Python's logits.
//!
//! By default the gemm kernels run every edge record and every 8th game record, and the
//! portable kernels a smaller sample; `LAYA_GOLDEN_ALL=1` runs every record with both.

mod common;

use laya_core::calibrate::{self, Temperatures};
use laya_core::checkpoint::Config;
use laya_core::engine::{Batch, Engine};
use laya_core::{WireStyle, decode_request};
use laya_cpu::KernelSet;
use serde_json::Value;

#[derive(Default)]
struct Stats {
    records: usize,
    seqs: usize,
    max_logit: f32,
    worst: String,
    max_logit_single: f32,
    max_act_rel: f32,
    argmax_checked: usize,
    argmax_same: usize,
    wire_fields: usize,
    wire_exact: usize,
    failures: Vec<String>,
}

fn compare_wire(got: &Value, want: &Value, stats: &mut Stats, path: &str) {
    match (got, want) {
        (Value::Object(got), Value::Object(want)) => {
            assert_eq!(got.keys().collect::<Vec<_>>(), want.keys().collect::<Vec<_>>(), "{path}: answer fields");
            for (key, value) in want {
                compare_wire(&got[key], value, stats, &format!("{path}.{key}"));
            }
        }
        (Value::Number(got), Value::Number(want)) => {
            let (got, want) = (got.as_f64().unwrap(), want.as_f64().unwrap());
            stats.wire_fields += 1;
            stats.wire_exact += usize::from(got == want);
            assert!((got - want).abs() <= 1e-4 + 1e-12, "{path}: got {got}, want {want}");
        }
        _ => assert_eq!(got, want, "{path}"),
    }
}

fn run(kernels: KernelSet, file: &str, every: usize, all: bool) -> (usize, usize) {
    let Some(engine) = common::engine(kernels) else { return (0, 0) };
    let every = if all { 1 } else { every };
    let text = std::fs::read_to_string(common::golden_dir().join(file)).unwrap();
    let config = Config::parse(&std::fs::read_to_string(common::golden_dir().join("meta.json")).unwrap()).unwrap();
    let (temperatures, _) = Temperatures::new(&config.temperature, &config.temperature_by_options);
    let mut st = Stats::default();
    let start = std::time::Instant::now();
    for (li, line) in text.lines().enumerate() {
        if li % every != 0 {
            continue;
        }
        let rec: serde_json::Value = serde_json::from_str(line).unwrap();
        if rec.get("error").is_some() {
            continue;
        }
        let id = rec["id"].as_str().unwrap();
        let seqs = rec["sequences"].as_array().unwrap();
        let mut batch = Batch::default();
        for s in seqs {
            batch.push(&common::u32s(&s["ids"]), common::u32s(&s["markers"]), s["qtype"].as_u64().unwrap() as u8);
        }
        // Decode the original ordered JSON. Re-serializing a serde object would reorder
        // choice criteria and change the relationship between their indices and logits.
        let ordered = laya_core::pyjson::parse(line.as_bytes()).unwrap();
        let request = ordered.get("request").unwrap().dumps(false);
        let prompts = decode_request(request.as_bytes()).unwrap();
        let questions = prompts.iter().flat_map(|prompt| &prompt.questions).collect::<Vec<_>>();
        assert_eq!(questions.len(), seqs.len(), "{id}");
        let out = engine.forward(&batch).unwrap_or_else(|e| panic!("{id}: {e}"));
        st.records += 1;
        for (i, s) in seqs.iter().enumerate() {
            st.seqs += 1;
            let want = common::f32s(&s["logits"]);
            let single = common::f32s(&s["logits_single"]);
            let got = &out.logits[i];
            assert_eq!(got.len(), want.len(), "{id} seq {i}: logit count");
            for j in 0..got.len() {
                let d = (got[j] - want[j]).abs();
                if d > st.max_logit {
                    st.max_logit = d;
                    st.worst = format!("{id}/{}.{}", s["key"].as_str().unwrap_or(""), s["question"]);
                }
                if single.len() == got.len() {
                    st.max_logit_single = st.max_logit_single.max((got[j] - single[j]).abs());
                }
            }
            let gap = common::top_gap(&want);
            if gap > 1e-3 {
                st.argmax_checked += 1;
                if common::argmax(got) == common::argmax(&want) {
                    st.argmax_same += 1;
                } else {
                    st.failures.push(format!("{id} seq {i}: argmax {}, python {} (gap {gap})", common::argmax(got), common::argmax(&want)));
                }
            }
            let key = s["key"].as_str().unwrap();
            let name = s["question"].as_str().unwrap();
            assert_eq!(questions[i].name, name, "{id} seq {i}");
            let answer = calibrate::answer(&temperatures, &questions[i].question, got, out.act_logits[i]);
            let wire = serde_json::from_str(&answer.to_wire_json(WireStyle::Batch)).unwrap();
            compare_wire(&wire, &rec["predict_many"][key][name], &mut st, &format!("{id}.{key}.{name}"));
            let act = common::f32s(&s["act_logits"]);
            for j in 0..act.len().min(2) {
                let rel = (out.act_logits[i][j] - act[j]).abs() / act[j].abs().max(1.0);
                st.max_act_rel = st.max_act_rel.max(rel);
            }
        }
    }
    println!(
        "{file} ({kernels:?}, {}): {} records, {} sequences in {:.1}s: logits max|Δ| {:.3e} (worst {}), vs single-state pass {:.3e}, act logit rel Δ {:.3e}, argmax {}/{} (where the Python top-two gap > 1e-3)",
        if every == 1 { "all".to_string() } else { format!("every {every}th") },
        st.records,
        st.seqs,
        start.elapsed().as_secs_f64(),
        st.max_logit,
        st.worst,
        st.max_logit_single,
        st.max_act_rel,
        st.argmax_same,
        st.argmax_checked,
    );
    println!("{file} ({kernels:?}): {}/{} rounded numeric fields identical", st.wire_exact, st.wire_fields);
    assert!(st.failures.is_empty(), "{}", st.failures.join("\n"));
    assert!(st.max_logit <= 1e-3, "logits differ by {:.3e} (> 1e-3) at {}", st.max_logit, st.worst);
    assert!(st.max_act_rel <= 1e-4, "act logits differ by {:.3e} relative", st.max_act_rel);
    (st.wire_exact, st.wire_fields)
}

fn parity(kernels: KernelSet, edge_every: usize, games_every: usize) {
    let all = std::env::var("LAYA_GOLDEN_ALL").is_ok_and(|v| v == "1");
    let edge = run(kernels, "edge.jsonl", edge_every, all);
    let games = run(kernels, "games.jsonl", games_every, all);
    let (exact, fields) = (edge.0 + games.0, edge.1 + games.1);
    // The percentage is a corpus-wide target. On the sampled suite, every numeric field
    // still has to meet the 1e-4 bound and every categorical field must match.
    if all && fields > 0 {
        assert!(exact * 1000 >= fields * 995, "{kernels:?}: only {exact}/{fields} rounded numeric fields match");
    }
}

#[test]
fn gemm() {
    parity(KernelSet::Gemm, 1, 8);
}

#[test]
fn portable() {
    parity(KernelSet::Portable, 4, 40);
}

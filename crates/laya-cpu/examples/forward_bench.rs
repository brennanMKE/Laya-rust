//! Times forward passes on game-sized batches taken from `tests/golden/games.jsonl`.
//!
//! ```sh
//! cargo run -p laya-cpu --release --example forward_bench -- [--kernels gemm|portable] [--threads N] [--runs N]
//! ```
//!
//! The checkpoint comes from `$LAYA_CHECKPOINT` or the Hugging Face cache.

use std::path::PathBuf;
use std::time::Instant;

use laya_core::engine::{Batch, Engine};
use laya_cpu::{CpuEngine, KernelSet, Options};

fn checkpoint() -> PathBuf {
    if let Some(d) = std::env::var_os("LAYA_CHECKPOINT") {
        return d.into();
    }
    PathBuf::from(std::env::var_os("HOME").expect("HOME"))
        .join(".cache/huggingface/hub/models--convaiinnovations--laya/snapshots/1c5edc17a7acd8701df6fc341c0d179f1c62c982")
}

/// The first `games.jsonl` record from `source` with `states` batch keys (any when 0).
fn batch(source: &str, states: usize) -> (Batch, usize) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/games.jsonl");
    let text = std::fs::read_to_string(path).expect("games.jsonl");
    for line in text.lines() {
        let rec: serde_json::Value = serde_json::from_str(line).unwrap();
        if rec["source"] != source || rec.get("error").is_some() {
            continue;
        }
        let seqs = rec["sequences"].as_array().unwrap();
        let mut keys: Vec<&str> = seqs.iter().map(|s| s["key"].as_str().unwrap()).collect();
        keys.dedup();
        if states != 0 && keys.len() != states {
            continue;
        }
        let mut b = Batch::default();
        for s in seqs {
            let ids: Vec<u32> = s["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            let markers = s["markers"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            b.push(&ids, markers, s["qtype"].as_u64().unwrap() as u8);
        }
        return (b, keys.len());
    }
    panic!("no {source} record with {states} states");
}

fn main() {
    let mut opts = Options::default();
    let mut runs = 10;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--kernels" => opts.kernels = args[i + 1].parse::<KernelSet>().unwrap(),
            "--threads" => opts.threads = args[i + 1].parse().unwrap(),
            "--runs" => runs = args[i + 1].parse().unwrap(),
            a => panic!("unknown argument {a}"),
        }
        i += 2;
    }
    let t0 = Instant::now();
    let engine = CpuEngine::load(&checkpoint(), opts).expect("load");
    println!("{}: loaded in {:.2}s", engine.name(), t0.elapsed().as_secs_f64());

    let mut single = Batch::default();
    let (f, _) = batch("frogger", 0);
    single.push(f.seq(0), f.markers[0].clone(), f.qtypes[0]);
    let mut cases = vec![("1 sequence".to_string(), single)];
    for (src, states) in [("frogger", 6), ("invaders", 0), ("tetris", 14)] {
        let (b, k) = batch(src, states);
        cases.push((format!("{src}, {k} states"), b));
    }
    for (name, b) in &cases {
        engine.forward(b).unwrap(); // warm up
        let mut ms: Vec<f64> = (0..runs)
            .map(|_| {
                let t = Instant::now();
                engine.forward(b).unwrap();
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        ms.sort_by(f64::total_cmp);
        println!(
            "{name:<20} {:>3} seqs {:>4} tokens: median {:7.1} ms  (min {:.1}, max {:.1}, {runs} runs)",
            b.len(),
            b.tokens.len(),
            ms[ms.len() / 2],
            ms[0],
            ms[ms.len() - 1]
        );
    }
}

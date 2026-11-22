//! Runs each trace sequence alone and compares every intermediate with PyTorch's.

mod common;

use std::collections::HashMap;

use laya_core::engine::Batch;
use laya_cpu::KernelSet;
use safetensors::{Dtype, SafeTensors};

fn run(kernels: KernelSet) {
    let Some(engine) = common::engine(kernels) else { return };
    let dir = common::golden_dir().join("trace");
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("index.json")).unwrap()).unwrap();
    let mut worst_logit = 0f32;
    let mut worst_act = 0f32;
    for entry in index.as_array().unwrap() {
        let file = entry["file"].as_str().unwrap();
        let bytes = std::fs::read(dir.join(file)).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let ids = common::u32s(&entry["ids"]);
        let markers = common::u32s(&entry["markers"]);
        let qtype = entry["qtype"].as_u64().unwrap() as u8;
        let mut batch = Batch::default();
        batch.push(&ids, markers, qtype);

        let mut got: HashMap<String, Vec<f32>> = HashMap::new();
        let mut order = Vec::new();
        let mut trace = |name: &str, v: &[f32]| {
            order.push(name.to_string());
            got.insert(name.to_string(), v.to_vec());
        };
        engine.forward_traced(&batch, Some(&mut trace)).unwrap();

        println!("{file} ({} tokens, {kernels:?}):", ids.len());
        for name in &order {
            let Ok(view) = st.tensor(name) else { continue };
            assert_eq!(view.dtype(), Dtype::F32);
            let want: Vec<f32> = view.data().as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
            let g = &got[name];
            assert_eq!(g.len(), want.len(), "{file} {name}: size");
            let mut worst = (0, 0.0f32);
            let mut ref_max = 0.0f32;
            for (i, (a, b)) in g.iter().zip(&want).enumerate() {
                let delta = (a - b).abs();
                if delta > worst.1 {
                    worst = (i, delta);
                }
                ref_max = ref_max.max(b.abs());
            }
            let (index, max_abs) = worst;
            println!(
                "  {name:<11} max|Δ| {max_abs:.3e} (max|ref| {ref_max:.3e}, flat index {index}, Rust {:e}, ref {:e})",
                g[index], want[index]
            );
            if !matches!(name.as_str(), "logits" | "act_logits") {
                // Same intermediate gate as laya-go. The logits have their own tighter limits.
                assert!(max_abs <= 5e-3 * ref_max.max(1.0), "{file} {name}: intermediate differs by {max_abs:.3e}");
            }
            match name.as_str() {
                "logits" => worst_logit = worst_logit.max(max_abs),
                "act_logits" => {
                    for (a, b) in g.iter().zip(&want) {
                        worst_act = worst_act.max((a - b).abs() / b.abs().max(1.0));
                    }
                }
                _ => {}
            }
        }
    }
    println!("{kernels:?}: worst logit max|Δ| {worst_logit:.3e}, act logit rel Δ {worst_act:.3e}");
    assert!(worst_logit <= 1e-3, "logits differ by {worst_logit}");
    assert!(worst_act <= 1e-4, "act logits differ by {worst_act} relative");
}

#[test]
fn trace_gemm() {
    run(KernelSet::Gemm);
}

#[test]
fn trace_portable() {
    run(KernelSet::Portable);
}

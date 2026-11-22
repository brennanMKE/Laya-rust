//! Loading the checkpoint and the forward pass (laya-go's `internal/modernbert/load.go` and
//! `model.go`): a ModernBERT encoder plus Laya's decision head (two norm-first transformer
//! layers, type embeddings, scorer and act head), run on packed, unpadded sequences.

use std::collections::HashMap;
use std::path::Path;

use half::{bf16, f16};
use laya_core::engine::{Batch, Output};
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

use crate::LoadError;
use crate::attention::{RopeTable, apply_rope, attention};
use crate::config::Config;
use crate::kernels::{Kernels, Weight};
use crate::ops;

/// The `nn.LayerNorm` eps of the head layers and the scorer (PyTorch's default).
const HEAD_EPS: f32 = 1e-5;
/// The act head's hidden width.
const ACT_HIDDEN: usize = 256;

struct EncLayer {
    attn_norm: Option<Vec<f32>>, // None for layer 0 (identity)
    wqkv: Weight,
    wo: Weight,
    mlp_norm: Vec<f32>,
    wi: Weight,
    wo2: Weight,
    global: bool,
}

struct HeadLayer {
    norm1_w: Vec<f32>,
    norm1_b: Vec<f32>,
    in_proj: Weight,
    in_bias: Vec<f32>,
    out_proj: Weight,
    out_bias: Vec<f32>,
    norm2_w: Vec<f32>,
    norm2_b: Vec<f32>,
    lin1: Weight,
    lin1_b: Vec<f32>,
    lin2: Weight,
    lin2_b: Vec<f32>,
}

/// The token table, kept in its stored precision; rows are widened as they are gathered.
enum Embeddings {
    F16(Vec<f16>),
    Bf16(Vec<bf16>),
    F32(Vec<f32>),
}

impl Embeddings {
    fn row(&self, tok: usize, dst: &mut [f32]) {
        let d = dst.len();
        let r = tok * d..tok * d + d;
        match self {
            Embeddings::F16(v) => dst.iter_mut().zip(&v[r]).for_each(|(o, h)| *o = h.to_f32()),
            Embeddings::Bf16(v) => dst.iter_mut().zip(&v[r]).for_each(|(o, h)| *o = h.to_f32()),
            Embeddings::F32(v) => dst.copy_from_slice(&v[r]),
        }
    }
}

/// The loaded network.
pub(crate) struct Model {
    pub cfg: Config,
    pub k: Box<dyn Kernels>,
    emb: Embeddings,
    emb_norm: Vec<f32>,
    layers: Vec<EncLayer>,
    final_norm: Vec<f32>,
    head: Vec<HeadLayer>,
    type_emb: Vec<f32>,
    sc_norm_w: Vec<f32>,
    sc_norm_b: Vec<f32>,
    sc1: Weight,
    sc1_b: Vec<f32>,
    sc3: Weight,
    sc3_b: Vec<f32>,
    act0: Weight,
    act0_b: Vec<f32>,
    act2: Weight,
    act2_b: Vec<f32>,
}

/// Activations of one forward pass, reused across calls.
#[derive(Default)]
pub(crate) struct Workspace {
    x: Vec<f32>,
    h: Vec<f32>,
    qkv: Vec<f32>,
    att: Vec<f32>,
    heads: Vec<f32>,
    tmp: Vec<f32>,
    u: Vec<f32>,
    g: Vec<f32>,
    pos: Vec<usize>,
    rope: Option<(RopeTable, RopeTable)>,
}

fn grow(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

/// Widens a tensor to f32 (in parallel for big tensors).
fn to_f32(t: &TensorView<'_>) -> Result<Vec<f32>, LoadError> {
    let b = t.data();
    let n: usize = t.shape().iter().product();
    let mut out = vec![0f32; n];
    const CH: usize = 1 << 16;
    match t.dtype() {
        Dtype::F16 => out.par_chunks_mut(CH).enumerate().for_each(|(c, o)| {
            let src = &b[c * CH * 2..];
            for (i, v) in o.iter_mut().enumerate() {
                *v = f16::from_le_bytes([src[2 * i], src[2 * i + 1]]).to_f32();
            }
        }),
        Dtype::BF16 => out.par_chunks_mut(CH).enumerate().for_each(|(c, o)| {
            let src = &b[c * CH * 2..];
            for (i, v) in o.iter_mut().enumerate() {
                *v = bf16::from_le_bytes([src[2 * i], src[2 * i + 1]]).to_f32();
            }
        }),
        Dtype::F32 => {
            for (v, c) in out.iter_mut().zip(b.as_chunks::<4>().0) {
                *v = f32::from_le_bytes(*c);
            }
        }
        Dtype::F64 => {
            for (v, c) in out.iter_mut().zip(b.as_chunks::<8>().0) {
                *v = f64::from_le_bytes(*c) as f32;
            }
        }
        dt => return Err(LoadError::Checkpoint(format!("unsupported dtype {dt:?}"))),
    }
    Ok(out)
}

impl Model {
    /// Reads `dir/model.safetensors` with `cfg`.
    pub fn load(dir: &Path, cfg: Config, k: Box<dyn Kernels>) -> Result<Model, LoadError> {
        if cfg.n_act != 2 {
            return Err(LoadError::Config(format!("act head with {} actions is not supported (want 2)", cfg.n_act)));
        }
        let file = std::fs::File::open(dir.join("model.safetensors"))?;
        // SAFETY: the checkpoint is read-only; we copy everything out before unmapping.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let st = SafeTensors::deserialize(&mmap).map_err(|e| LoadError::Checkpoint(format!("model.safetensors: {e}")))?;
        let (d, inter, ff) = (cfg.hidden, cfg.intermediate, cfg.head_ff);

        // Every tensor the model uses, with its shape; matches load_state_dict's strict check
        // (plus the unused temperature buffer).
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
        let mut want = |name: String, shape: &[usize]| {
            shapes.insert(name, shape.to_vec());
        };
        want("encoder.embeddings.tok_embeddings.weight".into(), &[cfg.vocab, d]);
        want("encoder.embeddings.norm.weight".into(), &[d]);
        want("encoder.final_norm.weight".into(), &[d]);
        for i in 0..cfg.layers {
            let p = format!("encoder.layers.{i}.");
            if i > 0 {
                want(p.clone() + "attn_norm.weight", &[d]);
            }
            want(p.clone() + "attn.Wqkv.weight", &[3 * d, d]);
            want(p.clone() + "attn.Wo.weight", &[d, d]);
            want(p.clone() + "mlp_norm.weight", &[d]);
            want(p.clone() + "mlp.Wi.weight", &[2 * inter, d]);
            want(p + "mlp.Wo.weight", &[d, inter]);
        }
        for i in 0..cfg.head_layers {
            let p = format!("head.layers.{i}.");
            for (n, s) in [
                ("norm1.weight", vec![d]),
                ("norm1.bias", vec![d]),
                ("self_attn.in_proj_weight", vec![3 * d, d]),
                ("self_attn.in_proj_bias", vec![3 * d]),
                ("self_attn.out_proj.weight", vec![d, d]),
                ("self_attn.out_proj.bias", vec![d]),
                ("norm2.weight", vec![d]),
                ("norm2.bias", vec![d]),
                ("linear1.weight", vec![ff, d]),
                ("linear1.bias", vec![ff]),
                ("linear2.weight", vec![d, ff]),
                ("linear2.bias", vec![d]),
            ] {
                want(p.clone() + n, &s);
            }
        }
        want("type_emb.weight".into(), &[3, d]);
        want("scorer.0.weight".into(), &[d]);
        want("scorer.0.bias".into(), &[d]);
        want("scorer.1.weight".into(), &[d, d]);
        want("scorer.1.bias".into(), &[d]);
        want("scorer.3.weight".into(), &[1, d]);
        want("scorer.3.bias".into(), &[1]);
        want("act_head.0.weight".into(), &[ACT_HIDDEN, d + 4]);
        want("act_head.0.bias".into(), &[ACT_HIDDEN]);
        want("act_head.2.weight".into(), &[2, ACT_HIDDEN]);
        want("act_head.2.bias".into(), &[2]);

        let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
        for (name, view) in st.iter() {
            match shapes.get(name) {
                Some(s) if s.as_slice() != view.shape() => {
                    return Err(LoadError::Checkpoint(format!("incompatible checkpoint: {name} has shape {:?}, want {s:?}", view.shape())));
                }
                Some(_) => {
                    if !matches!(view.dtype(), Dtype::F16 | Dtype::BF16 | Dtype::F32 | Dtype::F64) {
                        return Err(LoadError::Checkpoint(format!("{name} has unsupported dtype {:?}", view.dtype())));
                    }
                    views.insert(name.to_string(), view);
                }
                None if name == "temperature" => {}
                None => {
                    return Err(LoadError::Checkpoint(format!("incompatible checkpoint: unexpected tensor {name:?}")));
                }
            }
        }
        if let Some(missing) = shapes.keys().find(|n| !views.contains_key(*n)) {
            return Err(LoadError::Checkpoint(format!("incompatible checkpoint: missing {missing:?}")));
        }

        let vec = |name: &str| to_f32(&views[name]);
        let mat = |name: &str| -> Result<Weight, LoadError> {
            let v = &views[name];
            let (r, c) = (v.shape()[0], v.shape()[1]);
            Ok(k.new_weight(r, c, to_f32(v)?))
        };

        let et = &views["encoder.embeddings.tok_embeddings.weight"];
        let raw = et.data();
        let emb = match et.dtype() {
            Dtype::F16 => Embeddings::F16(raw.as_chunks::<2>().0.iter().map(|c| f16::from_le_bytes(*c)).collect()),
            Dtype::BF16 => Embeddings::Bf16(raw.as_chunks::<2>().0.iter().map(|c| bf16::from_le_bytes(*c)).collect()),
            _ => Embeddings::F32(to_f32(et)?),
        };

        // Encoder layers load in parallel; each tensor's widening is parallel too.
        let layers = (0..cfg.layers)
            .into_par_iter()
            .map(|i| -> Result<EncLayer, LoadError> {
                let p = format!("encoder.layers.{i}.");
                Ok(EncLayer {
                    attn_norm: if i > 0 { Some(vec(&(p.clone() + "attn_norm.weight"))?) } else { None },
                    wqkv: mat(&(p.clone() + "attn.Wqkv.weight"))?,
                    wo: mat(&(p.clone() + "attn.Wo.weight"))?,
                    mlp_norm: vec(&(p.clone() + "mlp_norm.weight"))?,
                    wi: mat(&(p.clone() + "mlp.Wi.weight"))?,
                    wo2: mat(&(p + "mlp.Wo.weight"))?,
                    global: cfg.global[i],
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let head = (0..cfg.head_layers)
            .map(|i| -> Result<HeadLayer, LoadError> {
                let p = |n: &str| format!("head.layers.{i}.{n}");
                Ok(HeadLayer {
                    norm1_w: vec(&p("norm1.weight"))?,
                    norm1_b: vec(&p("norm1.bias"))?,
                    in_proj: mat(&p("self_attn.in_proj_weight"))?,
                    in_bias: vec(&p("self_attn.in_proj_bias"))?,
                    out_proj: mat(&p("self_attn.out_proj.weight"))?,
                    out_bias: vec(&p("self_attn.out_proj.bias"))?,
                    norm2_w: vec(&p("norm2.weight"))?,
                    norm2_b: vec(&p("norm2.bias"))?,
                    lin1: mat(&p("linear1.weight"))?,
                    lin1_b: vec(&p("linear1.bias"))?,
                    lin2: mat(&p("linear2.weight"))?,
                    lin2_b: vec(&p("linear2.bias"))?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Model {
            emb,
            emb_norm: vec("encoder.embeddings.norm.weight")?,
            final_norm: vec("encoder.final_norm.weight")?,
            layers,
            head,
            type_emb: vec("type_emb.weight")?,
            sc_norm_w: vec("scorer.0.weight")?,
            sc_norm_b: vec("scorer.0.bias")?,
            sc1: mat("scorer.1.weight")?,
            sc1_b: vec("scorer.1.bias")?,
            sc3: mat("scorer.3.weight")?,
            sc3_b: vec("scorer.3.bias")?,
            act0: mat("act_head.0.weight")?,
            act0_b: vec("act_head.0.bias")?,
            act2: mat("act_head.2.weight")?,
            act2_b: vec("act_head.2.bias")?,
            cfg,
            k,
        })
    }

    /// Runs the network over a validated batch. `trace`, if set, receives named intermediates
    /// (see [`crate::TraceFn`]).
    pub fn forward(&self, ws: &mut Workspace, b: &Batch, mut trace: Option<&mut crate::TraceFn<'_>>) -> Output {
        let c = &self.cfg;
        let d = c.hidden;
        let t = b.tokens.len();
        let n = b.len();
        let starts = &b.starts;
        let wide = (2 * c.intermediate).max(c.head_ff);
        grow(&mut ws.x, t * d);
        grow(&mut ws.h, t * d);
        grow(&mut ws.qkv, t * 3 * d);
        grow(&mut ws.att, t * d);
        grow(&mut ws.heads, t * d);
        grow(&mut ws.tmp, t * d);
        grow(&mut ws.u, t * wide);
        grow(&mut ws.g, t * c.intermediate);
        ws.pos.clear();
        let mut max_l = 0;
        for i in 0..n {
            let l = starts[i + 1] - starts[i];
            max_l = max_l.max(l);
            ws.pos.extend(0..l);
        }
        let need = max_l.max(c.max_len);
        if ws.rope.as_ref().is_none_or(|(g, _)| g.n < need) {
            let dim = c.head_dim();
            ws.rope = Some((RopeTable::new(c.global_theta, dim, need), RopeTable::new(c.local_theta, dim, need)));
        }
        let mut emit = |name: &str, v: &[f32]| {
            if let Some(tr) = trace.as_mut() {
                tr(name, v);
            }
        };

        // Embeddings: gather, then LayerNorm without bias.
        {
            let x = &mut ws.x[..t * d];
            x.par_chunks_exact_mut(d).zip(b.tokens.par_iter()).with_min_len(32).for_each(|(row, &tok)| self.emb.row(tok as usize, row));
            ops::layer_norm(x, d, &self.emb_norm, None, c.norm_eps);
            emit("emb", x);
        }

        for (li, l) in self.layers.iter().enumerate() {
            self.encoder_layer(l, ws, starts, t);
            emit(&format!("layer.{li}"), &ws.x[..t * d]);
        }
        let x = &mut ws.x[..t * d];
        ops::layer_norm(x, d, &self.final_norm, None, c.norm_eps);
        emit("final_norm", x);

        // Question type embedding on every token.
        for i in 0..n {
            let q = b.qtypes[i] as usize;
            let te = &self.type_emb[q * d..q * d + d];
            for row in x[starts[i] * d..starts[i + 1] * d].chunks_exact_mut(d) {
                for (v, e) in row.iter_mut().zip(te) {
                    *v += e;
                }
            }
        }
        emit("head_in", x);

        for (hi, h) in self.head.iter().enumerate() {
            self.head_layer(h, ws, starts, t);
            emit(&format!("head.{hi}"), &ws.x[..t * d]);
        }

        let out = self.score_and_act(&ws.x[..t * d], b);
        let flat: Vec<f32> = out.logits.iter().flatten().copied().collect();
        emit("logits", &flat);
        let act: Vec<f32> = out.act_logits.iter().flatten().copied().collect();
        emit("act_logits", &act);
        out
    }

    fn encoder_layer(&self, l: &EncLayer, ws: &mut Workspace, starts: &[usize], t: usize) {
        let c = &self.cfg;
        let (d, inter) = (c.hidden, c.intermediate);
        let k = self.k.as_ref();
        let x = &mut ws.x[..t * d];
        let h = &mut ws.h[..t * d];
        let qkv = &mut ws.qkv[..t * 3 * d];
        let src: &[f32] = match &l.attn_norm {
            Some(w) => {
                ops::layer_norm_into(h, x, d, w, None, c.norm_eps);
                h
            }
            None => x,
        };
        k.matmul_t(qkv, src, &l.wqkv, t, None);
        let (rg, rl) = ws.rope.as_ref().unwrap();
        let (rt, window) = if l.global { (rg, None) } else { (rl, Some(c.window)) };
        apply_rope(qkv, &ws.pos, rt, d);
        let att = &mut ws.att[..t * d];
        attention(k, att, &mut ws.heads, qkv, starts, d, c.heads, window);
        // Residual added after the product, never accumulated inside the GEMM.
        let tmp = &mut ws.tmp[..t * d];
        k.matmul_t(tmp, att, &l.wo, t, None);
        ops::add_in_place(x, tmp);

        ops::layer_norm_into(h, x, d, &l.mlp_norm, None, c.norm_eps);
        let u = &mut ws.u[..t * 2 * inter];
        k.matmul_t(u, h, &l.wi, t, None);
        let g = &mut ws.g[..t * inter];
        ops::geglu(g, u, inter);
        k.matmul_t(tmp, g, &l.wo2, t, None);
        ops::add_in_place(x, tmp);
    }

    fn head_layer(&self, hl: &HeadLayer, ws: &mut Workspace, starts: &[usize], t: usize) {
        let c = &self.cfg;
        let d = c.hidden;
        let k = self.k.as_ref();
        let x = &mut ws.x[..t * d];
        let h = &mut ws.h[..t * d];
        let qkv = &mut ws.qkv[..t * 3 * d];
        let tmp = &mut ws.tmp[..t * d];
        ops::layer_norm_into(h, x, d, &hl.norm1_w, Some(&hl.norm1_b), HEAD_EPS);
        k.matmul_t(qkv, h, &hl.in_proj, t, Some(&hl.in_bias));
        let att = &mut ws.att[..t * d];
        attention(k, att, &mut ws.heads, qkv, starts, d, c.head_heads, None);
        k.matmul_t(tmp, att, &hl.out_proj, t, Some(&hl.out_bias));
        ops::add_in_place(x, tmp);

        ops::layer_norm_into(h, x, d, &hl.norm2_w, Some(&hl.norm2_b), HEAD_EPS);
        let u = &mut ws.u[..t * c.head_ff];
        k.matmul_t(u, h, &hl.lin1, t, Some(&hl.lin1_b));
        ops::relu_in_place(u);
        k.matmul_t(tmp, u, &hl.lin2, t, Some(&hl.lin2_b));
        ops::add_in_place(x, tmp);
    }

    /// Gathers the marker rows, runs the scorer, then the act head.
    fn score_and_act(&self, x: &[f32], b: &Batch) -> Output {
        let d = self.cfg.hidden;
        let n = b.len();
        let k = self.k.as_ref();
        let m: usize = b.markers.iter().map(Vec::len).sum();
        let mut sel = Vec::with_capacity(m * d);
        for i in 0..n {
            for &p in &b.markers[i] {
                let r = b.starts[i] + p as usize;
                sel.extend_from_slice(&x[r * d..r * d + d]);
            }
        }
        // scorer: LayerNorm → Linear(d) → GELU → Linear(1)
        ops::layer_norm(&mut sel, d, &self.sc_norm_w, Some(&self.sc_norm_b), HEAD_EPS);
        let mut s1 = vec![0f32; m * d];
        k.matmul_t(&mut s1, &sel, &self.sc1, m, Some(&self.sc1_b));
        ops::gelu_in_place(&mut s1);
        let mut flat = vec![0f32; m];
        k.matmul_t(&mut flat, &s1, &self.sc3, m, Some(&self.sc3_b));
        let mut logits = Vec::with_capacity(n);
        let mut r = 0;
        for mk in &b.markers {
            logits.push(flat[r..r + mk.len()].to_vec());
            r += mk.len();
        }

        // act head over [h[CLS], top1, top1 − top2, entropy / log k, k / 255], with
        // probabilities from an untempered softmax of the logits.
        let mut input = vec![0f32; n * (d + 4)];
        for (i, row) in input.chunks_exact_mut(d + 4).enumerate() {
            let s = b.starts[i];
            row[..d].copy_from_slice(&x[s * d..s * d + d]);
            row[d..].copy_from_slice(&act_features(&logits[i]));
        }
        let mut a1 = vec![0f32; n * ACT_HIDDEN];
        k.matmul_t(&mut a1, &input, &self.act0, n, Some(&self.act0_b));
        ops::gelu_in_place(&mut a1);
        let mut a2 = vec![0f32; n * 2];
        k.matmul_t(&mut a2, &a1, &self.act2, n, Some(&self.act2_b));
        let act_logits = a2.as_chunks::<2>().0.to_vec();
        Output { logits, act_logits }
    }
}

/// The act head's four extra inputs, the way `DecisionModel.forward` does in f32
/// (laya-go's `actFeatures`).
fn act_features(logits: &[f32]) -> [f32; 4] {
    let k = logits.len();
    let mx = logits.iter().copied().fold(logits[0], f32::max);
    let mut p: Vec<f64> = logits.iter().map(|&v| ((v - mx) as f64).exp()).collect();
    let sum: f64 = p.iter().sum();
    let (mut top1, mut top2, mut ent) = (0f64, 0f64, 0f64);
    for pi in &mut p {
        *pi /= sum;
        let pf = *pi as f32 as f64;
        if pf > top1 {
            (top1, top2) = (pf, top1);
        } else if pf > top2 {
            top2 = pf;
        }
        ent -= pf * pf.max(1e-9).ln();
    }
    let kk = k.max(2) as f64;
    [top1 as f32, (top1 - top2) as f32, (ent / kk.ln()) as f32, (kk / 255.0) as f32]
}

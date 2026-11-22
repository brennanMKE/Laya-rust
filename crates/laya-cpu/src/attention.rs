//! RoPE and per-sequence attention (laya-go's `internal/modernbert/attention.go`).

use rayon::prelude::*;

use crate::kernels::Kernels;
use crate::ops::softmax_masked;

/// cos and sin for positions `0..n`, `dim` wide, as `cat(freqs, freqs)`, computed the way
/// transformers does in f32.
#[derive(Debug, Clone)]
pub(crate) struct RopeTable {
    pub n: usize,
    dim: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl RopeTable {
    pub fn new(theta: f64, dim: usize, n: usize) -> Self {
        let half = dim / 2;
        // inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim)), in f32
        let inv: Vec<f32> = (0..half)
            .map(|j| {
                let e = (2 * j) as f32 / dim as f32;
                1.0 / theta.powf(e as f64) as f32
            })
            .collect();
        let mut cos = vec![0f32; n * dim];
        let mut sin = vec![0f32; n * dim];
        for p in 0..n {
            for j in 0..half {
                let f = p as f32 * inv[j];
                let (c, s) = ((f as f64).cos() as f32, (f as f64).sin() as f32);
                cos[p * dim + j] = c;
                cos[p * dim + j + half] = c;
                sin[p * dim + j] = s;
                sin[p * dim + j + half] = s;
            }
        }
        RopeTable { n, dim, cos, sin }
    }
}

/// Rotates q and k in place ("rotate half" form) for every token and head. `qkv` rows are
/// `[3, heads, dim]`: q, then k, then v.
pub(crate) fn apply_rope(qkv: &mut [f32], pos: &[usize], rt: &RopeTable, d: usize) {
    let dim = rt.dim;
    let half = dim / 2;
    qkv.par_chunks_exact_mut(3 * d).zip(pos.par_iter()).with_min_len(4).for_each(|(row, &p)| {
        let cs = &rt.cos[p * dim..p * dim + dim];
        let sn = &rt.sin[p * dim..p * dim + dim];
        let (c1, c2) = cs.split_at(half);
        let (s1, s2) = sn.split_at(half);
        for v in row[..2 * d].chunks_exact_mut(dim) {
            let (a, b) = v.split_at_mut(half);
            for j in 0..half {
                let (x1, x2) = (a[j], b[j]);
                a[j] = x1 * c1[j] - x2 * s1[j];
                b[j] = x2 * c2[j] + x1 * s2[j];
            }
        }
    });
}

/// Queries per attention task.
const QBLOCK: usize = 64;

/// `softmax(q·kᵀ/√dim)·v` per sequence and head. `window` `None` is full bidirectional
/// attention; `Some(w)` lets query `i` see key `j` only when `|i − j| ≤ w`. `out` rows are
/// `[heads, dim]`; `scratch` (at least `T·d`) holds the head-major result before it is
/// interleaved into `out`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention(
    k: &dyn Kernels,
    out: &mut [f32],
    scratch: &mut [f32],
    qkv: &[f32],
    starts: &[usize],
    d: usize,
    heads: usize,
    window: Option<usize>,
) {
    let dim = d / heads;
    let t_total = *starts.last().unwrap();
    let scale = (1.0 / (dim as f64).sqrt()) as f32;
    let ld = 3 * d;

    // Tasks are (head, sequence, block of queries), each owning a contiguous block of the
    // head-major scratch, so one long sequence still spreads across cores.
    let mut tasks: Vec<(usize, usize, usize, usize, &mut [f32])> = Vec::new();
    let mut rest = &mut scratch[..heads * t_total * dim];
    for h in 0..heads {
        for s in 0..starts.len() - 1 {
            let l = starts[s + 1] - starts[s];
            let mut q0 = 0;
            while q0 < l {
                let q1 = (q0 + QBLOCK).min(l);
                let (blk, r) = std::mem::take(&mut rest).split_at_mut((q1 - q0) * dim);
                rest = r;
                tasks.push((h, s, q0, q1, blk));
                q0 = q1;
            }
        }
    }
    tasks.into_par_iter().for_each(|(h, s, q0, q1, blk)| {
        let s0 = starts[s];
        let l = starts[s + 1] - s0;
        let (j0, j1) = match window {
            Some(w) => (q0.saturating_sub(w), (q1 + w).min(l)),
            None => (0, l),
        };
        let (nq, nk) = (q1 - q0, j1 - j0);
        let mut sc = vec![0f32; nq * nk];
        let hq = (s0 + q0) * ld + h * dim;
        let hk = (s0 + j0) * ld + d + h * dim;
        let hv = (s0 + j0) * ld + 2 * d + h * dim;
        k.gemm(true, nq, nk, dim, scale, &qkv[hq..], ld, &qkv[hk..], ld, &mut sc, nk);
        for (qi, row) in sc.chunks_exact_mut(nk).enumerate() {
            let (a, b) = match window {
                Some(w) => {
                    let i = q0 + qi;
                    (i.saturating_sub(w) - j0, (i + w + 1).min(l) - j0)
                }
                None => (0, nk),
            };
            softmax_masked(row, a, b);
        }
        k.gemm(false, nq, dim, nk, 1.0, &sc, nk, &qkv[hv..], ld, blk, dim);
    });

    out[..t_total * d].par_chunks_exact_mut(d).enumerate().with_min_len(8).for_each(|(t, row)| {
        for (h, o) in row.chunks_exact_mut(dim).enumerate() {
            let base = (h * t_total + t) * dim;
            o.copy_from_slice(&scratch[base..base + dim]);
        }
    });
}

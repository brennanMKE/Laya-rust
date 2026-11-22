//! Raw logits to calibrated probabilities: the temperature lookup and clamp, the softmax, and
//! the confidence and expected-score formulas of laya 0.3.6's `Agent.system_one` (laya-go's
//! `internal/calibrate`).
//!
//! The arithmetic follows numpy's float32 semantics step by step, because that is what the
//! Python reference computes: the logits are float32, a Python float temperature is cast to
//! float32 before dividing (NEP 50), `exp` and `log` are float32, and sums use numpy's pairwise
//! summation order. The answers are rounded to 4 decimals, so matching the float32 steps is
//! what makes the rounded answers agree.
//!
//! `exp` and `log` are `f32::exp` and `f32::ln`, which call the platform's `expf` and `logf`
//! just as numpy does. On macOS that reproduces the fixtures bit for bit; other platforms'
//! libm may differ by an ulp in rare cases.

use std::collections::BTreeMap;

use crate::question::{Answer, Question};

/// Converts raw scorer and act logits to a calibrated answer.
pub fn answer(temperatures: &Temperatures, question: &Question, logits: &[f32], act_logits: [f32; 2]) -> Answer {
    let probabilities = softmax(logits, temperatures.for_question(question.qtype as u8, logits.len()));
    Answer::new(question, &probabilities, act_probability(act_logits))
}

/// Question types, as the checkpoint numbers them.
pub const CHOICE: u8 = 0;
pub const SCORE: u8 = 1;
pub const NOUL: u8 = 2;

const TYPE_NAMES: [&str; 3] = ["choice", "score", "noul"];

/// Temperature bounds: Laya refuses fitted temperatures that sharpen logits hard (the
/// checkpoint's `choice:11+` bucket is 0.1006) or flatten them.
pub const TEMP_MIN: f64 = 0.5;
pub const TEMP_MAX: f64 = 5.0;

/// `clamp_temperature`: `t` confined to [0.5, 5.0], or 1.0 if `t` is NaN or infinite.
pub fn clamp(t: f64) -> f64 {
    if !t.is_finite() { 1.0 } else { t.clamp(TEMP_MIN, TEMP_MAX) }
}

/// `temp_bucket`: `"<type>:<size>"` with size 2, 3-5, 6-10 or 11+ by the number of options.
pub fn bucket(qtype: u8, k: usize) -> String {
    let size = match k {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    format!("{}:{size}", TYPE_NAMES[qtype as usize])
}

/// The clamped temperatures Laya applies.
#[derive(Debug, Clone, PartialEq)]
pub struct Temperatures {
    /// Per question type.
    pub by_type: [f64; 3],
    /// Per bucket; takes precedence over `by_type`.
    pub by_bucket: BTreeMap<String, f64>,
}

impl Temperatures {
    /// Clamps the checkpoint's fitted temperatures and returns them with a description of each
    /// value the clamp changed (Python warns about these), such as `"choice:11+=0.1006"`.
    pub fn new(by_type: &[f64], by_bucket: &BTreeMap<String, f64>) -> (Self, Vec<String>) {
        let mut changed = Vec::new();
        let mut t = [1.0; 3];
        for (i, slot) in t.iter_mut().enumerate() {
            let v = by_type.get(i).copied().unwrap_or(1.0);
            *slot = clamp(v);
            if *slot != v {
                changed.push(format!("temperature[{i}]={}", fmt_g4(v)));
            }
        }
        let mut b = BTreeMap::new();
        for (k, &v) in by_bucket {
            let c = clamp(v);
            if c != v {
                changed.push(format!("{k}={}", fmt_g4(v)));
            }
            b.insert(k.clone(), c);
        }
        (Temperatures { by_type: t, by_bucket: b }, changed)
    }

    /// The temperature for a question of type `qtype` with `k` options.
    pub fn for_question(&self, qtype: u8, k: usize) -> f64 {
        self.by_bucket.get(&bucket(qtype, k)).copied().unwrap_or(self.by_type[qtype as usize])
    }
}

/// Go's `%.4g` for the values the clamp reports.
fn fmt_g4(v: f64) -> String {
    if !v.is_finite() {
        return format!("{v}");
    }
    if v == 0.0 {
        return "0".into();
    }
    let exp = v.abs().log10().floor() as i32;
    if !(-4..4).contains(&exp) {
        let s = format!("{v:.3e}");
        let (m, e) = s.split_once('e').unwrap();
        let m = m.trim_end_matches('0').trim_end_matches('.');
        let e: i32 = e.parse().unwrap();
        return format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
    }
    let decimals = (3 - exp).max(0) as usize;
    let s = format!("{v:.decimals$}");
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

/// `softmax(logits / t)` in float32, as
/// `z = logits[:k] / t; p = np.exp(z - z.max()); p = p / p.sum()`.
pub fn softmax(logits: &[f32], temp: f64) -> Vec<f32> {
    let t = temp as f32;
    let mut z: Vec<f32> = logits.iter().map(|&l| l / t).collect();
    let zmax = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in &mut z {
        *v = (*v - zmax).exp();
    }
    let s = sum32(&z);
    for v in &mut z {
        *v /= s;
    }
    z
}

/// `np.argmax`: the first index of the largest value.
pub fn argmax(p: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in p.iter().enumerate() {
        if v > p[best] {
            best = i;
        }
    }
    best
}

/// `confidence_from_probs`: 1 − H(p)/log k clipped to [0, 1], and 1 when k < 2.
pub fn confidence(p: &[f32]) -> f64 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let terms: Vec<f32> = p.iter().map(|&v| v * v.clamp(1e-12, 1.0).ln()).collect();
    let ent = -sum32(&terms);
    let c = 1.0 - ent / ((k as f64).ln() as f32);
    c.clamp(0.0, 1.0) as f64
}

/// Σ i·pᵢ as numpy computes `(np.arange(k) * p).sum()`: float64 products, float64 pairwise sum.
pub fn expected_score(p: &[f32]) -> f64 {
    let terms: Vec<f64> = p.iter().enumerate().map(|(i, &v)| i as f64 * v as f64).collect();
    sum64(&terms)
}

/// P(true) and the noul confidence max(p1, 1 − p1) from a noul question's two probabilities.
pub fn noul_answer(p: &[f32]) -> (f64, f64) {
    let p1 = p[1] as f64;
    (p1, p1.max(1.0 - p1))
}

/// `torch.softmax(act_logits)[0]` for the act head's two logits.
pub fn act_probability(act: [f32; 2]) -> f64 {
    let m = act[0].max(act[1]);
    let e0 = (act[0] - m).exp();
    let e1 = (act[1] - m).exp();
    (e0 / (e0 + e1)) as f64
}

/// Python's `round(x, 4)`: the exact binary value rounded half-to-even at 4 decimals, then
/// the nearest float. Rust's fixed-precision formatting rounds the exact value the same way.
pub fn round4(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let r: f64 = format!("{x:.4}").parse().unwrap();
    if r == 0.0 { 0.0f64.copysign(x) } else { r } // round(-0.00001, 4) is -0.0
}

/// numpy's `PW_BLOCKSIZE`.
const PW_BLOCK: usize = 128;

macro_rules! pairwise {
    ($name:ident, $t:ty) => {
        /// numpy's `pairwise_sum` (numpy/_core/src/umath/loops_utils.h.src).
        fn $name(a: &[$t]) -> $t {
            let n = a.len();
            if n < 8 {
                let mut res: $t = 0.0;
                for &v in a {
                    res += v;
                }
                res
            } else if n <= PW_BLOCK {
                let mut r: [$t; 8] = a[..8].try_into().unwrap();
                let mut i = 8;
                while i < n - n % 8 {
                    for j in 0..8 {
                        r[j] += a[i + j];
                    }
                    i += 8;
                }
                let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
                while i < n {
                    res += a[i];
                    i += 1;
                }
                res
            } else {
                let mut n2 = n / 2;
                n2 -= n2 % 8;
                $name(&a[..n2]) + $name(&a[n2..])
            }
        }
    };
}

pairwise!(pairwise32, f32);
pairwise!(pairwise64, f64);

/// `np.sum` over a contiguous float32 array.
pub fn sum32(a: &[f32]) -> f32 {
    0.0 + pairwise32(a)
}

/// `np.sum` over a contiguous float64 array.
pub fn sum64(a: &[f64]) -> f64 {
    0.0 + pairwise64(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_and_buckets() {
        let by_bucket: BTreeMap<String, f64> = [
            ("choice:2".to_string(), 1.9063563346862793),
            ("choice:11+".to_string(), 0.10058280825614929),
            ("noul:2".to_string(), 1.983399510383606),
        ]
        .into();
        let (t, changed) = Temperatures::new(&[1.6369030475616455, 1.2514300346374512, 1.983399510383606], &by_bucket);
        assert_eq!(changed, vec!["choice:11+=0.1006"]);
        assert_eq!(t.for_question(CHOICE, 1), 1.9063563346862793);
        assert_eq!(t.for_question(CHOICE, 4), 1.6369030475616455);
        assert_eq!(t.for_question(CHOICE, 11), 0.5);
        assert_eq!(t.for_question(SCORE, 5), 1.2514300346374512);
        assert_eq!(clamp(f64::NAN), 1.0);
        assert_eq!(clamp(9.0), 5.0);
    }

    #[test]
    fn pairwise_order() {
        let a: Vec<f32> = (0..19).map(|i| 1.0 / (i + 3) as f32).collect();
        let mut r: [f32; 8] = a[..8].try_into().unwrap();
        for j in 0..8 {
            r[j] += a[8 + j];
        }
        let mut want = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        want += a[16];
        want += a[17];
        want += a[18];
        assert_eq!(sum32(&a), want);
    }

    #[test]
    fn softmax_confidence() {
        let p = softmax(&[0.0; 4], 1.7601518630981445);
        assert!(p.iter().all(|&v| v == 0.25));
        assert_eq!(confidence(&p), 0.0);
        assert_eq!(confidence(&[1.0]), 1.0);
        assert_eq!(expected_score(&[0.0, 0.5, 0.5]), 1.5);
        assert_eq!(argmax(&[0.2, 0.4, 0.4]), 1);
    }

    #[test]
    fn rounding() {
        assert_eq!(round4(0.12345), 0.1235); // 0.12345 is just above the tie in binary
        assert_eq!(round4(2.675e-1), 0.2675);
        assert!(round4(-0.00001).is_sign_negative());
        assert_eq!(round4(0.00005), 0.0001); // 5e-05 is just above the tie
    }
}

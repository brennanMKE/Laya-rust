//! Element-wise operations (laya-go's `internal/kernels/ops.go` and `erf.go`), parallel over
//! rows with rayon and written so LLVM can vectorize the inner loops.

use rayon::prelude::*;

/// LayerNorm in place over `rows` rows of width `d`: `(x − mean) / sqrt(var + eps) · w (+ b)`.
/// Statistics are accumulated in f64, as laya-go does.
pub fn layer_norm(x: &mut [f32], d: usize, w: &[f32], b: Option<&[f32]>, eps: f32) {
    x.par_chunks_exact_mut(d).with_min_len(8).for_each(|row| layer_norm_row(row, w, b, eps));
}

/// [`layer_norm`] from `src` into `dst`.
pub fn layer_norm_into(dst: &mut [f32], src: &[f32], d: usize, w: &[f32], b: Option<&[f32]>, eps: f32) {
    dst.par_chunks_exact_mut(d).zip(src.par_chunks_exact(d)).with_min_len(8).for_each(|(o, i)| {
        o.copy_from_slice(i);
        layer_norm_row(o, w, b, eps);
    });
}

fn layer_norm_row(x: &mut [f32], w: &[f32], b: Option<&[f32]>, eps: f32) {
    let n = x.len();
    let w = &w[..n];
    let mut s = [0f64; 4];
    let (chunks, rest) = x.as_chunks::<4>();
    for c in chunks {
        for l in 0..4 {
            s[l] += c[l] as f64;
        }
    }
    let mut sum = (s[0] + s[1]) + (s[2] + s[3]);
    for &v in rest {
        sum += v as f64;
    }
    let mean = sum / n as f64;
    let mut q = [0f64; 4];
    for c in chunks {
        for l in 0..4 {
            let dv = c[l] as f64 - mean;
            q[l] += dv * dv;
        }
    }
    let mut ss = (q[0] + q[1]) + (q[2] + q[3]);
    for &v in rest {
        let dv = v as f64 - mean;
        ss += dv * dv;
    }
    let rstd = 1.0 / (ss / n as f64 + eps as f64).sqrt();
    match b {
        None => {
            for (v, &wv) in x.iter_mut().zip(w) {
                *v = ((*v as f64 - mean) * rstd) as f32 * wv;
            }
        }
        Some(b) => {
            let b = &b[..n];
            for ((v, &wv), &bv) in x.iter_mut().zip(w).zip(b) {
                *v = ((*v as f64 - mean) * rstd) as f32 * wv + bv;
            }
        }
    }
}

/// A fast f32 error function: the rational minimax approximation Eigen and XLA use, on
/// `[-4, 4]`, evaluated in f64. Absolute error below 1e-7 (laya-go's `Erf32`).
#[inline]
pub fn erf32(xf: f32) -> f32 {
    if xf.is_nan() {
        return xf;
    }
    let x = (xf as f64).clamp(-4.0, 4.0);
    let x2 = x * x;
    let mut p = x2 * -2.72614225801306e-10 + 2.77068142495902e-08;
    p = x2 * p + -2.10102402082508e-06;
    p = x2 * p + -5.69250639462346e-05;
    p = x2 * p + -7.34990630326855e-04;
    p = x2 * p + -2.95459980854025e-03;
    p = x2 * p + -1.60960333262415e-02;
    p *= x;
    let mut q = x2 * -1.45660718464996e-05 + -2.13374055278905e-04;
    q = x2 * q + -1.68282697438203e-03;
    q = x2 * q + -7.37332916720468e-03;
    q = x2 * q + -1.42647390514189e-02;
    (p / q) as f32
}

const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Exact (erf) GELU, PyTorch's default: `0.5·x·(1 + erf(x/√2))`.
#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf32(x * INV_SQRT2))
}

/// GELU on every element, in place.
pub fn gelu_in_place(x: &mut [f32]) {
    x.par_chunks_mut(4096).for_each(|s| {
        for v in s {
            *v = gelu(*v);
        }
    });
}

/// `out[r, j] = GELU(u[r, j]) · u[r, inter + j]` for `u` of shape `[rows, 2·inter]`.
pub fn geglu(out: &mut [f32], u: &[f32], inter: usize) {
    out.par_chunks_exact_mut(inter).zip(u.par_chunks_exact(2 * inter)).with_min_len(2).for_each(|(o, row)| {
        let (input, gate) = row.split_at(inter);
        for ((o, &x), &g) in o.iter_mut().zip(input).zip(gate) {
            *o = gelu(x) * g;
        }
    });
}

/// `max(0, x)` in place (NaN stays NaN, as in PyTorch).
pub fn relu_in_place(x: &mut [f32]) {
    x.par_chunks_mut(16384).for_each(|s| {
        for v in s {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
    });
}

/// `dst += src`.
pub fn add_in_place(dst: &mut [f32], src: &[f32]) {
    dst.par_chunks_mut(16384).zip(src.par_chunks(16384)).for_each(|(d, s)| {
        for (d, s) in d.iter_mut().zip(s) {
            *d += s;
        }
    });
}

/// Replaces `row` with `softmax(row[a..b])` there and zero elsewhere. `exp` is computed in
/// f64 and rounded, as laya-go does.
pub fn softmax_masked(row: &mut [f32], a: usize, b: usize) {
    let mx = row[a..b].iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in &mut row[a..b] {
        let e = ((*v - mx) as f64).exp() as f32;
        *v = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for v in &mut row[a..b] {
        *v *= inv;
    }
    row[..a].fill(0.0);
    row[b..].fill(0.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_close_to_f64() {
        // libm-free reference: erf via a high-precision series for |x| < 3.
        fn erf_ref(x: f64) -> f64 {
            let mut sum = x;
            let mut term = x;
            for n in 1..200 {
                term *= -x * x / n as f64;
                sum += term / (2 * n + 1) as f64;
            }
            sum * 2.0 / std::f64::consts::PI.sqrt()
        }
        let mut worst = 0f64;
        for i in -3000..=3000 {
            let x = i as f32 / 1000.0;
            worst = worst.max((erf32(x) as f64 - erf_ref(x as f64)).abs());
        }
        assert!(worst < 2e-7, "erf32 error {worst}");
    }

    #[test]
    fn softmax_masks() {
        let mut r = [1.0, 2.0, 3.0, 4.0];
        softmax_masked(&mut r, 1, 3);
        assert_eq!(r[0], 0.0);
        assert_eq!(r[3], 0.0);
        assert!((r[1] + r[2] - 1.0).abs() < 1e-6);
    }
}

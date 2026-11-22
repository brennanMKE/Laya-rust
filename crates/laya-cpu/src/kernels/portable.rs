use rayon::prelude::*;

use super::{Kernels, Weight, check_matmul};

/// Plain-loop kernels: the correctness baseline. LLVM still vectorizes the dot products.
#[derive(Debug, Clone, Default)]
pub struct PortableKernels;

/// Output columns (weight rows) per task.
const PANEL: usize = 16;
/// Output rows per task.
const ROWS: usize = 16;

/// `Σ a[i]·b[i]` with eight independent accumulators, so it vectorizes.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let b = &b[..a.len()];
    let mut acc = [0f32; 8];
    let (ac, ar) = a.as_chunks::<8>();
    let (bc, br) = b.as_chunks::<8>();
    for (x, y) in ac.iter().zip(bc) {
        for l in 0..8 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (x, y) in ar.iter().zip(br) {
        s += x * y;
    }
    s
}

struct SendPtr(*mut f32);
// SAFETY: tasks write disjoint (row, column) blocks of C.
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl Kernels for PortableKernels {
    fn name(&self) -> &'static str {
        "portable"
    }

    fn matmul_t(&self, c: &mut [f32], a: &[f32], w: &Weight, m: usize, bias: Option<&[f32]>) {
        check_matmul(c, a, w, m, bias);
        let (n, k) = (w.rows, w.cols);
        if m == 0 || n == 0 {
            return;
        }
        let panels = n.div_ceil(PANEL);
        let row_blocks = m.div_ceil(ROWS);
        let cp = SendPtr(c.as_mut_ptr());
        let cp = &cp;
        (0..panels * row_blocks).into_par_iter().for_each(|task| {
            let (pj, rb) = (task % panels, task / panels);
            let (j0, j1) = (pj * PANEL, (pj * PANEL + PANEL).min(n));
            let (i0, i1) = (rb * ROWS, (rb * ROWS + ROWS).min(m));
            for i in i0..i1 {
                let ar = &a[i * k..i * k + k];
                for j in j0..j1 {
                    let mut v = dot(ar, &w.data[j * k..j * k + k]);
                    if let Some(b) = bias {
                        v += b[j];
                    }
                    // SAFETY: i < m, j < n, and no other task owns (i, j).
                    unsafe { *cp.0.add(i * n + j) = v };
                }
            }
        });
    }

    fn gemm(
        &self,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        c: &mut [f32],
        ldc: usize,
    ) {
        for i in 0..m {
            let crow = &mut c[i * ldc..i * ldc + n];
            let arow = &a[i * lda..i * lda + k];
            if trans_b {
                for (j, cv) in crow.iter_mut().enumerate() {
                    *cv = alpha * dot(arow, &b[j * ldb..j * ldb + k]);
                }
            } else {
                crow.fill(0.0);
                for (x, &av) in arow.iter().enumerate() {
                    let s = alpha * av;
                    let br = &b[x * ldb..x * ldb + n];
                    for (cv, bv) in crow.iter_mut().zip(br) {
                        *cv += s * bv;
                    }
                }
            }
        }
    }
}

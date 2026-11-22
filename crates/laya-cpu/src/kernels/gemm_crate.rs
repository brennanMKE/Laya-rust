use gemm::Parallelism;

use super::{Kernels, Weight, add_bias, check_matmul};

/// Kernels on the `gemm` crate (pure Rust, run-time dispatch to NEON / AVX2 / AVX-512).
#[derive(Debug, Clone, Default)]
pub struct GemmKernels {
    threads: usize,
}

impl GemmKernels {
    /// Kernels whose [`Kernels::matmul_t`] uses `threads` rayon threads (0 = the current pool).
    pub fn new(threads: usize) -> Self {
        GemmKernels { threads }
    }
}

impl Kernels for GemmKernels {
    fn name(&self) -> &'static str {
        "gemm"
    }

    fn matmul_t(&self, c: &mut [f32], a: &[f32], w: &Weight, m: usize, bias: Option<&[f32]>) {
        check_matmul(c, a, w, m, bias);
        let (n, k) = (w.rows, w.cols);
        if m == 0 || n == 0 {
            return;
        }
        // Small products are not worth waking the pool for.
        let par = if m * n * k < 64 * 1024 { Parallelism::None } else { Parallelism::Rayon(self.threads) };
        // SAFETY: the sizes were checked above; dst does not alias the inputs (distinct
        // borrows). rhs = Wᵀ is k×n: element (kk, j) is w.data[j*k + kk].
        unsafe {
            gemm::gemm(
                m,
                n,
                k,
                c.as_mut_ptr(),
                1,
                n as isize,
                false,
                a.as_ptr(),
                1,
                k as isize,
                w.data.as_ptr(),
                k as isize,
                1,
                0.0,
                1.0,
                false,
                false,
                false,
                par,
            );
        }
        if let Some(b) = bias {
            add_bias(c, b, m);
        }
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
        if m == 0 || n == 0 {
            return;
        }
        assert!(a.len() >= (m - 1) * lda + k && c.len() >= (m - 1) * ldc + n);
        let (rhs_cs, rhs_rs) = if trans_b {
            assert!(b.len() >= (n - 1) * ldb + k);
            (ldb as isize, 1)
        } else {
            assert!(b.len() >= (k.max(1) - 1) * ldb + n);
            (1, ldb as isize)
        };
        if k == 0 {
            for row in c.chunks_mut(ldc).take(m) {
                row[..n].fill(0.0);
            }
            return;
        }
        // SAFETY: bounds checked above; c is a distinct mutable borrow.
        unsafe {
            gemm::gemm(
                m,
                n,
                k,
                c.as_mut_ptr(),
                1,
                ldc as isize,
                false,
                a.as_ptr(),
                1,
                lda as isize,
                b.as_ptr(),
                rhs_cs,
                rhs_rs,
                0.0,
                alpha,
                false,
                false,
                false,
                Parallelism::None,
            );
        }
    }
}

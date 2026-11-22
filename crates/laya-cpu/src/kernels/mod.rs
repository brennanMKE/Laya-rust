//! The matrix multiplies the engine delegates (laya-go's `engine.Kernels`).
//!
//! Two sets: [`GemmKernels`] (the `gemm` crate, run-time SIMD dispatch, the default) and
//! [`PortableKernels`] (plain loops, the correctness baseline). Neither ever accumulates onto
//! its output: callers add residuals afterwards, as PyTorch does.

mod gemm_crate;
mod portable;

pub use gemm_crate::GemmKernels;
pub use portable::PortableKernels;

/// A Linear weight in PyTorch layout: `rows` out features by `cols` in features, row-major
/// fp32 (`data[n * cols + k]`).
#[derive(Debug, Clone)]
pub struct Weight {
    /// Out features.
    pub rows: usize,
    /// In features.
    pub cols: usize,
    /// Row-major fp32 values.
    pub data: Vec<f32>,
}

/// The hot operations of the forward pass.
pub trait Kernels: Send + Sync {
    /// A short name, such as `"gemm"`.
    fn name(&self) -> &'static str;

    /// Takes ownership of a weight in PyTorch layout and returns it in the form
    /// [`Kernels::matmul_t`] wants.
    fn new_weight(&self, rows: usize, cols: usize, data: Vec<f32>) -> Weight {
        assert_eq!(data.len(), rows * cols, "weight has {} values for {rows}x{cols}", data.len());
        Weight { rows, cols, data }
    }

    /// `C[m, n] = A[m, k] · W[n, k]ᵀ (+ bias)`, with `n = w.rows` and `k = w.cols`. `C` is
    /// overwritten, never accumulated onto. Parallel across the thread pool it runs in.
    fn matmul_t(&self, c: &mut [f32], a: &[f32], w: &Weight, m: usize, bias: Option<&[f32]>);

    /// `C = alpha · A · op(B)` on row-major strided matrices, single-threaded (attention runs
    /// many at once). `A` is `m×k` with row stride `lda`; `op(B)` is `k×n` (`B` is `n×k` with
    /// stride `ldb` when `trans_b`, else `k×n`); `C` is `m×n` with row stride `ldc`.
    #[allow(clippy::too_many_arguments)]
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
    );
}

pub(crate) fn check_matmul(c: &[f32], a: &[f32], w: &Weight, m: usize, bias: Option<&[f32]>) {
    let (n, k) = (w.rows, w.cols);
    assert!(
        a.len() >= m * k && c.len() >= m * n && bias.is_none_or(|b| b.len() == n),
        "matmul_t sizes: a {} (m {m} k {k}), c {} (n {n}), bias {:?}",
        a.len(),
        c.len(),
        bias.map(|b| b.len())
    );
}

/// Adds `bias` to every `n`-wide row of `c[..m*n]`.
pub(crate) fn add_bias(c: &mut [f32], bias: &[f32], m: usize) {
    let n = bias.len();
    for row in c[..m * n].chunks_exact_mut(n) {
        for (v, b) in row.iter_mut().zip(bias) {
            *v += b;
        }
    }
}

# The 195-token trace

The largest discrepancy in `tests/golden/trace/long.safetensors` is concentrated at token
38, using zero-based positions. It is already present at the end of the encoder. The scored
option markers are at 15 and 25, and the act head reads position 0 and the scorer's logits.

On the Apple M4 Pro with the pinned checkpoint, the maximum absolute difference from the
Python reference was:

| Tensor | GEMM | Portable |
|---|---:|---:|
| Encoder layer 3 | 0.000069 | 0.000065 |
| Encoder layer 27 | 0.5210 | 0.4924 |
| Head input after final LayerNorm and type embedding | 0.04893 | 0.04625 |
| First head layer | 0.5623 | 0.5304 |
| Second head layer | 0.6076 | 0.5732 |
| Scorer logits | 0.00004590 | 0.00002229 |

The larger errors occur in hidden activations. The largest GEMM discrepancy in the second
head layer is at row 38, channel 91, where Python reports about −989.38 and Rust reports
−989.99. Encoder layer 27 contains values as large as 29,130. The high-gain path through
the encoder and heads magnifies small numerical differences at this token.

To isolate the head, I reran both Rust kernels with the fixture's exact Python `head_in`
tensor substituted immediately before the first head layer. The second-head maximum error
fell from 0.6076 to 0.001831 for GEMM and from 0.5732 to 0.001770 for Portable. Scorer logits
then differed by at most 3.58e-7 and 1.20e-7 respectively, and both act logits matched
exactly. This locates the dominant drift upstream of the head.

A second diagnostic captured every encoder layer in a scratch trace with the same PyTorch
and checkpoint versions. The original encoder trace points reproduced exactly. In the
zero-based encoder layer 19, the residual after attention differed by at most 0.04453,
and the input to the MLP after normalization differed by 0.001452. The MLP amplified
that to 0.9858, and the layer output differed by 1.030. With the exact Python layer-18
output as Rust's input to layer 19, the layer-19 difference fell to 0.02734 and the
second-head difference fell to 0.001953. The main amplification comes from the encoder
MLP acting on drift accumulated in earlier layers.

`cargo test -p laya-cpu --release --test trace -- --nocapture` reports the worst flat index and
values for each traced tensor. The test checks the same scale-relative intermediate bound
as laya-go, `0.005 × max(1, max(abs(reference)))`, and retains the existing final-logit and
act-logit limits. The fixture and production forward pass are unchanged.

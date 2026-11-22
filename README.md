# laya-rust

Laya, a System 1 decision model, in Rust: fast CPU inference with SIMD (NEON on Arm, AVX2 and
AVX-512 on x86) and a drop-in `laya-server`, starting with the Raspberry Pi 5.

**Status: in progress.** See [docs/rust-port.md](docs/rust-port.md) for the goals, design, parity
tests and phases. Phase 0 (text side) passes every golden fixture on macOS; the CPU engine and
`laya-server` are in place.

It ports [laya-go](../laya-go) (the most complete port, and the source of the golden fixtures),
takes `/v1/decisions` and request merging from [laya-swift](../laya-swift), and treats the Python
package `laya` 0.3.6 and [System 1 Arcade](../System1Arcade)'s `agents/laya_server.py` as the
reference for behaviour.

## Layout

| Crate | What |
|---|---|
| `crates/laya` | Public API for loading a checkpoint and making predictions from Rust |
| `crates/laya-core` | Text side: tokenizer (`tokenizers` 0.23.2), prompts, Python-compatible JSON, calibration, wire format, answer cache, the `Engine` trait and `Model` |
| `crates/laya-cpu` | The CPU engine: safetensors loading, ModernBERT forward pass, head, `gemm` and `portable` kernels |
| `crates/laya-server` | The `laya-server` binary: `/predict`, `/v1/decisions`, `/health`, `/stats`, laya-go's flags |
| `crates/laya-hub` | Reuses or downloads the pinned checkpoint in the Hugging Face cache |
| `xtask` | `cargo xtask sync-fixtures [../laya-go]` copies laya-go's golden fixtures to `tests/golden` and records `LAYA_GO_COMMIT` |

## Building and testing

```sh
cargo test --workspace            # model tests skip without the checkpoint; they never download
cargo test -p laya-cpu --release  # the forward-pass parity tests, much faster optimized
cargo run --release -p laya-server -- --port 8001
```

Model tests find the checkpoint through `$LAYA_CHECKPOINT` or the Hugging Face cache
(`convaiinnovations/laya@1c5edc17…`). `LAYA_GOLDEN_ALL=1` runs every golden record.

Build the server and parity tests with `cargo xtask cross`. See the [Pi 5 handoff](docs/pi5.md)
for deployment and measurement. The server downloads missing checkpoint files unless
`--offline` or `HF_HUB_OFFLINE=1` is set.

Requests queued during a forward pass are merged into the next pass. Set `--key` or
`SYSTEM1_LAYA_KEY` to require bearer authentication, and `--max-queue` or
`SYSTEM1_LAYA_MAX_QUEUE` to bound admitted requests. Requests waiting for inference can carry an `X-Deadline-Ms` Unix
millisecond deadline. Set `LAYA_COALESCE=0` to run queued requests one at a time.

On the Mac, both kernels passed the full 3,670-sequence corpus. Every checked argmax
matched. GEMM matched 11,299 of 11,317 rounded batch fields exactly, and Portable
matched 11,293 of 11,317. The remaining fields differed by at most 0.0001. A ten-request
HTTP smoke comparison against native laya-go matched byte for byte.

## Remaining milestones

- Run full parity and the Frogger latency benchmark on the Pi 5.
- Evaluate fp16 weight packing, measure AVX2 and AVX-512 dispatch on x86, and build the
  Phase 4 release binaries.

## Mac CPU benchmarks

Apple M4 Pro, arm64, pinned checkpoint, default thread counts. Rust uses its GEMM kernel
and Go uses its native Accelerate backend (laya-go `f5196db`). Forward rows are synthetic
batches with the same sequence lengths and option-marker layout in both ports.

| Case | Rust median | Rust p99 | Go native median |
|---|---:|---:|---:|
| `forward/1x56` | 136.59 ms | 146.74 ms | 45.69 ms |
| `forward/6x50` | 438.04 ms | 483.78 ms | 131.60 ms |
| `forward/14x56` | 952.71 ms | 1,152.26 ms | 326.70 ms |
| `forward/1x512` | 719.03 ms | 754.70 ms | 273.40 ms |
| `load` | 65.19 ms | 91.54 ms | 81.07 ms (mean) |

Rust process peak RSS was 2,875 MiB across the benchmark suite. Go reported up to 1,711 MiB
during its forward benchmarks. The p99 is the nearest-rank percentile of 11–70 timings
for these Rust cases.

| `matmul_t` shape (m × n × k) | GEMM GFLOP/s | Portable GFLOP/s |
|---|---:|---:|
| 56 × 3072 × 1024 | 372.74 | 216.92 |
| 56 × 1024 × 1024 | 212.02 | 179.25 |
| 56 × 5248 × 1024 | 396.43 | 134.96 |
| 56 × 1024 × 2624 | 203.95 | 71.76 |
| 784 × 3072 × 1024 | 570.56 | 211.01 |
| 784 × 1024 × 1024 | 556.72 | 187.30 |
| 784 × 5248 × 1024 | 643.98 | 233.71 |
| 784 × 1024 × 2624 | 534.06 | 201.50 |

Reproduce the Rust run with `cargo bench -p laya-cpu --bench cpu --locked`. The Go comparison uses
`go test ./internal/modernbert -run '^$' -bench '^(BenchmarkForward|BenchmarkLoad)' -benchtime 10x`.

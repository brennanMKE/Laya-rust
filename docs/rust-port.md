# Laya in Rust: port plan

**Status: plan.** No code yet. This page says what laya-rust is for, what it ports from where,
the crate and design choices, how parity is proven, and the phases. Anything marked
**Estimate** or **Unverified** has not been measured or run; correct it here when it is.

## Summary

- **Goal: fast Laya on CPUs, starting with the Raspberry Pi 5.** On yoyo (Pi 5, 4× Cortex-A76,
  no GPU), laya-go's portable Go kernels take 7.2 s for a Frogger decision against PyTorch's
  3.4 s, because Go can't use NEON without assembly. Rust gets NEON (and AVX2/AVX-512) from
  LLVM and from pure-Rust GEMM crates, with no C toolchain and no Python.
- **A drop-in `laya-server`.** Same HTTP contract as System 1 Arcade's `agents/laya_server.py`
  and laya-go's `laya-server` (`/predict`, `/health`, `/stats`), plus laya-swift's
  `/v1/decisions`. One static binary per platform.
- **Same answers as Python.** Proven with laya-go's golden fixtures, unchanged: identical token
  ids and markers, identical 4-decimal answers from Python's logits, logits within 1e-3.
- **Port from laya-go, line by line where it matters.** laya-go is the most complete and
  best-documented port (8,100 lines plus 4,500 lines of tests). Python is the reference for
  behaviour; Swift contributes the `/v1/decisions` endpoint and request merging.
- **About 3–4 weeks for one person** (**Estimate**), phases below. The Pi target is to at least
  match PyTorch: a Frogger decision in about 2 s (**Estimate**, from the GEMM arithmetic below).

## What we measured that motivates this

Benchmark on 21 Nov 2026, same checkpoint, answer cache off, real System 1 Arcade requests
([artifact](https://claude.ai/artifact/3i6uYWbVGnE5hwNpWkRdCc)):

| Server | Frogger p50 | Tetris p50 | States/s | Peak memory | Ready in |
|---|---|---|---|---|---|
| gordon (M2) laya-go Metal | 117 ms | 304 ms | 49.9 | 931 MB | 1.2 s |
| gordon (M2) laya-swift Metal | 121 ms | 304 ms | 50.9 | 1,029 MB | 0.7 s |
| joe (M1) laya-go Metal | 162 ms | 437 ms | 35.1 | 926 MB | 1.6 s |
| yoyo (Pi 5) Python / PyTorch | 3.4 s | 10.0 s | 1.45 | 2,168 MB | ~70 s |
| yoyo (Pi 5) laya-go portable | 7.2 s | 20.9 s | 0.71 | 1,664 MB | 1.1 s |

Both yoyo servers kept all four cores busy (Go 368%, Python 327% average CPU), so Go's gap is
work per cycle, not idle cores. Macs are already well served by laya-go and laya-swift; laya-rust
is for Linux and Windows CPUs first.

### How fast can a Pi 5 go?

A Frogger request is 6 sequences, about 300 tokens. ModernBERT-large has about 343M non-embedding
parameters, so a pass is about 2 × 343M × 300 ≈ 206 GFLOP, nearly all in GEMMs.

| | GFLOP/s | Frogger pass |
|---|---|---|
| Cortex-A76 peak: 4 cores × 2.4 GHz × 2 NEON FMA pipes × 4 lanes × 2 | 154 | 1.3 s |
| A well-tuned fp32 GEMM at 70–75% of peak (**Estimate**) | 110–115 | 1.8–1.9 s |
| PyTorch as measured | ~61 | 3.4 s |
| laya-go portable as measured | ~28 | 7.2 s |

So a good GEMM should beat PyTorch by about 1.5–1.8×, but not by 10×. Bigger wins on the Pi
would need fp16 arithmetic or int8 weights, with looser parity (phase 5).

## Goals and non-goals

Goals:

- Parity with laya 0.3.6 on checkpoint `convaiinnovations/laya@1c5edc17…`, at laya-go's
  tolerances.
- A CPU engine with SIMD on aarch64 (NEON) and x86_64 (AVX2, AVX-512 when present), chosen at run
  time.
- `laya-server` compatible with laya-go's flags, endpoints, JSON formatting and `/stats`.
- One self-contained binary per target, cross-compiled from a Mac: `aarch64-unknown-linux-gnu`,
  `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`, `aarch64-apple-darwin`.
- A library crate others can embed (`laya`), like laya-go's `laya` package.

Non-goals, at least at first:

- A GPU engine. Metal is covered by laya-go and laya-swift. `wgpu` or CUDA can come later.
- Training or fine-tuning. `--weights` (heads tuned by System 1 Arcade's `agents/finetune.py`)
  is unsupported in laya-go too; it is listed under phase 5.
- Other checkpoints (`typed-decisions`, `multilingual`) until the base one is done, as in laya-go.

## What Laya computes

As in laya-go's [CONTRACT.md](../../laya-go/CONTRACT.md) and laya-swift's design:

- **One sequence per (state, question):**
  `[CLS] "<type> question: <instructions>" [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]`,
  at most `max_len` (512) tokens, options fitted to `head_max_len` (192). Each option's `[MASK]`
  position is a marker.
- **The network:** 28-layer ModernBERT-large (hidden 1024, 16 heads, GeGLU MLP, RoPE θ 160000 on
  global layers 0, 3, … 27, θ 10000 with a ±64 sliding window elsewhere), final LayerNorm, a
  per-type embedding added to every token, two pre-norm `nn.TransformerEncoderLayer` head layers
  with **ReLU**, a scorer (LayerNorm, Linear, GELU, Linear(1)) at each marker, and an act head over
  `[h[CLS], top1, top1 − top2, entropy / log k, k / 255]`.
- **Calibration:** temperature per `"<type>:<size>"` bucket clamped to [0.5, 5.0], float32
  softmax, then `choice` argmax with `confidence = 1 − H(p)/log k`, `score` `Σ i·pᵢ`, `noul`
  `p[1]`; rounded like Python's `round(x, 4)`.
- **Engine boundary:** the engine returns raw scorer logits at each marker and the two raw act
  logits. Temperatures, softmax, confidence and rounding stay in the text side.

## Where each piece comes from

| Piece | Port from | Lines there | Rust approach |
|---|---|---|---|
| Tokenizer | Python's `tokenizers` (the reference itself) | n/a | The `tokenizers` crate, pinned to the version Python uses; laya-go's `internal/tokenizer` as fallback |
| Prompt builder | `internal/prompt/prompt.go` | 157 | Direct port |
| Python JSON (`json.loads`/`dumps`) | `internal/pyjson` | 361 + 173 | Direct port; `ryu` for shortest float digits |
| Python dict helpers | `internal/pydict` | 160 | Direct port |
| Calibration | `internal/calibrate` | 182 | Direct port, `f32` throughout |
| Checkpoint config | `internal/checkpoint` | 149 | `serde` |
| safetensors | `internal/safetensors` | 376 | `safetensors` crate + `memmap2` |
| Request decoding, wire format | `wire.go`, `laya.go`, `model.go` | 262 + 252 + 394 | Direct port |
| Forward pass (CPU) | `internal/modernbert` | ~950 | Port the structure; GEMMs through a `Kernels` trait |
| Kernels | `internal/kernels` | 209 + 206 | New: `gemm` crate, plus Accelerate and OpenBLAS options |
| Answer cache | `answercache` | 259 | Direct port (same keys and stats) |
| Hub download | `hub` | 538 | Port (pinned revision, resumable, sha256-checked, HF cache layout) |
| HTTP server | `cmd/laya-server` | 257 + 346 | `axum` + `tokio` |
| `/v1/decisions` | laya-swift `Decisions.swift` | 339 | Direct port |
| Merging requests that arrive during a pass | laya-swift 803cd83 | small | Port the idea |
| Fixtures | laya-go `testdata/golden` | 20 MB | Copied by a sync script |

Python (`laya` 0.3.6, `agents/laya_server.py`, `agents/laya_batch.py`) stays the behavioural
reference: when Go and Python disagree, Python wins and laya-go gets a bug report.

## Crate layout

A Cargo workspace:

| Crate | What |
|---|---|
| `laya-core` | Text side: tokenizer wrapper, prompt, pyjson, pydict, calibrate, checkpoint config, wire format, the `Engine` trait. No heavy dependencies; builds for `wasm32` too (**Unverified**, nice to have) |
| `laya-cpu` | The CPU engine: safetensors loading, ModernBERT forward, head, `Kernels` trait and kernel sets |
| `laya-hub` | Find or download the checkpoint in the Hugging Face cache |
| `laya` | Public API: `Model::load`, `predict`, `predict_many`, the answer cache |
| `laya-server` | The binary |
| `xtask` | Fixture sync, cross builds, benchmarks |

The `Engine` trait mirrors laya-go's `engine.Engine`:

```rust
pub struct Batch {
    pub tokens: Vec<u32>,       // all sequences concatenated, no padding
    pub starts: Vec<usize>,     // n + 1 offsets
    pub markers: Vec<Vec<u32>>, // option [MASK] positions per sequence
    pub qtypes: Vec<u8>,        // 0 choice, 1 score, 2 noul
}

pub struct Output {
    pub logits: Vec<Vec<f32>>,  // raw scorer output at each marker
    pub act_logits: Vec<[f32; 2]>,
}

pub trait Engine: Send + Sync {
    fn forward(&self, batch: &Batch) -> Result<Output, EngineError>;
}
```

Keeping it identical means laya-go's CONTRACT.md applies as written, and a future engine
(`wgpu`, ONNX Runtime through `ort`, `candle`) plugs in without touching the text side.

## Design decisions

### Tokenizer: use the reference implementation

Python's `tokenizers` package is a binding to the Rust `tokenizers` crate. Using the crate means
running the same code Python ran when the fixtures were made, which removes the hardest parity
problem laya-go and laya-swift had to solve by hand (added-token matching, NFC, the GPT-2
pre-tokenizer pattern, byte-level BPE).

- **Pin the version to Python's**: the fixtures were made with tokenizers 0.23.2. Use crate
  `0.23.x` (**Unverified** that the Python version maps one to one onto a crate version; check
  the Python wheel's `Cargo.lock`). The local registry has 0.22.2 cached, which is too old.
- **Regex backend.** The crate's default `onig` feature builds Oniguruma from C, which is what
  Python uses. That needs a C cross-compiler for each target. The `fancy-regex` backend is pure
  Rust but may differ on Unicode classes (**Unverified**). Start with `onig` for parity; try
  `fancy-regex` only if cross-compiling hurts, and accept it only if `tokenizer.jsonl` still
  matches exactly.
- **Fallback:** if the crate drags in too much or misbehaves, port laya-go's
  `internal/tokenizer` (444 + 121 lines), which already matches on every fixture.
- Rust makes the Swift traps easy: `&str` is UTF-8 bytes, `==` is byte equality (no canonical
  equivalence), and `char` is a Unicode scalar.

Gate: identical ids on all 6,511 `tokenizer.jsonl` strings and identical ids and markers on all
390 requests in `games.jsonl` and `edge.jsonl`.

### Python-compatible JSON: port, don't configure serde

States and non-string criteria are `json.dumps`-ed before tokenizing, so bytes must match Python.
`serde_json` with `preserve_order` and `arbitrary_precision` gets close but not there: Python keeps
a repeated key at its **first** position with its **last** value, accepts `NaN`/`Infinity`, writes
`", "` and `": "`, keeps non-ASCII, and lays floats out with `repr`'s rules (`1e-05`, `1e+16`,
`123.0`, `-0.0`). Port `internal/pyjson` as its own parser and writer. Take shortest round-trip
digits from `ryu` and apply laya-go's layout rules from `dumps.go`. Gate:
`pyjson/testdata/floats.json` and `dumps.jsonl` byte-identical.

### Calibration in `f32`, and `expf`

Port `internal/calibrate` with `f32` everywhere and numpy's pairwise summation order.

- **`exp`.** numpy's float32 `exp` calls the platform `expf`, and the fixtures came from macOS
  arm64. Rust's `f32::exp` lowers to the platform libm `expf`. On macOS it should match the
  fixtures bit for bit (laya-go can't, and is off by one ulp on about 0.6% of inputs). On Linux
  it's glibc's `expf`, which can differ from Apple's (**Unverified**). Repeat laya-go's
  200,000-question random comparison on both, and if Linux differs, consider a bundled
  correctly-rounded `expf` (for example the `libm` crate, **Unverified** accuracy) so every
  platform gives the same answers.
- **`round(x, 4)`.** Format with `format!("{:.4}", x)` and parse back; Rust's precision formatting
  rounds from the exact binary value (**Unverified** that it's half-to-even in every case the
  fixtures hit). `calibration.jsonl` checks it.

Gate: identical answer JSON on `calibration.jsonl` (264 questions), `games.jsonl` and
`edge.jsonl`, both answer shapes, from Python's logits.

### CPU engine

Port `internal/modernbert`'s structure (packed sequences, per-sequence attention, sliding window,
RoPE, GeGLU, the head), with the matrix multiplies behind a trait like laya-go's
`engine.Kernels`:

```rust
pub trait Kernels: Send + Sync {
    fn new_weight(&self, w: WeightView) -> PackedWeight; // may repack into SIMD panels
    fn matmul_t(&self, a: &[f32], w: &PackedWeight, bias: Option<&[f32]>, c: &mut [f32], m: usize);
    fn gemm(&self, /* strided row-major, used by attention from several threads */);
}
```

Kernel sets, chosen at load (`--kernels` / `LAYA_KERNELS`):

| Kernels | Where | Notes |
|---|---|---|
| `gemm` crate (default) | everywhere | Pure Rust, run-time dispatch to NEON, AVX2, AVX-512; f32 and f16 inputs. Expected default on the Pi |
| `accelerate` | macOS | For comparison with laya-go's native numbers on the same Mac |
| `openblas` | Linux, Windows | `cblas_sgemm` from a system library, optional feature. A useful second opinion on the Pi |
| `portable` | everywhere | Plain loops, the correctness baseline (LLVM still vectorizes them) |

Rules carried over from laya-go:

- **Never let a GEMM accumulate onto a residual** (`beta = 1`). Add the residual afterwards, or
  ModernBERT's outlier channels cost about 4× in logit error.
- **Weights:** the checkpoint's linears are fp16. Phase 1 widens them to fp32 at load (about
  1.6 GB, like laya-go native). Phase 4 tries keeping fp16 and widening inside the GEMM's packing
  step (about 0.9 GB, like the Metal engines), if the `gemm` crate's f16 path accumulates in f32
  (**Unverified**).
- **Threads:** `rayon` with one pool sized by `--threads` (default: all cores). Parallelize inside
  each GEMM and across attention heads and sequences, not across requests: one pass at a time
  already uses every core.
- **The non-GEMM work** (LayerNorm, GELU, softmax, RoPE) is a real share of the time on a slow
  CPU. Write those loops so LLVM vectorizes them (`chunks_exact`, no bounds checks in the inner
  loop) and check with `cargo asm` on aarch64.

**Alternative to evaluate first: `candle`.** Hugging Face's `candle-transformers` has a ModernBERT
model (**Unverified** that it handles Laya's packed sequences and sliding window, and whether its
CPU matmul is competitive on A76). A one-day spike in phase 1 decides: if candle's encoder matches
the trace fixtures and is at least as fast as the `gemm` path, use it for the encoder and keep the
hand-written head. Otherwise port laya-go's forward pass as planned. `tract` (pure-Rust ONNX) and
`ort` (ONNX Runtime) are further options, through the same `Engine` trait.

### Server

`axum` on `tokio`; the model runs on a dedicated thread (or `spawn_blocking`) so the async
runtime never blocks on a pass.

| Endpoint | Source |
|---|---|
| `POST /predict` (single and `batch`) | laya-go `cmd/laya-server`, Python `laya_server.py` |
| `GET /health`, `GET /stats` | same; `/stats` keys and order identical to laya-go's |
| `POST /v1/decisions` | laya-swift `Decisions.swift` |

Also:

- **Flags** identical to laya-go's `laya-server` (`--host`, `--port`, `--addr`, `--model`,
  `--revision`, `--cache-size`, `--threads`, `--stats-every`, `--offline`), plus `--kernels`.
  Accept and ignore `--device` and `--engine` values that make sense for scripts written for the
  others.
- **Ready line** `agent ready http://…/predict`, as System 1 Arcade expects.
- **Merging:** requests that arrive while a pass runs are packed into the next pass, as laya-swift
  does since 803cd83. On the Pi, where a pass takes seconds, this matters more than on a Mac.
- **Optional bearer key and a bounded queue.** yoyo's production Python server requires a key
  (`SYSTEM1_LAYA_KEY`) and its `/stats` reports `requests: answered, busy, expired, unauthorized,
  bad_request, waiting, max_waiting, max_queue`. Match that shape so Parallax can switch servers
  without changes (**Unverified**: find that server's source on yoyo and port its exact rules).

### Hub

Port laya-go's `hub` (pinned revision `1c5edc17…`, `$HF_HUB_CACHE` / `$HF_HOME/hub` /
`~/.cache/huggingface/hub`, resumable downloads checked against sha256, `HF_TOKEN` only to
huggingface.co, `HF_HUB_OFFLINE=1`). The `hf-hub` crate is an option if it supports resuming and
verification (**Unverified**); a port with `ureq` + `sha2` keeps behaviour identical to laya-go
and Python.

## Parity testing

Fixtures come from laya-go's `testdata/golden` (laya 0.3.6, torch 2.14 fp32 CPU, tokenizers
0.23.2, numpy 2.5.3), copied by `cargo xtask sync-fixtures ../laya-go`, which records the commit
in `LAYA_GO_COMMIT`, as laya-swift does. Never regenerate fixtures to make a Rust test pass;
regenerate in laya-go only when the reference or the checkpoint changes.

| Fixture | Records | Test |
|---|---|---|
| `tokenizer.jsonl` | 6,511 strings | ids identical |
| `games.jsonl`, `edge.jsonl` | 343 + 47 requests, 3,670 sequences | ids, markers, errors identical; answers from Python's logits identical; engine logits vs `logits` and `logits_single` |
| `calibration.jsonl` | 264 questions | answer JSON identical, both shapes |
| `trace/*.safetensors` | 5 sequences, one of 195 tokens | every intermediate, to bisect a mismatch |
| `pyjson/testdata` | floats, dumps | bytes identical |

Tolerances, the same as laya-go's native engine:

| Check | Required |
|---|---|
| Token ids, markers, option counts, error cases | identical |
| Answers from Python's logits | identical JSON text |
| Engine logits | max abs difference ≤ 1e-3 (expect ~1e-5 in fp32) |
| Act logits | relative difference ≤ 1e-4 |
| Every argmax | same as Python's wherever the top-two gap > 1e-3 |
| 4-decimal answers end to end | ≥ 99.5% of fields identical, the rest within 1e-4, every choice the same |

Model tests find the checkpoint through `LAYA_CHECKPOINT` or the HF cache and skip when it's
missing; they never download. `LAYA_GOLDEN_ALL=1` runs every record. Fixture logits are float32
written as shortest decimals: parse them as `f32`.

A **cross-port check** runs the same requests through laya-go and laya-rust on the same machine
and diffs the 4-decimal answers; any difference names a record to look at.

## Benchmarks

The same cases as laya-go and laya-swift, so tables line up:

- Criterion benches: `forward/1x56`, `forward/6x50`, `forward/14x56`, `forward/1x512`,
  `load`, `matmul_t` GFLOP/s per shape and kernel set. Median and p99, peak RSS.
- End to end: `tools/laya_bench.py` (the script used for the 21 Nov homelab run: golden game
  requests, parity sample, per-game latency, concurrency 1/2/4) against `laya-server` on yoyo,
  next to the existing numbers.

Targets on yoyo (**Estimate**):

| | laya-go portable | PyTorch | laya-rust target |
|---|---|---|---|
| Frogger p50 | 7.2 s | 3.4 s | ≤ 2.2 s |
| Tetris p50 | 20.9 s | 10.0 s | ≤ 6.5 s |
| Peak memory | 1.66 GB | 2.17 GB | ≤ 1.7 GB fp32 weights, ~1.0 GB with fp16 |
| Ready in | 1.1 s | ~70 s | ≤ 2 s |

## Plan

Durations are for one person who knows Rust (**Estimate**). Each phase ends with something
testable.

**Phase 0: workspace and text side (3–4 days)**

- Cargo workspace, `xtask sync-fixtures`, CI on macOS and Linux (aarch64 and x86_64).
- `tokenizers` crate wired up and pinned; port pyjson, pydict, prompt, calibrate, checkpoint
  config, wire format.
- *Milestone:* identical ids on all 6,511 strings and 390 requests; identical answers from
  Python's logits on `games`, `edge` and `calibration`, on macOS and Linux aarch64.

**Phase 1: CPU engine (1–1.5 weeks)**

- Day 1: the `candle` spike, then decide.
- safetensors with `memmap2`, weight loading, the forward pass, the head, the `portable` and
  `gemm` kernel sets, a trace hook.
- *Milestone:* trace intermediates within tolerance; every sequence within 1e-3; ≥ 99.5% of
  4-decimal fields and every argmax equal end to end; `forward/*` benches on the Mac next to
  laya-go native.

**Phase 2: server, cache and hub (3–4 days)**

- `laya-server` with `/predict`, `/health`, `/stats`, `/v1/decisions`, the answer cache, request
  merging, optional key and queue; `laya-hub`.
- *Milestone:* laya-go's `cmd/laya-server` server tests ported and passing (same JSON bytes for
  the same requests); System 1 Arcade's custom agent plays against it on the Mac.

**Phase 3: Pi 5 deployment and tuning (3–5 days)**

- Cross-build `aarch64-unknown-linux-gnu` (`cargo zigbuild` or `cross`; Oniguruma needs a C
  cross-compiler unless the tokenizer moves to `fancy-regex`).
- Ask the Homelab agent to deploy it on yoyo as a transient unit beside production, as in the
  21 Nov run, and rerun `laya_bench.py`. Profile with `perf` on the Pi; tune GEMM blocking and
  the non-GEMM loops.
- *Milestone:* full fixture parity on the Pi (`cargo test` on yoyo, or the test binary copied
  over); Frogger p50 at or below PyTorch's 3.4 s, aiming for 2.2 s; the benchmark table above
  filled in with measurements.

**Phase 4: memory and reach (about 1 week)**

- fp16 weights widened in the GEMM pack step, if parity holds.
- x86_64 Linux and Windows builds, AVX2/AVX-512 dispatch measured on whatever x86 machine is
  available; the `openblas` kernel set as a comparison.
- Release binaries for every target, README with parity and benchmark tables.
- *Milestone:* 0.1.0 released, recording the laya-go commit, laya version and checkpoint
  revision.

**Phase 5: options (spikes, then pick)**

- int8 weight-only or fp16-arithmetic kernels on A76 (FEAT_FP16), with looser, documented
  tolerances.
- `--weights` for heads tuned by `agents/finetune.py` (also missing in laya-go).
- A `wgpu` engine, ONNX Runtime via `ort`, or `candle` if not already chosen.
- Other checkpoints, following laya-go.

Phases 0–4: about 3–4 weeks (**Estimate**). laya-go's own history (first commit 23 Sep, Metal
engine merged by 3 Oct) suggests agent-assisted work can be faster.

## Risks and open questions

- **The Pi speed-up is bounded.** The GEMM arithmetic says about 1.5–1.8× over PyTorch, not more,
  in fp32. If the goal is realtime games on a Pi, that's not enough on its own; reduced precision
  is the only bigger lever, and it costs parity.
- **The cheaper alternative.** laya-go could call OpenBLAS through purego (its
  `docs/windows-linux.md`, option 2) in about a day. If that lands near PyTorch or better, the
  Pi case for Rust weakens to binary size, memory and a third independent implementation. Worth
  doing first or in parallel as a baseline.
- **Tokenizer crate version and regex backend** must reproduce tokenizers 0.23.2 exactly; only the
  fixtures prove it.
- **`expf` differs across platforms.** Linux answers may differ from macOS answers in rare
  4-decimal cases unless a bundled `expf` is used.
- **Three ports to keep in step.** A change to prompts, calibration or the server contract must
  reach laya-go, laya-swift and laya-rust. `LAYA_GO_COMMIT` makes drift visible; a cross-port
  answer diff in CI would make it fail loudly.
- **Production `/stats` shape on yoyo** (key, queue, request counters) isn't in any repository
  checked so far; find its source before claiming drop-in compatibility with Parallax.
- **Name.** `laya-rust` for the repository; crates `laya`, `laya-core`, … may collide on
  crates.io (**Unverified**).

# laya-rust

Laya, a System 1 decision model, in Rust: fast CPU inference with SIMD (NEON on Arm, AVX2 and
AVX-512 on x86) and a drop-in `laya-server`, starting with the Raspberry Pi 5.

**Status: plan only.** See [docs/rust-port.md](docs/rust-port.md) for the goals, design, parity
tests and phases.

It ports [laya-go](../laya-go) (the most complete port, and the source of the golden fixtures),
takes `/v1/decisions` and request merging from [laya-swift](../laya-swift), and treats the Python
package `laya` 0.3.6 and [System 1 Arcade](../System1Arcade)'s `agents/laya_server.py` as the
reference for behaviour.

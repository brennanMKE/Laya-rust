# Raspberry Pi 5 handoff

Build the package on the Mac with Rust, Zig and cargo-zigbuild installed.

```sh
rustup target add aarch64-unknown-linux-gnu
cargo install cargo-zigbuild --locked
cargo xtask cross
file target/cross/aarch64-unknown-linux-gnu.2.28/laya-server
```

The package contains the AArch64 server, Linux test executables, unchanged golden fixtures and
the System 1 Arcade benchmark script. The default target uses glibc 2.28 as its minimum.

## Homelab agent

Copy the package and use the existing checkpoint on yoyo. Its snapshot is `fd4f9dbc…`, whose files are byte-identical to the pinned `1c5edc17…`. Run the trial alongside
production on port 8001. The transient service binds to loopback.

```sh
scp -r target/cross/aarch64-unknown-linux-gnu.2.28 pi@yoyo:laya-rust-trial
ssh pi@yoyo
cd "$HOME/laya-rust-trial"
export LAYA_CHECKPOINT=/srv/yoyo-data/laya/hf/hub/models--convaiinnovations--laya/snapshots/fd4f9dbcf4fde222fd43174bdd4c3b08b789e4e1
export LAYA_GOLDEN_DIR="$PWD/golden"
export LAYA_GOLDEN_ALL=1
export RAYON_NUM_THREADS=4

test -r "$LAYA_CHECKPOINT/model.safetensors" || exit 1
ldd ./laya-server
for test_binary in ./parity-tests/*; do
  "$test_binary" --test-threads=1 --nocapture || exit 1
done

sudo systemd-run --unit=laya-rust-trial --uid=pi \
  --property=WorkingDirectory="$PWD" --property=MemoryAccounting=yes \
  "$PWD/laya-server" --host 127.0.0.1 --port 8001 \
  --model "$LAYA_CHECKPOINT" --offline --threads 4 --cache-size 0
for attempt in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8001/health; then break; fi
  sleep 1
done
curl -fsS http://127.0.0.1:8001/health || exit 1
sudo journalctl -u laya-rust-trial -n 20 -o short-monotonic --no-pager
python3 laya_bench.py --url http://127.0.0.1:8001 --label rust-pi5-gemm \
  --golden "$LAYA_GOLDEN_DIR" --out rust-pi5-gemm.json --parity-n 0
pid=$(systemctl show laya-rust-trial -p MainPID --value)
awk '/^VmHWM:/ {print}' "/proc/$pid/status"
sudo systemctl show laya-rust-trial -p MemoryPeak
sudo systemctl stop laya-rust-trial
```

Record the fixture results, Frogger and Tetris p50, throughput at concurrency 1, 2 and 4,
peak service RSS, and the time from launch to the ready line. The Phase 3 Frogger acceptance
threshold is p50 ≤ 3.4 seconds. The tuning target is ≤ 2.2 seconds. Compare against the PyTorch
and Go baselines in [the port plan](rust-port.md). Run profiling separately from the timed run
if the Pi misses the target.

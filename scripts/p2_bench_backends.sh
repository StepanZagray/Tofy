#!/usr/bin/env bash
# A/B compare --features cuda vs --features cudnn on a short p2-train microbench.
# Prefer the faster backend for overnight runs; re-run after candle/backend changes.
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS="${STEPS:-15}"
BATCH="${PHYSICAL_BATCH:-1024}"
HIDDEN="${HIDDEN_DIM:-128}"
ACTION="${ACTION_DIM:-32}"
PROFILE_EVERY="${TOFY_P2_STEP_PROFILE:-5}"

run_one() {
  local feat="$1"
  local out="/tmp/tofy-p2-bench-${feat}"
  echo "===== features=${feat} ====="
  cargo build --release --features "$feat" --bin tofy
  rm -rf "$out"
  TOFY_P2_STEP_PROFILE="$PROFILE_EVERY" ./target/release/tofy p2-train \
    --device cuda \
    --lessons dynamics \
    --steps-per-lesson "$STEPS" \
    --physical-batch "$BATCH" \
    --hidden-dim "$HIDDEN" \
    --action-dim "$ACTION" \
    --checkpoint-every-steps 0 \
    --output-dir "$out" \
    2>&1 | tee "/tmp/tofy-bench-${feat}.log" | rg '\[profile |status=' || true
}

run_one cuda
run_one cudnn
echo
echo "Compare last steady [profile] lines above. Use the faster feature flag."
echo "Layer probe: cargo test --release --features cudnn --test cuda_conv_probe -- --ignored --nocapture"

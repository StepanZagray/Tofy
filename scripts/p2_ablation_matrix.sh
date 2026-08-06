#!/usr/bin/env bash
# Phase C ablation harness: D=1 baseline vs full recursion (5 seeds).
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/p2_runs.sh
source "$script_dir/p2_runs.sh"
ROOT="$(cd "$script_dir/.." && pwd)"
cd "$ROOT"
BASE="${1:-$(p2_run_dir readiness)}"
FEATURES=""
if command -v nvidia-smi >/dev/null 2>&1; then
  FEATURES="--features cuda"
fi

for SEED in 1 2 3 4 5; do
  OUT="${BASE}-ablation-d1-s${SEED}"
  cargo run --release $FEATURES -- p2-train \
    --device "${P2_DEVICE:-cuda}" \
    --seed "$SEED" \
    --lessons dynamics,exploration \
    --steps-per-lesson 512 \
    --physical-batch 64 \
    --grad-accum 2 \
    --baseline-d1 \
    --prefix-weight 0.1 \
    --max-steps-this-run 1024 \
    --output-dir "$OUT"
done

echo "Ablation matrix seeds 1–5 complete under ${BASE}-ablation-d1-s*"

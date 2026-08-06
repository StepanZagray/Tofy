#!/usr/bin/env bash
# After v17-stable finishes: PTRM sweep on stable ckpt, then start exp run if needed.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
# shellcheck source=scripts/p2_runs.sh
source "$script_dir/p2_runs.sh"
cd -- "$repo_root"

STABLE_DIR="${STABLE_DIR:-$(p2_run_dir v17-stable)}"
EXP_DIR="${EXP_DIR:-$(p2_run_dir v17-exp-qnoise)}"
RUN_ROOT="${RUN_ROOT:-$(p2_run_dir v17-night)}"
PHYSICAL_BATCH="${PHYSICAL_BATCH:-512}"
LOG="$RUN_ROOT/watch-continue.log"

log() { printf '[%s] watch: %s\n' "$(date -Iseconds)" "$*" | tee -a "$LOG"; }

mkdir -p -- "$RUN_ROOT"

log "waiting for stable eval: $STABLE_DIR/eval_report_64ep_v5.json"
while [[ ! -f "$STABLE_DIR/eval_report_64ep_v5.json" ]]; do
  sleep 120
done
log "stable eval ready"

while pgrep -f 'target/release/tofy' >/dev/null 2>&1; do
  sleep 30
done
log "GPU idle — stable PTRM sweep"
DEVICE=cuda PRETEST_DIR="$RUN_ROOT/pretest" scripts/p2_v17_pretest.sh stable-sweep "$STABLE_DIR" 2>&1 | tee -a "$LOG"

if [[ -d "$EXP_DIR/checkpoints" ]] || [[ -f "$EXP_DIR/model.safetensors" ]]; then
  log "exp run already present — done"
  exit 0
fi

log "starting v17-exp-qnoise train+eval"
{
  cargo run --release --features cudnn -- p2-train \
    --device cuda \
    --lessons dynamics,exploration,sequential,q_calibration,falsification \
    --physical-batch "$PHYSICAL_BATCH" --grad-accum 1 \
    --steps-per-lesson 4096 --checkpoint-every-steps 100 \
    --output-dir "$EXP_DIR" \
    --hidden-dim 128 --action-dim 32 \
    --outer-steps 8 --inner-steps 2 \
    --sigreg-weight 0.003 --sigreg-projections 32 \
    --randomize-depth --stop-grad-q-y --q-quantile-targets --train-z-noise 0.03
  ckpt="$EXP_DIR/model.safetensors"
  [[ -f "$EXP_DIR/model.best.safetensors" ]] && ckpt="$EXP_DIR/model.best.safetensors"
  q="$(python3 -c "import json; print(json.load(open('${EXP_DIR}/config.json'))['q_mse_threshold'])")"
  cargo run --release --features cudnn -- p2-eval \
    --device cuda --physical-batch 64 \
    --checkpoint "$ckpt" --train-config "$EXP_DIR/config.json" \
    --synthetic-episodes 64 --ptrm-k 1,2,4,8 --q-mse-threshold "$q" \
    --output "$EXP_DIR/eval_report_64ep_v5.json"
} 2>&1 | tee -a "$LOG"

log "watch-continue finished"

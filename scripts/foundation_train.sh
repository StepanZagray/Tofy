#!/usr/bin/env bash
# Foundation-run launcher: train the full-v4 recipe, then run the held-out
# evals (synthetic + public ARC-AGI-3). No qualification gates — the point of
# the foundation run is debugging the model, not the launcher.
#
# Usage: scripts/foundation_train.sh <output-dir> [physical-batch] [steps-per-lesson]
# Env:   DEVICE (default cuda), SEED (default 2).
# The live ARC-AGI-3 eval needs ARC_AGI_3_API_KEY in the environment or .env.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

RUN_DIR="${1:?usage: foundation_train.sh <output-dir> [physical-batch] [steps-per-lesson]}"
PHYSICAL_BATCH="${2:-2048}"
STEPS_PER_LESSON="${3:-4096}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-2}"

cargo build --release --features cudnn --locked
BIN=target/release/tofy

"$BIN" p2-train \
  --recipe full-v4 --device "$DEVICE" \
  --seed "$SEED" --init-seed "$SEED" \
  --physical-batch "$PHYSICAL_BATCH" --grad-accum 1 \
  --steps-per-lesson "$STEPS_PER_LESSON" \
  --checkpoint-every-steps 100 \
  --output-dir "$RUN_DIR"

if [[ ! -f "$RUN_DIR/train_report.json" ]] || \
  ! jq -e '.status == "completed"' "$RUN_DIR/train_report.json" >/dev/null; then
  printf 'training did not complete; skipping synthetic and live evals for %s\n' "$RUN_DIR"
  exit 0
fi

# Eval flags mirror the proven full-v4 boundary evals so reports stay comparable.
"$BIN" p2-eval \
  --device "$DEVICE" \
  --checkpoint "$RUN_DIR/model.safetensors" \
  --train-config "$RUN_DIR/config.json" \
  --physical-batch 64 --synthetic-episodes 64 --ptrm-k 1,2,4,8 \
  --seed 1000002 --iid-seed 1000003 \
  --output "$RUN_DIR/eval_report.json"

# Held-out generalization check: live public ARC-AGI-3 games. The model never
# trains on these; p2-train is synthetic-curriculum only (enforced by test).
"$BIN" p2-arc3-live-eval \
  --device "$DEVICE" \
  --checkpoint "$RUN_DIR/model.safetensors" \
  --train-config "$RUN_DIR/config.json" \
  --output "$RUN_DIR/arc3_live_report.json"

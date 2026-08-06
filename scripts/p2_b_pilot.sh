#!/usr/bin/env bash
# Phase B pilot: 256x2 accumulation stress on dynamics-only lesson, then full eval.
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/p2_runs.sh
source "$script_dir/p2_runs.sh"
ROOT="$(cd "$script_dir/.." && pwd)"
cd "$ROOT"
OUT="${1:-$(p2_run_dir b-pilot)}"
mkdir -p "$OUT"
exec >"$OUT/run.log" 2>&1
set -x

cargo build --release --features cuda

cargo run --release --features cuda -- p2-train \
  --device cuda \
  --lessons dynamics \
  --steps-per-lesson 1000 \
  --physical-batch 256 \
  --grad-accum 2 \
  --max-steps-this-run 1000 \
  --checkpoint-every-steps 250 \
  --randomize-depth \
  --residual-y-update \
  --warm-start-y \
  --sigreg-spatial \
  --outer-steps 8 \
  --inner-steps 2 \
  --hidden-dim 128 \
  --output-dir "$OUT"

CKPT_DIR="$(python3 - <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
latest = out / "checkpoints" / "latest.json"
if latest.is_file():
    d = json.loads(latest.read_text())["directory"]
    print(out / "checkpoints" / d / "model.safetensors")
else:
    print(out / "model.safetensors")
PY
"$OUT")"

cargo run --release --features cuda -- p2-eval \
  --checkpoint "$CKPT_DIR" \
  --train-config "$OUT/config.json" \
  --device cuda \
  --seed 2 \
  --synthetic-episodes 16 \
  --physical-batch 8 \
  --output "$OUT/eval_report.json"

echo "Pilot complete: $OUT/eval_report.json"

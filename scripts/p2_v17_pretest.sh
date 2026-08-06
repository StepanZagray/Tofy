#!/usr/bin/env bash
# Quick pre-training experiments (eval sweeps + micro-trains) before v17 overnight runs.
# Called from scripts/p2_v17_night.sh when starting fresh; skipped on resume.
#
# Env: DEVICE (cuda), PHYSICAL_BATCH (512), PRETEST_DIR (default runs/p2/v17-night/pretest)
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
# shellcheck source=scripts/p2_runs.sh
source "$script_dir/p2_runs.sh"
cd -- "$repo_root"

DEVICE="${DEVICE:-cuda}"
PHYSICAL_BATCH="${PHYSICAL_BATCH:-512}"
PRETEST_DIR="${PRETEST_DIR:-$(p2_run_dir v17-night)/pretest}"
V11_CKPT="${V11_CKPT:-$(p2_run_dir v11-control)/model.safetensors}"
V11_CFG="${V11_CFG:-$(p2_run_dir v11-control)/config.json}"
MICRO_STEPS="${MICRO_STEPS:-512}"

log() { printf '[%s] pretest: %s\n' "$(date -Iseconds)" "$*"; }

mkdir -p -- "$PRETEST_DIR"

# E0 — PTRM noise sweep on v11-control (eval-only baseline for noise hypothesis)
run_v11_ptrm_sweep() {
  local out="$PRETEST_DIR/v11_ptrm_sweep.jsonl"
  : >"$out"
  if [[ ! -f "$V11_CKPT" ]]; then
    log "skip v11 PTRM sweep (no checkpoint at $V11_CKPT)"
    return 0
  fi
  log "E0: PTRM noise sweep on v11-control → $out"
  local q noise k
  q="$(python3 -c "import json; print(json.load(open('$V11_CFG'))['q_mse_threshold'])")"
  for noise in 0.003 0.01 0.03 0.1; do
    for k in 2 4 8; do
      local ep="$PRETEST_DIR/eval_v11_noise${noise}_k${k}.json"
      cargo run --release --features cudnn -- p2-eval \
        --device "$DEVICE" --physical-batch 64 \
        --checkpoint "$V11_CKPT" --train-config "$V11_CFG" \
        --synthetic-episodes 64 --ptrm-k "$k" --ptrm-noise "$noise" \
        --q-mse-threshold "$q" --output "$ep"
      python3 - <<PY >>"$out"
import json
r = json.load(open("$ep"))
dyn = r.get("synthetic_dynamics") or {}
ptrm = {x["k"]: x for x in dyn.get("ptrm") or []}
row = ptrm.get($k, {})
print(json.dumps({"noise": $noise, "k": $k, "pass_at_k": row.get("pass_at_k"),
  "disagreement": row.get("disagreement"), "q_oracle_rank": row.get("q_oracle_rank_accuracy")}))
PY
    done
  done
  log "E0 done"
}

# Micro-train ablation: spatial baseline vs residual+warm (512 steps each)
run_micro_train() {
  local name="$1"
  local out_dir="$PRETEST_DIR/micro-$name"
  shift
  local -a extra=("$@")
  log "micro-train $name → $out_dir (${MICRO_STEPS} steps)"
  cargo run --release --features cudnn -- p2-train \
    --device "$DEVICE" \
    --lessons dynamics,sequential \
    --physical-batch "$PHYSICAL_BATCH" --grad-accum 1 \
    --steps-per-lesson 4096 \
    --checkpoint-every-steps 0 \
    --max-steps-this-run "$MICRO_STEPS" \
    --hidden-dim 128 --action-dim 32 \
    --outer-steps 8 --inner-steps 2 \
    --sigreg-weight 0.003 --sigreg-projections 32 \
    --output-dir "$out_dir" \
    "${extra[@]}"
  cargo run --release --features cudnn -- p2-eval \
    --device "$DEVICE" --physical-batch 64 \
    --checkpoint "$out_dir/model.safetensors" \
    --train-config "$out_dir/config.json" \
    --synthetic-episodes 32 --ptrm-k 1,2,4 \
    --q-mse-threshold 0.05 \
    --output "$out_dir/eval_micro.json"
}

run_micro_ablations() {
  log "E1/E2: micro-train ablations (${MICRO_STEPS} steps each)"
  run_micro_train "spatial-base" --randomize-depth
  run_micro_train "residual-warm" \
    --randomize-depth --residual-y-update --warm-start-y --sigreg-spatial
  PRETEST_DIR="$PRETEST_DIR" python3 - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ["PRETEST_DIR"])
out = {}
for name in ("spatial-base", "residual-warm"):
    p = root / f"micro-{name}" / "eval_micro.json"
    if not p.is_file():
        continue
    r = json.loads(p.read_text())
    dyn = r.get("synthetic_dynamics") or {}
    roll = dyn.get("rollout") or {}
    out[name] = {
        "one_step_mse": dyn.get("one_step_latent_mse"),
        "rollout_mse_4": roll.get("mse_4"),
        "rollout_mse_8": roll.get("mse_8"),
    }
(root / "micro_summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PY
  log "micro ablations done → $PRETEST_DIR/micro_summary.json"
}

# Between-run experiments on a finished stable checkpoint (PTRM sweep on trained v17)
run_stable_ptrm_sweep() {
  local stable_dir="$1"
  local ckpt="$stable_dir/model.safetensors"
  [[ -f "$stable_dir/model.best.safetensors" ]] && ckpt="$stable_dir/model.best.safetensors"
  if [[ ! -f "$ckpt" ]]; then
    log "skip stable PTRM sweep (no checkpoint in $stable_dir)"
    return 0
  fi
  local out="$PRETEST_DIR/stable_ptrm_sweep.jsonl"
  log "between-runs: PTRM sweep on $stable_dir → $out"
  : >"$out"
  local q
  q="$(python3 -c "import json; print(json.load(open('${stable_dir}/config.json'))['q_mse_threshold'])")"
  for noise in 0.003 0.01 0.03 0.1; do
    local ep="$PRETEST_DIR/eval_stable_noise${noise}.json"
    cargo run --release --features cudnn -- p2-eval \
      --device "$DEVICE" --physical-batch 64 \
      --checkpoint "$ckpt" --train-config "$stable_dir/config.json" \
      --synthetic-episodes 64 --ptrm-k 1,2,4,8 --ptrm-noise "$noise" \
      --q-mse-threshold "$q" --output "$ep"
    python3 - <<PY >>"$out"
import json
r = json.load(open("$ep"))
dyn = r.get("synthetic_dynamics") or {}
for row in dyn.get("ptrm") or []:
    print(json.dumps({"noise": $noise, "k": row["k"], "pass_at_k": row.get("pass_at_k"),
      "disagreement": row.get("disagreement"), "q_oracle_rank": row.get("q_oracle_rank_accuracy")}))
PY
  done
}

case "${1:-all}" in
  v11-sweep) run_v11_ptrm_sweep ;;
  micro) run_micro_ablations ;;
  stable-sweep) run_stable_ptrm_sweep "${2:?stable dir required}" ;;
  all)
    run_v11_ptrm_sweep
    run_micro_ablations
    ;;
  *) echo "usage: $0 {all|v11-sweep|micro|stable-sweep <dir>}" >&2; exit 1 ;;
esac

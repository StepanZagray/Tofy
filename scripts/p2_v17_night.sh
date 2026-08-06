#!/usr/bin/env bash
# Two sequential overnight P2 runs (v17 stable + experimental Q/noise).
#
# Usage:
#   scripts/p2_v17_night.sh
#   DEVICE=cuda scripts/p2_v17_night.sh
#   SKIP_RUN=1 scripts/p2_v17_night.sh   # compile check only
#
# Env:
#   DEVICE           cuda | cpu (default: cuda)
#   PHYSICAL_BATCH   default 512 (8GB GPU safe max for outer-steps=8 + spatial SIGReg)
#   STEPS_PER_LESSON default 4096
#   RUN_ROOT         default runs/p2/v17-night (logs + summary)
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
# shellcheck source=scripts/p2_runs.sh
source "$script_dir/p2_runs.sh"
cd -- "$repo_root"

DEVICE="${DEVICE:-cuda}"
PHYSICAL_BATCH="${PHYSICAL_BATCH:-512}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
STEPS_PER_LESSON="${STEPS_PER_LESSON:-4096}"
RUN_ROOT="${RUN_ROOT:-$(p2_run_dir v17-night)}"
SKIP_RUN="${SKIP_RUN:-0}"
PRETEST="${PRETEST:-1}"

STABLE_DIR="${STABLE_DIR:-$(p2_run_dir v17-stable)}"
EXP_DIR="${EXP_DIR:-$(p2_run_dir v17-exp-qnoise)}"

LESSONS="${LESSONS:-dynamics,exploration,sequential,q_calibration,falsification}"

mkdir -p -- "$RUN_ROOT"
NIGHT_LOG="$RUN_ROOT/night-$(date +%Y%m%dT%H%M%S).log"

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*"; }

run_pretest_if_fresh() {
  if [[ "$PRETEST" != "1" ]]; then
    log "PRETEST=0 — skipping pre-training experiments"
    return 0
  fi
  if [[ -d "$STABLE_DIR/checkpoints" ]] && [[ -f "$STABLE_DIR/checkpoints/latest.json" ]]; then
    log "stable run in progress/resume (checkpoints present) — skipping pretest"
    return 0
  fi
  log "=== pre-training experiments (before v17-stable) ==="
  DEVICE="$DEVICE" PHYSICAL_BATCH="$PHYSICAL_BATCH" PRETEST_DIR="$RUN_ROOT/pretest" \
    scripts/p2_v17_pretest.sh all
}

run_between_runs_experiments() {
  log "=== between-runs experiments (on v17-stable checkpoint) ==="
  DEVICE="$DEVICE" PRETEST_DIR="$RUN_ROOT/pretest" \
    scripts/p2_v17_pretest.sh stable-sweep "$STABLE_DIR"
}

preflight_build() {
  log "preflight: cargo build --release --features cudnn"
  cargo build --release --features cudnn
}

run_train_eval() {
  local name="$1"
  local output_dir="$2"
  shift 2
  local -a extra=("$@")
  local log_dir="$output_dir/logs"
  mkdir -p -- "$output_dir" "$log_dir"
  local log_file="$log_dir/pipeline-$(date +%Y%m%dT%H%M%S).log"

  log "=== $name → $output_dir ==="
  log "extra args: ${extra[*]}"

  local -a train=(
    cargo run --release --features cudnn -- p2-train
    --device "$DEVICE"
    --lessons "$LESSONS"
    --physical-batch "$PHYSICAL_BATCH"
    --grad-accum "$GRAD_ACCUM"
    --steps-per-lesson "$STEPS_PER_LESSON"
    --checkpoint-every-steps 100
    --output-dir "$output_dir"
    "${extra[@]}"
  )
  if [[ -d "$output_dir/checkpoints" ]] && [[ -f "$output_dir/checkpoints/latest.json" ]]; then
    train+=(--resume "$output_dir/checkpoints")
    log "resuming from $output_dir/checkpoints"
  fi

  {
    "${train[@]}"
    local ckpt q
    if [[ -f "$output_dir/model.best.safetensors" ]]; then
      ckpt="$output_dir/model.best.safetensors"
    else
      ckpt="$output_dir/model.safetensors"
    fi
    q="$(python3 -c "import json; print(json.load(open('${output_dir}/config.json'))['q_mse_threshold'])")"
    cargo run --release --features cudnn -- p2-eval \
      --device "$DEVICE" --physical-batch 64 \
      --checkpoint "$ckpt" \
      --train-config "$output_dir/config.json" \
      --synthetic-episodes 64 --ptrm-k 1,2,4,8 \
      --q-mse-threshold "$q" \
      --output "$output_dir/eval_report_64ep_v5.json"
  } 2>&1 | tee "$log_file"

  log "$name complete → $log_file"
}

write_summary() {
  local summary="$RUN_ROOT/summary.json"
  python3 - <<PY
import json
from pathlib import Path

def load_eval(path):
    p = Path(path)
    if not p.is_file():
        return None
    r = json.loads(p.read_text())
    dyn = r.get("synthetic_dynamics") or {}
    roll = dyn.get("rollout") or {}
    closed = dyn.get("closed_loop") or {}
    qs = dyn.get("q_surprise") or {}
    ident = dyn.get("identifiability") or {}
    ptrm = dyn.get("ptrm") or []
    qrank4 = next((x.get("q_oracle_rank_accuracy") for x in ptrm if x.get("k") == 4), None)
    return {
        "one_step_mse": dyn.get("one_step_latent_mse"),
        "rollout_mse_4": roll.get("mse_4"),
        "rollout_mse_8": roll.get("mse_8"),
        "closed_loop_mse_8": closed.get("mse_8"),
        "confident_error_rate": qs.get("confident_error_rate"),
        "cov_frobenius": ident.get("latent_covariance_frobenius"),
        "q_oracle_rank_at_4": qrank4,
    }

out = {
    "stable": {"dir": "$STABLE_DIR", "metrics": load_eval("$STABLE_DIR/eval_report_64ep_v5.json")},
    "experimental": {"dir": "$EXP_DIR", "metrics": load_eval("$EXP_DIR/eval_report_64ep_v5.json")},
}
Path("$summary").write_text(json.dumps(out, indent=2) + "\n")
print("wrote", "$summary")
PY
}

preflight_build

if [[ "$SKIP_RUN" == "1" ]]; then
  log "SKIP_RUN=1 — build OK, skipping training"
  exit 0
fi

{
  log "v17 overnight: pretest (if fresh) → stable train → between-run eval → exp train"
  log "night log: $NIGHT_LOG"
  log "physical_batch=$PHYSICAL_BATCH grad_accum=$GRAD_ACCUM steps_per_lesson=$STEPS_PER_LESSON"

  run_pretest_if_fresh

  # Run 1 — stability pack from planning session (residual + warm-start + spatial SIGReg + depth rand)
  run_train_eval "v17-stable" "$STABLE_DIR" \
    --hidden-dim 128 --action-dim 32 \
    --outer-steps 8 --inner-steps 2 \
    --sigreg-weight 0.003 --sigreg-projections 32 \
    --randomize-depth \
    --residual-y-update \
    --warm-start-y \
    --sigreg-spatial

  run_between_runs_experiments

  # Run 2 — experimental: Q observer + quantile labels + train-time z-noise (no arch residual flags)
  run_train_eval "v17-exp-qnoise" "$EXP_DIR" \
    --hidden-dim 128 --action-dim 32 \
    --outer-steps 8 --inner-steps 2 \
    --sigreg-weight 0.003 --sigreg-projections 32 \
    --randomize-depth \
    --stop-grad-q-y \
    --q-quantile-targets \
    --train-z-noise 0.03

  write_summary
  log "both runs finished; summary at $RUN_ROOT/summary.json"
} 2>&1 | tee -a "$NIGHT_LOG"

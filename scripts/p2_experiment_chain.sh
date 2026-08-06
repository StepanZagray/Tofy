#!/usr/bin/env bash
# Sequential P2 train→eval experiments with Grok 4.5 repair on failure.
#
# Usage:
#   scripts/p2_experiment_chain.sh run
#   scripts/p2_experiment_chain.sh run --from 3        # skip first two experiments
#   OUTPUT_ROOT=runs/p2/v12 scripts/p2_experiment_chain.sh run
#
# Env:
#   MAX_REPAIR_ATTEMPTS   per-experiment repair loops (default: 3)
#   AGENT_BIN             cursor agent binary (default: cursor-agent)
#   AGENT_MODEL           model slug (default: cursor-grok-4.5-high)
#   DEVICE                cuda | cpu (default: cuda)
#   PHYSICAL_BATCH        default 1024
#   STEPS_PER_LESSON      default 4096
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
# shellcheck source=scripts/p2_runs.sh
source "$script_dir/p2_runs.sh"
cd -- "$repo_root"

OUTPUT_ROOT="${OUTPUT_ROOT:-$(p2_run_dir v12)}"
p2_ensure_runs_root
DEVICE="${DEVICE:-cuda}"
PHYSICAL_BATCH="${PHYSICAL_BATCH:-1024}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
STEPS_PER_LESSON="${STEPS_PER_LESSON:-4096}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
ACTION_DIM="${ACTION_DIM:-32}"
MAX_REPAIR_ATTEMPTS="${MAX_REPAIR_ATTEMPTS:-3}"
AGENT_BIN="${AGENT_BIN:-cursor-agent}"
AGENT_MODEL="${AGENT_MODEL:-cursor-grok-4.5-high}"
FROM_INDEX="${FROM_INDEX:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export TOFY_MALLOC_TRIM_EVERY="${TOFY_MALLOC_TRIM_EVERY:-100}"

LOG_DIR="$OUTPUT_ROOT/chain-logs"
mkdir -p -- "$OUTPUT_ROOT" "$LOG_DIR"

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*"; }

# name | output_subdir | extra train args
read -r -d '' EXPERIMENTS <<'EOF' || true
v12-fix|fix|--lessons dynamics,sequential,q_calibration,falsification,retarget --q-mse-threshold 0.25
v12-h256|exp-h256|--lessons dynamics,sequential,q_calibration,falsification,retarget --q-mse-threshold 0.25 --hidden-dim 256
v12-explore|exp-explore|--lessons dynamics,exploration,sequential,q_calibration,falsification,retarget --q-mse-threshold 0.25
v12-q005|exp-q005|--lessons dynamics,sequential,q_calibration,falsification,retarget --q-mse-threshold 0.05
v12-deep|exp-deep|--lessons dynamics,sequential,q_calibration,falsification,retarget --q-mse-threshold 0.25 --inner-steps 3 --outer-steps 3
v12-h256-explore|exp-h256-explore|--lessons dynamics,exploration,sequential,q_calibration,falsification,retarget --q-mse-threshold 0.25 --hidden-dim 256
v12-stage1b|exp-stage1b|--lessons dynamics,exploration,sequential,q_calibration,falsification,retarget --q-mse-threshold 0.05
EOF

invoke_repair_agent() {
  local exit_code="$1"
  local log_file="$2"
  local attempt="$3"
  local exp_name="$4"
  local output_dir="$5"
  local agent_log="$LOG_DIR/agent-${exp_name}-${attempt}-$(date +%Y%m%dT%H%M%S).log"
  local tail_file="$LOG_DIR/failure-${exp_name}-${attempt}.txt"

  tail -n 400 -- "$log_file" >"$tail_file"
  log "failure in $exp_name exit=$exit_code; repair via $AGENT_BIN model=$AGENT_MODEL (attempt $attempt/$MAX_REPAIR_ATTEMPTS)"
  log "tail: $tail_file agent log: $agent_log"

  local prompt
  prompt="$(cat <<EOF
You are repairing a P2 experiment chain failure in this workspace.

Experiment: $exp_name
Output dir: $output_dir
Exit code: $exit_code
Full log: $log_file
Failure tail: $tail_file
Repo: $repo_root

Hard constraints:
- Fix only what is needed for train/eval to succeed.
- Do not change experiment hyperparameters (batch, steps, lessons, dims) unless the failure is OOM and a smaller batch is required for THIS experiment only — document any change.
- Prefer resuming from $output_dir/checkpoints when present.
- After fixing, run cargo test and cargo clippy --all-targets -- -D warnings (or the smallest check that proves the fix).
- Do not commit or push.

The v12 model uses separate block_z/block_y, dual-pool encoder, delta dynamics + stop-grad targets by default.
Read the log, apply a minimal fix, leave the tree ready for the chain script to retry this experiment.
EOF
)"

  if ! command -v "$AGENT_BIN" >/dev/null 2>&1; then
    log "ERROR: agent binary not found: $AGENT_BIN"
    return 127
  fi

  "$AGENT_BIN" --print --trust --force --sandbox disabled \
    --workspace "$repo_root" \
    --model "$AGENT_MODEL" \
    "$prompt" 2>&1 | tee -a -- "$agent_log"
}

run_train_eval() {
  local output_dir="$1"
  shift
  local -a extra=("$@")
  local extra_join="${extra[*]}"
  mkdir -p -- "$output_dir"

  local -a train_cmd=(
    cargo run --release --features cudnn -- p2-train
    --device "$DEVICE"
    --action-dim "$ACTION_DIM"
    --physical-batch "$PHYSICAL_BATCH"
    --grad-accum "$GRAD_ACCUM"
    --steps-per-lesson "$STEPS_PER_LESSON"
    --checkpoint-every-steps 100
    --output-dir "$output_dir"
  )
  if [[ "$extra_join" != *"--hidden-dim"* ]]; then
    train_cmd+=(--hidden-dim "$HIDDEN_DIM")
  fi
  train_cmd+=("${extra[@]}")

  if [[ -d "$output_dir/checkpoints" ]] || [[ -f "$output_dir/checkpoints/latest.json" ]]; then
    train_cmd+=(--resume "$output_dir/checkpoints")
  fi

  # Fail fast if a stale checkpoint used the wrong hidden dim (common after script fixes).
  if [[ -f "$output_dir/checkpoints/latest.json" ]]; then
    local ckpt_dir
    ckpt_dir="$(python3 -c "import json; print(json.load(open('${output_dir}/checkpoints/latest.json'))['directory'])")"
    local ckpt_hidden
    ckpt_hidden="$(python3 -c "import json; print(json.load(open('${output_dir}/checkpoints/${ckpt_dir}/trainer_state.json'))['contract']['hidden_dim'])")"
    local want_hidden="$HIDDEN_DIM"
    if [[ "$extra_join" == *"--hidden-dim 256"* ]]; then
      want_hidden=256
    fi
    if [[ "$ckpt_hidden" != "$want_hidden" ]]; then
      log "ERROR: $output_dir checkpoint hidden_dim=$ckpt_hidden != expected $want_hidden; remove checkpoints and retry"
      return 2
    fi
  fi

  "${train_cmd[@]}"

  [[ -f "$output_dir/model.safetensors" ]] || {
    log "ERROR: missing $output_dir/model.safetensors after train"
    return 1
  }
  [[ -f "$output_dir/config.json" ]] || {
    log "ERROR: missing $output_dir/config.json after train"
    return 1
  }

  local q_thresh
  q_thresh="$(python3 -c "import json; print(json.load(open('${output_dir}/config.json'))['q_mse_threshold'])")"

  cargo run --release --features cudnn -- p2-eval \
    --device "$DEVICE" \
    --physical-batch 64 \
    --checkpoint "$output_dir/model.safetensors" \
    --train-config "$output_dir/config.json" \
    --synthetic-episodes 64 \
    --ptrm-k 1,2,4,8 \
    --q-mse-threshold "$q_thresh" \
    --output "$output_dir/eval_report_64ep_v3.json"

  [[ -f "$output_dir/eval_report_64ep_v3.json" ]] || {
    log "ERROR: missing eval report for $output_dir"
    return 1
  }
}

run_one_experiment() {
  local index="$1"
  local name="$2"
  local subdir="$3"
  local extra_args="$4"
  local output_dir="$OUTPUT_ROOT/$subdir"
  local log_file="$LOG_DIR/${index}-${name}-$(date +%Y%m%dT%H%M%S).log"

  log "=== experiment $index: $name -> $output_dir ==="
  log "log: $log_file"

  local attempt=0
  while true; do
    set +o pipefail
    set +e
    # shellcheck disable=SC2086
    {
      echo "=== $name start $(date -Iseconds) ==="
      run_train_eval "$output_dir" $extra_args
      echo "=== $name ok $(date -Iseconds) ==="
    } 2>&1 | tee -a -- "$log_file"
    local rc=${PIPESTATUS[0]}
    set -e
    set -o pipefail
    if [[ "$rc" -eq 0 ]]; then
      log "$name completed"
      return 0
    fi

    attempt=$((attempt + 1))
    if [[ "$attempt" -gt "$MAX_REPAIR_ATTEMPTS" ]]; then
      log "ERROR: $name failed after $MAX_REPAIR_ATTEMPTS repairs (exit $rc)"
      return "$rc"
    fi

    set +e
    invoke_repair_agent "$rc" "$log_file" "$attempt" "$name" "$output_dir"
    local agent_rc=$?
    set -e
    if [[ "$agent_rc" -ne 0 ]]; then
      log "ERROR: repair agent failed exit $agent_rc"
      return "$agent_rc"
    fi
    log "repair done; retrying $name"
  done
}

cmd_run() {
  local from="$FROM_INDEX"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --from)
        from="${2:?}"
        shift 2
        ;;
      *)
        log "unknown arg: $1"
        exit 2
        ;;
    esac
  done

  log "chain root=$OUTPUT_ROOT device=$DEVICE from=$from repair_model=$AGENT_MODEL"
  local index=0
  while IFS='|' read -r name subdir extra; do
    [[ -z "$name" ]] && continue
    index=$((index + 1))
    if [[ "$index" -lt "$from" ]]; then
      log "skip $index: $name"
      continue
    fi
    run_one_experiment "$index" "$name" "$subdir" "$extra" || exit $?
  done <<<"$EXPERIMENTS"

  log "all experiments completed"
  write_summary
}

write_summary() {
  local summary="$OUTPUT_ROOT/chain_summary.json"
  python3 - <<'PY' "$OUTPUT_ROOT" "$summary"
import json, glob, os, sys
root, out = sys.argv[1], sys.argv[2]
rows = []
for path in sorted(glob.glob(os.path.join(root, "*/eval_report_64ep_v3.json"))):
    d = json.load(open(path))
    s = d.get("synthetic", {})
    r = s.get("rollout", {})
    i = s.get("identifiability", {})
    q = s.get("q", {})
    rows.append({
        "dir": os.path.dirname(path),
        "one_step_mse": s.get("one_step_latent_mse"),
        "rollout_mse_8": r.get("mse_8"),
        "rollout_mse_4": r.get("mse_4"),
        "events_acc": s.get("events", {}).get("accuracy"),
        "r2_h_to_z": i.get("r2_h_to_z"),
        "q_saturated": q.get("saturated"),
    })
json.dump({"experiments": rows}, open(out, "w"), indent=2)
print(f"wrote {out} ({len(rows)} eval reports)")
PY
}

usage() {
  cat <<EOF
Usage: $0 run [--from N]

Runs v12-fix then 6 experimental train→eval jobs sequentially.
On failure, spawns Cursor CLI agent ($AGENT_MODEL) to repair and retry.

Experiments (in order):
  1 v12-fix          dual-block encoder + delta/stop-grad (v11 curriculum)
  2 v12-h256         hidden_dim=256
  3 v12-explore      +exploration lesson
  4 v12-q005         q_mse_threshold=0.05
  5 v12-deep         inner/outer steps=3
  6 v12-h256-explore 256 + exploration
  7 v12-stage1b      exploration + q=0.05
EOF
}

main() {
  local mode="${1:-}"
  shift || true
  case "$mode" in
    run) cmd_run "$@" ;;
    -h|--help|help|"") usage; exit 2 ;;
    *) log "unknown mode: $mode"; usage; exit 2 ;;
  esac
}

main "$@"

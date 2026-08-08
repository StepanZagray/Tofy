#!/usr/bin/env bash
# Run one preregistered SIGReg/action-conditioning arm to updates 1,000 and 2,000.
# Usage: scripts/p2_sigreg_action_ab.sh <control|projector> <training-seed>
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
ab_root="${P2_AB_ROOT:-$repo_root/runs/p2/ab-sigreg-action-v1}"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
eval_seed="${P2_AB_EVAL_SEED:-424242}"
target_update="${P2_AB_TARGET_UPDATE:-2000}"

arm="${1:?usage: $0 <control|projector> <training-seed>}"
seed="${2:?usage: $0 <control|projector> <training-seed>}"
case "$arm" in
  control|projector) ;;
  *) printf 'invalid arm: %s\n' "$arm" >&2; exit 2 ;;
esac
[[ "$seed" =~ ^[1-9][0-9]*$ ]] || { printf 'seed must be a positive integer\n' >&2; exit 2; }
[[ "$target_update" == 2000 || "$target_update" == 4000 ]] || {
  printf 'P2_AB_TARGET_UPDATE must be 2000 or 4000\n' >&2
  exit 2
}
[[ -x "$tofy_bin" ]] || { printf 'missing reviewed binary: %s\n' "$tofy_bin" >&2; exit 2; }

git_sha="$(git -C "$repo_root" rev-parse HEAD)"
if [[ -n "${P2_EXPECTED_SHA:-}" && "$git_sha" != "$P2_EXPECTED_SHA" ]]; then
  printf 'HEAD %s does not match P2_EXPECTED_SHA %s\n' "$git_sha" "$P2_EXPECTED_SHA" >&2
  exit 2
fi

arm_dir="$ab_root/seed-$seed/$arm"
mkdir -p -- "$arm_dir" "$arm_dir/telemetry"
commands_log="$arm_dir/commands.log"
phase_log="$arm_dir/phases.jsonl"

record_command() {
  local stage="$1"
  shift
  {
    printf '[%s] %s' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$stage"
    printf ' %q' "$@"
    printf '\n'
  } >>"$commands_log"
}

record_phase() {
  local stage="$1" started="$2" finished="$3" status="$4"
  jq -nc \
    --arg stage "$stage" \
    --arg started_utc "$started" \
    --arg finished_utc "$finished" \
    --arg status "$status" \
    --arg git_sha "$git_sha" \
    --arg arm "$arm" \
    --argjson seed "$seed" \
    '{stage:$stage,started_utc:$started_utc,finished_utc:$finished_utc,status:$status,git_sha:$git_sha,arm:$arm,seed:$seed}' \
    >>"$phase_log"
}

run_phase() {
  local stage="$1" log_path="$2"
  shift 2
  local started finished rc
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  record_command "$stage" "$@"
  set +e
  /usr/bin/time -f 'wall_seconds=%e max_rss_kib=%M exit_status=%x' -o "$arm_dir/$stage.time" \
    "$@" > >(tee "$log_path") 2>&1
  rc=$?
  set -e
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  record_phase "$stage" "$started" "$finished" "$rc"
  return "$rc"
}

sample_gpu() {
  while true; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep 15
  done
}

sampler_pid=""
stop_sampler() {
  if [[ -n "$sampler_pid" ]] && kill -0 "$sampler_pid" 2>/dev/null; then
    kill "$sampler_pid"
    wait "$sampler_pid" 2>/dev/null || true
  fi
}
trap stop_sampler EXIT INT TERM
sample_gpu >>"$arm_dir/telemetry/gpu.csv" &
sampler_pid=$!

common_train=(
  "$tofy_bin" p2-train
  --device cuda
  --seed "$seed"
  --lessons dynamics
  # Dynamics receives 2× this base budget: 4,000 total updates are fixed in
  # the exact contract so a preregistered extension needs no schedule mutation.
  --steps-per-lesson 2000
  --physical-batch 1024
  --grad-accum 1
  --checkpoint-every-steps 500
  --randomize-depth
  --supervise-last-outer-only
  --residual-y-update
  --warm-start-y
  --sigreg-spatial
  --sigreg-spatial-pool
  --sigreg-max-rows 32768
  --shuffled-episodes
  --outer-steps 8
  --inner-steps 2
  --hidden-dim 128
  --action-dim 8
  --event-weight 0
  --q-weight 0
  --rollout-weight 0
  --prefix-weight 0
  --reliability-weight 0
  --ensemble-members 1
  --output-dir "$arm_dir"
)
if [[ "$arm" == projector ]]; then
  common_train+=(--sigreg-projector --sigreg-projector-dim 128)
fi

eval_update() {
  local update="$1" checkpoint_dir eval_dir
  printf -v checkpoint_dir '%s/checkpoints/step-%012d' "$arm_dir" "$update"
  printf -v eval_dir '%s/eval-update-%04d' "$arm_dir" "$update"
  if [[ -e "$eval_dir/eval_report.json" ]]; then
    printf 'preserving existing evaluation: %s\n' "$eval_dir/eval_report.json"
    return 0
  fi
  mkdir -p -- "$eval_dir"
  local eval_cmd=(
    "$tofy_bin" p2-eval
    --checkpoint "$checkpoint_dir/model.safetensors"
    --train-config "$arm_dir/config.json"
    --device cuda
    --seed "$eval_seed"
    --synthetic-episodes 64
    --physical-batch 1024
    --ptrm-k 1
    --ptrm-noise 0
    --ensemble-members 1
    --episode-jsonl "$eval_dir/episodes.jsonl"
    --output "$eval_dir/eval_report.json"
  )
  run_phase "eval-$update" "$eval_dir/eval.log" "${eval_cmd[@]}"
  sha256sum \
    "$checkpoint_dir/model.safetensors" \
    "$checkpoint_dir/optimizer.safetensors" \
    "$checkpoint_dir/trainer_state.json" \
    "$arm_dir/config.json" \
    "$eval_dir/eval_report.json" \
    >"$eval_dir/sha256.txt"
}

checkpoint_1000="$arm_dir/checkpoints/step-000000001000/model.safetensors"
if [[ ! -f "$checkpoint_1000" ]]; then
  run_phase train-1000 "$arm_dir/train-1000.log" \
    "${common_train[@]}" --max-steps-this-run 1000
fi
eval_update 1000

checkpoint_2000="$arm_dir/checkpoints/step-000000002000/model.safetensors"
if [[ ! -f "$checkpoint_2000" ]]; then
  run_phase train-2000 "$arm_dir/train-2000.log" \
    "${common_train[@]}" --resume "$arm_dir/checkpoints" --max-steps-this-run 1000
fi
eval_update 2000

if [[ "$target_update" == 4000 ]]; then
  checkpoint_4000="$arm_dir/checkpoints/step-000000004000/model.safetensors"
  if [[ ! -f "$checkpoint_4000" ]]; then
    run_phase train-4000 "$arm_dir/train-4000.log" \
      "${common_train[@]}" --resume "$arm_dir/checkpoints"
  fi
  eval_update 4000
fi

jq -nc \
  --arg schema p2.sigreg_action_arm.v1 \
  --arg git_sha "$git_sha" \
  --arg arm "$arm" \
  --argjson seed "$seed" \
  --argjson eval_seed "$eval_seed" \
  --arg output_dir "$arm_dir" \
  --arg config_sha256 "$(sha256sum "$arm_dir/config.json" | awk '{print $1}')" \
  --argjson complete_through_update "$target_update" \
  '{schema:$schema,git_sha:$git_sha,arm:$arm,seed:$seed,eval_seed:$eval_seed,physical_batch:1024,grad_accum:1,projector_dim:(if $arm=="projector" then 128 else null end),output_dir:$output_dir,config_sha256:$config_sha256,complete_through_update:$complete_through_update}' \
  >"$arm_dir/manifest.json"

stop_sampler
trap - EXIT INT TERM
printf 'completed %s seed %s at %s\n' "$arm" "$seed" "$arm_dir"

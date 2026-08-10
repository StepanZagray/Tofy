#!/usr/bin/env bash
# Phase 1B TC-SIGReg pilot: serialized paired seed-1 control/treatment only.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
candle_root="${P2_CANDLE_ROOT:-$repo_root/../candle_graph}"
run_root="${P2_TC_SIGREG_ROOT:-$repo_root/runs/p2/tc-sigreg-v1}"
physical_batch="${P2_TC_PHYSICAL_BATCH:-512}"
grad_accum="${P2_TC_GRAD_ACCUM:-1}"
eval_batch="${P2_TC_EVAL_BATCH:-$physical_batch}"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"

: "${P2_EXPECTED_SHA:?set P2_EXPECTED_SHA to the reviewed Tofy commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set P2_EXPECTED_CANDLE_SHA to the reviewed candle_graph commit}"
: "${P2_EXPECTED_BINARY_SHA:?set P2_EXPECTED_BINARY_SHA to the reviewed tofy binary hash}"
for command in git jq nvidia-smi sha256sum awk realpath tee; do
  command -v "$command" >/dev/null || { printf 'missing required command: %s\n' "$command" >&2; exit 2; }
done
[[ -x "$tofy_bin" ]] || { printf 'missing reviewed binary: %s\n' "$tofy_bin" >&2; exit 2; }
[[ -d "$candle_root/.git" ]] || { printf 'missing candle_graph checkout: %s\n' "$candle_root" >&2; exit 2; }
[[ "$physical_batch" =~ ^[0-9]+$ ]] && ((10#$physical_batch >= 2)) || {
  printf 'P2_TC_PHYSICAL_BATCH must be an integer >= 2\n' >&2; exit 2;
}
[[ "$grad_accum" =~ ^[1-9][0-9]*$ ]] || { printf 'P2_TC_GRAD_ACCUM must be >= 1\n' >&2; exit 2; }
[[ "$eval_batch" =~ ^[0-9]+$ ]] && ((10#$eval_batch >= 1)) || {
  printf 'P2_TC_EVAL_BATCH must be >= 1\n' >&2; exit 2;
}

git_sha="$(git -C "$repo_root" rev-parse HEAD)"
candle_sha="$(git -C "$candle_root" rev-parse HEAD)"
binary_sha="$(sha256sum "$tofy_bin" | awk '{print $1}')"
[[ "$git_sha" == "$P2_EXPECTED_SHA" ]] || { printf 'Tofy SHA mismatch\n' >&2; exit 2; }
[[ "$candle_sha" == "$P2_EXPECTED_CANDLE_SHA" ]] || { printf 'candle_graph SHA mismatch\n' >&2; exit 2; }
[[ "$binary_sha" == "$P2_EXPECTED_BINARY_SHA" ]] || { printf 'binary SHA mismatch\n' >&2; exit 2; }
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || { printf 'dirty Tofy worktree\n' >&2; exit 2; }
[[ -z "$(git -C "$candle_root" status --porcelain)" ]] || { printf 'dirty candle_graph worktree\n' >&2; exit 2; }

mkdir -p -- "$run_root"
run_root="$(realpath "$run_root")"

sample_gpu() {
  while true; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep "$gpu_interval"
  done
}

sampler_pid=""
stop_sampler() {
  if [[ -n "$sampler_pid" ]] && kill -0 "$sampler_pid" 2>/dev/null; then
    kill "$sampler_pid"
    wait "$sampler_pid" 2>/dev/null || true
  fi
  sampler_pid=""
}
trap stop_sampler EXIT INT TERM

run_arm() {
  local seed="$1" arm="$2" target arm_dir update checkpoint eval_dir
  case "$arm" in
    control) target=marginal ;;
    temporal-residual) target=temporal-residual ;;
    *) printf 'invalid TC-SIGReg arm: %s\n' "$arm" >&2; return 2 ;;
  esac
  arm_dir="$run_root/seed-$seed/$arm"
  mkdir -p -- "$arm_dir/telemetry"
  jq -nc \
    --arg schema p2.tc_sigreg_arm.v1 \
    --arg arm "$arm" --arg target "$target" \
    --arg git_sha "$git_sha" --arg candle_git_sha "$candle_sha" --arg binary_sha256 "$binary_sha" \
    --argjson seed "$seed" --argjson physical_batch "$physical_batch" --argjson grad_accum "$grad_accum" \
    --argjson effective_batch "$((physical_batch * grad_accum))" \
    '{schema:$schema,arm:$arm,sigreg_target:$target,seed:$seed,physical_batch:$physical_batch,grad_accum:$grad_accum,effective_batch:$effective_batch,git_sha:$git_sha,candle_git_sha:$candle_git_sha,binary_sha256:$binary_sha256,checkpoints:[250,500,750,1000],evaluation_updates:[250,500,750,1000]}' \
    >"$arm_dir/run.json"

  sample_gpu >>"$arm_dir/telemetry/gpu.csv" &
  sampler_pid=$!
  "$tofy_bin" p2-train \
    --device cuda --seed "$seed" --lessons sequential --steps-per-lesson 1000 \
    --physical-batch "$physical_batch" --grad-accum "$grad_accum" \
    --checkpoint-every-steps 250 --profile-update 250 \
    --sigreg-target "$target" --sigreg-temporal-window 8 \
    --sigreg-spatial --sigreg-spatial-pool --sigreg-max-rows 32768 \
    --randomize-depth --supervise-last-outer-only --residual-y-update --warm-start-y \
    --outer-steps 8 --inner-steps 2 --hidden-dim 128 --action-dim 8 \
    --event-weight 0 --q-weight 0 --rollout-weight 0 --prefix-weight 0 --reliability-weight 0 \
    --ensemble-members 1 --output-dir "$arm_dir" \
    > >(tee "$arm_dir/train.log") 2>&1
  # Evaluation is intentionally after the trainer exits: no concurrent GPU-heavy process.
  for update in 250 500 750 1000; do
    printf -v checkpoint '%s/checkpoints/step-%012d/model.safetensors' "$arm_dir" "$update"
    eval_dir="$arm_dir/eval-update-$update"
    [[ -s "$checkpoint" ]] || { printf 'missing checkpoint: %s\n' "$checkpoint" >&2; return 1; }
    mkdir -p -- "$eval_dir"
    "$tofy_bin" p2-eval --checkpoint "$checkpoint" --train-config "$arm_dir/config.json" \
      --device cuda --seed 424242 --synthetic-episodes 64 --physical-batch "$eval_batch" \
      --ptrm-k 1 --ptrm-noise 0 --ensemble-members 1 \
      --episode-jsonl "$eval_dir/episodes.jsonl" --output "$eval_dir/eval_report.json" \
      > >(tee "$eval_dir/eval.log") 2>&1
    sha256sum "$eval_dir/eval_report.json" >"$eval_dir/sha256.txt"
  done
  stop_sampler
}

# The two seed-1 arms are always serialized, preserving matched physical batch,
# accumulation, ordered-window count, and one encoder-pair call per microbatch.
run_arm 1 control
run_arm 1 temporal-residual

# Replication is opt-in and cannot start without an external gate result written by
# the preregistered analysis. The default exits here, before seeds 2/3.
if [[ "${P2_TC_RUN_SEEDS_2_3:-0}" != 1 ]]; then
  printf 'seed-1 pilot complete; seeds 2/3 locked pending explicit gate result\n'
  exit 0
fi
gate_result="${P2_TC_GATE_RESULT:?set P2_TC_GATE_RESULT to a passed seed-1 gate JSON}"
jq -e '.schema == "p2.tc_sigreg_gate.v1" and .status == "passed" and .promotion == "run_seeds_2_3"' \
  "$gate_result" >/dev/null || { printf 'seed-1 gate did not authorize seeds 2/3\n' >&2; exit 2; }
for seed in 2 3; do
  run_arm "$seed" control
  run_arm "$seed" temporal-residual
done

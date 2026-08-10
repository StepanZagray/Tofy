#!/usr/bin/env bash
# Phase 1B TC-SIGReg pilot: serialized paired seed-1 control/treatment only.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
candle_root="${P2_CANDLE_ROOT:-$repo_root/../candle_graph}"
run_root="${P2_TC_SIGREG_ROOT:-$repo_root/runs/p2/tc-sigreg-v1}"
eval_batch="${P2_TC_EVAL_BATCH:-1024}"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"

: "${P2_EXPECTED_SHA:?set P2_EXPECTED_SHA to the reviewed Tofy commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set P2_EXPECTED_CANDLE_SHA to the reviewed candle_graph commit}"
: "${P2_EXPECTED_BINARY_SHA:?set P2_EXPECTED_BINARY_SHA to the reviewed tofy binary hash}"
: "${P2_TC_PHYSICAL_BATCH:?set P2_TC_PHYSICAL_BATCH from the passed A40 probe}"
: "${P2_TC_GRAD_ACCUM:?set P2_TC_GRAD_ACCUM from the passed A40 probe}"
: "${P2_TC_BATCH_PROBE:?set P2_TC_BATCH_PROBE to the passed probe.json}"
physical_batch="$P2_TC_PHYSICAL_BATCH"
grad_accum="$P2_TC_GRAD_ACCUM"

for command in git jq nvidia-smi sha256sum awk realpath tee; do
  command -v "$command" >/dev/null || { printf 'missing required command: %s\n' "$command" >&2; exit 2; }
done
[[ -x "$tofy_bin" ]] || { printf 'missing reviewed binary: %s\n' "$tofy_bin" >&2; exit 2; }
[[ -d "$candle_root/.git" ]] || { printf 'missing candle_graph checkout: %s\n' "$candle_root" >&2; exit 2; }
[[ "$physical_batch" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid physical batch\n' >&2; exit 2; }
[[ "$grad_accum" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid gradient accumulation\n' >&2; exit 2; }
[[ "$eval_batch" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid evaluation batch\n' >&2; exit 2; }
((physical_batch % 8 == 0)) || { printf 'physical batch must contain complete W=8 windows\n' >&2; exit 2; }
((physical_batch * grad_accum == 1024)) || {
  printf 'physical_batch * grad_accum must preserve effective batch 1024\n' >&2; exit 2;
}

git_sha="$(git -C "$repo_root" rev-parse HEAD)"
candle_sha="$(git -C "$candle_root" rev-parse HEAD)"
binary_sha="$(sha256sum "$tofy_bin" | awk '{print $1}')"
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#gpu_names[@]} == 1 )) || { printf 'pilot requires exactly one visible physical GPU\n' >&2; exit 2; }
gpu_name="${gpu_names[0]}"
[[ "$git_sha" == "$P2_EXPECTED_SHA" ]] || { printf 'Tofy SHA mismatch\n' >&2; exit 2; }
[[ "$candle_sha" == "$P2_EXPECTED_CANDLE_SHA" ]] || { printf 'candle_graph SHA mismatch\n' >&2; exit 2; }
[[ "$binary_sha" == "$P2_EXPECTED_BINARY_SHA" ]] || { printf 'binary SHA mismatch\n' >&2; exit 2; }
[[ "$gpu_name" == "NVIDIA A40" ]] || { printf 'pilot requires NVIDIA A40, found %s\n' "$gpu_name" >&2; exit 2; }
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || { printf 'dirty Tofy worktree\n' >&2; exit 2; }
[[ -z "$(git -C "$candle_root" status --porcelain)" ]] || { printf 'dirty candle_graph worktree\n' >&2; exit 2; }
[[ -s "$P2_TC_BATCH_PROBE" ]] || { printf 'missing batch probe: %s\n' "$P2_TC_BATCH_PROBE" >&2; exit 2; }
probe_dir="$(cd -- "$(dirname -- "$P2_TC_BATCH_PROBE")" && pwd)"
probe_manifest="$probe_dir/artifacts.sha256"
[[ -s "$probe_manifest" ]] || { printf 'missing probe artifact manifest\n' >&2; exit 2; }
required_probe_artifacts=(
  "$probe_dir/run/config.json"
  "$probe_dir/run/train_report.json"
  "$probe_dir/run/checkpoints/step-000000000002/model.safetensors"
  "$probe_dir/run/checkpoints/step-000000000002/optimizer.safetensors"
  "$probe_dir/run/checkpoints/step-000000000002/trainer_state.json"
)
mapfile -t manifest_paths < <(awk '{print $2}' "$probe_manifest")
(( ${#manifest_paths[@]} == ${#required_probe_artifacts[@]} )) || {
  printf 'probe artifact manifest has the wrong number of entries\n' >&2; exit 2;
}
for index in "${!required_probe_artifacts[@]}"; do
  [[ "${manifest_paths[$index]}" == "${required_probe_artifacts[$index]}" ]] || {
    printf 'probe artifact manifest entry %s is not the required artifact\n' "$index" >&2; exit 2;
  }
done
(cd / && sha256sum --quiet -c "$probe_manifest") || { printf 'probe artifact verification failed\n' >&2; exit 2; }
probe_manifest_sha="$(sha256sum "$probe_manifest" | awk '{print $1}')"
jq -e \
  --arg git_sha "$git_sha" --arg candle_sha "$candle_sha" --arg binary_sha "$binary_sha" \
  --arg manifest_sha "$probe_manifest_sha" \
  --argjson physical "$physical_batch" --argjson accum "$grad_accum" \
  '.schema == "p2.tc_sigreg_batch_probe.v1" and .status == "passed"
    and .git_sha == $git_sha and .candle_git_sha == $candle_sha
    and .binary_sha256 == $binary_sha and .physical_batch == $physical
    and .grad_accum == $accum and .effective_batch == 1024
    and .gpu_name == "NVIDIA A40" and .artifact_manifest_sha256 == $manifest_sha
    and .sigreg_target == "temporal_residual" and .sigreg_temporal_window == 8
    and .completed_updates >= 2' "$P2_TC_BATCH_PROBE" >/dev/null || {
  printf 'batch probe does not authorize this exact experiment\n' >&2; exit 2;
}
probe_sha="$(sha256sum "$P2_TC_BATCH_PROBE" | awk '{print $1}')"

run_parent="$(dirname -- "$run_root")"
mkdir -p -- "$run_parent"
mkdir -- "$run_root" || {
  printf 'pilot run root already exists; refusing mixed or relabeled artifacts: %s\n' "$run_root" >&2
  exit 2
}
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
  local arm="$1" target arm_dir update checkpoint eval_dir started finished
  case "$arm" in
    control) target=marginal ;;
    temporal-residual) target=temporal-residual ;;
    *) printf 'invalid TC-SIGReg arm: %s\n' "$arm" >&2; return 2 ;;
  esac
  arm_dir="$run_root/seed-1/$arm"
  mkdir -p -- "$arm_dir/telemetry"
  jq -nc \
    --arg schema p2.tc_sigreg_arm.v1 --arg arm "$arm" --arg target "$target" \
    --arg git_sha "$git_sha" --arg candle_git_sha "$candle_sha" \
    --arg binary_sha256 "$binary_sha" --arg batch_probe_sha256 "$probe_sha" \
    --argjson physical_batch "$physical_batch" --argjson grad_accum "$grad_accum" \
    '{schema:$schema,arm:$arm,sigreg_target:$target,seed:1,physical_batch:$physical_batch,
      grad_accum:$grad_accum,effective_batch:1024,git_sha:$git_sha,candle_git_sha:$candle_git_sha,
      binary_sha256:$binary_sha256,batch_probe_sha256:$batch_probe_sha256,
      checkpoints:[250,500,750,1000],evaluation_updates:[250,500,750,1000]}' \
    >"$arm_dir/run.json.tmp"
  mv -- "$arm_dir/run.json.tmp" "$arm_dir/run.json"

  sample_gpu >>"$arm_dir/telemetry/gpu.csv" &
  sampler_pid=$!
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$tofy_bin" p2-train \
    --device cuda:0 --seed 1 --lessons sequential --steps-per-lesson 1000 \
    --physical-batch "$physical_batch" --grad-accum "$grad_accum" \
    --checkpoint-every-steps 250 --profile-update 250 \
    --sigreg-target "$target" --sigreg-temporal-window 8 \
    --sigreg-spatial --sigreg-spatial-pool --sigreg-max-rows 32768 \
    --randomize-depth --supervise-last-outer-only --residual-y-update --warm-start-y \
    --outer-steps 8 --inner-steps 2 --hidden-dim 128 --action-dim 8 \
    --event-weight 0 --q-weight 0 --rollout-weight 0 --prefix-weight 0 --reliability-weight 0 \
    --ensemble-members 1 --output-dir "$arm_dir" \
    > >(tee "$arm_dir/train.log") 2>&1
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  jq -nc --arg stage train --arg started_utc "$started" --arg finished_utc "$finished" \
    '{stage:$stage,started_utc:$started_utc,finished_utc:$finished_utc,status:"passed"}' \
    >>"$arm_dir/phases.jsonl"

  # Evaluation starts only after the trainer exits; every GPU-heavy process is serialized.
  for update in 250 500 750 1000; do
    printf -v checkpoint '%s/checkpoints/step-%012d/model.safetensors' "$arm_dir" "$update"
    eval_dir="$arm_dir/eval-update-$update"
    [[ -s "$checkpoint" ]] || { printf 'missing checkpoint: %s\n' "$checkpoint" >&2; return 1; }
    mkdir -p -- "$eval_dir"
    started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "$tofy_bin" p2-eval --checkpoint "$checkpoint" --train-config "$arm_dir/config.json" \
      --device cuda:0 --seed 424242 --synthetic-episodes 64 --physical-batch "$eval_batch" \
      --ptrm-k 1 --ptrm-noise 0 --ensemble-members 1 \
      --episode-jsonl "$eval_dir/episodes.jsonl" --output "$eval_dir/eval_report.json" \
      > >(tee "$eval_dir/eval.log") 2>&1
    finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sha256sum "$checkpoint" "$arm_dir/config.json" "$eval_dir/eval_report.json" \
      "$eval_dir/episodes.jsonl" >"$eval_dir/sha256.txt"
    jq -nc --arg stage "eval-$update" --arg started_utc "$started" --arg finished_utc "$finished" \
      '{stage:$stage,started_utc:$started_utc,finished_utc:$finished_utc,status:"passed"}' \
      >>"$arm_dir/phases.jsonl"
  done
  stop_sampler
}

run_arm control
run_arm temporal-residual
jq -nc --arg schema p2.tc_sigreg_pilot.v1 --arg git_sha "$git_sha" \
  --arg binary_sha256 "$binary_sha" --arg batch_probe_sha256 "$probe_sha" \
  '{schema:$schema,status:"seed_1_complete",promotion:"locked_pending_gate_analysis",seed:1,
    arms:["control","temporal-residual"],git_sha:$git_sha,binary_sha256:$binary_sha256,
    batch_probe_sha256:$batch_probe_sha256}' >"$run_root/pilot.json.tmp"
mv -- "$run_root/pilot.json.tmp" "$run_root/pilot.json"
printf 'seed-1 pilot complete; seeds 2/3 are not runnable from this launcher\n'

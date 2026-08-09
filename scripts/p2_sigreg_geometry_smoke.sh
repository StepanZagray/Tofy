#!/usr/bin/env bash
# Exercise training, checkpoint loading, and rollout evaluation for both geometry arms.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
smoke_root="${P2_SMOKE_ROOT:-$repo_root/runs/p2/smoke-sigreg-geometry}"
seed="${P2_SMOKE_SEED:-991}"
train_batch="${P2_SMOKE_TRAIN_BATCH:-1024}"
eval_batch="${P2_SMOKE_EVAL_BATCH:-1024}"
eval_episodes="${P2_SMOKE_EVAL_EPISODES:-64}"

[[ -x "$tofy_bin" ]] || { printf 'missing smoke binary: %s\n' "$tofy_bin" >&2; exit 2; }
[[ "$seed" =~ ^[1-9][0-9]*$ ]] || { printf 'P2_SMOKE_SEED must be positive\n' >&2; exit 2; }
[[ "$train_batch" =~ ^[1-9][0-9]*$ ]] || { printf 'P2_SMOKE_TRAIN_BATCH must be positive\n' >&2; exit 2; }
[[ "$eval_batch" =~ ^[1-9][0-9]*$ ]] || { printf 'P2_SMOKE_EVAL_BATCH must be positive\n' >&2; exit 2; }
[[ "$eval_episodes" =~ ^[1-9][0-9]*$ ]] || { printf 'P2_SMOKE_EVAL_EPISODES must be positive\n' >&2; exit 2; }
for required in jq sha256sum awk; do
  command -v "$required" >/dev/null || { printf 'missing required command: %s\n' "$required" >&2; exit 2; }
done
mkdir -p -- "$smoke_root"
binary_sha256="$(sha256sum "$tofy_bin" | awk '{print $1}')"
if [[ -n "${P2_EXPECTED_BINARY_SHA:-}" && "$binary_sha256" != "$P2_EXPECTED_BINARY_SHA" ]]; then
  printf 'smoke binary SHA-256 %s does not match P2_EXPECTED_BINARY_SHA %s\n' \
    "$binary_sha256" "$P2_EXPECTED_BINARY_SHA" >&2
  exit 2
fi
summary="$smoke_root/summary.json"
if [[ -s "$summary" \
  && -s "$smoke_root/control/eval/eval_report.json" \
  && -s "$smoke_root/pre-rms-spatial/eval/eval_report.json" ]] \
  && jq -e \
    --arg binary_sha256 "$binary_sha256" \
    --argjson seed "$seed" \
    --argjson train_batch "$train_batch" \
    --argjson eval_batch "$eval_batch" \
    --argjson eval_episodes "$eval_episodes" \
    --arg control_report_sha256 "$(sha256sum "$smoke_root/control/eval/eval_report.json" | awk '{print $1}')" \
    --arg treatment_report_sha256 "$(sha256sum "$smoke_root/pre-rms-spatial/eval/eval_report.json" | awk '{print $1}')" \
    '.schema == "p2.sigreg_geometry_smoke.v1"
      and .status == "passed"
      and .binary_sha256 == $binary_sha256
      and .seed == $seed
      and .train_batch == $train_batch
      and .eval_batch == $eval_batch
      and .eval_episodes == $eval_episodes
      and .report_sha256.control == $control_report_sha256
      and .report_sha256["pre-rms-spatial"] == $treatment_report_sha256' \
    "$summary" >/dev/null; then
  printf 'preserving verified geometry smoke: %s\n' "$smoke_root"
  exit 0
fi

run_arm() {
  local arm="$1" arm_dir checkpoint eval_dir
  arm_dir="$smoke_root/$arm"
  checkpoint="$arm_dir/checkpoints/step-000000000004/model.safetensors"
  eval_dir="$arm_dir/eval"
  mkdir -p -- "$arm_dir" "$eval_dir"
  local train_cmd=(
    "$tofy_bin" p2-train
    --device cuda
    --seed "$seed"
    --lessons dynamics
    --steps-per-lesson 2000
    --max-steps-this-run 4
    --physical-batch "$train_batch"
    --grad-accum 1
    --checkpoint-every-steps 2
    --profile-update 2
    --randomize-depth
    --supervise-last-outer-only
    --residual-y-update
    --warm-start-y
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
  if [[ "$arm" == control ]]; then
    train_cmd+=(--sigreg-spatial --sigreg-spatial-pool)
  else
    train_cmd+=(--sigreg-pre-rms-spatial)
  fi
  "${train_cmd[@]}" >"$arm_dir/train.log" 2>&1
  [[ -f "$checkpoint" ]] || { printf 'smoke checkpoint missing: %s\n' "$checkpoint" >&2; exit 1; }
  "$tofy_bin" p2-eval \
    --checkpoint "$checkpoint" \
    --train-config "$arm_dir/config.json" \
    --device cuda \
    --seed 424242 \
    --synthetic-episodes "$eval_episodes" \
    --physical-batch "$eval_batch" \
    --ptrm-k 1 \
    --ptrm-noise 0 \
    --ensemble-members 1 \
    --episode-jsonl "$eval_dir/episodes.jsonl" \
    --output "$eval_dir/eval_report.json" \
    >"$eval_dir/eval.log" 2>&1
  [[ -s "$eval_dir/eval_report.json" ]] || {
    printf 'smoke evaluation report missing: %s\n' "$eval_dir/eval_report.json" >&2
    exit 1
  }
}

run_arm control
run_arm pre-rms-spatial
jq -nc \
  --arg schema p2.sigreg_geometry_smoke.v1 \
  --arg binary_sha256 "$binary_sha256" \
  --argjson seed "$seed" \
  --argjson train_batch "$train_batch" \
  --argjson eval_batch "$eval_batch" \
  --argjson eval_episodes "$eval_episodes" \
  --arg control_report_sha256 "$(sha256sum "$smoke_root/control/eval/eval_report.json" | awk '{print $1}')" \
  --arg treatment_report_sha256 "$(sha256sum "$smoke_root/pre-rms-spatial/eval/eval_report.json" | awk '{print $1}')" \
  '{schema:$schema,status:"passed",binary_sha256:$binary_sha256,seed:$seed,train_batch:$train_batch,eval_batch:$eval_batch,eval_episodes:$eval_episodes,arms:["control","pre-rms-spatial"],report_sha256:{control:$control_report_sha256,"pre-rms-spatial":$treatment_report_sha256}}' \
  >"$summary"
printf 'geometry smoke passed: %s\n' "$smoke_root"

#!/usr/bin/env bash
# Seed-1 dual-scale TC-SIGReg dose-response. Never promotes seeds automatically.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
candle_root="${P2_CANDLE_ROOT:-$repo_root/../candle_graph}"
run_root="${P2_TC_GLOBAL_ROOT:-$repo_root/runs/p2/tc-global-mix-v1}"
eval_batch="${P2_TC_EVAL_BATCH:-1024}"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"

: "${P2_EXPECTED_SHA:?set reviewed Tofy commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set reviewed candle_graph commit}"
: "${P2_EXPECTED_BINARY_SHA:?set reviewed tofy binary hash}"
: "${P2_TC_PHYSICAL_BATCH:?set physical batch from the passed A40 probe}"
: "${P2_TC_GRAD_ACCUM:?set accumulation from the passed A40 probe}"
: "${P2_TC_BATCH_PROBE:?set path to the passed worst-case mix probe.json}"
physical_batch="$P2_TC_PHYSICAL_BATCH"
grad_accum="$P2_TC_GRAD_ACCUM"

for command in git jq nvidia-smi sha256sum awk realpath tee; do
  command -v "$command" >/dev/null || { printf 'missing required command: %s\n' "$command" >&2; exit 2; }
done
[[ -x "$tofy_bin" ]] || { printf 'missing reviewed binary: %s\n' "$tofy_bin" >&2; exit 2; }
[[ -d "$candle_root/.git" ]] || { printf 'missing candle_graph checkout: %s\n' "$candle_root" >&2; exit 2; }
[[ "$physical_batch" =~ ^[1-9][0-9]*$ && "$grad_accum" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$eval_batch" =~ ^[1-9][0-9]*$ ]] || exit 2
((physical_batch % 8 == 0)) || { printf 'physical batch must contain complete W=8 windows\n' >&2; exit 2; }
((physical_batch * grad_accum == 1024)) || {
  printf 'effective batch must remain 1024\n' >&2; exit 2;
}

git_sha="$(git -C "$repo_root" rev-parse HEAD)"
candle_sha="$(git -C "$candle_root" rev-parse HEAD)"
binary_sha="$(sha256sum "$tofy_bin" | awk '{print $1}')"
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#gpu_names[@]} == 1 )) || { printf 'campaign requires one visible GPU\n' >&2; exit 2; }
gpu_name="${gpu_names[0]}"
[[ "$git_sha" == "$P2_EXPECTED_SHA" && "$candle_sha" == "$P2_EXPECTED_CANDLE_SHA" \
  && "$binary_sha" == "$P2_EXPECTED_BINARY_SHA" ]] || { printf 'reviewed SHA mismatch\n' >&2; exit 2; }
[[ "$gpu_name" == "NVIDIA A40" ]] || { printf 'campaign requires NVIDIA A40, found %s\n' "$gpu_name" >&2; exit 2; }
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || { printf 'dirty Tofy worktree\n' >&2; exit 2; }
[[ -z "$(git -C "$candle_root" status --porcelain)" ]] || { printf 'dirty candle_graph worktree\n' >&2; exit 2; }

probe_dir="$(cd -- "$(dirname -- "$P2_TC_BATCH_PROBE")" && pwd)"
probe_manifest="$probe_dir/artifacts.sha256"
[[ -s "$P2_TC_BATCH_PROBE" && -s "$probe_manifest" ]] || { printf 'missing probe evidence\n' >&2; exit 2; }
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
    and .steady_gpu == true and .sigreg_target == "temporal_residual"
    and .sigreg_temporal_window == 8 and .sigreg_global_mix == 0.5
    and .completed_updates >= 2' "$P2_TC_BATCH_PROBE" >/dev/null || {
  printf 'probe does not authorize the worst-case mixed objective\n' >&2; exit 2;
}
probe_sha="$(sha256sum "$P2_TC_BATCH_PROBE" | awk '{print $1}')"

mkdir -p -- "$(dirname -- "$run_root")"
mkdir -- "$run_root" || { printf 'run root already exists: %s\n' "$run_root" >&2; exit 2; }
run_root="$(realpath "$run_root")"
campaign_started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
jq -nc --arg schema p2.tc_global_mix_campaign.v1 --arg status running \
  --arg started_utc "$campaign_started" --arg git_sha "$git_sha" \
  --arg candle_git_sha "$candle_sha" --arg binary_sha256 "$binary_sha" \
  --arg batch_probe_sha256 "$probe_sha" \
  '{schema:$schema,status:$status,started_utc:$started_utc,seed:1,
    arms:["marginal-control","tc-cell","tc-mix-025","tc-mix-050","tc-global"],
    git_sha:$git_sha,candle_git_sha:$candle_git_sha,binary_sha256:$binary_sha256,
    batch_probe_sha256:$batch_probe_sha256,promotion:"locked_pending_gate_analysis"}' \
  >"$run_root/campaign.json"

sample_gpu() {
  while true; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep "$gpu_interval"
  done
}

run_arm() {
  local arm="$1" target="$2" mix="$3" arm_dir update checkpoint eval_dir started finished sampler_pid
  arm_dir="$run_root/seed-1/$arm"
  mkdir -p -- "$arm_dir/telemetry"
  jq -nc --arg schema p2.tc_global_mix_arm.v1 --arg arm "$arm" --arg target "$target" \
    --arg git_sha "$git_sha" --arg candle_git_sha "$candle_sha" \
    --arg binary_sha256 "$binary_sha" --arg batch_probe_sha256 "$probe_sha" \
    --argjson mix "$mix" --argjson physical_batch "$physical_batch" --argjson grad_accum "$grad_accum" \
    '{schema:$schema,arm:$arm,sigreg_target:$target,sigreg_global_mix:$mix,seed:1,
      physical_batch:$physical_batch,grad_accum:$grad_accum,effective_batch:1024,
      git_sha:$git_sha,candle_git_sha:$candle_git_sha,binary_sha256:$binary_sha256,
      batch_probe_sha256:$batch_probe_sha256,checkpoints:[250,500,750,1000],
      evaluation_updates:[250,500,750,1000]}' >"$arm_dir/run.json"

  sample_gpu >>"$arm_dir/telemetry/gpu.csv" &
  sampler_pid=$!
  trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT INT TERM
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$tofy_bin" p2-train \
    --device cuda:0 --seed 1 --lessons sequential --steps-per-lesson 1000 \
    --physical-batch "$physical_batch" --grad-accum "$grad_accum" \
    --checkpoint-every-steps 250 --profile-update 250 \
    --sigreg-target "$target" --sigreg-temporal-window 8 --sigreg-global-mix "$mix" \
    --sigreg-spatial --sigreg-spatial-pool --sigreg-max-rows 32768 \
    --randomize-depth --supervise-last-outer-only --residual-y-update --warm-start-y \
    --outer-steps 8 --inner-steps 2 --hidden-dim 128 --action-dim 8 \
    --event-weight 0 --q-weight 0 --rollout-weight 0 --prefix-weight 0 --reliability-weight 0 \
    --ensemble-members 1 --output-dir "$arm_dir" > >(tee "$arm_dir/train.log") 2>&1
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  jq -nc --arg stage train --arg started_utc "$started" --arg finished_utc "$finished" \
    '{stage:$stage,started_utc:$started_utc,finished_utc:$finished_utc,status:"passed"}' \
    >>"$arm_dir/phases.jsonl"

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
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  trap - EXIT INT TERM
}

arms=(
  'marginal-control marginal 0'
  'tc-cell temporal-residual 0'
  'tc-mix-025 temporal-residual 0.25'
  'tc-mix-050 temporal-residual 0.5'
  'tc-global temporal-residual 1.0'
)
failed=0
for definition in "${arms[@]}"; do
  read -r arm target mix <<<"$definition"
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if (set -euo pipefail; run_arm "$arm" "$target" "$mix"); then
    status=passed
  else
    status=failed
    failed=$((failed + 1))
  fi
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  jq -nc --arg arm "$arm" --arg status "$status" --arg started_utc "$started" \
    --arg finished_utc "$finished" \
    '{arm:$arm,status:$status,started_utc:$started_utc,finished_utc:$finished_utc}' \
    >>"$run_root/arms.jsonl"
done

campaign_finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
jq --arg status "$([[ "$failed" -eq 0 ]] && printf complete || printf partial_failure)" \
  --arg finished_utc "$campaign_finished" --argjson failed_arms "$failed" \
  '.status=$status | .finished_utc=$finished_utc | .failed_arms=$failed_arms' \
  "$run_root/campaign.json" >"$run_root/campaign.json.tmp"
mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"
printf 'campaign finished with %s failed arms; promotion remains locked\n' "$failed"
exit "$([[ "$failed" -eq 0 ]] && printf 0 || printf 1)"

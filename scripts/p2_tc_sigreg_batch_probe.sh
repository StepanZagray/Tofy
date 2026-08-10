#!/usr/bin/env bash
# Measure one candidate A40 physical-batch/accumulation pair for the TC-SIGReg pilot.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
candle_root="${P2_CANDLE_ROOT:-$repo_root/../candle_graph}"
: "${P2_EXPECTED_SHA:?set P2_EXPECTED_SHA}"
: "${P2_EXPECTED_CANDLE_SHA:?set P2_EXPECTED_CANDLE_SHA}"
: "${P2_EXPECTED_BINARY_SHA:?set P2_EXPECTED_BINARY_SHA}"
: "${P2_TC_PHYSICAL_BATCH:?set candidate physical batch}"
: "${P2_TC_GRAD_ACCUM:?set candidate gradient accumulation}"
physical_batch="$P2_TC_PHYSICAL_BATCH"
grad_accum="$P2_TC_GRAD_ACCUM"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-1}"

[[ "$physical_batch" =~ ^[1-9][0-9]*$ && "$grad_accum" =~ ^[1-9][0-9]*$ ]] || exit 2
((physical_batch % 8 == 0)) || { printf 'physical batch must be divisible by W=8\n' >&2; exit 2; }
((physical_batch * grad_accum == 1024)) || { printf 'effective batch must remain 1024\n' >&2; exit 2; }
git_sha="$(git -C "$repo_root" rev-parse HEAD)"
candle_sha="$(git -C "$candle_root" rev-parse HEAD)"
binary_sha="$(sha256sum "$tofy_bin" | awk '{print $1}')"
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#gpu_names[@]} == 1 )) || { printf 'probe requires exactly one visible physical GPU\n' >&2; exit 2; }
gpu_name="${gpu_names[0]}"
[[ "$git_sha" == "$P2_EXPECTED_SHA" && "$candle_sha" == "$P2_EXPECTED_CANDLE_SHA" \
  && "$binary_sha" == "$P2_EXPECTED_BINARY_SHA" ]] || { printf 'reviewed SHA mismatch\n' >&2; exit 2; }
[[ "$gpu_name" == "NVIDIA A40" ]] || { printf 'probe requires NVIDIA A40, found %s\n' "$gpu_name" >&2; exit 2; }
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || { printf 'dirty Tofy worktree\n' >&2; exit 2; }
[[ -z "$(git -C "$candle_root" status --porcelain)" ]] || { printf 'dirty candle_graph worktree\n' >&2; exit 2; }

probe_root="${P2_TC_BATCH_PROBE_ROOT:-$repo_root/runs/p2/tc-sigreg-batch-probe/${git_sha:0:12}/batch-$physical_batch-accum-$grad_accum}"
probe_parent="$(dirname -- "$probe_root")"
mkdir -p -- "$probe_parent"
mkdir -- "$probe_root" || { printf 'probe root already exists: %s\n' "$probe_root" >&2; exit 2; }
mkdir -- "$probe_root/telemetry"
probe_root="$(realpath "$probe_root")"
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
}
trap stop_sampler EXIT INT TERM
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits >"$probe_root/telemetry/before.csv"
sample_gpu >>"$probe_root/telemetry/gpu.csv" &
sampler_pid=$!
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$tofy_bin" p2-train \
  --device cuda:0 --seed 1 --lessons sequential --steps-per-lesson 1000 --max-steps-this-run 2 \
  --physical-batch "$physical_batch" --grad-accum "$grad_accum" \
  --checkpoint-every-steps 2 --profile-update 2 \
  --sigreg-target temporal-residual --sigreg-temporal-window 8 \
  --sigreg-spatial --sigreg-spatial-pool --sigreg-max-rows 32768 \
  --randomize-depth --supervise-last-outer-only --residual-y-update --warm-start-y \
  --outer-steps 8 --inner-steps 2 --hidden-dim 128 --action-dim 8 \
  --event-weight 0 --q-weight 0 --rollout-weight 0 --prefix-weight 0 --reliability-weight 0 \
  --ensemble-members 1 --output-dir "$probe_root/run" > >(tee "$probe_root/train.log") 2>&1
finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
stop_sampler
sampler_pid=""
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits >"$probe_root/telemetry/after.csv"
completed="$(jq -r '.global_step' "$probe_root/run/train_report.json")"
sha256sum "$probe_root/run/config.json" \
  "$probe_root/run/train_report.json" \
  "$probe_root/run/checkpoints/step-000000000002/model.safetensors" \
  "$probe_root/run/checkpoints/step-000000000002/optimizer.safetensors" \
  "$probe_root/run/checkpoints/step-000000000002/trainer_state.json" \
  >"$probe_root/artifacts.sha256.tmp"
(cd / && sha256sum --quiet -c "$probe_root/artifacts.sha256.tmp")
mv -- "$probe_root/artifacts.sha256.tmp" "$probe_root/artifacts.sha256"
artifact_manifest_sha="$(sha256sum "$probe_root/artifacts.sha256" | awk '{print $1}')"
jq -nc --arg schema p2.tc_sigreg_batch_probe.v1 --arg status passed \
  --arg git_sha "$git_sha" --arg candle_git_sha "$candle_sha" --arg binary_sha256 "$binary_sha" \
  --arg gpu_name "$gpu_name" --arg started_utc "$started" --arg finished_utc "$finished" \
  --arg artifact_manifest_sha256 "$artifact_manifest_sha" \
  --argjson physical_batch "$physical_batch" --argjson grad_accum "$grad_accum" \
  --argjson completed_updates "$completed" \
  '{schema:$schema,status:$status,git_sha:$git_sha,candle_git_sha:$candle_git_sha,
    binary_sha256:$binary_sha256,gpu_name:$gpu_name,physical_batch:$physical_batch,
    grad_accum:$grad_accum,effective_batch:1024,sigreg_target:"temporal_residual",
    sigreg_temporal_window:8,completed_updates:$completed_updates,
    artifact_manifest_sha256:$artifact_manifest_sha256,
    started_utc:$started_utc,finished_utc:$finished_utc}' >"$probe_root/probe.json.tmp"
mv -- "$probe_root/probe.json.tmp" "$probe_root/probe.json"
printf '%s\n' "$probe_root/probe.json"

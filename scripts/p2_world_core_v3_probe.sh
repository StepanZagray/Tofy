#!/usr/bin/env bash
# Worst-case A40 population/VRAM and displacement-gradient probe for V3.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
candle_root="${P2_CANDLE_ROOT:-$repo_root/../candle_graph}"
probe_root="${P2_V3_PROBE_ROOT:-$repo_root/runs/p2/world-core-v3-probe}"
physical_batch="${P2_V3_PHYSICAL_BATCH:-1024}"
variance_weight="${P2_V3_PROBE_VARIANCE_WEIGHT:-0.02}"
covariance_weight="${P2_V3_PROBE_COVARIANCE_WEIGHT:-0.002}"

: "${P2_EXPECTED_SHA:?set reviewed Tofy commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set reviewed candle_graph commit}"
: "${P2_EXPECTED_BINARY_SHA:?set reviewed tofy binary hash}"
for command in git jq nvidia-smi sha256sum awk; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 2; }
done
[[ "$physical_batch" =~ ^[1-9][0-9]*$ ]] && ((physical_batch % 4 == 0)) || exit 2
[[ -x "$tofy_bin" && -d "$candle_root/.git" ]] || exit 2

git_sha="$(git -C "$repo_root" rev-parse HEAD)"
candle_sha="$(git -C "$candle_root" rev-parse HEAD)"
binary_sha="$(sha256sum "$tofy_bin" | awk '{print $1}')"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
[[ "$git_sha" == "$P2_EXPECTED_SHA" && "$candle_sha" == "$P2_EXPECTED_CANDLE_SHA" \
  && "$binary_sha" == "$P2_EXPECTED_BINARY_SHA" && "$gpu_name" == "NVIDIA A40" ]] || exit 2
[[ -z "$(git -C "$repo_root" status --porcelain)" && -z "$(git -C "$candle_root" status --porcelain)" ]] || exit 2
mkdir -p -- "$(dirname -- "$probe_root")"
mkdir -- "$probe_root" || { printf 'probe root already exists: %s\n' "$probe_root" >&2; exit 2; }

nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits >"$probe_root/gpu-before.csv"
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$tofy_bin" p2-train --device cuda:0 --seed 1 --world-core-v3 \
  --spatial-action-field --spatial-action-residual --spatial-action-residual-scale 0.25 \
  --lessons factual_branches --steps-per-lesson 2 --max-steps-this-run 2 \
  --physical-batch "$physical_batch" --grad-accum 1 --checkpoint-every-steps 0 \
  --profile-update 2 --sigreg-weight 0 --outcome-pull-weight 0.05 \
  --outcome-push-weight 0.05 --action-recovery-weight 0.05 \
  --coordinate-recovery-weight 0.05 --changed-margin-weight 0.05 \
  --displacement-variance-weight "$variance_weight" \
  --displacement-covariance-weight "$covariance_weight" --displacement-norm-floor 0.05 \
  --health-minimum-std 0.1 --health-maximum-rows 16384 --shuffled-episodes \
  --steady-gpu --supervise-last-outer-only --residual-y-update --warm-start-y \
  --outer-steps 8 --inner-steps 2 --hidden-dim 128 --action-dim 8 \
  --event-weight 0 --q-weight 0 --rollout-weight 0.1 --prefix-weight 0.05 \
  --reliability-weight 0 --ensemble-members 1 --output-dir "$probe_root" \
  >"$probe_root/train.log" 2>&1
finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits >"$probe_root/gpu-after.csv"

ratio="$(jq -er '.gradient_pressure.displacement_health_to_next_ratio' "$probe_root/train_report.json")"
jq -e --argjson physical "$physical_batch" \
  '.world_core_schema == "world_core_v3" and .physical_batch == $physical
   and .grad_accum == 1 and .global_step == 2
   and (.gradient_pressure.displacement_health_to_next_ratio >= 0.01)
   and (.gradient_pressure.displacement_health_to_next_ratio <= 0.5)' \
  "$probe_root/train_report.json" >/dev/null
jq -nc --arg schema p2.world_core_v3_batch_probe.v1 --arg status passed \
  --arg started_utc "$started" --arg finished_utc "$finished" --arg git_sha "$git_sha" \
  --arg candle_git_sha "$candle_sha" --arg binary_sha256 "$binary_sha" --arg gpu_name "$gpu_name" \
  --arg config_sha256 "$(sha256sum "$probe_root/config.json" | awk '{print $1}')" \
  --arg report_sha256 "$(sha256sum "$probe_root/train_report.json" | awk '{print $1}')" \
  --argjson physical_batch "$physical_batch" --argjson displacement_health_to_next_ratio "$ratio" \
  '{schema:$schema,status:$status,started_utc:$started_utc,finished_utc:$finished_utc,
    world_core_schema:"world_core_v3",physical_batch:$physical_batch,grad_accum:1,global_step:2,
    displacement_health_to_next_ratio:$displacement_health_to_next_ratio,
    accepted_ratio_interval:[0.01,0.5],desired_ratio_interval:[0.05,0.25],
    git_sha:$git_sha,candle_git_sha:$candle_git_sha,binary_sha256:$binary_sha256,gpu_name:$gpu_name,
    config_sha256:$config_sha256,report_sha256:$report_sha256}' >"$probe_root/probe.json"
sha256sum "$probe_root/config.json" "$probe_root/train_report.json" "$probe_root/train.log" \
  "$probe_root/probe.json" >"$probe_root/sha256.txt"
printf 'passed world-core-v3 batch probe: %s (gradient ratio %s)\n' "$probe_root/probe.json" "$ratio"

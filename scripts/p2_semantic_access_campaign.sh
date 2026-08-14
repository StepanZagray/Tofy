#!/usr/bin/env bash
# Fail-closed six-checkpoint frozen semantic-access campaign.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
parent_root="${P2_AUDIT_PARENT_ROOT:?set the pressure-grounding parent root}"
run_root="${P2_SEMANTIC_AUDIT_ROOT:-$repo_root/runs/p2/semantic-access-v1-$(date -u +%Y%m%dT%H%M%SZ)}"
eval_batch="${P2_SEMANTIC_EVAL_BATCH:-256}"
decoder_batch="${P2_SEMANTIC_DECODER_BATCH:-4096}"
permutations="${P2_SEMANTIC_PERMUTATIONS:-39}"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"

: "${P2_EXPECTED_SHA:?set the reviewed audit commit}"
: "${P2_EXPECTED_BINARY_SHA:?set the reviewed audit binary hash}"
: "${P2_EXPECTED_PARENT_SHA:?set the frozen parent commit}"
: "${P2_EXPECTED_PARENT_BINARY_SHA:?set the frozen parent binary hash}"
for command in git jq nvidia-smi sha256sum awk realpath seq sleep tee timeout; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 2; }
done
for value in "$eval_batch" "$decoder_batch" "$permutations" "$gpu_interval"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid positive integer: %s\n' "$value" >&2; exit 2; }
done
(( permutations >= 39 )) || { printf 'at least 39 permutations are required\n' >&2; exit 2; }
[[ -x "$tofy_bin" ]] || { printf 'missing audit binary: %s\n' "$tofy_bin" >&2; exit 2; }
[[ -d "$parent_root" ]] || { printf 'missing parent root: %s\n' "$parent_root" >&2; exit 2; }
parent_root="$(realpath "$parent_root")"
run_root="$(realpath -m "$run_root")"

git_sha="$(git -C "$repo_root" rev-parse HEAD)"
binary_sha="$(sha256sum "$tofy_bin" | awk '{print $1}')"
[[ "$git_sha" == "$P2_EXPECTED_SHA" ]] || { printf 'Tofy SHA mismatch\n' >&2; exit 2; }
[[ "$binary_sha" == "$P2_EXPECTED_BINARY_SHA" ]] || { printf 'binary SHA mismatch\n' >&2; exit 2; }
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || { printf 'dirty Tofy checkout\n' >&2; exit 2; }

parent_sha="$(jq -er '.git_sha' "$parent_root/campaign.json")"
[[ "$parent_sha" == "$P2_EXPECTED_PARENT_SHA" ]] || { printf 'parent SHA mismatch\n' >&2; exit 2; }
[[ "$(jq -er '.binary_sha256' "$parent_root/campaign.json")" == "$P2_EXPECTED_PARENT_BINARY_SHA" ]] || {
  printf 'parent binary SHA mismatch\n' >&2; exit 2;
}
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#gpu_names[@]} == 1 )) || { printf 'campaign requires one visible GPU\n' >&2; exit 2; }
read -r initial_memory initial_utilization < <(
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, 'NR==1 {gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
)
(( initial_memory <= 1024 && initial_utilization == 0 )) || {
  printf 'GPU is not idle (memory=%s MiB utilization=%s%%)\n' "$initial_memory" "$initial_utilization" >&2
  exit 2
}

mkdir -p -- "$(dirname -- "$run_root")"
mkdir -- "$run_root"
started_epoch="$(date +%s)"
telemetry_pid=""
finalized=false

cleanup() {
  local rc="$?"
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  if [[ "$finalized" != true && -s "$run_root/campaign.json" ]]; then
    jq --arg at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson rc "$rc" \
      '.status="failed_integrity_or_infrastructure" | .finished_utc=$at | .exit_code=$rc' \
      "$run_root/campaign.json" >"$run_root/campaign.json.tmp" 2>/dev/null &&
      mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json" || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

jq -nc --arg started "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg git_sha "$git_sha" \
  --arg binary_sha "$binary_sha" --arg parent_root "$parent_root" --arg parent_sha "$parent_sha" \
  --arg gpu "${gpu_names[0]}" --argjson permutations "$permutations" \
  --argjson eval_batch "$eval_batch" --argjson decoder_batch "$decoder_batch" \
  '{schema:"p2.semantic_access_campaign.v1",status:"running",started_utc:$started,
    git_sha:$git_sha,binary_sha256:$binary_sha,parent_root:$parent_root,parent_git_sha:$parent_sha,
    gpu_name:$gpu,arms:["S0G0","ScalG0","ScurG0","S0G1","ScalG1","ScurG1"],
    frozen_protocol:{checkpoint_update:500,population_seed:424243,episodes_per_source:64,
      physical_batch:$eval_batch,decoder_batch:$decoder_batch,decoder_hidden:64,
      max_epochs:40,patience:6,permutations:$permutations,
      split:"episode-disjoint nested train/selection/final",final_accesses_per_decoder:1},
    promotion:"locked_pending_analysis"}' >"$run_root/campaign.json"

sample_gpu() {
  local sampler_sleep_pid=""
  trap 'if [[ -n "$sampler_sleep_pid" ]]; then kill "$sampler_sleep_pid" 2>/dev/null || true; fi' EXIT INT TERM
  while true; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep "$gpu_interval" &
    sampler_sleep_pid=$!
    wait "$sampler_sleep_pid" || exit
    sampler_sleep_pid=""
  done
}
sample_gpu >>"$run_root/gpu.csv" &
telemetry_pid=$!
printf '%s\n' "$telemetry_pid" >"$run_root/telemetry.pid"
: >"$run_root/INPUTS.sha256"

wait_for_idle_gpu() {
  local attempt used utilization
  for attempt in $(seq 1 120); do
    read -r used utilization < <(
      nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits |
        awk -F, 'NR==1 {gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
    )
    if (( used <= 1024 && utilization == 0 )); then return 0; fi
    sleep 5
  done
  return 1
}

audit_arm() {
  local arm="$1" arm_dir checkpoint config manifest primary_dir recovery_dir selected
  arm_dir="$parent_root/seed-1/$arm"
  checkpoint="$arm_dir/checkpoints/step-000000000500/model.safetensors"
  config="$arm_dir/config.json"
  manifest="$arm_dir/train-500.sha256"
  [[ -s "$checkpoint" && -s "$config" && -s "$manifest" ]] || return 1
  sha256sum -c "$manifest" >/dev/null || return 1
  awk -v path="$checkpoint" '$2==path {found=1} END {exit !found}' "$manifest" || return 1
  awk -v path="$config" '$2==path {found=1} END {exit !found}' "$manifest" || return 1
  sha256sum "$checkpoint" "$config" "$manifest" >>"$run_root/INPUTS.sha256" || return 1
  primary_dir="$run_root/checkpoints/$arm"
  mkdir -p -- "$primary_dir"
  local -a command=("$tofy_bin" p2-semantic-access-audit
    --checkpoint "$checkpoint" --train-config "$config" --device cuda:0
    --seed 424243 --synthetic-episodes 64 --physical-batch "$eval_batch"
    --decoder-hidden 64 --decoder-epochs 40 --decoder-patience 6
    --decoder-batch "$decoder_batch" --permutations "$permutations")
  if timeout --foreground 12h "${command[@]}" --output "$primary_dir/semantic.json" \
    >"$primary_dir/audit.log" 2>&1; then
    selected="$primary_dir/semantic.json"
    jq -nc --arg arm "$arm" --arg status primary_succeeded --arg report "$selected" \
      '{arm:$arm,status:$status,selected_report:$report}' >"$primary_dir/receipt.json"
  else
    wait_for_idle_gpu || return 1
    recovery_dir="$run_root/recoveries/$arm/attempt-1"
    mkdir -p -- "$recovery_dir"
    if CUDA_LAUNCH_BLOCKING=1 timeout --foreground 12h "${command[@]}" \
      --output "$recovery_dir/semantic.json" \
      >"$recovery_dir/audit.log" 2>&1; then
      selected="$recovery_dir/semantic.json"
      jq -nc --arg arm "$arm" --arg status recovered_after_primary_failure --arg report "$selected" \
        --arg primary_log "$primary_dir/audit.log" \
        '{arm:$arm,status:$status,selected_report:$report,primary_failure_log:$primary_log}' \
        >"$primary_dir/receipt.json"
    else
      jq -nc --arg arm "$arm" --arg status recovery_failed \
        '{arm:$arm,status:$status}' >"$primary_dir/receipt.json"
      return 1
    fi
  fi
  jq -e '.schema=="p2.semantic_access.v1" and .model_weights_updated==false
    and .real_target_final_scores_per_decoder==1
    and .final_partition_used_for_decoder_selection==false
    and .controls.observable_positive_mse>=0 and .controls.observable_positive_mse<=0.001
    and (.protocol.permutation_seeds|length)==$permutations
    and .protocol.decoder_hidden==64 and .protocol.decoder_max_epochs==40
    and .protocol.decoder_patience==6 and .protocol.parameter_cap==100000
    and .protocol.permutation_movable_row_fraction>=0.95
    and (.population_fingerprint|startswith("sha256:"))
    and (.any_bounded_decoder_trusted|type)=="boolean"
    and (.decoders|length)==3
    and ([.decoders[].final_mse]|all(type=="number" and .>=0 and .<1e300))' \
    --argjson permutations "$permutations" "$selected" >/dev/null || return 1
  [[ "$(jq -r '.checkpoint_sha256' "$selected")" == "$(sha256sum "$checkpoint" | awk '{print $1}')" ]] || return 1
  [[ "$(jq -r '.train_config_sha256' "$selected")" == "$(sha256sum "$config" | awk '{print $1}')" ]] || return 1
  sha256sum "$checkpoint" "$config" "$selected" "$primary_dir/receipt.json" \
    >"$primary_dir/SHA256SUMS"
}

# Counterbalanced relative to the original training/evaluation order.
for arm in ScurG1 ScalG1 S0G1 ScurG0 ScalG0 S0G0; do
  kill -0 "$telemetry_pid" 2>/dev/null || { printf 'telemetry process failed\n' >&2; exit 1; }
  audit_arm "$arm"
done
sha256sum -c "$run_root/INPUTS.sha256" >/dev/null

mapfile -t selected_reports < <(
  for arm in S0G0 ScalG0 ScurG0 S0G1 ScalG1 ScurG1; do
    jq -er '.selected_report' "$run_root/checkpoints/$arm/receipt.json"
  done
)
fingerprint="$(jq -r '.population_fingerprint' "${selected_reports[0]}")"
for report in "${selected_reports[@]}"; do
  [[ "$(jq -r '.population_fingerprint' "$report")" == "$fingerprint" ]] || {
    printf 'population fingerprint mismatch\n' >&2; exit 1;
  }
done

jq -s '
  map({arm:.arm,receipt_status:.receipt.status,report:.report}) as $rows
  | def trusted($name): ($rows | map(select(.arm==$name))[0].report.any_bounded_decoder_trusted);
    ($rows | map(.report.population_fingerprint) | unique) as $fingerprints
  | if ($fingerprints|length)!=1 then error("fingerprint mismatch") else
      {schema:"p2.semantic_access_summary.v1",arms:$rows,population_fingerprint:$fingerprints[0]}
      | .decision = (
          if ((trusted("S0G0") and (trusted("ScalG0")|not) and (trusted("ScurG0")|not))
             or (trusted("S0G1") and (trusted("ScalG1")|not) and (trusted("ScurG1")|not)))
          then "trajectory_pressure_control"
          elif ($rows | all(.report.any_bounded_decoder_trusted|not))
          then "richer_exact_semantic_grounding"
          else "stop_inconclusive"
          end)
    end' \
  <(
    for arm in S0G0 ScalG0 ScurG0 S0G1 ScalG1 ScurG1; do
      receipt="$run_root/checkpoints/$arm/receipt.json"
      report="$(jq -er '.selected_report' "$receipt")"
      jq -nc --arg arm "$arm" --slurpfile receipt "$receipt" --slurpfile report "$report" \
        '{arm:$arm,receipt:$receipt[0],report:$report[0]}'
    done
  ) >"$run_root/summary.json.tmp"
mv -- "$run_root/summary.json.tmp" "$run_root/summary.json"

jq --arg finished "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg decision "$(jq -r '.decision' "$run_root/summary.json")" \
  --argjson elapsed "$(( $(date +%s) - started_epoch ))" \
  '.status="complete_pending_analysis" | .finished_utc=$finished | .elapsed_seconds=$elapsed
    | .decision=$decision' "$run_root/campaign.json" >"$run_root/campaign.json.tmp"
mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"
kill "$telemetry_pid" 2>/dev/null || true
wait "$telemetry_pid" 2>/dev/null || true
telemetry_pid=""
sha256sum "$run_root/campaign.json" "$run_root/summary.json" "$run_root/INPUTS.sha256" \
  "$run_root/gpu.csv" "${selected_reports[@]}" \
  "$run_root"/checkpoints/*/receipt.json >"$run_root/SHA256SUMS"
finalized=true
printf 'semantic-access campaign complete; decision=%s root=%s\n' \
  "$(jq -r '.decision' "$run_root/summary.json")" "$run_root"

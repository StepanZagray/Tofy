#!/usr/bin/env bash
# Fail-closed queued 2x2 grounding-mechanism factorial at seeds 2 and 3.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
tofy_bin="${TOFY_BIN:-$repo_root/target/grounding-mechanism-v1/release/tofy}"
parent_tofy_bin="${P2_PARENT_TOFY_BIN:-$repo_root/target/release/tofy}"
candle_root="${P2_CANDLE_ROOT:-$repo_root/../candle_graph}"
run_root="${P2_GROUNDING_MECHANISM_ROOT:-$repo_root/runs/p2/grounding-mechanism-v1-$(date -u +%Y%m%dT%H%M%SZ)}"
parent_root="${P2_PRESSURE_GROUNDING_PARENT_ROOT:?set the exact parent campaign root}"
eval_batch="${P2_GROUNDING_MECHANISM_EVAL_BATCH:-256}"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"
parent_poll_interval="${P2_PARENT_POLL_INTERVAL:-30}"

: "${P2_EXPECTED_SHA:?set the reviewed queue commit}"
: "${P2_EXPECTED_PARENT_SHA:?set the reviewed parent commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set the reviewed candle_graph commit}"
: "${P2_EXPECTED_BINARY_SHA:?set the reviewed queue binary hash}"
: "${P2_EXPECTED_PARENT_BINARY_SHA:?set the reviewed parent binary hash}"
for command in git jq nvidia-smi sha256sum awk realpath cmp cp tee sleep; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 2; }
done
for value in "$eval_batch" "$gpu_interval" "$parent_poll_interval"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid positive integer: %s\n' "$value" >&2; exit 2; }
done
[[ -x "$tofy_bin" ]] || { printf 'missing queue binary: %s\n' "$tofy_bin" >&2; exit 2; }
[[ -x "$parent_tofy_bin" ]] || { printf 'missing parent binary: %s\n' "$parent_tofy_bin" >&2; exit 2; }
[[ -d "$parent_root" ]] || { printf 'missing parent root: %s\n' "$parent_root" >&2; exit 2; }
parent_root="$(realpath "$parent_root")"
tofy_bin="$(realpath "$tofy_bin")"
parent_tofy_bin="$(realpath "$parent_tofy_bin")"
[[ "$tofy_bin" != "$parent_tofy_bin" && ! "$tofy_bin" -ef "$parent_tofy_bin" ]] || {
  printf 'queue binary must be path/inode-isolated from live parent binary\n' >&2; exit 2;
}

git_sha="$(git -C "$repo_root" rev-parse HEAD)"
candle_sha="$(git -C "$candle_root" rev-parse HEAD)"
binary_sha="$(sha256sum "$tofy_bin" | awk '{print $1}')"
parent_binary_sha="$(sha256sum "$parent_tofy_bin" | awk '{print $1}')"
[[ "$git_sha" == "$P2_EXPECTED_SHA" ]] || { printf 'queue SHA mismatch\n' >&2; exit 2; }
[[ "$candle_sha" == "$P2_EXPECTED_CANDLE_SHA" ]] || { printf 'candle_graph SHA mismatch\n' >&2; exit 2; }
[[ "$binary_sha" == "$P2_EXPECTED_BINARY_SHA" ]] || { printf 'queue binary SHA mismatch\n' >&2; exit 2; }
[[ "$parent_binary_sha" == "$P2_EXPECTED_PARENT_BINARY_SHA" ]] || { printf 'live parent binary SHA mismatch\n' >&2; exit 2; }
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || { printf 'dirty Tofy checkout\n' >&2; exit 2; }
[[ -z "$(git -C "$candle_root" status --porcelain)" ]] || { printf 'dirty candle_graph checkout\n' >&2; exit 2; }

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#gpu_names[@]} == 1 )) || { printf 'queue requires one visible GPU\n' >&2; exit 2; }
gpu_name="${gpu_names[0]}"
[[ "$gpu_name" == "NVIDIA A40" ]] || { printf 'expected NVIDIA A40, got %s\n' "$gpu_name" >&2; exit 2; }

mkdir -p -- "$(dirname -- "$run_root")"
mkdir -- "$run_root" || { printf 'run root exists: %s\n' "$run_root" >&2; exit 2; }
run_root="$(realpath "$run_root")"
started_epoch="$(date +%s)"
campaign_finalized=false
telemetry_pid=""

write_json_atomic() {
  local source="$1" destination="$2"
  [[ -s "$source" ]] || return 1
  mv -- "$source" "$destination" || return 1
}

finish_campaign() {
  local status="$1"
  jq --arg status "$status" --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --argjson elapsed_seconds "$(( $(date +%s) - started_epoch ))" \
    '.status=$status | .finished_utc=$finished_utc | .elapsed_seconds=$elapsed_seconds' \
    "$run_root/campaign.json" >"$run_root/campaign.json.tmp" || return 1
  write_json_atomic "$run_root/campaign.json.tmp" "$run_root/campaign.json" || return 1
  campaign_finalized=true
}

cleanup() {
  local rc="$?"
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
    telemetry_pid=""
  fi
  if [[ "$campaign_finalized" != true && -s "$run_root/campaign.json" ]]; then
    jq --arg status failed_integrity_or_infrastructure \
      --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson exit_code "$rc" \
      '.status=$status | .finished_utc=$finished_utc | .exit_code=$exit_code' \
      "$run_root/campaign.json" >"$run_root/campaign.json.tmp" 2>/dev/null &&
      mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json" || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

jq -nc --arg started_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg git_sha "$git_sha" \
  --arg parent_root "$parent_root" --arg parent_git_sha "$P2_EXPECTED_PARENT_SHA" \
  --arg candle_sha "$candle_sha" --arg binary_sha "$binary_sha" --arg gpu "$gpu_name" \
  '{schema:"p2.grounding_mechanism_campaign.v1",status:"waiting_parent",started_utc:$started_utc,
    git_sha:$git_sha,parent_root:$parent_root,parent_git_sha:$parent_git_sha,
    candle_git_sha:$candle_sha,binary_sha256:$binary_sha,gpu_name:$gpu,seeds:[2,3],
    training_seed_equals_initialization_seed:true,
    cells:{G00:{target:0,predicted:0},GT0:{target:0.038966035989056355,predicted:0},
      G0P:{target:0,predicted:0.038966035989056355},
      GTP:{target:0.038966035989056355,predicted:0.038966035989056355}},
    fixed:{sigreg_weight:0.008883956672433376,updates:500,effective_batch:1024,
      consumer_readout:"global_mean",action_conditioning:"global_additive"},
    checkpoints:[250,500],evaluation_updates:[500],evaluation_seed:424243,
    efficacy_early_stop:false,promotion:"locked_pending_completed_analysis"}' >"$run_root/campaign.json"

record_stage() {
  jq -nc --arg stage "$1" --arg status "$2" --arg detail "${3:-}" \
    --arg at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{stage:$stage,status:$status,detail:$detail,at_utc:$at}' >>"$run_root/stages.jsonl" || return 1
}

while true; do
  parent_status="$(jq -er '.status' "$parent_root/campaign.json")" || exit 1
  case "$parent_status" in
    running) sleep "$parent_poll_interval" ;;
    complete_pending_analysis) break ;;
    *) record_stage parent failed "unexpected parent status=$parent_status" || true; exit 1 ;;
  esac
done

[[ "$(sha256sum "$parent_tofy_bin" | awk '{print $1}')" == "$P2_EXPECTED_PARENT_BINARY_SHA" ]] || {
  record_stage parent failed "live parent binary changed while campaign was running" || true; exit 1;
}

jq -e --arg parent_sha "$P2_EXPECTED_PARENT_SHA" --arg candle_sha "$P2_EXPECTED_CANDLE_SHA" \
  --arg binary_sha "$P2_EXPECTED_PARENT_BINARY_SHA" \
  '.status=="complete_pending_analysis" and .git_sha==$parent_sha
    and .candle_git_sha==$candle_sha and .binary_sha256==$binary_sha' \
  "$parent_root/campaign.json" >/dev/null || exit 1
jq -e 'select(.stage=="final_integrity" and .status=="passed")' \
  "$parent_root/stages.jsonl" >/dev/null || exit 1
for arm in S0G1 ScalG0 ScalG1 S0G0 ScurG1 ScurG0; do
  [[ -s "$parent_root/seed-1/$arm/train-500.sha256" \
    && -s "$parent_root/seed-1/$arm/eval-update-500/eval_report.json" \
    && -s "$parent_root/seed-1/$arm/eval-update-500/sha256.txt" ]] || exit 1
  sha256sum -c "$parent_root/seed-1/$arm/train-500.sha256" >/dev/null || exit 1
  sha256sum -c "$parent_root/seed-1/$arm/eval-update-500/sha256.txt" >/dev/null || exit 1
done
[[ "$(jq -r '.sigreg.weight' "$parent_root/calibration/calibration.json")" == "0.008883956672433376" ]] || exit 1
[[ "$(jq -r '.grounding.weight' "$parent_root/calibration/calibration.json")" == "0.07793207197811271" ]] || exit 1
jq -e '.seed==1 and (.init_seed // .seed)==1 and .sigreg_weight==0.008883956672433376
  and .patch_grounding_weight==0.07793207197811271
  and (.patch_grounding_mode // "both")=="both"
  and .physical_batch==1024 and .grad_accum==1 and .consumer_readout=="global_mean"' \
  "$parent_root/seed-1/ScalG1/config.json" >/dev/null || exit 1
record_stage parent passed "final integrity and frozen weights verified without efficacy inspection" || exit 1

attempt=0
used_memory=999999
utilization=100
while (( attempt < 120 )); do
  read -r used_memory utilization < <(
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F, 'NR==1 {gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
  )
  if (( used_memory <= 1024 && utilization == 0 )); then break; fi
  sleep 5
  attempt=$((attempt + 1))
done
(( used_memory <= 1024 && utilization == 0 )) || { record_stage gpu_idle failed || true; exit 1; }
record_stage gpu_idle passed "memory=${used_memory}MiB utilization=${utilization}%" || exit 1

jq '.status="running" | .parent_completed_utc=(now|todate)' "$run_root/campaign.json" \
  >"$run_root/campaign.json.tmp" || exit 1
write_json_atomic "$run_root/campaign.json.tmp" "$run_root/campaign.json" || exit 1

sample_gpu() {
  while true; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep "$gpu_interval"
  done
}
sample_gpu >>"$run_root/gpu.csv" &
telemetry_pid=$!
printf '%s\n' "$telemetry_pid" >"$run_root/telemetry.pid"

sigreg_weight="0.008883956672433376"
grounding_weight="0.07793207197811271"
grounding_half="0.038966035989056355"
physical_batch=1024
grad_accum=1
forward_order=(G00 GT0 G0P GTP)
reverse_order=(GTP G0P GT0 G00)

cell_args() {
  case "$1" in
    G00) printf '0 both\n' ;;
    GT0) printf '%s target\n' "$grounding_half" ;;
    G0P) printf '%s predicted\n' "$grounding_half" ;;
    GTP) printf '%s both\n' "$grounding_weight" ;;
    *) return 1 ;;
  esac
}

train_phase() {
  local seed="$1" cell="$2" target_step="$3" resume_flag="$4"
  local patch_weight patch_mode cell_dir checkpoint
  read -r patch_weight patch_mode < <(cell_args "$cell") || return 1
  cell_dir="$run_root/seed-$seed/$cell"
  mkdir -p -- "$cell_dir" || return 1
  local -a resume_args=()
  if [[ "$resume_flag" == resume ]]; then resume_args=(--resume "$cell_dir/checkpoints"); fi
  "$tofy_bin" p2-train --seed "$seed" --init-seed "$seed" --physical-batch "$physical_batch" \
    --grad-accum "$grad_accum" --checkpoint-every-steps 250 --profile-update 250 \
    --pressure-updates 1,249,499 --max-steps-this-run 250 --sigreg-weight "$sigreg_weight" \
    --patch-grounding-weight "$patch_weight" --patch-grounding-mode "$patch_mode" \
    "${resume_args[@]}" --device cuda:0 --lessons q_calibration --steps-per-lesson 500 \
    --steady-gpu --supervise-last-outer-only --residual-y-update --warm-start-y \
    --outer-steps 8 --inner-steps 2 --hidden-dim 128 --action-dim 8 \
    --event-weight 0 --q-weight 0.1 --reliability-weight 0 --rollout-weight 0 --prefix-weight 0 \
    --ptrm-rank-every 0 --ensemble-members 1 --shuffled-episodes --consumer-readout global-mean \
    --sigreg-target temporal-residual --sigreg-statistic quantile --sigreg-temporal-window 8 \
    --sigreg-global-mix 0 --sigreg-spatial --sigreg-spatial-pool --sigreg-max-rows 32768 \
    --output-dir "$cell_dir" > >(tee -a "$cell_dir/train.log") 2>&1 || return 1
  jq -e --argjson step "$target_step" --argjson patch_weight "$patch_weight" \
    --arg patch_mode "$patch_mode" '
      .global_step==$step and .physical_batch==1024 and .grad_accum==1
      and .experiment.consumer_readout=="global_mean"
      and .experiment.sigreg.enabled and .experiment.patch_grounding_weight==$patch_weight
      and .experiment.patch_grounding_mode==$patch_mode
      and (.training_content_fingerprint|startswith("sha256:"))
      and .training_population_rows==($step*1024)' "$cell_dir/train_report.json" >/dev/null || return 1
  printf -v checkpoint '%s/checkpoints/step-%012d/model.safetensors' "$cell_dir" "$target_step"
  [[ -s "$checkpoint" ]] || return 1
  cp -- "$cell_dir/config.json" "$cell_dir/config-update-$target_step.json" || return 1
  cp -- "$cell_dir/train_report.json" "$cell_dir/train-report-update-$target_step.json" || return 1
  sha256sum "$cell_dir/config-update-$target_step.json" \
    "$cell_dir/train-report-update-$target_step.json" "$checkpoint" \
    >"$cell_dir/train-$target_step.sha256" || return 1
}

run_eval() {
  local seed="$1" cell="$2" cell_dir checkpoint eval_dir
  cell_dir="$run_root/seed-$seed/$cell"
  printf -v checkpoint '%s/checkpoints/step-%012d/model.safetensors' "$cell_dir" 500
  eval_dir="$cell_dir/eval-update-500"
  [[ -s "$checkpoint" && -s "$cell_dir/config.json" ]] || return 1
  mkdir -- "$eval_dir" || return 1
  "$tofy_bin" p2-eval --checkpoint "$checkpoint" --train-config "$cell_dir/config.json" \
    --device cuda:0 --seed 424243 --synthetic-episodes 64 --physical-batch "$eval_batch" \
    --ptrm-k 1 --ptrm-noise 0 --ensemble-members 1 --episode-jsonl "$eval_dir/episodes.jsonl" \
    --output "$eval_dir/eval_report.json" >"$eval_dir/eval.log" 2>&1 || return 1
  jq -e '.synthetic_dynamics.rollout.h8.finite_n==.synthetic_dynamics.rollout.h8.n
    and (.board_probe.population_fingerprint|length)>0' "$eval_dir/eval_report.json" >/dev/null || return 1
  sha256sum "$checkpoint" "$cell_dir/config.json" "$eval_dir/eval_report.json" \
    "$eval_dir/episodes.jsonl" >"$eval_dir/sha256.txt" || return 1
}

verify_seed() {
  local seed="$1" reference_content reference_rows reference_params parent_content eval_ref cell
  reference_content="$(jq -r .training_content_fingerprint "$run_root/seed-$seed/G00/train_report.json")"
  reference_rows="$(jq -r .training_population_rows "$run_root/seed-$seed/G00/train_report.json")"
  reference_params="$(jq -r .parameter_count "$run_root/seed-$seed/G00/train_report.json")"
  parent_content="$(jq -r .training_content_fingerprint "$parent_root/seed-1/S0G0/train_report.json")"
  [[ "$reference_content" != "$parent_content" ]] || return 1
  if [[ "$seed" == 3 ]]; then
    [[ "$reference_content" != "$(jq -r .training_content_fingerprint "$run_root/seed-2/G00/train_report.json")" ]] || return 1
  fi
  [[ "$reference_params" == "$(jq -r .parameter_count "$parent_root/seed-1/S0G0/train_report.json")" ]] || return 1
  eval_ref="$(jq -r .board_probe.population_fingerprint "$parent_root/seed-1/S0G0/eval-update-500/eval_report.json")"
  for cell in "${forward_order[@]}"; do
    sha256sum -c "$run_root/seed-$seed/$cell/train-250.sha256" >/dev/null || return 1
    sha256sum -c "$run_root/seed-$seed/$cell/train-500.sha256" >/dev/null || return 1
    sha256sum -c "$run_root/seed-$seed/$cell/eval-update-500/sha256.txt" >/dev/null || return 1
    [[ "$(jq -r .training_content_fingerprint "$run_root/seed-$seed/$cell/train_report.json")" == "$reference_content" ]] || return 1
    [[ "$(jq -r .training_population_rows "$run_root/seed-$seed/$cell/train_report.json")" == "$reference_rows" ]] || return 1
    [[ "$(jq -r .parameter_count "$run_root/seed-$seed/$cell/train_report.json")" == "$reference_params" ]] || return 1
    jq -e '([.gradient_pressure_samples[].update]==[1,249,499])
      and ([.gradient_pressure_samples[].encoder_next_latent_l2] | all(type=="number" and .>0 and .<1e300))
      and (.lessons|length)==1 and (.lessons[0].mean_losses.pre_clip_gradient_norm>0)
      and (.lessons[0].mean_losses.gradient_clip_scale>0)
      and (.lessons[0].mean_losses.gradient_clip_scale<=1)
      and (.lessons[0].mean_losses.clipped_updates>=0)
      and (.lessons[0].mean_losses.clipped_updates<=1)' "$run_root/seed-$seed/$cell/train_report.json" >/dev/null || return 1
    [[ "$(jq -r .board_probe.population_fingerprint "$run_root/seed-$seed/$cell/eval-update-500/eval_report.json")" == "$eval_ref" ]] || return 1
    jq -S 'del(.output_dir,.patch_grounding_weight,.patch_grounding_mode)' \
      "$run_root/seed-$seed/$cell/config.json" >"$run_root/seed-$seed/$cell/normalized-config.json" || return 1
  done
  for cell in "${forward_order[@]}"; do
    cmp -s "$run_root/seed-$seed/G00/normalized-config.json" \
      "$run_root/seed-$seed/$cell/normalized-config.json" || return 1
  done
  jq -nc --argjson seed "$seed" --arg content "$reference_content" --argjson rows "$reference_rows" \
    --argjson params "$reference_params" --arg eval_population "$eval_ref" \
    '{seed:$seed,status:"complete",training_content_fingerprint:$content,training_rows:$rows,
      parameter_count:$params,evaluation_population_fingerprint:$eval_population}' \
    >"$run_root/seed-$seed/seed-report.json.tmp" || return 1
  write_json_atomic "$run_root/seed-$seed/seed-report.json.tmp" "$run_root/seed-$seed/seed-report.json" || return 1
}

run_seed_block() {
  local seed="$1" order_kind="$2" cell
  local -a first second
  if [[ "$order_kind" == forward ]]; then
    first=("${forward_order[@]}"); second=("${reverse_order[@]}")
  else
    first=("${reverse_order[@]}"); second=("${forward_order[@]}")
  fi
  mkdir -- "$run_root/seed-$seed" || return 1
  record_stage "seed_${seed}_train_0_250" running "$order_kind" || return 1
  for cell in "${first[@]}"; do train_phase "$seed" "$cell" 250 fresh || return 1; done
  record_stage "seed_${seed}_train_0_250" passed || return 1
  record_stage "seed_${seed}_train_250_500" running || return 1
  for cell in "${second[@]}"; do train_phase "$seed" "$cell" 500 resume || return 1; done
  record_stage "seed_${seed}_train_250_500" passed || return 1
  record_stage "seed_${seed}_eval_500" running "population_seed=424243" || return 1
  for cell in "${first[@]}"; do run_eval "$seed" "$cell" || return 1; done
  record_stage "seed_${seed}_eval_500" passed || return 1
  verify_seed "$seed" || return 1
  record_stage "seed_${seed}_integrity" passed || return 1
}

run_seed_block 2 forward || exit 1
run_seed_block 3 reverse || exit 1

kill -0 "$telemetry_pid" 2>/dev/null || exit 1
(( $(awk 'END {print NR}' "$run_root/gpu.csv") >= 10 )) || exit 1
jq '.selected_batch={physical:1024,accumulation:1,effective:1024}
    | .verified_seeds=[2,3]' "$run_root/campaign.json" >"$run_root/campaign.json.tmp" || exit 1
write_json_atomic "$run_root/campaign.json.tmp" "$run_root/campaign.json" || exit 1
record_stage final_integrity passed || exit 1
finish_campaign complete_pending_analysis || exit 1
printf 'grounding-mechanism-v1 complete; promotion remains locked: %s\n' "$run_root"

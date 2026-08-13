#!/usr/bin/env bash
# Fail-closed, counterbalanced seed-1 SIGReg-pressure x patch-grounding screen.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
candle_root="${P2_CANDLE_ROOT:-$repo_root/../candle_graph}"
run_root="${P2_PRESSURE_GROUNDING_ROOT:-$repo_root/runs/p2/pressure-grounding-v1-$(date -u +%Y%m%dT%H%M%SZ)}"
eval_batch="${P2_PRESSURE_GROUNDING_EVAL_BATCH:-256}"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"

: "${P2_EXPECTED_SHA:?set the reviewed Tofy commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set the reviewed candle_graph commit}"
: "${P2_EXPECTED_BINARY_SHA:?set the reviewed release binary hash}"
for command in git jq nvidia-smi sha256sum awk realpath cmp tee; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 2; }
done
[[ -x "$tofy_bin" ]] || { printf 'missing release binary: %s\n' "$tofy_bin" >&2; exit 2; }
[[ "$eval_batch" =~ ^[1-9][0-9]*$ && "$gpu_interval" =~ ^[1-9][0-9]*$ ]] || {
  printf 'eval batch and telemetry interval must be positive integers\n' >&2; exit 2;
}

git_sha="$(git -C "$repo_root" rev-parse HEAD)"
candle_sha="$(git -C "$candle_root" rev-parse HEAD)"
binary_sha="$(sha256sum "$tofy_bin" | awk '{print $1}')"
[[ "$git_sha" == "$P2_EXPECTED_SHA" ]] || { printf 'Tofy SHA mismatch\n' >&2; exit 2; }
[[ "$candle_sha" == "$P2_EXPECTED_CANDLE_SHA" ]] || { printf 'candle_graph SHA mismatch\n' >&2; exit 2; }
[[ "$binary_sha" == "$P2_EXPECTED_BINARY_SHA" ]] || { printf 'binary SHA mismatch\n' >&2; exit 2; }
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || { printf 'dirty Tofy checkout\n' >&2; exit 2; }
[[ -z "$(git -C "$candle_root" status --porcelain)" ]] || { printf 'dirty candle_graph checkout\n' >&2; exit 2; }

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#gpu_names[@]} == 1 )) || { printf 'campaign requires one visible GPU\n' >&2; exit 2; }
gpu_name="${gpu_names[0]}"
[[ "$gpu_name" == "NVIDIA A40" ]] || { printf 'expected NVIDIA A40, got %s\n' "$gpu_name" >&2; exit 2; }
read -r initial_memory initial_utilization < <(
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, 'NR==1 {gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
)
(( initial_memory <= 1024 && initial_utilization == 0 )) || {
  printf 'A40 is not idle (memory=%s MiB utilization=%s%%)\n' "$initial_memory" "$initial_utilization" >&2
  exit 2
}

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
    jq --arg status failed_integrity_or_infrastructure --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      --argjson exit_code "$rc" '.status=$status | .finished_utc=$finished_utc | .exit_code=$exit_code' \
      "$run_root/campaign.json" >"$run_root/campaign.json.tmp" 2>/dev/null &&
      mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json" || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

jq -nc --arg started_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg git_sha "$git_sha" \
  --arg candle_sha "$candle_sha" --arg binary_sha "$binary_sha" --arg gpu "$gpu_name" \
  '{schema:"p2.pressure_grounding_campaign.v1",status:"running",started_utc:$started_utc,
    git_sha:$git_sha,candle_git_sha:$candle_sha,binary_sha256:$binary_sha,gpu_name:$gpu,
    initialization_seed:1,training_seed:1,calibration_data_seeds:[1001,1002,1003,1004,1005,1006,1007,1008],
    arm_order:["S0G1","ScalG0","ScalG1","S0G0","ScurG1","ScurG0"],
    phase_order:{train_0_250:"forward",eval_250:"reverse",train_250_500:"reverse",eval_500:"forward"},
    factors:{sigreg:[0,"pressure_calibrated",0.003],patch_grounding:[0,"pressure_calibrated"]},
    pressure_target:{median:0.275,accepted_median:[0.20,0.35],maximum:0.50},
    checkpoints:[250,500],evaluation_seeds:{"250":424242,"500":424243},
    fixed:{updates:500,effective_batch:1024,consumer_readout:"global_mean",lesson:"q_calibration",
      q_definition:"checkpoint_dependent_diagnostic_only",action_conditioning:"global_additive",
      grounding:"shared_linear_16_colour_patch_histogram_target_and_prediction_balanced_by_change_class_status_row_excluded"},
    efficacy_early_stop:false,promotion:"locked_pending_completed_analysis"}' >"$run_root/campaign.json"

record_stage() {
  jq -nc --arg stage "$1" --arg status "$2" --arg detail "${3:-}" \
    --arg at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{stage:$stage,status:$status,detail:$detail,at_utc:$at}' >>"$run_root/stages.jsonl" || return 1
}

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

common_args=(
  --device cuda:0 --init-seed 1 --lessons q_calibration --steps-per-lesson 500
  --steady-gpu --supervise-last-outer-only --residual-y-update --warm-start-y
  --outer-steps 8 --inner-steps 2 --hidden-dim 128 --action-dim 8
  --event-weight 0 --q-weight 0.1 --reliability-weight 0 --rollout-weight 0 --prefix-weight 0
  --ptrm-rank-every 0 --ensemble-members 1 --shuffled-episodes --consumer-readout global-mean
  --sigreg-target temporal-residual --sigreg-statistic quantile --sigreg-temporal-window 8
  --sigreg-global-mix 0 --sigreg-spatial --sigreg-spatial-pool --sigreg-max-rows 32768
)

physical_batch=1024
grad_accum=1
probe_dir="$run_root/capacity-1024-1"
"$tofy_bin" p2-train --seed 1 --physical-batch "$physical_batch" --grad-accum "$grad_accum" \
  --max-steps-this-run 1 --checkpoint-every-steps 0 --profile-update 2 --pressure-updates 1 \
  --sigreg-weight 1 --patch-grounding-weight 1 "${common_args[@]}" --output-dir "$probe_dir" \
  >"$probe_dir.log" 2>&1 || { record_stage capacity_probe failed "1024x1 is required; see $probe_dir.log" || true; finish_campaign failed_capacity || true; exit 1; }
jq -e '.global_step==1 and .status=="paused" and .physical_batch==1024 and .grad_accum==1
  and ([.gradient_pressure_samples[].update]==[1])
  and (.training_content_fingerprint|startswith("sha256:")) and .training_population_rows==1024' \
  "$probe_dir/train_report.json" >/dev/null || { finish_campaign failed_capacity_integrity || true; exit 1; }
record_stage capacity_probe passed "physical_batch=1024 grad_accum=1 effective_batch=1024" || exit 1

mkdir -- "$run_root/calibration"
for data_seed in 1001 1002 1003 1004 1005 1006 1007 1008; do
  probe_dir="$run_root/calibration/seed-$data_seed"
  "$tofy_bin" p2-train --seed "$data_seed" --physical-batch "$physical_batch" \
    --grad-accum "$grad_accum" --max-steps-this-run 1 --checkpoint-every-steps 0 \
    --profile-update 2 --pressure-updates 1 --sigreg-weight 1 --patch-grounding-weight 1 \
    "${common_args[@]}" --output-dir "$probe_dir" >"$probe_dir.log" 2>&1 || exit 1
  jq -e '.global_step==1 and .status=="paused" and .gradient_pressure.update==1
    and (.gradient_pressure.sigreg_to_next_ratio > 0)
    and (.gradient_pressure.grounding_to_next_ratio > 0)
    and (.training_content_fingerprint | startswith("sha256:"))' \
    "$probe_dir/train_report.json" >/dev/null || exit 1
  jq -c --argjson data_seed "$data_seed" \
    '{data_seed:$data_seed,content_fingerprint:.training_content_fingerprint,
      next_l2:.gradient_pressure.encoder_next_latent_l2,
      sigreg_ratio:.gradient_pressure.sigreg_to_next_ratio,
      grounding_ratio:.gradient_pressure.grounding_to_next_ratio,
      grounding_head_l2:.gradient_pressure.grounding_head_weighted_l2,
      sigreg_next_cosine:.gradient_pressure.sigreg_next_cosine,
      grounding_next_cosine:.gradient_pressure.grounding_next_cosine,
      grounding_sigreg_cosine:.gradient_pressure.grounding_sigreg_cosine}' \
    "$probe_dir/train_report.json" >>"$run_root/calibration/samples.jsonl" || exit 1
done

jq -s '
  def median: sort as $s | ($s|length) as $n |
    if ($n%2)==1 then $s[($n/2|floor)] else (($s[$n/2-1]+$s[$n/2])/2) end;
  def resolve($xs): ($xs|median) as $median | ($xs|max) as $maximum |
    ([0.275/$median,0.50/$maximum]|min) as $weight |
    {unweighted_median:$median,unweighted_maximum:$maximum,weight:$weight,
      weighted_median:($weight*$median),weighted_maximum:($weight*$maximum)};
  {schema:"p2.pressure_calibration.v1",samples:.,sigreg:resolve(map(.sigreg_ratio)),
    grounding:resolve(map(.grounding_ratio))}' "$run_root/calibration/samples.jsonl" \
    >"$run_root/calibration/calibration.json.tmp" || exit 1
write_json_atomic "$run_root/calibration/calibration.json.tmp" \
  "$run_root/calibration/calibration.json" || exit 1
jq -e '.sigreg.weight > 0 and .grounding.weight > 0
  and .sigreg.weighted_median >= 0.20 and .sigreg.weighted_median <= 0.35
  and .grounding.weighted_median >= 0.20 and .grounding.weighted_median <= 0.35
  and .sigreg.weighted_maximum <= 0.50 and .grounding.weighted_maximum <= 0.50
  and ((.sigreg.weight-0.003)|abs) >= 0.0006
  and ([.samples[].content_fingerprint]|unique|length)==8' \
  "$run_root/calibration/calibration.json" >/dev/null || { finish_campaign failed_calibration || true; exit 1; }
sigreg_calibrated="$(jq -r .sigreg.weight "$run_root/calibration/calibration.json")"
grounding_calibrated="$(jq -r .grounding.weight "$run_root/calibration/calibration.json")"
record_stage pressure_calibration passed \
  "sigreg_weight=$sigreg_calibrated grounding_weight=$grounding_calibrated" || exit 1

arm_order=(S0G1 ScalG0 ScalG1 S0G0 ScurG1 ScurG0)
reverse_order=(ScurG0 ScurG1 S0G0 ScalG1 ScalG0 S0G1)

arm_weights() {
  case "$1" in
    S0G0) printf '0 0\n' ;;
    S0G1) printf '0 %s\n' "$grounding_calibrated" ;;
    ScalG0) printf '%s 0\n' "$sigreg_calibrated" ;;
    ScalG1) printf '%s %s\n' "$sigreg_calibrated" "$grounding_calibrated" ;;
    ScurG0) printf '0.003 0\n' ;;
    ScurG1) printf '0.003 %s\n' "$grounding_calibrated" ;;
    *) return 1 ;;
  esac
}

train_phase() {
  local arm="$1" target_step="$2" resume_flag="$3" sigreg_weight grounding_weight arm_dir
  read -r sigreg_weight grounding_weight < <(arm_weights "$arm") || return 1
  arm_dir="$run_root/seed-1/$arm"
  mkdir -p -- "$arm_dir" || return 1
  local -a resume_args=()
  if [[ "$resume_flag" == resume ]]; then resume_args=(--resume "$arm_dir/checkpoints"); fi
  "$tofy_bin" p2-train --seed 1 --physical-batch "$physical_batch" --grad-accum "$grad_accum" \
    --checkpoint-every-steps 250 --profile-update 250 --pressure-updates 1,249,499 \
    --max-steps-this-run 250 --sigreg-weight "$sigreg_weight" \
    --patch-grounding-weight "$grounding_weight" "${resume_args[@]}" "${common_args[@]}" \
    --output-dir "$arm_dir" > >(tee -a "$arm_dir/train.log") 2>&1 || return 1
  jq -e --argjson step "$target_step" --argjson batch "$physical_batch" --argjson accum "$grad_accum" \
    --argjson sigreg "$sigreg_weight" --argjson grounding "$grounding_weight" '
      .global_step==$step and .physical_batch==$batch and .grad_accum==$accum
      and .experiment.consumer_readout=="global_mean"
      and .experiment.patch_grounding_weight==$grounding
      and .experiment.sigreg.enabled==($sigreg>0)
      and (.training_content_fingerprint|startswith("sha256:"))
      and .training_population_rows==($step*1024)' "$arm_dir/train_report.json" >/dev/null || return 1
  local checkpoint
  printf -v checkpoint '%s/checkpoints/step-%012d/model.safetensors' "$arm_dir" "$target_step"
  [[ -s "$checkpoint" ]] || return 1
  sha256sum "$arm_dir/config.json" "$arm_dir/train_report.json" "$checkpoint" \
    >"$arm_dir/train-$target_step.sha256" || return 1
  [[ -s "$arm_dir/train-$target_step.sha256" ]] || return 1
}

run_eval() {
  local arm="$1" update="$2" eval_seed="$3" arm_dir checkpoint eval_dir
  arm_dir="$run_root/seed-1/$arm"
  printf -v checkpoint '%s/checkpoints/step-%012d/model.safetensors' "$arm_dir" "$update"
  eval_dir="$arm_dir/eval-update-$update"
  [[ -s "$checkpoint" && -s "$arm_dir/config.json" ]] || return 1
  mkdir -- "$eval_dir" || return 1
  "$tofy_bin" p2-eval --checkpoint "$checkpoint" --train-config "$arm_dir/config.json" \
    --device cuda:0 --seed "$eval_seed" --synthetic-episodes 64 --physical-batch "$eval_batch" \
    --ptrm-k 1 --ptrm-noise 0 --ensemble-members 1 --episode-jsonl "$eval_dir/episodes.jsonl" \
    --output "$eval_dir/eval_report.json" >"$eval_dir/eval.log" 2>&1 || return 1
  [[ -s "$eval_dir/eval_report.json" && -s "$eval_dir/episodes.jsonl" ]] || return 1
  jq -e '.synthetic_dynamics.rollout.h8.finite_n==.synthetic_dynamics.rollout.h8.n
    and (.board_probe.population_fingerprint|length)>0' "$eval_dir/eval_report.json" >/dev/null || return 1
  sha256sum "$checkpoint" "$arm_dir/config.json" "$eval_dir/eval_report.json" \
    "$eval_dir/episodes.jsonl" >"$eval_dir/sha256.txt" || return 1
  [[ -s "$eval_dir/sha256.txt" ]] || return 1
}

for arm in "${arm_order[@]}"; do train_phase "$arm" 250 fresh || exit 1; done
record_stage train_0_250 passed || exit 1
for arm in "${reverse_order[@]}"; do run_eval "$arm" 250 424242 || exit 1; done
record_stage eval_250 passed || exit 1
for arm in "${reverse_order[@]}"; do train_phase "$arm" 500 resume || exit 1; done
record_stage train_250_500 passed || exit 1
for arm in "${arm_order[@]}"; do run_eval "$arm" 500 424243 || exit 1; done
record_stage eval_500 passed || exit 1

reference_content="$(jq -r .training_content_fingerprint "$run_root/seed-1/ScurG0/train_report.json")"
reference_rows="$(jq -r .training_population_rows "$run_root/seed-1/ScurG0/train_report.json")"
reference_params="$(jq -r .parameter_count "$run_root/seed-1/ScurG0/train_report.json")"
for arm in "${arm_order[@]}"; do
  [[ "$(jq -r .training_content_fingerprint "$run_root/seed-1/$arm/train_report.json")" == "$reference_content" ]] || exit 1
  [[ "$(jq -r .training_population_rows "$run_root/seed-1/$arm/train_report.json")" == "$reference_rows" ]] || exit 1
  [[ "$(jq -r .parameter_count "$run_root/seed-1/$arm/train_report.json")" == "$reference_params" ]] || exit 1
  jq -e '([.gradient_pressure_samples[].update]==[1,249,499])
    and ([.gradient_pressure_samples[].encoder_next_latent_l2] | all(type=="number" and .>0 and .<1e300))
    and (.lessons|length)==1
    and (.lessons[0].mean_losses.pre_clip_gradient_norm>0)
    and (.lessons[0].mean_losses.gradient_clip_scale>0)
    and (.lessons[0].mean_losses.gradient_clip_scale<=1)
    and (.lessons[0].mean_losses.clipped_updates>=0)
    and (.lessons[0].mean_losses.clipped_updates<=1)' \
    "$run_root/seed-1/$arm/train_report.json" >/dev/null || exit 1
  jq -S 'del(.output_dir,.sigreg_weight,.patch_grounding_weight)' "$run_root/seed-1/$arm/config.json" \
    >"$run_root/seed-1/$arm/normalized-config.json" || exit 1
done

kill -0 "$telemetry_pid" 2>/dev/null || exit 1
(( $(awk 'END {print NR}' "$run_root/gpu.csv") >= 10 )) || exit 1
for arm in "${arm_order[@]}"; do
  cmp -s "$run_root/seed-1/ScurG0/normalized-config.json" \
    "$run_root/seed-1/$arm/normalized-config.json" || exit 1
done
for update in 250 500; do
  eval_ref="$(jq -r .board_probe.population_fingerprint "$run_root/seed-1/ScurG0/eval-update-$update/eval_report.json")"
  for arm in "${arm_order[@]}"; do
    [[ "$(jq -r .board_probe.population_fingerprint "$run_root/seed-1/$arm/eval-update-$update/eval_report.json")" == "$eval_ref" ]] || exit 1
  done
done

jq --argjson batch "$physical_batch" --argjson accum "$grad_accum" \
  --argjson sigreg "$sigreg_calibrated" --argjson grounding "$grounding_calibrated" \
  --arg content "$reference_content" --argjson rows "$reference_rows" --argjson params "$reference_params" '
    .selected_batch={physical:$batch,accumulation:$accum,effective:1024}
    | .calibrated_weights={sigreg:$sigreg,patch_grounding:$grounding}
    | .verified={training_content_fingerprint:$content,training_rows:$rows,parameter_count:$params}' \
  "$run_root/campaign.json" >"$run_root/campaign.json.tmp" || exit 1
write_json_atomic "$run_root/campaign.json.tmp" "$run_root/campaign.json" || exit 1
record_stage final_integrity passed || exit 1
finish_campaign complete_pending_analysis || exit 1
printf 'pressure-grounding-v1 complete; promotion remains locked: %s\n' "$run_root"

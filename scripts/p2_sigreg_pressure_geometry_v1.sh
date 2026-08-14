#!/usr/bin/env bash
# Serial pressure x population-geometry diagnostic. This is not a paper-faithful SIGReg run.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
candle_root="${P2_CANDLE_ROOT:-$repo_root/../candle_graph}"
run_root="${P2_SIGREG_PRESSURE_GEOMETRY_ROOT:-$repo_root/runs/p2/sigreg-pressure-geometry-v1-$(date -u +%Y%m%dT%H%M%SZ)}"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"
eval_batch="${P2_SIGREG_EVAL_BATCH:-256}"
updates="${P2_SIGREG_UPDATES:-250}"
checkpoint_every="${P2_SIGREG_CHECKPOINT_EVERY:-125}"

: "${P2_EXPECTED_SHA:?set the reviewed Tofy commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set the reviewed candle_graph commit}"
: "${P2_EXPECTED_BINARY_SHA:?set the reviewed release binary SHA-256}"
for command in git jq nvidia-smi sha256sum awk realpath date mkdir mv env cmp sleep; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 2; }
done
[[ -x "$tofy_bin" ]] || { printf 'missing release binary: %s\n' "$tofy_bin" >&2; exit 2; }
[[ "$updates" == 250 && "$checkpoint_every" == 125 ]] || {
  printf 'v1 fixes updates=250 and checkpoint_every=125\n' >&2; exit 2;
}
[[ "$eval_batch" =~ ^[1-9][0-9]*$ && "$gpu_interval" =~ ^[1-9][0-9]*$ ]] || {
  printf 'evaluation batch and telemetry interval must be positive integers\n' >&2; exit 2;
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
active_pid=""

write_json_atomic() {
  local source="$1" destination="$2"
  [[ -s "$source" ]] || return 1
  mv -- "$source" "$destination"
}

record_stage() {
  jq -nc --arg stage "$1" --arg status "$2" --arg detail "${3:-}" \
    --arg at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{stage:$stage,status:$status,detail:$detail,at_utc:$at_utc}' >>"$run_root/stages.jsonl"
}

finish_campaign() {
  local status="$1"
  jq --arg status "$status" --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --argjson elapsed_seconds "$(( $(date +%s) - started_epoch ))" \
    '.status=$status | .finished_utc=$finished_utc | .elapsed_seconds=$elapsed_seconds' \
    "$run_root/campaign.json" >"$run_root/campaign.json.tmp"
  write_json_atomic "$run_root/campaign.json.tmp" "$run_root/campaign.json"
  campaign_finalized=true
}

sample_gpu() {
  local delay_pid=""
  trap 'if [[ -n "$delay_pid" ]] && kill -0 "$delay_pid" 2>/dev/null; then
    kill "$delay_pid" 2>/dev/null || true
    wait "$delay_pid" 2>/dev/null || true
  fi
  exit 0' TERM INT
  while true; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep "$gpu_interval" &
    delay_pid=$!
    wait "$delay_pid" || true
    delay_pid=""
  done
}

run_tracked() {
  local log_path="$1" rc
  shift
  "$@" >"$log_path" 2>&1 &
  active_pid=$!
  if wait "$active_pid"; then rc=0; else rc=$?; fi
  active_pid=""
  return "$rc"
}

cleanup() {
  local rc="$?"
  if [[ -n "$active_pid" ]] && kill -0 "$active_pid" 2>/dev/null; then
    kill "$active_pid" 2>/dev/null || true
    wait "$active_pid" 2>/dev/null || true
  fi
  active_pid=""
  if [[ -n "$telemetry_pid" ]] && kill -0 "$telemetry_pid" 2>/dev/null; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  telemetry_pid=""
  if [[ "$campaign_finalized" != true && -s "$run_root/campaign.json" ]]; then
    jq --arg status failed_integrity_or_training --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      --argjson exit_code "$rc" --argjson elapsed_seconds "$(( $(date +%s) - started_epoch ))" \
      '.status=$status | .finished_utc=$finished_utc | .exit_code=$exit_code | .elapsed_seconds=$elapsed_seconds' \
      "$run_root/campaign.json" >"$run_root/campaign.json.tmp" 2>/dev/null &&
      mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json" || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

jq -nc --arg started_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg git_sha "$git_sha" --arg candle_sha "$candle_sha" --arg binary_sha "$binary_sha" \
  --arg gpu "$gpu_name" --argjson updates "$updates" --argjson checkpoint_every "$checkpoint_every" '
  {schema:"p2.sigreg_pressure_geometry.v1",status:"running",started_utc:$started_utc,
   git_sha:$git_sha,candle_git_sha:$candle_sha,binary_sha256:$binary_sha,gpu_name:$gpu,
   physical_batch:1024,gradient_accumulation:1,updates_per_arm:$updates,
   checkpoints:[$checkpoint_every,$updates],training_seeds:[1],evaluation_seeds:{"1":424244},
   calibration:{maximum_shared_median_ratio:0.01,max_ratio:0.02,data_seeds_per_geometry:8},
   seed_1_order:["S0","cell-high","global-high","global-matched","cell-matched"],
   interpretation:"Tofy-specific TC-QQ pressure-by-population diagnostic; not paper-faithful",
   promotion:"locked_pending_completed_analysis"}' >"$run_root/campaign.json"

sample_gpu >>"$run_root/gpu.csv" &
telemetry_pid=$!
printf '%s\n' "$telemetry_pid" >"$run_root/telemetry.pid"

common_args=(
  --device cuda:0 --lessons q_calibration --steps-per-lesson "$updates"
  --physical-batch 1024 --grad-accum 1 --steady-gpu
  --supervise-last-outer-only --residual-y-update --warm-start-y
  --outer-steps 8 --inner-steps 2 --hidden-dim 128 --action-dim 8
  --event-weight 0 --q-weight 0.1 --reliability-weight 0 --rollout-weight 0 --prefix-weight 0
  --patch-grounding-weight 0 --ptrm-rank-every 0 --ensemble-members 1
  --shuffled-episodes --consumer-readout global-mean
  --sigreg-target temporal-residual --sigreg-statistic quantile --sigreg-temporal-window 8
  --sigreg-spatial --sigreg-spatial-pool --sigreg-max-rows 32768
)

collect_geometry_calibration() {
  local init_seed="$1" geometry="$2" data_seed probe_dir mix
  case "$geometry" in
    cell) mix=0 ;;
    global) mix=1 ;;
    *) return 2 ;;
  esac
  mkdir -p -- "$run_root/calibration/seed-$init_seed/$geometry"
  : >"$run_root/calibration/seed-$init_seed/$geometry/samples.jsonl"
  for offset in 1 2 3 4 5 6 7 8; do
    data_seed=$((init_seed * 1000 + offset))
    probe_dir="$run_root/calibration/seed-$init_seed/$geometry/data-$data_seed"
    run_tracked "$probe_dir.log" "$tofy_bin" p2-train --seed "$data_seed" --init-seed "$init_seed" \
      --max-steps-this-run 1 --checkpoint-every-steps 0 --profile-update 2 \
      --pressure-updates 1 --sigreg-weight 1 --sigreg-global-mix "$mix" \
      "${common_args[@]}" --output-dir "$probe_dir"
    jq -e '.global_step==1 and .status=="paused" and .physical_batch==1024 and .grad_accum==1
      and (.gradient_pressure.sigreg_to_next_ratio>0)
      and (.training_content_fingerprint|startswith("sha256:"))' \
      "$probe_dir/train_report.json" >/dev/null
    jq -c --argjson data_seed "$data_seed" \
      '{data_seed:$data_seed,content_fingerprint:.training_content_fingerprint,
        unweighted_ratio:.gradient_pressure.sigreg_to_next_ratio,
        next_l2:.gradient_pressure.encoder_next_latent_l2,
        sigreg_l2:.gradient_pressure.encoder_sigreg_weighted_l2}' \
      "$probe_dir/train_report.json" >>"$run_root/calibration/seed-$init_seed/$geometry/samples.jsonl"
  done
}

resolve_joint_calibration() {
  local init_seed="$1" calibration_root
  calibration_root="$run_root/calibration/seed-$init_seed"
  jq -n \
    --slurpfile cell "$calibration_root/cell/samples.jsonl" \
    --slurpfile global "$calibration_root/global/samples.jsonl" '
    def median: sort as $s | ($s|length) as $n |
      if ($n%2)==1 then $s[($n/2|floor)] else (($s[$n/2-1]+$s[$n/2])/2) end;
    ($cell|map(.unweighted_ratio)) as $cell_ratios |
    ($global|map(.unweighted_ratio)) as $global_ratios |
    ($cell_ratios|median) as $cell_median |
    ($global_ratios|median) as $global_median |
    ($cell_ratios|max) as $cell_maximum |
    ($global_ratios|max) as $global_maximum |
    ([0.01,0.02/($cell_maximum/$cell_median),0.02/($global_maximum/$global_median)]|min) as $target |
    ($target/$cell_median) as $cell_weight |
    ($target/$global_median) as $global_weight |
    {schema:"p2.sigreg_joint_pressure_calibration.v1",shared_weighted_median:$target,
     maximum_weighted_ratio:0.02,
     cell:{samples:$cell,unweighted_median:$cell_median,unweighted_maximum:$cell_maximum,
       weight:$cell_weight,weighted_median:($cell_weight*$cell_median),
       weighted_maximum:($cell_weight*$cell_maximum)},
     global:{samples:$global,unweighted_median:$global_median,unweighted_maximum:$global_maximum,
       weight:$global_weight,weighted_median:($global_weight*$global_median),
       weighted_maximum:($global_weight*$global_maximum)}}' \
    >"$calibration_root/calibration.json.tmp"
  write_json_atomic "$calibration_root/calibration.json.tmp" "$calibration_root/calibration.json"
  jq -e '
    .shared_weighted_median>0 and .shared_weighted_median<=0.010000001
    and .cell.weight>0 and .global.weight>0
    and ((.cell.weighted_median-.shared_weighted_median)|abs)<1e-12
    and ((.global.weighted_median-.shared_weighted_median)|abs)<1e-12
    and .cell.weighted_maximum<=0.020000001 and .global.weighted_maximum<=0.020000001
    and ([.cell.samples[].content_fingerprint]|unique|length)==8
    and ([.global.samples[].content_fingerprint]|unique|length)==8' \
    "$calibration_root/calibration.json" >/dev/null
}

collect_geometry_calibration 1 cell
collect_geometry_calibration 1 global
resolve_joint_calibration 1
record_stage calibration passed "seed-1 cell/global initial gradient pressure matched"

run_arm() {
  local seed="$1" arm="$2" weight mix eval_seed arm_dir checkpoint eval_dir
  local -a eval_cmd
  case "$arm" in
    S0) weight=0; mix=0 ;;
    cell-high) weight=0.003; mix=0 ;;
    global-high) weight=0.003; mix=1 ;;
    cell-matched)
      weight="$(jq -r .cell.weight "$run_root/calibration/seed-$seed/calibration.json")"; mix=0 ;;
    global-matched)
      weight="$(jq -r .global.weight "$run_root/calibration/seed-$seed/calibration.json")"; mix=1 ;;
    *) return 2 ;;
  esac
  eval_seed=$((424243 + seed))
  arm_dir="$run_root/seed-$seed/$arm"
  mkdir -p -- "$arm_dir"
  jq -nc --arg arm "$arm" --argjson seed "$seed" --argjson sigreg_weight "$weight" \
    --argjson global_mix "$mix" --argjson eval_seed "$eval_seed" \
    '{arm:$arm,training_seed:$seed,init_seed:$seed,sigreg_weight:$sigreg_weight,
      sigreg_global_mix:$global_mix,evaluation_seed:$eval_seed,status:"running"}' \
    >"$arm_dir/arm.json"
  run_tracked "$arm_dir/train.log" "$tofy_bin" p2-train --seed "$seed" --init-seed "$seed" \
    --checkpoint-every-steps "$checkpoint_every" --profile-update 125 \
    --pressure-updates 1,124,249 --sigreg-weight "$weight" --sigreg-global-mix "$mix" \
    "${common_args[@]}" --output-dir "$arm_dir"
  jq -e --argjson updates "$updates" --argjson weight "$weight" --argjson mix "$mix" '
    .global_step==$updates and .status=="completed" and .physical_batch==1024 and .grad_accum==1
    and .experiment.sigreg.enabled==($weight>0) and .experiment.sigreg.global_mix==$mix
    and ([.gradient_pressure_samples[].update] == [1,124,249])
    and ((if $weight>0 then [.gradient_pressure_samples[].sigreg_to_next_ratio]
      | all(type=="number" and .>0 and .<1e300)
      else [.gradient_pressure_samples[].sigreg_to_next_ratio] | all(.==null) end))
    and (.lessons|length)==1
    and (.lessons[0].mean_losses.gradient_clip_scale>0)
    and (.lessons[0].mean_losses.gradient_clip_scale<=1)
    and (.lessons[0].mean_losses.clipped_updates>=0)
    and (.lessons[0].mean_losses.clipped_updates<=1)
    and (.training_content_fingerprint|startswith("sha256:"))' "$arm_dir/train_report.json" >/dev/null
  jq -e --argjson weight "$weight" --argjson mix "$mix" '
    .sigreg_weight==$weight and .sigreg_global_mix==$mix' "$arm_dir/config.json" >/dev/null

  for update in "$checkpoint_every" "$updates"; do
    printf -v checkpoint '%s/checkpoints/step-%012d' "$arm_dir" "$update"
    eval_dir="$arm_dir/eval-update-$update"
    mkdir -p -- "$eval_dir"
    eval_cmd=(
      "$tofy_bin" p2-eval --checkpoint "$checkpoint/model.safetensors"
      --train-config "$arm_dir/config.json" --device cuda:0 --seed "$eval_seed"
      --synthetic-episodes 64 --physical-batch "$eval_batch" --ptrm-k 1
      --ptrm-noise 0 --ensemble-members 1 --episode-jsonl "$eval_dir/episodes.jsonl"
      --output "$eval_dir/eval_report.json"
    )
    if ! run_tracked "$eval_dir/eval.log" "${eval_cmd[@]}"; then
      [[ ! -e "$eval_dir/eval_report.json" ]] ||
        mv -- "$eval_dir/eval_report.json" "$eval_dir/eval_report.failed-attempt-1.json"
      [[ ! -e "$eval_dir/episodes.jsonl" ]] ||
        mv -- "$eval_dir/episodes.jsonl" "$eval_dir/episodes.failed-attempt-1.jsonl"
      run_tracked "$eval_dir/eval.retry.log" env CUDA_LAUNCH_BLOCKING=1 "${eval_cmd[@]}"
    fi
    (
      cd "$run_root"
      sha256sum \
        "$(realpath --relative-to="$run_root" "$checkpoint/model.safetensors")" \
        "$(realpath --relative-to="$run_root" "$checkpoint/optimizer.safetensors")" \
        "$(realpath --relative-to="$run_root" "$checkpoint/trainer_state.json")" \
        "$(realpath --relative-to="$run_root" "$arm_dir/config.json")" \
        "$(realpath --relative-to="$run_root" "$eval_dir/eval_report.json")" \
        "$(realpath --relative-to="$run_root" "$eval_dir/episodes.jsonl")"
    ) >"$eval_dir/sha256.txt"
    (cd "$run_root" && sha256sum --quiet -c "$(realpath --relative-to="$run_root" "$eval_dir/sha256.txt")")
  done
  jq '.status="complete"' "$arm_dir/arm.json" >"$arm_dir/arm.json.tmp"
  write_json_atomic "$arm_dir/arm.json.tmp" "$arm_dir/arm.json"
  record_stage "seed-$seed/$arm" passed "weight=$weight global_mix=$mix"
}

for arm in S0 cell-high global-high global-matched cell-matched; do run_arm 1 "$arm"; done

for seed in 1; do
  reference_content="$(jq -r .training_content_fingerprint "$run_root/seed-$seed/S0/train_report.json")"
  reference_rows="$(jq -r .training_population_rows "$run_root/seed-$seed/S0/train_report.json")"
  reference_params="$(jq -r .parameter_count "$run_root/seed-$seed/S0/train_report.json")"
  for arm in S0 cell-high global-high cell-matched global-matched; do
    [[ "$(jq -r .training_content_fingerprint "$run_root/seed-$seed/$arm/train_report.json")" == "$reference_content" ]]
    [[ "$(jq -r .training_population_rows "$run_root/seed-$seed/$arm/train_report.json")" == "$reference_rows" ]]
    [[ "$(jq -r .parameter_count "$run_root/seed-$seed/$arm/train_report.json")" == "$reference_params" ]]
    jq -S 'del(.output_dir,.sigreg_weight,.sigreg_global_mix)' "$run_root/seed-$seed/$arm/config.json" \
      >"$run_root/seed-$seed/$arm/normalized-config.json"
    cmp -s "$run_root/seed-$seed/S0/normalized-config.json" \
      "$run_root/seed-$seed/$arm/normalized-config.json"
  done
  for update in "$checkpoint_every" "$updates"; do
    eval_ref="$(jq -r .board_probe.population_fingerprint "$run_root/seed-$seed/S0/eval-update-$update/eval_report.json")"
    [[ "$eval_ref" != null && -n "$eval_ref" ]]
    for arm in S0 cell-high global-high cell-matched global-matched; do
      [[ "$(jq -r .board_probe.population_fingerprint "$run_root/seed-$seed/$arm/eval-update-$update/eval_report.json")" == "$eval_ref" ]]
    done
  done
done
kill -0 "$telemetry_pid" 2>/dev/null
(( $(awk 'END {print NR}' "$run_root/gpu.csv") >= 10 ))
record_stage final_integrity passed "paired content, rows, parameters, checkpoints, and evaluations verified"
finish_campaign complete_pending_analysis
printf 'SIGReg pressure x geometry v1 complete: %s\n' "$run_root"

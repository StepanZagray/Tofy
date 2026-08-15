#!/usr/bin/env bash
# Paired multi-seed cell-TC-QQ dose response. Descriptive total-treatment screen;
# it does not identify clipping mediation or reproduce a published SIGReg recipe.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
eval_validator="$script_dir/p2_validate_eval.sh"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
candle_root="${P2_CANDLE_ROOT:-$repo_root/../candle_graph}"
run_root="${P2_SIGREG_CELL_DOSE_ROOT:-$repo_root/runs/p2/sigreg-cell-dose-response-v1-$(date -u +%Y%m%dT%H%M%SZ)}"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"
eval_batch="${P2_SIGREG_EVAL_BATCH:-256}"

: "${P2_EXPECTED_SHA:?set the reviewed Tofy commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set the reviewed candle_graph commit}"
: "${P2_EXPECTED_BINARY_SHA:?set the reviewed release binary SHA-256}"
for command in bash git jq nvidia-smi sha256sum awk realpath date mkdir mv env cmp sleep timeout find sort xargs; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 2; }
done
[[ -x "$tofy_bin" ]] || { printf 'missing release binary: %s\n' "$tofy_bin" >&2; exit 2; }
[[ -r "$eval_validator" ]] || { printf 'missing evaluation validator: %s\n' "$eval_validator" >&2; exit 2; }
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
  --arg gpu "$gpu_name" '
  {schema:"p2.sigreg_cell_dose_response.v1",status:"running",started_utc:$started_utc,
   git_sha:$git_sha,candle_git_sha:$candle_sha,binary_sha256:$binary_sha,gpu_name:$gpu,
   physical_batch:1024,gradient_accumulation:1,updates_per_arm:250,checkpoints:[125,250],
   training_seeds:[2,3],evaluation_seed:424248,
   pressure_updates:[1,31,62,93,124,155,186,217,249],
   weights:[0,0.00004,0.00008,0.00016,0.00032305295536180014],
   order:{"2":["S0","w004","w008","w016","w0323"],
          "3":["w0323","w016","w008","w004","S0"]},
   interpretation:"paired multi-seed cell-TC-QQ fixed-dose response; not clipping mediation or paper reproduction",
   promotion:"locked_pending_completed_analysis"}' >"$run_root/campaign.json"

sample_gpu >>"$run_root/gpu.csv" &
telemetry_pid=$!
printf '%s\n' "$telemetry_pid" >"$run_root/telemetry.pid"

common_args=(
  --device cuda:0 --lessons q_calibration --steps-per-lesson 250
  --physical-batch 1024 --grad-accum 1 --steady-gpu
  --supervise-last-outer-only --residual-y-update --warm-start-y
  --outer-steps 8 --inner-steps 2 --hidden-dim 128 --action-dim 8
  --event-weight 0 --q-weight 0.1 --reliability-weight 0 --rollout-weight 0 --prefix-weight 0
  --patch-grounding-weight 0 --ptrm-rank-every 0 --ensemble-members 1
  --shuffled-episodes --consumer-readout global-mean
  --sigreg-target temporal-residual --sigreg-statistic quantile --sigreg-temporal-window 8
  --sigreg-spatial --sigreg-spatial-pool --sigreg-global-mix 0 --sigreg-max-rows 32768
)

arm_weight() {
  case "$1" in
    S0) printf '0\n' ;;
    w004) printf '0.00004\n' ;;
    w008) printf '0.00008\n' ;;
    w016) printf '0.00016\n' ;;
    w0323) printf '0.00032305295536180014\n' ;;
    *) return 2 ;;
  esac
}

run_arm() {
  local seed="$1" arm="$2" weight eval_seed arm_dir checkpoint eval_dir update
  local -a eval_cmd
  weight="$(arm_weight "$arm")"
  eval_seed=424248
  arm_dir="$run_root/seed-$seed/$arm"
  mkdir -p -- "$arm_dir"
  jq -nc --arg arm "$arm" --argjson seed "$seed" --argjson weight "$weight" \
    --argjson eval_seed "$eval_seed" \
    '{arm:$arm,training_seed:$seed,init_seed:$seed,sigreg_weight:$weight,
      sigreg_global_mix:0,evaluation_seed:$eval_seed,status:"running"}' >"$arm_dir/arm.json"

  run_tracked "$arm_dir/train.log" timeout --signal=TERM --kill-after=60s 90m \
    "$tofy_bin" p2-train --seed "$seed" --init-seed "$seed" \
    --checkpoint-every-steps 125 --profile-update 125 \
    --pressure-updates 1,31,62,93,124,155,186,217,249 --sigreg-weight "$weight" \
    "${common_args[@]}" --output-dir "$arm_dir"
  jq -e --argjson weight "$weight" '
    .global_step==250 and .status=="completed" and .physical_batch==1024 and .grad_accum==1
    and .parameter_count>0 and .training_population_rows>0
    and .experiment.sigreg.enabled==($weight>0) and .experiment.sigreg.global_mix==0
    and ([.gradient_pressure_samples[].update] == [1,31,62,93,124,155,186,217,249])
    and (if $weight>0 then [.gradient_pressure_samples[].sigreg_to_next_ratio]
      | all(type=="number" and .>0 and .<1e300)
      else [.gradient_pressure_samples[].sigreg_to_next_ratio] | all(.==null) end)
    and (.lessons|length)==1
    and (.lessons[0].mean_losses.gradient_clip_scale>0)
    and (.lessons[0].mean_losses.gradient_clip_scale<=1)
    and (.lessons[0].mean_losses.clipped_updates>=0)
    and (.lessons[0].mean_losses.clipped_updates<=1)
    and (.training_content_fingerprint|startswith("sha256:"))' "$arm_dir/train_report.json" >/dev/null
  jq -e --argjson weight "$weight" --argjson seed "$seed" '
    .sigreg_weight==$weight and .sigreg_global_mix==0
    and .seed==$seed and .init_seed==$seed' "$arm_dir/config.json" >/dev/null

  for update in 125 250; do
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
    if run_tracked "$eval_dir/eval.log" timeout --signal=TERM --kill-after=60s 30m \
      "${eval_cmd[@]}" && \
      bash "$eval_validator" "$eval_dir/eval_report.json" "$eval_dir/episodes.jsonl" "$eval_seed"; then
      :
    else
      [[ ! -e "$eval_dir/eval_report.json" ]] ||
        mv -- "$eval_dir/eval_report.json" "$eval_dir/eval_report.failed-attempt-1.json"
      [[ ! -e "$eval_dir/episodes.jsonl" ]] ||
        mv -- "$eval_dir/episodes.jsonl" "$eval_dir/episodes.failed-attempt-1.jsonl"
      run_tracked "$eval_dir/eval.retry.log" timeout --signal=TERM --kill-after=60s 30m \
        env CUDA_LAUNCH_BLOCKING=1 "${eval_cmd[@]}"
      bash "$eval_validator" "$eval_dir/eval_report.json" "$eval_dir/episodes.jsonl" "$eval_seed"
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
  record_stage "seed-$seed/$arm" passed "weight=$weight"
}

for arm in S0 w004 w008 w016 w0323; do run_arm 2 "$arm"; done
for arm in w0323 w016 w008 w004 S0; do run_arm 3 "$arm"; done

for seed in 2 3; do
  reference_content="$(jq -r .training_content_fingerprint "$run_root/seed-$seed/S0/train_report.json")"
  reference_rows="$(jq -r .training_population_rows "$run_root/seed-$seed/S0/train_report.json")"
  reference_params="$(jq -r .parameter_count "$run_root/seed-$seed/S0/train_report.json")"
  [[ "$reference_rows" == 256000 ]]
  for arm in S0 w004 w008 w016 w0323; do
    [[ "$(jq -r .training_content_fingerprint "$run_root/seed-$seed/$arm/train_report.json")" == "$reference_content" ]]
    [[ "$(jq -r .training_population_rows "$run_root/seed-$seed/$arm/train_report.json")" == "$reference_rows" ]]
    [[ "$(jq -r .parameter_count "$run_root/seed-$seed/$arm/train_report.json")" == "$reference_params" ]]
    jq -S 'del(.output_dir,.sigreg_weight)' "$run_root/seed-$seed/$arm/config.json" \
      >"$run_root/seed-$seed/$arm/normalized-config.json"
    cmp -s "$run_root/seed-$seed/S0/normalized-config.json" \
      "$run_root/seed-$seed/$arm/normalized-config.json"
  done
  for update in 125 250; do
    eval_ref="$(jq -r .board_probe.population_fingerprint "$run_root/seed-$seed/S0/eval-update-$update/eval_report.json")"
    [[ "$eval_ref" != null && -n "$eval_ref" ]]
    for arm in S0 w004 w008 w016 w0323; do
      [[ "$(jq -r .board_probe.population_fingerprint "$run_root/seed-$seed/$arm/eval-update-$update/eval_report.json")" == "$eval_ref" ]]
    done
  done
done
[[ "$(jq -r .training_content_fingerprint "$run_root/seed-2/S0/train_report.json")" != \
   "$(jq -r .training_content_fingerprint "$run_root/seed-3/S0/train_report.json")" ]]

evaluation_fingerprint="$(jq -r .board_probe.population_fingerprint \
  "$run_root/seed-2/S0/eval-update-125/eval_report.json")"
for seed in 2 3; do
  for arm in S0 w004 w008 w016 w0323; do
    for update in 125 250; do
      [[ "$(jq -r .board_probe.population_fingerprint \
        "$run_root/seed-$seed/$arm/eval-update-$update/eval_report.json")" == "$evaluation_fingerprint" ]]
    done
  done
done

for arm in S0 w004 w008 w016 w0323; do
  jq -S 'del(.output_dir,.seed,.init_seed)' "$run_root/seed-2/$arm/config.json" \
    >"$run_root/seed-2/$arm/cross-seed-config.json"
  jq -S 'del(.output_dir,.seed,.init_seed)' "$run_root/seed-3/$arm/config.json" \
    >"$run_root/seed-3/$arm/cross-seed-config.json"
  cmp -s "$run_root/seed-2/$arm/cross-seed-config.json" \
    "$run_root/seed-3/$arm/cross-seed-config.json"
done

: >"$run_root/dose-gates.jsonl"
: >"$run_root/outcome-gates.jsonl"
for seed in 2 3; do
  for arm in S0 w004 w008 w016 w0323; do
    weight="$(arm_weight "$arm")"
    jq -nc --arg arm "$arm" --argjson seed "$seed" --argjson weight "$weight" \
      --slurpfile treated "$run_root/seed-$seed/$arm/train_report.json" \
      --slurpfile control "$run_root/seed-$seed/S0/train_report.json" '
      def median: sort as $s | ($s|length) as $n |
        if $n==0 then null elif ($n%2)==1 then $s[($n/2|floor)]
        else (($s[$n/2-1]+$s[$n/2])/2) end;
      ($treated[0].lessons[0].mean_losses) as $tm |
      ($control[0].lessons[0].mean_losses) as $cm |
      ([$treated[0].gradient_pressure_samples[] | select(.sigreg_to_next_ratio!=null)
        | {update,ratio:.sigreg_to_next_ratio}]) as $samples |
      ($samples|map(.ratio)) as $ratios |
      (if ($samples|length)>1 then
        reduce range(0;($samples|length)-1) as $i (0;
          . + ((($samples[$i].ratio+$samples[$i+1].ratio)/2)
            *(($samples[$i+1].update-$samples[$i].update)/248)))
       else null end) as $auc |
      {seed:$seed,arm:$arm,weight:$weight,pressure_samples:$samples,pressure_auc:$auc,
       pressure_median:($ratios|median),pressure_max:(if ($ratios|length)>0 then ($ratios|max) else null end),
       clipped_fraction:$tm.clipped_updates,s0_clipped_fraction:$cm.clipped_updates,
       clipped_fraction_delta:($tm.clipped_updates-$cm.clipped_updates),
       mean_clip_scale:$tm.gradient_clip_scale,s0_mean_clip_scale:$cm.gradient_clip_scale,
       clip_scale_ratio:($tm.gradient_clip_scale/$cm.gradient_clip_scale),
       low_pressure_candidate:($weight>0
         and $tm.clipped_updates<=($cm.clipped_updates+0.10)
         and $tm.gradient_clip_scale>=($cm.gradient_clip_scale*0.75)
         and $auc<=0.10 and ($ratios|median)<=0.10 and ($ratios|max)<=0.25)}' >>"$run_root/dose-gates.jsonl"

    for update in 125 250; do
      jq -c --arg arm "$arm" --argjson seed "$seed" --argjson weight "$weight" \
        --argjson update "$update" '
        def finite_number: type=="number" and .>-1e300 and .<1e300;
        .synthetic_dynamics as $d |
        {seed:$seed,arm:$arm,weight:$weight,update:$update,
         noncollapse:$d.representation.noncollapse_pass,
         rank_fraction:$d.representation.effective_rank_fraction,
         changed_improvement:$d.changed_transitions.improvement_fraction,
         changed_ci95_low:$d.changed_transitions.improvement_ci95_low,
         changed_pass:$d.changed_transitions.ten_percent_improvement_pass,
         action_ratio:$d.action_diagnostics.aggregate.shuffle.ratio,
         action_ci95_low:$d.action_diagnostics.aggregate.shuffle.ratio_ci95_low,
         action_pass:$d.action_diagnostics.aggregate.shuffle.action_conditioning_pass,
         h4_mean:$d.rollout.h4.normalized_mean,h4_cvar95:$d.rollout.h4.normalized_cvar95,
         h8_mean:$d.rollout.h8.normalized_mean,h8_cvar95:$d.rollout.h8.normalized_cvar95,
         h8_fraction_beating_copy:$d.rollout.h8.fraction_beating_copy,
         board_trusted:.board_probe.metrics.trusted,q_saturated:$d.q.saturated,
         q_positive_label_rate:$d.q.positive_label_rate,
         q_balanced_accuracy:$d.q.balanced_accuracy,q_brier:$d.q.brier,
         q_pass:($d.q.saturated==false and (($d.q.balanced_accuracy // 0)>0.5)),
         promotion_pass:($d.representation.noncollapse_pass==true
           and $d.changed_transitions.ten_percent_improvement_pass==true
           and $d.action_diagnostics.aggregate.shuffle.action_conditioning_pass==true
           and ($d.rollout.h4.normalized_mean|finite_number) and $d.rollout.h4.normalized_mean<=1
           and ($d.rollout.h4.normalized_cvar95|finite_number) and $d.rollout.h4.normalized_cvar95<=1
           and ($d.rollout.h8.normalized_mean|finite_number) and $d.rollout.h8.normalized_mean<=1
           and ($d.rollout.h8.normalized_cvar95|finite_number) and $d.rollout.h8.normalized_cvar95<=1
           and $d.q.saturated==false and (($d.q.balanced_accuracy // 0)>0.5)
           and .board_probe.metrics.trusted==true)}' \
        "$run_root/seed-$seed/$arm/eval-update-$update/eval_report.json" >>"$run_root/outcome-gates.jsonl"
    done
  done
done
jq -s '{schema:"p2.sigreg_cell_dose_gates.v1",arms:.}' "$run_root/dose-gates.jsonl" \
  >"$run_root/dose-gates.json.tmp"
write_json_atomic "$run_root/dose-gates.json.tmp" "$run_root/dose-gates.json"
jq -s '{schema:"p2.sigreg_cell_outcome_gates.v1",checkpoints:.}' "$run_root/outcome-gates.jsonl" \
  >"$run_root/outcome-gates.json.tmp"
write_json_atomic "$run_root/outcome-gates.json.tmp" "$run_root/outcome-gates.json"
jq -e '.arms|length==10' "$run_root/dose-gates.json" >/dev/null
jq -e '.checkpoints|length==20' "$run_root/outcome-gates.json" >/dev/null

jq -n --slurpfile doses "$run_root/dose-gates.jsonl" \
  --slurpfile outcomes "$run_root/outcome-gates.jsonl" '
  ["S0","w004","w008","w016","w0323"] as $arms |
  {schema:"p2.sigreg_cell_cross_seed_gates.v1",automatic_promotion:false,
   requires_completed_causal_analysis:true,
   persistence_rule:"all registered promotion gates pass at updates 125 and 250 in both seeds",
   arms:[$arms[] as $arm |
     ($doses|map(select(.arm==$arm))) as $dose |
     ($outcomes|map(select(.arm==$arm))) as $outcome |
     {arm:$arm,weight:$dose[0].weight,
      low_pressure_both_seeds:(($dose|length)==2
        and ($dose|all(.low_pressure_candidate==true))),
      outcomes_all_checkpoints_both_seeds:(($outcome|length)==4
        and ($outcome|all(.promotion_pass==true))),
      selection_candidate:($dose[0].weight>0 and ($dose|length)==2
        and ($dose|all(.low_pressure_candidate==true)) and ($outcome|length)==4
        and ($outcome|all(.promotion_pass==true))) }]}' >"$run_root/cross-seed-gates.json.tmp"
write_json_atomic "$run_root/cross-seed-gates.json.tmp" "$run_root/cross-seed-gates.json"
jq -e '.automatic_promotion==false and (.arms|length)==5' "$run_root/cross-seed-gates.json" >/dev/null

kill -0 "$telemetry_pid" 2>/dev/null
(( $(awk 'END {print NR}' "$run_root/gpu.csv") >= 10 ))
kill "$telemetry_pid"
wait "$telemetry_pid" || true
telemetry_pid=""
record_stage final_integrity passed "all artifacts, paired data/config/evaluations, and cross-seed treatments verified"
finish_campaign complete_pending_analysis
(cd "$run_root" && find . -type f ! -name root-sha256.txt -print0 | sort -z | xargs -0 sha256sum) \
  >"$run_root/root-sha256.txt"
(cd "$run_root" && sha256sum --quiet -c root-sha256.txt)
campaign_finalized=true
printf 'SIGReg cell dose response v1 complete: %s\n' "$run_root"

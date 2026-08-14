#!/usr/bin/env bash
# Run one checkpoint-only P2 evaluation and preserve an isolated blocking retry.
set -euo pipefail

if (( $# < 3 )); then
  printf 'usage: %s CHECKPOINT TRAIN_CONFIG OUTPUT_DIR [p2-eval flags...]\n' "$0" >&2
  exit 2
fi
checkpoint="$1"
train_config="$2"
output_dir="$3"
shift 3
tofy_bin="${TOFY_BIN:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/target/release/tofy}"
eval_timeout="${P2_EVAL_TIMEOUT:-6h}"

for command in jq sha256sum timeout nvidia-smi awk seq sleep; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 2; }
done
[[ -x "$tofy_bin" && -s "$checkpoint" && -s "$train_config" ]] || exit 2
[[ ! -e "$output_dir" ]] || { printf 'output exists: %s\n' "$output_dir" >&2; exit 2; }
mkdir -p -- "$output_dir/primary" "$output_dir/recovery"
checkpoint_sha="$(sha256sum "$checkpoint" | awk '{print $1}')"
config_sha="$(sha256sum "$train_config" | awk '{print $1}')"

run_attempt() {
  local attempt_dir="$1" blocking="$2"
  shift 2
  local -a command=("$tofy_bin" p2-eval --checkpoint "$checkpoint"
    --train-config "$train_config" --output "$attempt_dir/eval_report.json"
    --episode-jsonl "$attempt_dir/episodes.jsonl" "$@")
  if [[ "$blocking" == true ]]; then
    CUDA_LAUNCH_BLOCKING=1 timeout --foreground "$eval_timeout" "${command[@]}" \
      >"$attempt_dir/eval.log" 2>&1
  else
    timeout --foreground "$eval_timeout" "${command[@]}" >"$attempt_dir/eval.log" 2>&1
  fi
}

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

selected=""
status=""
if run_attempt "$output_dir/primary" false "$@"; then
  selected="$output_dir/primary"
  status="primary_succeeded"
else
  wait_for_idle_gpu
  if run_attempt "$output_dir/recovery" true "$@"; then
    selected="$output_dir/recovery"
    status="recovered_after_primary_failure"
  else
    status="recovery_failed"
  fi
fi

jq -nc --arg status "$status" --arg selected "$selected" --arg checkpoint "$checkpoint" \
  --arg checkpoint_sha256 "$checkpoint_sha" --arg train_config "$train_config" \
  --arg train_config_sha256 "$config_sha" --arg binary "$tofy_bin" \
  '{schema:"p2.eval_recovery_receipt.v1",status:$status,selected_attempt:$selected,
    checkpoint:$checkpoint,checkpoint_sha256:$checkpoint_sha256,
    train_config:$train_config,train_config_sha256:$train_config_sha256,binary:$binary,
    primary_failure_preserved:($status=="recovered_after_primary_failure")}' \
  >"$output_dir/receipt.json.tmp"
mv -- "$output_dir/receipt.json.tmp" "$output_dir/receipt.json"
[[ "$status" != recovery_failed ]] || exit 1
[[ -s "$selected/eval_report.json" && -f "$selected/episodes.jsonl" ]] || exit 1
[[ "$(sha256sum "$checkpoint" | awk '{print $1}')" == "$checkpoint_sha" ]] || exit 1
[[ "$(sha256sum "$train_config" | awk '{print $1}')" == "$config_sha" ]] || exit 1
sha256sum "$checkpoint" "$train_config" "$selected/eval_report.json" \
  "$selected/episodes.jsonl" "$output_dir/receipt.json" >"$output_dir/SHA256SUMS"
printf 'p2 evaluation %s; selected=%s\n' "$status" "$selected"

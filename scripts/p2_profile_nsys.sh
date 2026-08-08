#!/usr/bin/env bash
# Capture optional Nsight evidence for one Tofy representative-update bundle.
# Usage: scripts/p2_profile_nsys.sh OUTPUT_DIR [--] COMMAND [ARG...]
set -euo pipefail

usage() { printf 'usage: %s OUTPUT_DIR [--] COMMAND [ARG...]\n' "${0##*/}" >&2; }
[[ $# -ge 2 ]] || { usage; exit 64; }
output_dir=$1
shift
[[ "${1:-}" == -- ]] && shift
[[ $# -gt 0 ]] || { usage; exit 64; }
command=("$@")

profile_update=${P2_PROFILE_UPDATE:-2}
[[ "$profile_update" =~ ^[1-9][0-9]*$ ]] || {
  printf 'P2_PROFILE_UPDATE must be a one-based integer, got %q\n' "$profile_update" >&2
  exit 64
}
profile_update=$((10#$profile_update))
printf -v update_name 'update-%012d' "$profile_update"

nsys_mode=${P2_NSYS:-auto}
case "$nsys_mode" in off|auto|require) ;; *) printf 'P2_NSYS must be off, auto, or require\n' >&2; exit 64 ;; esac
nsys_bin=${NSYS_BIN:-nsys}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
candle_graph_manifest="$repo_root/../candle_graph/Cargo.toml"
profile_root="$output_dir/profile"
bundle_dir="$profile_root/$update_name"
trace_file="$bundle_dir/application.jsonl"
nvtx_root="tofy.p2/$update_name"

# A published representative update will never emit its NVTX capture range again on resume.
# Preserve normal resume behavior instead of placing the remaining run under a dead capture.
if [[ -f "$trace_file" && -f "$bundle_dir/evidence.json" && -f "$bundle_dir/viewer.html" ]]; then
  exec "${command[@]}"
fi

mkdir -p -- "$profile_root"
nsight_stage=$(mktemp -d "$profile_root/.${update_name}.nsight.XXXXXX")
status_file="$nsight_stage/status.txt"
child_status=$(mktemp -d "$profile_root/.${update_name}.child.XXXXXX")
child_started="$child_status/started"
child_exit="$child_status/exit"

cleanup() {
  local rc=$?
  rm -rf -- "$child_status"
  # Retain an unpublished Nsight staging directory after early failure for diagnosis.
  if [[ -d "$nsight_stage" ]] && [[ -z "$(find "$nsight_stage" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    rmdir -- "$nsight_stage"
  fi
  exit "$rc"
}
trap cleanup EXIT

write_status() {
  local state=$1 reason=$2
  {
    printf 'state=%s\nreason=%s\nmode=%s\nnvtx_root=%s\nupdated_at=%s\n' \
      "$state" "$reason" "$nsys_mode" "$nvtx_root" "$(date -Iseconds)"
  } >"$status_file"
}
append_status() { printf '%s=%s\n' "$1" "$2" >>"$status_file"; }

export P2_PROFILE_UPDATE="$profile_update"
export TOFY_NSYS_NVTX_ROOT="$nvtx_root"
child_rc=0
profile_rc=0

run_direct() {
  set +e
  "${command[@]}"
  child_rc=$?
  set -e
}

if [[ "$nsys_mode" == off ]]; then
  write_status disabled 'P2_NSYS=off; application evidence only'
  run_direct
elif ! command -v "$nsys_bin" >/dev/null 2>&1; then
  write_status unavailable "Nsight executable not found: $nsys_bin"
  if [[ "$nsys_mode" == require ]]; then exit 127; fi
  run_direct
else
  raw_stem="$nsight_stage/capture-$(date -u +%Y%m%dT%H%M%SZ)"
  write_status capturing "capturing $nvtx_root"
  set +e
  P2_NSYS_CHILD_STARTED="$child_started" P2_NSYS_CHILD_EXIT="$child_exit" \
    "$nsys_bin" profile \
      --trace=cuda,nvtx,osrt,cudnn,cublas \
      --capture-range=nvtx \
      --nvtx-capture="$nvtx_root" \
      --stop-on-range-end=true \
      --output "$raw_stem" \
      bash -c '
        printf "started\n" >"$P2_NSYS_CHILD_STARTED"
        set +e
        "$@"
        rc=$?
        printf "%s\n" "$rc" >"$P2_NSYS_CHILD_EXIT"
        exit "$rc"
      ' p2-profile-child "${command[@]}"
  profile_rc=$?
  set -e

  if [[ -s "$child_exit" ]]; then
    child_rc=$(<"$child_exit")
    raw_report="$raw_stem.nsys-rep"
    write_status captured "raw report retained: ${raw_report##*/}"
    append_status profiler_exit "$profile_rc"
    append_status child_exit "$child_rc"
    if [[ -f "$raw_report" ]]; then
      set +e
      "$nsys_bin" stats --format csv --output "$nsight_stage/nsys" \
        --report cuda_gpu_trace \
        --report cuda_gpu_kern_sum \
        --report cuda_api_sum \
        --report cuda_gpu_mem_time_sum \
        --report nvtx_gpu_proj_trace \
        "$raw_report"
      stats_rc=$?
      set -e
      append_status stats_exit "$stats_rc"
      [[ "$stats_rc" -eq 0 ]] || profile_rc=$stats_rc
    else
      append_status stats_error 'raw .nsys-rep was not produced'
      profile_rc=1
    fi
  elif [[ "$nsys_mode" == auto ]]; then
    write_status unavailable "Nsight ended without a child result (exit $profile_rc); reran training without it"
    run_direct
  else
    write_status failed "Nsight ended before child status was recorded (exit $profile_rc)"
    child_rc=$profile_rc
  fi
fi

if [[ -d "$bundle_dir" ]]; then
  augment_dir=$(mktemp -d "$profile_root/.${update_name}.augment.XXXXXX")
  cp -a -- "$bundle_dir/." "$augment_dir/"
  augmented_nsight="$augment_dir/nsight"
  mkdir -p -- "$augmented_nsight"
  shopt -s dotglob nullglob
  staged=("$nsight_stage"/*)
  shopt -u dotglob nullglob
  if [[ ${#staged[@]} -gt 0 ]]; then mv -- "${staged[@]}" "$augmented_nsight/"; fi
  rmdir -- "$nsight_stage"
  status_file="$augmented_nsight/status.txt"

  augmented_trace="$augment_dir/application.jsonl"
  if [[ -f "$augmented_trace" ]]; then
    set +e
    (
      cd -- "$repo_root"
      cargo run --quiet --manifest-path "$candle_graph_manifest" --bin candle-graph -- \
        report "$augmented_trace" --nsight-dir "$augmented_nsight" \
        --json "$augment_dir/evidence.json" --markdown "$augment_dir/EVIDENCE.md" &&
      cargo run --quiet --manifest-path "$candle_graph_manifest" --bin candle-graph -- \
        view "$augmented_trace" --nsight-dir "$augmented_nsight" \
        --output "$augment_dir/viewer.html"
    )
    evidence_rc=$?
    set -e
    append_status evidence_exit "$evidence_rc"
    if [[ "$evidence_rc" -eq 0 ]]; then
      mv --exchange --no-copy --no-target-directory -- "$augment_dir" "$bundle_dir"
      case "$augment_dir" in
        "$profile_root"/."$update_name".augment.*) rm -rf -- "$augment_dir" ;;
        *) printf 'refusing to remove unexpected augmentation path: %s\n' "$augment_dir" >&2; exit 70 ;;
      esac
    else
      profile_rc=$evidence_rc
    fi
  else
    append_status evidence_error 'application.jsonl was not produced'
  fi
else
  append_status bundle_error "profile bundle was not published: $bundle_dir"
fi

[[ "$child_rc" -eq 0 ]] || exit "$child_rc"
if [[ "$nsys_mode" == require && "$profile_rc" -ne 0 ]]; then exit "$profile_rc"; fi
exit 0

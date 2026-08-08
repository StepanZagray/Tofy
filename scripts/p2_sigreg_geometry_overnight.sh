#!/usr/bin/env bash
# Run the preregistered geometry A/B adaptively, pairing arms on one large GPU.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
ab_root="${P2_AB_ROOT:-$repo_root/runs/p2/ab-sigreg-geometry-v2}"
gate="$script_dir/p2_ab_gate.py"
runner="$script_dir/p2_sigreg_geometry_ab.sh"

: "${P2_EXPECTED_SHA:?set P2_EXPECTED_SHA to the reviewed Tofy commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set P2_EXPECTED_CANDLE_SHA to the reviewed candle_graph commit}"
: "${P2_EXPECTED_BINARY_SHA:?set P2_EXPECTED_BINARY_SHA to the reviewed tofy binary hash}"
mkdir -p -- "$ab_root/gates"

run_pair() {
  local seed="$1" target="$2" control_pid treatment_pid control_status treatment_status
  printf '[%s] starting paired geometry arms seed=%s target=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$seed" "$target"
  P2_AB_ROOT="$ab_root" P2_AB_TARGET_UPDATE="$target" \
    P2_GEOMETRY_EXTENSION_APPROVED="$(if [[ "$target" == 4000 ]]; then printf 1; fi)" \
    "$runner" control "$seed" &
  control_pid=$!
  P2_AB_ROOT="$ab_root" P2_AB_TARGET_UPDATE="$target" \
    P2_GEOMETRY_EXTENSION_APPROVED="$(if [[ "$target" == 4000 ]]; then printf 1; fi)" \
    "$runner" pre-rms-spatial "$seed" &
  treatment_pid=$!
  if wait "$control_pid"; then control_status=0; else control_status=$?; fi
  if wait "$treatment_pid"; then treatment_status=0; else treatment_status=$?; fi
  if ((control_status != 0 || treatment_status != 0)); then
    printf 'paired run failed: seed=%s control=%s treatment=%s\n' \
      "$seed" "$control_status" "$treatment_status" >&2
    return 1
  fi
}

run_gate() {
  local label="$1" final_update="$2"
  shift 2
  python3 "$gate" \
    --root "$ab_root" \
    --treatment-arm pre-rms-spatial \
    --seeds "$@" \
    --final-update "$final_update" \
    --output-json "$ab_root/gates/$label.json" \
    --output-md "$ab_root/gates/$label.md"
}

run_pair 1 2000
run_gate pilot 2000 1
action="$(jq -r '.decision.action' "$ab_root/gates/pilot.json")"
if [[ "$action" == stop_after_pilot ]]; then
  printf '[%s] preregistered pilot stop reached\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 0
fi
[[ "$action" == replicate_both_arms_seeds_2_and_3 ]] || {
  printf 'unexpected pilot action: %s\n' "$action" >&2; exit 2;
}

# Each historical arm peaked near 17.3 GiB, so one control/treatment pair fits
# the 46 GiB L40S. Seeds remain sequential to avoid exceeding memory.
run_pair 2 2000
run_pair 3 2000
run_gate three-seed 2000 1 2 3
action="$(jq -r '.decision.action' "$ab_root/gates/three-seed.json")"
if [[ "$action" == extend_all_six_to_4000 ]]; then
  for seed in 1 2 3; do
    run_pair "$seed" 4000
  done
  run_gate post-extension 4000 1 2 3
elif [[ "$action" != stop_at_2000 ]]; then
  printf 'unexpected three-seed action: %s\n' "$action" >&2
  exit 2
fi

printf '[%s] geometry overnight queue complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

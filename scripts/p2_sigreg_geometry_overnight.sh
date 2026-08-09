#!/usr/bin/env bash
# Run the preregistered geometry A/B adaptively, one GPU-heavy arm at a time.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
ab_root="${P2_AB_ROOT:-$repo_root/runs/p2/ab-sigreg-geometry-v2}"
gate="$script_dir/p2_ab_gate.py"
runner="$script_dir/p2_sigreg_geometry_ab.sh"
smoke="$script_dir/p2_sigreg_geometry_smoke.sh"

: "${P2_EXPECTED_SHA:?set P2_EXPECTED_SHA to the reviewed Tofy commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set P2_EXPECTED_CANDLE_SHA to the reviewed candle_graph commit}"
: "${P2_EXPECTED_BINARY_SHA:?set P2_EXPECTED_BINARY_SHA to the reviewed tofy binary hash}"
mkdir -p -- "$ab_root/gates"

run_arm_with_recovery() {
  local arm="$1" seed="$2" target="$3" status stage update phase_log phase_count recovered_updates pending_stage
  phase_log="$ab_root/seed-$seed/$arm/phases.jsonl"
  recovered_updates=" "
  while true; do
    phase_count=0
    pending_stage=""
    if [[ -f "$phase_log" ]]; then
      phase_count="$(wc -l <"$phase_log")"
      if ! pending_stage="$(jq -s -r \
        'last | select((.status | tostring) != "0") | .stage // empty' "$phase_log")"; then
        printf 'cannot parse phase history: %s\n' "$phase_log" >&2
        return 2
      fi
    fi
    if [[ "$pending_stage" == eval-* \
      && "$recovered_updates" != *" ${pending_stage#eval-} "* ]]; then
      stage="$pending_stage"
      status=1
      printf '[%s] found pending failed %s; entering isolated recovery: seed=%s arm=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$stage" "$seed" "$arm" >&2
    else
      if P2_AB_ROOT="$ab_root" P2_AB_TARGET_UPDATE="$target" \
        P2_GEOMETRY_EXTENSION_APPROVED="$(if [[ "$target" == 4000 ]]; then printf 1; fi)" \
        "$runner" "$arm" "$seed"; then
        return 0
      else
        status=$?
      fi
      stage=""
      if [[ -f "$phase_log" ]] \
        && ! stage="$(jq -s -r --argjson before "$phase_count" \
          '.[ $before:] | last | select((.status | tostring) != "0") | .stage // empty' \
          "$phase_log")"; then
        printf 'cannot parse phases appended by failed arm invocation: %s\n' "$phase_log" >&2
        return "$status"
      fi
    fi
    if [[ "$stage" != eval-* ]]; then
      printf 'arm failed outside evaluation; not retrying: seed=%s arm=%s target=%s status=%s stage=%s\n' \
        "$seed" "$arm" "$target" "$status" "${stage:-unknown}" >&2
      return "$status"
    fi
    update="${stage#eval-}"
    if [[ "$recovered_updates" == *" $update "* ]]; then
      printf 'evaluation already received its one recovery sequence: seed=%s arm=%s update=%s\n' \
        "$seed" "$arm" "$update" >&2
      return "$status"
    fi
    recovered_updates+="$update "
    printf '[%s] retrying failed %s with synchronous CUDA diagnostics: seed=%s arm=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$stage" "$seed" "$arm" >&2
    if CUDA_LAUNCH_BLOCKING=1 RUST_BACKTRACE=full \
      P2_AB_ROOT="$ab_root" P2_AB_TARGET_UPDATE="$target" \
      P2_AB_RECOVERY_ONLY_UPDATE="$update" \
      P2_GEOMETRY_EXTENSION_APPROVED="$(if [[ "$target" == 4000 ]]; then printf 1; fi)" \
      "$runner" "$arm" "$seed"; then
      :
    else
      status=$?
      printf 'diagnostic recovery failed: seed=%s arm=%s update=%s status=%s\n' \
        "$seed" "$arm" "$update" "$status" >&2
      return "$status"
    fi
    printf '[%s] repeating recovered %s for deterministic checksum verification\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$stage" >&2
    if CUDA_LAUNCH_BLOCKING=1 RUST_BACKTRACE=full \
      P2_AB_ROOT="$ab_root" P2_AB_TARGET_UPDATE="$target" \
      P2_AB_RECOVERY_ONLY_UPDATE="$update" P2_AB_REPEAT_EVAL_UPDATE="$update" \
      P2_GEOMETRY_EXTENSION_APPROVED="$(if [[ "$target" == 4000 ]]; then printf 1; fi)" \
      "$runner" "$arm" "$seed"; then
      :
    else
      status=$?
      printf 'deterministic evaluation repeat failed: seed=%s arm=%s update=%s status=%s\n' \
        "$seed" "$arm" "$update" "$status" >&2
      return "$status"
    fi
  done
}

run_seed() {
  local seed="$1" target="$2"
  printf '[%s] starting sequential geometry arms seed=%s target=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$seed" "$target"
  run_arm_with_recovery control "$seed" "$target"
  run_arm_with_recovery pre-rms-spatial "$seed" "$target"
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

if [[ "${P2_GEOMETRY_SKIP_SMOKE:-0}" != 1 ]]; then
  P2_SMOKE_ROOT="${P2_SMOKE_ROOT:-$ab_root/preflight/${P2_EXPECTED_BINARY_SHA:0:12}}" "$smoke"
fi

run_seed 1 2000
run_gate pilot 2000 1
action="$(jq -r '.decision.action' "$ab_root/gates/pilot.json")"
if [[ "$action" == stop_after_pilot ]]; then
  printf '[%s] preregistered pilot stop reached\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 0
fi
[[ "$action" == replicate_both_arms_seeds_2_and_3 ]] || {
  printf 'unexpected pilot action: %s\n' "$action" >&2; exit 2;
}

# Serialize every arm: the failed run showed that nominal aggregate capacity did
# not protect concurrent evaluation from allocator or kernel-launch failures.
run_seed 2 2000
run_seed 3 2000
run_gate three-seed 2000 1 2 3
action="$(jq -r '.decision.action' "$ab_root/gates/three-seed.json")"
if [[ "$action" == extend_all_six_to_4000 ]]; then
  for seed in 1 2 3; do
    run_seed "$seed" 4000
  done
  run_gate post-extension 4000 1 2 3
elif [[ "$action" != stop_at_2000 ]]; then
  printf 'unexpected three-seed action: %s\n' "$action" >&2
  exit 2
fi

printf '[%s] geometry overnight queue complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

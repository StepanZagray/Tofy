#!/usr/bin/env bash
# Shared P2 output root. All train/eval run directories live here (gitignored via runs/).
#
# Usage in other scripts:
#   source "$script_dir/p2_runs.sh"
#   OUT="${P2_OUTPUT_DIR:-$(p2_run_dir readiness-v2)}"
#
# Override root: P2_RUNS_ROOT=artifacts/p2 scripts/p2_readiness_train.sh run

P2_RUNS_ROOT="${P2_RUNS_ROOT:-runs/p2}"

p2_run_dir() {
  local name="${1:?p2 run name required}"
  echo "${P2_RUNS_ROOT}/${name}"
}

p2_ensure_runs_root() {
  mkdir -p -- "$P2_RUNS_ROOT"
}

p2_legacy_run_dir() {
  local name="${1:?p2 run name required}"
  echo "p2-output-${name}"
}

# Prefer runs/p2/<name>; auto-move legacy p2-output-<name> on first resume.
p2_migrate_legacy_run_dir() {
  local name="${1:?p2 run name required}"
  local new_dir legacy_dir
  new_dir="$(p2_run_dir "$name")"
  legacy_dir="$(p2_legacy_run_dir "$name")"
  p2_ensure_runs_root
  if [[ -d "$legacy_dir" && ! -e "$new_dir" ]]; then
    mv -- "$legacy_dir" "$new_dir"
  fi
  if [[ -d "$new_dir" ]]; then
    echo "$new_dir"
  elif [[ -d "$legacy_dir" ]]; then
    echo "$legacy_dir"
  else
    echo "$new_dir"
  fi
}

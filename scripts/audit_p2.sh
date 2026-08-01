#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
analyzer_manifest="$repo_root/../candleModelAnalyzer/Cargo.toml"
output_dir="${1:-$repo_root/runs/p2/analyzer}"
checkpoint="${2:-}"
runtime_trace="${3:-}"

if [[ ! -f "$analyzer_manifest" ]]; then
  echo "candleModelAnalyzer not found at $analyzer_manifest" >&2
  exit 2
fi

mkdir -p -- "$output_dir"
analyzer=(cargo run --quiet --release --manifest-path "$analyzer_manifest" -- "$repo_root")

"${analyzer[@]}" --query summary --format json --output "$output_dir/summary.json"
"${analyzer[@]}" --query doctor --format json --output "$output_dir/doctor.json"
"${analyzer[@]}" --model-ir --format json --output "$output_dir/model-ir.json"
"${analyzer[@]}" \
  --root WorldModel \
  --entry WorldModel::forward \
  --format json \
  --output "$output_dir/world-model.json"

if [[ -n "$checkpoint" ]]; then
  if [[ ! -f "$checkpoint" ]]; then
    echo "checkpoint not found: $checkpoint" >&2
    exit 2
  fi
  "${analyzer[@]}" \
    --root WorldModel \
    --verify "$checkpoint" \
    --verify-root vb \
    --format json \
    --output "$output_dir/checkpoint.json"
fi

if [[ -n "$runtime_trace" ]]; then
  if [[ ! -f "$runtime_trace" ]]; then
    echo "runtime trace not found: $runtime_trace" >&2
    exit 2
  fi
  "${analyzer[@]}" \
    --runtime-trace "$runtime_trace" \
    --query runtime \
    --format json \
    --output "$output_dir/runtime.json"
fi

echo "P2 analyzer reports: $output_dir"

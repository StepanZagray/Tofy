#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
validator="$script_dir/p2_validate_replay.sh"
fixture_dir="$(mktemp -d /tmp/tofy-replay-test.XXXXXX)"
cleanup() {
  rm -rf -- "$fixture_dir"
}
trap cleanup EXIT

printf '%s\n' '{"schema":"test.v1","id":7,"diagnostic_rows":12,"metric":1.0,"nested":{"pass":false,"score":2.0}}' \
  >"$fixture_dir/reference.json"
printf '%s\n' '{"schema":"test.v1","id":7,"diagnostic_rows":12,"metric":4.0,"nested":{"pass":false,"score":2.0}}' \
  >"$fixture_dir/nondiagnostic-drift.json"
printf '%s\n' '{"schema":"test.v1","id":8,"diagnostic_rows":12,"metric":1.0,"nested":{"pass":false,"score":2.0}}' \
  >"$fixture_dir/identity-drift.json"
printf '%s\n' '{"schema":"test.v1","id":7,"diagnostic_rows":12,"metric":1.0,"nested":{"pass":true,"score":2.0}}' \
  >"$fixture_dir/decision-drift.json"
printf '%s\n' '{"schema":"test.v1","id":7,"diagnostic_rows":13,"metric":1.0,"nested":{"pass":false,"score":2.0}}' \
  >"$fixture_dir/count-drift.json"

bash "$validator" structure-identity "$fixture_dir/reference.json" \
  "$fixture_dir/nondiagnostic-drift.json" "$fixture_dir/structure"
if bash "$validator" numeric "$fixture_dir/reference.json" \
  "$fixture_dir/nondiagnostic-drift.json" "$fixture_dir/numeric" 0.000001 0.10 \
  >/dev/null 2>&1; then
  printf 'numeric replay unexpectedly accepted out-of-envelope drift\n' >&2
  exit 1
fi
if bash "$validator" structure-identity "$fixture_dir/reference.json" \
  "$fixture_dir/identity-drift.json" "$fixture_dir/identity" >/dev/null 2>&1; then
  printf 'structure replay unexpectedly accepted identity drift\n' >&2
  exit 1
fi
if bash "$validator" structure-identity "$fixture_dir/reference.json" \
  "$fixture_dir/decision-drift.json" "$fixture_dir/decision" >/dev/null 2>&1; then
  printf 'structure replay unexpectedly accepted decision drift\n' >&2
  exit 1
fi
if bash "$validator" structure-identity "$fixture_dir/reference.json" \
  "$fixture_dir/count-drift.json" "$fixture_dir/count" >/dev/null 2>&1; then
  printf 'structure replay unexpectedly accepted count drift\n' >&2
  exit 1
fi

printf 'p2 replay validator tests passed\n'

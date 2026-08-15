#!/usr/bin/env bash
# Run the repaired exploratory parent and, only after it seals successfully,
# launch the fixed fresh-evaluator-seed confirmation.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
parent_root="${P2_ACTION_PREMISE_ROOT:?set a never-used exploratory parent root}"
confirmation_root="${P2_ACTION_CONFIRM_ROOT:?set a never-used confirmation root}"

[[ "$parent_root" != "$confirmation_root" ]] || {
  printf 'parent and confirmation roots must differ\n' >&2
  exit 2
}
[[ "${P2_ACTION_PREMISE_UPDATES:-}" == 250 ]] || {
  printf 'the confirmation sequence requires P2_ACTION_PREMISE_UPDATES=250\n' >&2
  exit 2
}

bash "$script_dir/p2_action_premise_rescore_v1.sh"

P2_ACTION_CONFIRM_PARENT_RUN="$parent_root" \
P2_EXPECTED_PARENT_SHA="$P2_EXPECTED_SHA" \
  bash "$script_dir/p2_action_premise_confirm_v1.sh"

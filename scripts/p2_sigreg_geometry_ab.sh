#!/usr/bin/env bash
# Run the preregistered SIGReg geometry-isolation control or treatment arm.
set -euo pipefail

arm="${1:?usage: $0 <control|pre-rms-spatial> <training-seed>}"
case "$arm" in
  control|pre-rms-spatial) ;;
  *) printf 'invalid geometry arm: %s\n' "$arm" >&2; exit 2 ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
if [[ "${P2_AB_TARGET_UPDATE:-2000}" == 4000 && "${P2_GEOMETRY_EXTENSION_APPROVED:-}" != 1 ]]; then
  printf 'geometry update 4000 requires P2_GEOMETRY_EXTENSION_APPROVED=1 after the three-seed gate\n' >&2
  exit 2
fi
export P2_AB_EXPERIMENT=geometry
export P2_AB_ROOT="${P2_AB_ROOT:-$repo_root/runs/p2/ab-sigreg-geometry-v2}"
exec "$script_dir/p2_sigreg_action_ab.sh" "$@"

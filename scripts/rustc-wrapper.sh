#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "rustc-wrapper: missing rustc path" >&2
  exit 1
fi

RUSTC_BIN="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIAG_DIR="${REPO_ROOT}/target/rustc-diagnostics"

set +e
"${RUSTC_BIN}" "$@"
status=$?
set -e

mkdir -p "${DIAG_DIR}"
shopt -s nullglob
for path in "${REPO_ROOT}"/*.long-type*.txt; do
  [[ -f "${path}" ]] || continue
  mv -f "${path}" "${DIAG_DIR}/"
done
shopt -u nullglob

exit "${status}"

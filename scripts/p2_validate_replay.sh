#!/usr/bin/env bash
# Compare cross-binary evaluation artifacts at an explicitly selected scope.
set -euo pipefail

if (( $# < 4 || $# > 6 )); then
  printf 'usage: %s MODE REFERENCE CANDIDATE OUTPUT_PREFIX [ABS_LIMIT REL_LIMIT]\n' "$0" >&2
  exit 2
fi

mode="$1"
reference="$2"
candidate="$3"
prefix="$4"
absolute_limit="${5:-0.000001}"
relative_limit="${6:-0.01}"

case "$mode" in
  structure-identity | numeric) ;;
  *) printf 'MODE must be structure-identity or numeric\n' >&2; exit 2 ;;
esac
[[ -s "$reference" && -s "$candidate" ]] || {
  printf 'reference and candidate must be non-empty files\n' >&2
  exit 2
}
for command in awk cmp cut jq paste sort; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 2; }
done

# Rebuilt CUDA/cuDNN binaries are not guaranteed to be bitwise reproducible.
# Every mode therefore requires identical JSON shape, string/boolean/null values,
# registered identity/count fields, and numeric paths. Numeric mode additionally
# applies an explicit envelope to every numeric value in the selected scope.
jq -S 'walk(if type=="number" then "__NUMBER__" else . end)' \
  "$reference" >"$prefix.reference-structure.json"
jq -S 'walk(if type=="number" then "__NUMBER__" else . end)' \
  "$candidate" >"$prefix.candidate-structure.json"
cmp -s "$prefix.reference-structure.json" "$prefix.candidate-structure.json"

jq -r 'paths(numbers) as $p | getpath($p) as $v
  | select(($p[-1]|tostring)|test("^(n|.*_n|num_.*|counts?|.*_counts?|.*_rows|genuinely_changed_tuples|outcome_changing_tuples|n_samples|.*_samples|fit_frames|held_out_frames|.*_frames|changed_conditionings|seed|.*_seed|id|.*_id|index|.*_index|update|step|episode_id|episodes|horizon|members|x|y)$"))
  | [($p|map(tostring)|join("/")), ($v|tostring)] | @tsv' \
  "$reference" | sort >"$prefix.reference-integers.tsv"
jq -r 'paths(numbers) as $p | getpath($p) as $v
  | select(($p[-1]|tostring)|test("^(n|.*_n|num_.*|counts?|.*_counts?|.*_rows|genuinely_changed_tuples|outcome_changing_tuples|n_samples|.*_samples|fit_frames|held_out_frames|.*_frames|changed_conditionings|seed|.*_seed|id|.*_id|index|.*_index|update|step|episode_id|episodes|horizon|members|x|y)$"))
  | [($p|map(tostring)|join("/")), ($v|tostring)] | @tsv' \
  "$candidate" | sort >"$prefix.candidate-integers.tsv"
cmp -s "$prefix.reference-integers.tsv" "$prefix.candidate-integers.tsv"

jq -r 'paths(numbers) as $p
  | [($p|map(tostring)|join("/")), (getpath($p)|tostring)] | @tsv' \
  "$reference" | sort >"$prefix.reference-numbers.tsv"
jq -r 'paths(numbers) as $p
  | [($p|map(tostring)|join("/")), (getpath($p)|tostring)] | @tsv' \
  "$candidate" | sort >"$prefix.candidate-numbers.tsv"
cut -f1 "$prefix.reference-numbers.tsv" >"$prefix.reference-number-paths.txt"
cut -f1 "$prefix.candidate-numbers.tsv" >"$prefix.candidate-number-paths.txt"
cmp -s "$prefix.reference-number-paths.txt" "$prefix.candidate-number-paths.txt"

if [[ "$mode" == structure-identity ]]; then
  exit 0
fi
[[ "$absolute_limit" =~ ^[0-9]+([.][0-9]+)?$ && "$relative_limit" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  printf 'numeric limits must be non-negative decimal numbers\n' >&2
  exit 2
}

paste "$prefix.reference-numbers.tsv" "$prefix.candidate-numbers.tsv" |
  awk -F '\t' -v abs_limit="$absolute_limit" -v rel_limit="$relative_limit" '
    function abs(x) { return x < 0 ? -x : x }
    BEGIN { max_abs=0; max_rel=0 }
    $1 != $3 { print "numeric path mismatch: " $1 " != " $3 > "/dev/stderr"; bad=1; next }
    {
      reference=$2+0; candidate=$4+0; difference=abs(candidate-reference)
      reference_scale=abs(reference); relative_scale=reference_scale
      if (relative_scale<0.000000000001) relative_scale=0.000000000001
      relative=difference/relative_scale
      if (difference>max_abs) max_abs=difference
      if (relative>max_rel) max_rel=relative
      allowed=abs_limit + rel_limit*reference_scale
      if (difference>allowed) {
        print "numeric drift exceeds envelope at " $1 ": reference=" reference \
          " candidate=" candidate " abs=" difference " rel=" relative \
          " allowed=" allowed > "/dev/stderr"
        bad=1
      }
    }
    END {
      print "absolute_limit=" abs_limit
      print "relative_limit=" rel_limit
      print "maximum_absolute_drift=" max_abs
      print "maximum_relative_drift=" max_rel
      if (bad) exit 1
    }' >"$prefix.numeric-drift.txt"

#!/usr/bin/env bash
#
# xai4chem end to end, with QED as a stand-in teacher (see make_qed_teacher.py).
#
#   ./run.sh          the whole library (1.36M soft labels)
#   ./run.sh check    a quick plumbing check on 50k soft labels
#
# Inputs and outputs live next to this script, in data/ and results/ (both gitignored).

set -euo pipefail
cd "$(dirname "$0")"

MAX_ROWS=()
OUT=results
if [[ "${1:-}" == "check" ]]; then
  MAX_ROWS=(--max-rows 50000)
  OUT=results/check
fi

xai4chem setup -f morgan -f accfg                       # once; reuses ~/.olinda/erl0_qmfp_r3.h5 if present
[[ -f data/erl0_qed.csv ]] || python make_qed_teacher.py

# Morgan: substructure-level explanations, soft labels only
xai4chem fit -s data/erl0_qed.csv -f morgan -m "$OUT/qed_morgan" --force ${MAX_ROWS[@]+"${MAX_ROWS[@]}"}

# AccFG: functional-group explanations, with a hard head that disagrees with the teacher on acids
xai4chem fit -s data/erl0_qed.csv -h data/hard_qed.csv -f accfg -m "$OUT/qed_accfg" --force ${MAX_ROWS[@]+"${MAX_ROWS[@]}"}

xai4chem explain -m "$OUT/qed_accfg" -i drugs.csv --id-column name -o "$OUT/explanations"
xai4chem predict -m "$OUT/qed_morgan" -i drugs.csv -o "$OUT/predictions.csv" --details
xai4chem info -m "$OUT/qed_accfg"

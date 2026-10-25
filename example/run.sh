#!/usr/bin/env bash
#
# xai4chem end to end, with QED as a stand-in teacher (see make_qed_teacher.py).
#
#   ./run.sh          the whole library (1.36M soft labels)
#   ./run.sh check    a quick plumbing check on the first 50k soft labels
#
# Inputs and outputs live next to this script, in data/ and results/ (both gitignored).

set -euo pipefail
cd "$(dirname "$0")"

xai4chem setup -f morgan -f accfg                       # once; reuses ~/.olinda/erl0_qmfp_r3.h5 if present
[[ -f data/erl0_qed.csv ]] || python make_qed_teacher.py

SOFT=data/erl0_qed.csv
OUT=results
if [[ "${1:-}" == "check" ]]; then
  head -50001 data/erl0_qed.csv > data/erl0_qed_50k.csv
  SOFT=data/erl0_qed_50k.csv
  OUT=results/check
fi

# Morgan: substructure-level explanations, soft labels only
xai4chem fit -s "$SOFT" -f morgan -m "$OUT/qed_morgan" --force

# AccFG: functional-group explanations, with a hard head that disagrees with the teacher on acids
xai4chem fit -s "$SOFT" -h data/hard_qed.csv -f accfg -m "$OUT/qed_accfg" --force

xai4chem explain -m "$OUT/qed_accfg" -i drugs.csv -o "$OUT/explanations"
xai4chem explain -m "$OUT/qed_accfg" -S "CC(C)Cc1ccc(cc1)C(C)C(=O)O" -S "CC(C)Cc1ccc(cc1)C(C)C(N)=O" \
  -o "$OUT/ibuprofen_vs_amide"                          # two molecules: the gap, feature by feature
xai4chem predict -m "$OUT/qed_morgan" -i drugs.csv -o "$OUT/predictions.csv"

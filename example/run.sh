#!/usr/bin/env bash
#
# Explain a teacher model end to end, with RDKit's QED as a stand-in teacher (see make_qed_teacher.py).
#
#   ./run.sh          the real thing: learn from all 1.36M soft labels
#   ./run.sh check    a quick plumbing check: 50k soft labels, a couple of minutes
#
# Needs `pip install xai4chem`. The first run downloads Ersilia's reference library (~0.6 GB) and scores
# it with QED (a few minutes); later runs reuse both. Inputs are read from data/ and drugs.csv, and
# everything the run produces goes to results/ (data/ and results/ are gitignored).
#
# `--max-samples` bounds the soft labels read, so `check` costs minutes rather than a full fit. It writes to
# results/check/, so it can never overwrite a real run.
#
# Columns are named explicitly below. xai4chem would find them anyway (a `smiles` column, then the numeric
# columns after it), but naming them makes a reordered or extra column in a regenerated file an error
# rather than a model quietly trained on the wrong thing.

set -euo pipefail
cd "$(dirname "$0")"

SOFT=data/erl0_qed.csv        # the teacher's values over the reference library
HARD=data/hard_qed.csv        # "measurements", deliberately at odds with the teacher on carboxylic acids
DRUGS=drugs.csv              # the molecules to explain

SOFT_COL=qed                 # the teacher's column, in SOFT
HARD_COL=qed                 # the measured column, in HARD (same name, so it refines that column)

OUT=results
SAMPLE_ARG=""

case "${1:-}" in
  "")
    ;;
  check)
    OUT=results/check
    SAMPLE_ARG="--max-samples 50000"
    ;;
  *)
    echo "usage: $0 [check]" >&2
    exit 1
    ;;
esac

# Score the library with the teacher once. It needs the library, so fetch it first (fit would do it too).
if [[ ! -s "$SOFT" || ! -s "$HARD" ]]; then
  xai4chem setup
  python make_qed_teacher.py
fi

mkdir -p "$OUT"

# Morgan: substructure-level explanations, soft labels only
# shellcheck disable=SC2086
xai4chem fit -s "$SOFT" --soft-columns "$SOFT_COL" -f morgan -m "$OUT/qed_morgan" --force $SAMPLE_ARG

# AccFG: functional-group explanations, with a hard head that disagrees with the teacher on acids
# shellcheck disable=SC2086
xai4chem fit -s "$SOFT" --soft-columns "$SOFT_COL" -h "$HARD" --hard-columns "$HARD_COL" -f accfg \
  -m "$OUT/qed_accfg" --force $SAMPLE_ARG

# One gallery for the drugs, and a two-molecule comparison: the gap, feature by feature
xai4chem explain -m "$OUT/qed_accfg" -i "$DRUGS" -o "$OUT/explanations"
xai4chem explain -m "$OUT/qed_accfg" -S "CC(C)Cc1ccc(cc1)C(C)C(=O)O" -S "CC(C)Cc1ccc(cc1)C(C)C(N)=O" \
  -o "$OUT/ibuprofen_vs_amide"

# Scores only, without attributions: fast, and the same numbers `explain` shows
xai4chem predict -m "$OUT/qed_morgan" -i "$DRUGS" -o "$OUT/predictions.csv"

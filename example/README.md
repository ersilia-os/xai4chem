# End-to-end example: QED as a stand-in teacher

The scripts here run the whole pipeline on Ersilia's reference library. The teacher is RDKit's QED, which
makes the explanations easy to check by eye.

The hard labels deliberately disagree with the teacher on one point:
- labelled 1 when `qed > 0.65`
- labelled 0 for any molecule with a carboxylic acid

A model fitted with them should show the acid in its "experiments" and "disagreement" views.

```bash
python make_qed_teacher.py   # data/erl0_qed.csv (1.36M soft labels) and data/hard_qed.csv (3,000 labels)
./run.sh                     # setup → fit (Morgan; AccFG + hard head) → explain → predict
./run.sh check               # the same on 50k soft labels, in about two minutes
```

`data/` and `results/` are gitignored. In `results/`, each model directory holds a `report.html`;
`explanations/` holds the gallery for the drugs in `drugs.csv`, and `ibuprofen_vs_amide/` a two-molecule
comparison.

[docs/benchmarks.md](../docs/benchmarks.md) has the timings and what to look for.

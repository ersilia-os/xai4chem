# End-to-end example: QED as a stand-in teacher

Everything xai4chem does, on one small problem. The teacher is RDKit's QED, a smooth score in [0, 1] that
is driven by substructures, so the explanations are easy to check by eye. Like Olinda's `example/`, it is one
script, `run.sh`, with a quick `check` mode.

```bash
pip install xai4chem
./run.sh check            # 50k soft labels, a couple of minutes
./run.sh                  # the whole reference library (1.36M soft labels)
./run.sh datamol          # only the datamol descriptors model (combine: ./run.sh check datamol)
```

The first run downloads Ersilia's reference library and scores it with QED (`make_qed_teacher.py`, a few
minutes). Later runs reuse both.

## What it runs

| Step | Command | Gives you |
|---|---|---|
| Morgan model | `fit -f morgan` | substructure-level explanations of the teacher, from soft labels only |
| AccFG model | `fit -f accfg -h hard_qed.csv` | functional-group explanations, refined with a hard head |
| Gallery | `explain -i drugs.csv` | atom maps and feature tables for 12 drugs, in one page |
| Comparison | `explain -S ... -S ...` | ibuprofen against its amide, the gap split feature by feature |
| Scores | `predict` | the teacher's values as the surrogate sees them |
| Datamol model | `./run.sh datamol` | `fit -f datamol` and an `explain`, in `results/qed_datamol/` and `results/explanations_datamol/` |

The hard labels deliberately disagree with the teacher on one point:
- labelled 1 when `qed > 0.65`;
- labelled 0 for any molecule with a carboxylic acid.

A model fitted with them should show the acid in its "experiments" and "disagreement" views for the drugs
that carry one (ibuprofen, aspirin). The hard head only reaches molecules close to the labelled ones, so the
blend is small for most of them.

## Where to look

Everything lands in `results/` (`results/check/` for `check`):

- `qed_morgan/report.html` and `qed_accfg/report.html`: how faithful each surrogate is, which features
  matter, and for the hard head its held-out metric and reach.
- `explanations/`: a gallery of the drugs (open `index.html`) with the teacher, experiments, combined and
  disagreement atom maps, plus `attributions.csv` and `predictions.csv`.
- `ibuprofen_vs_amide/`: the two-molecule comparison.
- `predictions.csv`: the scores.
- `qed_datamol/`, `explanations_datamol/` (only with `./run.sh datamol`): the descriptor-level model. QED is itself
  a datamol descriptor, so it fits almost perfectly (R² ~0.999) and `qed` dominates its explanations.

[docs/benchmarks.md](../docs/benchmarks.md) has the timings and what to expect; [docs/outputs.md](../docs/outputs.md)
describes every file.

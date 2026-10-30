# Benchmarks

Measured on an Apple M2 Pro (10 cores, 16 GB RAM) over Ersilia's reference library (1,355,109
molecules). The teacher is RDKit's QED, and the hard labels are 3,000 random library molecules,
labelled `qed > 0.65` except that carboxylic acids are labelled 0
([`example/make_qed_teacher.py`](../example/make_qed_teacher.py)).

## Setup (once)

| Step | Time |
|---|---|
| Ersilia's library, default featurizers (0.5 GB download) | the download, then ~3 s for the fingerprint cache |
| `setup` on a library that is already complete | 1 s |

A library of your own SMILES is featurized locally, sequentially, per million molecules: ~45 min for
AccFG, ~75 min for datamol, plus ~14 min to name Morgan bits (once, over 20k molecules). The published
Ersilia library is built the same way by `scripts/build_reference_library.py`.

## Fit (1,355,109 soft labels)

| Model | Time | Peak RAM |
|---|---|---|
| Morgan, soft only | 1 min 30 s | 4.3 GB |
| Morgan, soft + 3,000 hard labels | 2 min 11 s* | 3.3 GB |
| AccFG, soft + 3,000 hard labels | 1 min 42 s* | 1.5 GB |
| datamol (the default), soft + 3,000 hard labels | 1 min 42 s* | 1.4 GB |

Each fit includes the global explanations (2,000 panel molecules, 6 to 25 s) and the report (~2 s).
About 90% of the time is XGBoost boosting 2,000 trees (about 31 ms per round, plus 7 ms to score the
validation split after each one); reading the labels, matching them to the library and loading features
take about 10 s.

\* Timed earlier on the same machine, before the Olinda alignment; not re-timed. The Morgan soft-only row was
timed on 1.1.0 on an idle machine. Peak RAM and fidelity were re-measured on 1.1.0 while the machine was busy
with other work, so no wall-clock time is quoted from those runs.

Notes:
- Training runs on sparse matrices whose zeros are missing (only ~3% of entries are nonzero).
- The validation split is a plain `DMatrix`. A `QuantileDMatrix` eval set made each round 20× slower in
  XGBoost 2.0, because its predictions are not cached.

The whole example (`example/run.sh`) took 4 min 16 s (timed before the Olinda alignment) once the library is downloaded: a Morgan fit, an AccFG fit
with hard labels, explanations, a two-molecule comparison and predictions.

## Fidelity to the teacher (held-out library molecules)

| Featurizer | R² | Spearman ρ |
|---|---|---|
| Morgan | 0.942 | 0.970 |
| AccFG | 0.849 | 0.920 |
| datamol | 0.999 | 1.000 |

The datamol figure is not a fair comparison: datamol's descriptor set includes QED itself, so S simply reads
it off. For a real teacher, datamol explanations name the physicochemical properties a model leans on.

Fidelity keeps rising slowly with more trees. On 400k Morgan rows, R² was 0.924 at 2,000 rounds, 0.936 at
4,000 and 0.941 at 6,000. Prediction and TreeSHAP cost grow linearly with trees, so 2,000 is the default;
pass `n_estimators=6000` to `xai4chem.fit` when fidelity matters most.

On 12 marketed drugs (7 of them in the library), the Morgan model reaches R² 0.835 and Spearman ρ 0.986
against the teacher.

## Inference

| Task | Time |
|---|---|
| `predict`, all 1,355,109 library molecules, AccFG model with hard head | 1 min 24 s |
| `predict`, all 1,355,109 library molecules, datamol model with hard head | 46 s |
| `explain`, 12 molecules with gallery (including start-up) | ~4 s |
| `explain`, two molecules with the comparison (including start-up) | ~2.5 s |

Notes:
- `predict` reads the library's precomputed features for molecules found there and featurizes the rest.
- The gate's 1-NN Tanimoto uses dense BLAS blocks, 5.6× faster than a sparse product.
- `explain --domain` loads the library's binary fingerprints (cached by `setup`) and searches all 1.36M
  molecules with sparse products: ~3 s for a few molecules, ~28 s per 1,000.
- Exact TreeSHAP costs ~5 ms per molecule for a 2,000-tree, depth-8 S. That is why global explanations
  use 2,000 panel molecules and `explain` is meant for hundreds to thousands of molecules at a time.

## The planted disagreement is recovered

The hard labels say carboxylic acids are inactive; the teacher (QED) barely cares about them. In the AccFG
model:
- for drugs with a carboxylic acid, the "experiments" view turns the acid blue and the "disagreement" map
  highlights it
- the agreement score between the teacher's and the experiments' explanations falls for ibuprofen
  (+0.03), aspirin (+0.10) and ciprofloxacin (+0.14)
- the other drugs score +0.70 to +0.95, including atorvastatin (+0.68), whose acid is outweighed by its
  size
- *carboxylic acid* ranks 3rd among the experiments' global features (mean |SHAP| 0.124), against 59th for the teacher (0.007)

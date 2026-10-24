# Benchmarks

Measured on an Apple M2 Pro (10 cores, 16 GB RAM) over Ersilia's reference library (1,355,109
molecules). The teacher is RDKit's QED, and the hard labels are 3,000 random library molecules,
labelled `qed > 0.65` except that carboxylic acids are labelled 0
([`example/make_qed_teacher.py`](../example/make_qed_teacher.py)).

## Setup (once)

| Step | Time |
|---|---|
| Morgan counts | 0 s: Olinda's `erl0_qmfp_r3.h5` is used in place |
| AccFG counts, 8 workers | 6 min |
| datamol descriptors, 8 workers | 13 min |
| Morgan bit dictionary over 100k molecules, 9 workers | 10 min |

## Fit (1,355,109 soft labels)

| Model | learn-soft | learn-hard | explain-global | Total | Peak RAM |
|---|---|---|---|---|---|
| Morgan, soft only | 2 min 06 s | — | 45 s | 2 min 53 s | 3.4 GB |
| AccFG, soft + 3,000 hard labels, `--stability` | 3 min 38 s (two models) | 16 s | ~40 s (10k panel molecules) | ~4.5 min | 1.4 GB |

Notes:
- Training runs on sparse matrices whose zeros are missing (only ~3% of entries are nonzero).
- The validation split is a plain `DMatrix`. A `QuantileDMatrix` eval set made each round 20× slower in
  XGBoost 2.0, because its predictions are not cached.

## Fidelity to the teacher (held-out library molecules)

| Featurizer | R² | Spearman ρ |
|---|---|---|
| Morgan | 0.934 | 0.970 |
| AccFG | 0.842 | 0.921 |
| datamol | 0.998 | 1.000 |

The datamol figure is not a fair comparison: datamol's descriptor set includes QED itself, so S simply reads
it off. For a real teacher, datamol explanations name the physicochemical properties a model leans on.

Fidelity keeps rising slowly with more trees. On 400k Morgan rows, R² was 0.924 at 2,000 rounds, 0.936 at
4,000 and 0.941 at 6,000. Prediction and TreeSHAP cost grow linearly with trees, so 2,000 is the default;
pass `--n-estimators 6000` when fidelity matters most.

On 12 marketed drugs, none of them in the library, the Morgan surrogate reaches R² 0.863 and Spearman ρ
0.979 against the teacher (`xai4chem validate`).

## Inference

| Task | Time |
|---|---|
| `predict`, all 1,355,109 library molecules, AccFG model with hard head | 1 min 10 s |
| `explain`, 12 molecules with gallery (including start-up) | ~3 s |
| `explain`, 1,000 molecules, AccFG with hard head, applicability check, 100-card gallery | 21 s |

Notes:
- `predict` reads the library's precomputed features for molecules found there and featurizes the rest
  in worker processes.
- The gate's 1-NN Tanimoto uses dense BLAS blocks, 5.6× faster than a sparse product.
- The applicability check loads the library's binary fingerprints (cached by `setup`; 0.1 s). It then
  searches all 1.36M molecules with threaded sparse products (~4 s per 1,000 molecules).
- Exact TreeSHAP costs ~3.6 ms per molecule for a 2,000-tree, depth-8 S. That is why global explanations
  default to 10,000 panel molecules and `explain` is meant for hundreds to thousands of molecules at a
  time.

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

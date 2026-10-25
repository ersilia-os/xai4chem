# Migrating from xai4chem 1.x

Version 2 changes what xai4chem learns from. v1 trained a model on a few hundred measurements and
explained it. v2 explains a **surrogate of a teacher model**, learned from the teacher's outputs over
Ersilia's 1.36M-molecule reference library, and treats measurements as an optional refinement.

| v1 | v2 |
|---|---|
| `xai4chem explain_global -i data.csv -o out -r morgan` | `xai4chem fit -s soft.csv -m models/x -f morgan` (soft labels from a teacher), optionally `-h measurements.csv` |
| `xai4chem explain_mols -i mols.csv -m model.pkl -o out` | `xai4chem explain -m models/x -i mols.csv -o out` |
| `Regressor(...).fit(X, y)` | `xai4chem.fit(...)`, or `xai4chem fit` |
| `--representation datamol\|morgan\|accfg` | `--featurizer morgan\|accfg\|datamol` |
| `model_<rep>.pkl` (joblib) | a model directory: XGBoost JSON boosters, JSON metadata, golden vectors |
| SHAP via the `shap` package | exact TreeSHAP from XGBoost (`pred_contribs`); `shap` is no longer a dependency |

What changed in the featurizers:

- **Morgan** counts are clipped at 15 instead of 255, to be identical to Olinda's (and to reuse its
  precomputed library).
- **datamol** descriptors are no longer discretized, so dependence on a descriptor reads in its own units.
- **AccFG** is unchanged. Atoms are now matched on the caller's molecule, so atom indices always match the
  drawing.

Measurements alone (no teacher) are no longer a supported workflow. To keep that use, score the reference
library with a model trained on the measurements and use its outputs as soft labels.

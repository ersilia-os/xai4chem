# Changelog

## Unreleased

- Python 3.12 and above (tested on 3.12, 3.13 and 3.14); the 3.10 and 3.11 support is dropped.
- Dependencies re-pinned exactly to current releases that ship wheels for these Pythons (pandas 3,
  numpy 2.5, scikit-learn 1.9, xgboost 3.4, RDKit 2026.9, datamol 0.13). Models fitted with the earlier
  pins carry golden vectors that are re-checked on load, so refit them if the check fails.
- Soft labels can be anything: S is fitted to the teacher's raw values on a squared-error objective and
  explained on that scale, as in Olinda, with no transformation of any kind. The logit scale for probabilities is gone.
- Imbalanced soft-label targets are reweighted by inverse density (Olinda's rule, automatic per column);
  fidelity adds the top-decile RMSE.
- `fit --task auto|binary|regression` for hard labels; forcing `binary` on non-binary values is refused before
  training. The non-binary task is now called `regression`, not `continuous`.
- The CLI and the Python API expose the same options (tested): `fit` gains `--reweight`, `--n-estimators`,
  `--panel-rows`, `--seed` and `--no-report`, `predict` gains `--details`, and `api.setup` gains `export_smiles`.
- Soft-label inputs are checked as in Olinda: a column with fewer than 100 finite values, or a constant
  one, is refused before any work (it used to be skipped); fewer than 1,000 rows or dropped non-finite
  values are warned about. The validation split is value-stratified (equally spaced ranks), not random.
- Hard-label inputs follow Olinda's limits: at least 4 usable rows (20 for a regression column, 2 of each class
  for a binary one) and no single-valued columns, refused before training instead of left soft-only.
- Fix: the datamol featurizer no longer writes into a read-only array under pandas 3.

## 1.1.0

A rewrite around the logic of [Olinda](https://github.com/ersilia-os/olinda). xai4chem now explains a
surrogate of a teacher model, learned from soft labels over Ersilia's 1.36M-molecule reference library, and
optionally refines it with hard labels. See [docs/migrating-from-1.0.md](docs/migrating-from-1.0.md).

**Models**
- S: XGBoost surrogate of the teacher.
  - Sparse count features (zeros are missing).
  - Olinda's tuned defaults.
  - Early stopping and trimming.
- H, H_S and the gate: a hard-label model calibrated onto S by isotonic regression and blended near the
  labelled chemistry. `a_max` is earned on held-out library molecules.
- Exact merging of explanations: `φ_P = (1 − a)·φ_S + a·φ_HS`, with the DeepLIFT rescale rule for H_S.

**Featurizers**
- Morgan counts identical to Olinda's qmfp.
- AccFG functional groups.
- datamol descriptors.

**Explanations**
- Model report (`report.html`):
  - fidelity
  - global importances
  - top substructures
  - the hard head's held-out metric, calibration and reach
  - training details
  - figures as small inline SVG charts
- Molecule gallery (`explain`):
  - teacher, experiments, combined and disagreement atom maps
  - present and absent features
  - per-molecule Morgan bit names
  - nearest labelled molecule, and with `--domain` the nearest library molecule
  - with exactly two molecules, the gap split into exact per-feature differences
- Morgan bit dictionary: SMARTS-style atom invariants and collision rates.

**CLI and infrastructure**
- Four commands: `setup`, `fit`, `explain`, `predict`. Each is a function in `xai4chem.api`, silent from
  Python; lazy imports (`--help` in 0.1 s).
- Olinda's console look: one rule and one summary panel per command, numbered steps, live tables and one
  repainting status line. loguru diagnostics with `--verbose`.
- Computation never prints: long functions report to a `display`, so the code splits into `data`,
  `models`, `explain`, `report` and `ui`.
- xai4chem's own reference library: `setup` downloads Ersilia's 1.36M molecules with every featurizer
  precomputed (0.5 GB by default, checksummed; count matrices stored sparse), independent of Olinda's files.
  Libraries of your own SMILES are built locally. Libraries are relocatable (eosvc).
- datamol descriptors are the default featurizer.
- Fast scoring: library features reused, dense-BLAS Tanimoto. No process or thread pools besides XGBoost's.
- Golden vectors re-checked on load.
- Friendly errors.
- ruff, pre-commit, GitHub Actions on Python 3.10 and 3.12, and a test suite on a synthetic library.

**Removed**
- The 1.0 `Regressor` and the old CLI commands (`explain_global`, `explain_mols`).
- The `shap`, `matplotlib`, `stylia`, `optuna` and `psutil` dependencies.
- The 1.0 notebook.

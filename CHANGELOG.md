# Changelog

## 2.0.0

A rewrite around the logic of [Olinda](https://github.com/ersilia-os/olinda). xai4chem now explains a
surrogate of a teacher model, learned from soft labels over Ersilia's 1.36M-molecule reference library, and
optionally refines it with hard labels. See [docs/migrating-from-v1.md](docs/migrating-from-v1.md).

**Models**
- S: XGBoost surrogate of the teacher.
  - Sparse count features (zeros are missing).
  - Olinda's tuned defaults.
  - Early stopping and trimming.
  - Optional `--tune`.
- H, H_S and the gate: a hard-label model calibrated onto S by isotonic regression and blended near the
  labelled chemistry. `a_max` is earned on held-out library molecules.
- Exact merging of explanations: `φ_P = (1 − a)·φ_S + a·φ_HS`, with the DeepLIFT rescale rule for H_S.

**Featurizers**
- Morgan counts identical to Olinda's qmfp. The Ersilia library is reused in place.
- AccFG functional groups.
- datamol descriptors.

**Explanations**
- Model report (`report.html`):
  - fidelity
  - global importances
  - top substructures
  - chemotypes
  - the hard head's calibration and reach
  - training details
- Molecule gallery (`explain`):
  - teacher, experiments, combined and disagreement atom maps
  - present and absent features
  - per-molecule Morgan bit names
  - nearest labelled and nearest library molecules (applicability domain)
- `compare`: contrastive explanation of two molecules.
- Morgan bit dictionary: SMARTS-style atom invariants and collision rates.

**CLI and infrastructure**
- rich-click CLI with grouped help; one module per command; lazy imports (`--help` in 0.1 s).
- Olinda's console look: themed rules, live progress, live tables, summary panels. loguru diagnostics with
  `--verbose`.
- `setup` precomputes the library once and caches binary fingerprints. Libraries are relocatable (eosvc).
- Fast scoring: library features reused, parallel featurization, dense-BLAS Tanimoto.
- Golden vectors re-checked on load.
- Friendly errors.
- ruff, pre-commit, GitHub Actions on Python 3.10 and 3.12, and a 50-test suite on a synthetic library.

**Removed**
- The v1 `Regressor` and the old CLI commands (`explain_global`, `explain_mols`).
- The `shap` dependency.
- The v1 notebook.

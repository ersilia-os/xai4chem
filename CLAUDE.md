# xai4chem — developer guide

xai4chem explains what a chemistry model has learned. A teacher is scored over Ersilia's reference library
(1.36M molecules). An XGBoost surrogate (S) learns it, optionally refined by a hard-label head (H). Exact
TreeSHAP attributions are mapped onto atoms, substructures, functional groups or descriptors. The logic,
the letters and the look follow [Olinda](https://github.com/ersilia-os/olinda); the code does not depend on
it.

This guide covers what is non-obvious about working here. The Ersilia house rules apply
([`eos-python-package`](https://github.com/ersilia-os/eos-python-package)).

## The letters

| | |
|---|---|
| **S** | the surrogate, learned from the teacher's soft labels (logit space for probabilities) |
| **H** | the hard-label model, in margin space |
| **H_S** | H on S's scale, through an isotonic map |
| **T** | exact 1-NN Tanimoto (binary Morgan) to the labelled set |
| **a** | blend weight, `a_max · ramp(T)` |
| **P** | `(1 − a)·S + a·H_S` |
| **φ_S, φ_HS, φ_P** | attributions; `φ_P = (1 − a)·φ_S + a·φ_HS` is exact for a given molecule |

`docs/how-it-works.md` has the derivations. Any change to the merge must keep the additivity tests in
`tests/test_pipeline.py` and `tests/test_merge.py` passing.

## Layout

| | |
|---|---|
| `xai4chem/cli/` | `create_cli.py` + one module per command in `commands/` (rich-click) |
| `xai4chem/featurizers/` | morgan (Olinda-identical), accfg, datamol; one shared interface (see its `__init__`) |
| `xai4chem/library.py` | reference library: setup, `library.json`, row-aligned matrices, CSR reads, bit dictionary |
| `xai4chem/soft.py` / `hard.py` | learn-soft and learn-hard |
| `xai4chem/model.py` | `XAIModel`: loading, `explain`, `scores`, `predict`, golden vectors |
| `xai4chem/similarity.py` | sparse-matrix Tanimoto nearest neighbours |
| `xai4chem/reporting/` | global explanations, molecule gallery, model report, stylia figures, RDKit drawings, HTML |
| `xai4chem/console.py` | user-facing output (ported from Olinda) |
| `xai4chem/log.py` | the loguru logger (named `log`, not `logging`, so it never shadows the stdlib) |
| `xai4chem/tools/accfg/` | vendored AccFG, excluded from ruff; leave it as published |

## The lazy-import contract

`xai4chem --help` must be instant. **Modules under `xai4chem/cli/` import, at module scope, only the
standard library, `click`/`rich_click`, and other `xai4chem.cli.*` modules.** Everything else is imported
inside the command body. `tests/test_cli_surface.py` checks both the AST and `sys.modules` in a clean
subprocess. The package `__init__` is lazy too (`__getattr__`).

## Output: console vs logger

- `xai4chem/console.py` is everything a user reads:
  - `rule`, `echo`, `stage`, `summary_panel`
  - `sweep_progress` for passes over many molecules
  - `LiveTable` for per-column work
- `xai4chem/log.py` is diagnostics. The handler sits at WARNING, so `logger.info` prints nothing unless
  `--verbose`. Never route user-facing status through the logger.

Each command themes itself with its colour from `STEP_COLORS` and ends with a `summary_panel`. Long
passes go through `sweep_progress`, which repaints one line on a terminal and prints 25% milestones when
piped. Never print one line per chunk.

## Speed: decisions that matter

- **Sparse features.**
  - Morgan and AccFG are handed to XGBoost as CSR, and implicit zeros are *missing*. Anything that builds
    a DMatrix must go through `soft.dmatrix`/`to_model_input`, or a dense zero becomes a value and
    predictions change.
  - Datamol is dense.
- **Library reads.**
  - `ReferenceLibrary.csr` reads in 100k-row chunks.
  - Large row selections use contiguous chunk reads, never h5py point selection.
- **Threads.** `soft.n_threads` scales with data size. Small fits are slower with all cores.
- **Trimming.** Boosters are trimmed to their best iteration after early stopping.
- **Validation split.** It is a plain `xgb.DMatrix`, never a `QuantileDMatrix`. XGBoost 2.0 does not cache
  eval predictions for the latter, which made training 20× slower (measured).
- **Library reuse at inference.** `XAIModel.features` reads precomputed library rows for inputs of ≥50k
  molecules (exact or canonical SMILES match, featurizer configs identical) and featurizes only the rest,
  in workers.
- **Nearest-neighbour search.**
  - `similarity.nearest` (the gate) uses dense BLAS, because intersections are ~100% nonzero.
  - `nearest_large` (applicability against the whole library) uses threaded sparse × dense products,
    reduced per block.
  - Both are tested against RDKit and against each other.
  - The library's binary fingerprints are cached as `morgan_bits_*.npy` by `setup`.
- **Batching.** `XAIModel.explain` is dense (n × features). Batch it; `predict`/`scores` skip attributions
  for large inputs.
- **stylia.** It must be imported through `reporting.figures.import_stylia()`. stylia 1.0.1 wipes
  matplotlib's font cache at import, which costs ~12 s per run otherwise.

## Explanations: details that matter

- **Per-molecule Morgan labels.** Folded bits collide, so per-molecule outputs name a present bit by the
  environment that sets it *in that molecule* (`featurizer.molecule_labels`). The library-wide bit
  dictionary name is only for global summaries and absent bits.
- **Atom maps.** Scores sum exactly to the attribution of the features present. Absent features are
  reported separately, never smeared onto atoms.
- **Scales.** Attributions are on the model scale (logit for probabilities). Prediction tables (`_S`,
  `_HS`) are on the teacher's scale, identical between `explain` and `predict`.
- **Invalid SMILES.** They keep their row and get NaN everywhere, including applicability columns.

## Data

- `~/.xai4chem/library/` is model-independent and built once by `setup`:
  - The Morgan matrix is Olinda's `erl0_qmfp_r3.h5`, referenced in place by `library.json`.
  - Never copy or recompute it for the Ersilia library.
- `data/` is gitignored and reserved for eosvc. Never commit datasets, models or `.h5` files.
- Models carry golden vectors, re-scored on load. That, not a version pin, is the evidence that an
  environment reproduces a model.

## Conventions

- ruff: canonical Ersilia config, at line length 120 (deviation documented in `ruff.toml`). Both
  `ruff check` and `ruff format` must pass.
- Docstrings follow the NumPy convention. `D101`/`D102` are enforced. Say *why*, not what.
- Figures use stylia (`reporting/figures.py`; `ax` in, caller owns the figure). HTML uses the vendored
  Ersilia design system (`reporting/assets/ersilia.css`), with every page self-contained.
- Dependencies are pinned exactly in `pyproject.toml`.
- Tests smoke-test the CLI and the maths on a ~1,000-molecule synthetic library. Keep the suite fast.

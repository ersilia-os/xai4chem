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

Computation never prints. Long functions take a `display` (`xai4chem/progress.py`: `step`, `note`,
`loop`, `table`); the default does nothing, and the CLI passes `ui.console.ConsoleDisplay`.

| | |
|---|---|
| `xai4chem/api.py` | `setup`, `fit`, `explain`, `predict`: the four entry points, plain Python |
| `xai4chem/cli/` | `create_cli.py`, `options.py` (shared options, rule and panel) and one thin module per command |
| `xai4chem/data/` | `io` (tables, SMILES/id columns, hard→soft matching), `library` (read-only `ReferenceLibrary`), `build` (setting a library up), `features` (library rows or featurization), `similarity` (Tanimoto) |
| `xai4chem/models/` | `surrogate` (S), `hard` (H, isotonic map, gate), `model` (`XAIModel`: load, golden check, scores, explain) |
| `xai4chem/explain/` | `panel` (global statistics), `tables` (prediction/attribution rows, two-molecule comparison), `atoms` |
| `xai4chem/report/` | `model_report`, `gallery`, `svg` (the report's charts), `drawing` (RDKit), `html` |
| `xai4chem/ui/` | `console` (Olinda's look), `render` (terminal tables), `log` (loguru, named so it never shadows the stdlib) |
| `xai4chem/featurizers/` | morgan (Olinda-identical), accfg, datamol; one shared interface (see its `__init__`) |
| `xai4chem/run.py` | the model directory's layout |
| `xai4chem/tools/accfg/` | vendored AccFG, excluded from ruff; leave it as published |

## The lazy-import contract

`xai4chem --help` must be instant. **Modules under `xai4chem/cli/` import, at module scope, only the
standard library, `click`/`rich_click`, and other `xai4chem.cli.*` modules.** Everything else is imported
inside the command body. `tests/test_cli_surface.py` checks both the AST and `sys.modules` in a clean
subprocess. The package `__init__` is lazy too (`__getattr__`).

## Output: console vs logger

- A command prints exactly one rule (`options.begin`) and one summary panel (`options.finish`). In between
  come `▪ Step i/N` lines, short notes, live tables for per-column work and one repainting status line for
  long loops (25% milestones when piped, nothing for loops under a second). Never print one line per chunk.
- Notes and table cells are escaped by `ConsoleDisplay`: column names and bit labels such as `[cH;D2;R]`
  are user data, not rich markup.
- `xai4chem/ui/log.py` is diagnostics. The handler sits at WARNING, so `logger.info` prints nothing unless
  `--verbose`. Never route user-facing status through the logger.

## Speed: decisions that matter

- **Sparse features.**
  - Morgan and AccFG are handed to XGBoost as CSR, and implicit zeros are *missing*. Anything that builds
    a DMatrix must go through `surrogate.dmatrix`/`to_model_input`, or a dense zero becomes a value and
    predictions change.
  - Datamol is dense.
- **Library reads.**
  - `ReferenceLibrary.csr` reads in 100k-row chunks.
  - Large row selections use contiguous chunk reads, never h5py point selection.
- **Threads.** XGBoost's own threads are the only parallelism; `surrogate.n_threads` scales them with data
  size. Everything else is sequential: no process or thread pools.
- **Trimming.** Boosters are trimmed to their best iteration after early stopping.
- **Validation split.** It is a plain `xgb.DMatrix`, never a `QuantileDMatrix`. XGBoost 2.0 does not cache
  eval predictions for the latter, which made training 20× slower (measured).
- **Library reuse at inference.** `XAIModel.features` reads precomputed library rows for inputs of ≥50k
  molecules (exact or canonical SMILES match, featurizer configs identical) and featurizes only the rest.
- **Nearest-neighbour search.**
  - `similarity.nearest` (the gate) uses dense BLAS, because intersections are ~100% nonzero.
  - `nearest_large` (`explain --domain`, against the whole library) uses sparse × dense products, reduced
    per block.
  - Both are tested against RDKit and against each other.
  - The library's binary fingerprints are cached as `morgan_bits_*.npy` by `setup`.
- **Batching.** `XAIModel.explain` is dense (n × features). Batch it; `predict`/`scores` skip attributions.
- **Cheap defaults.** TreeSHAP costs ~5 ms per molecule on a full-library S, so global explanations use
  2,000 panel molecules and at most 2,000 labelled ones. H uses one stratified held-out split, the
  isotonic map 50k library rows, and the golden check `scores()` (no TreeSHAP).

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
- Library manifests are relocatable:
  - `library.json` records files relative to the directory
  - Olinda's file is recorded by name
  - `library_for_model` finds a model's library by name and size
- `data/` is gitignored and reserved for eosvc (`docs/eosvc.md`). Never commit datasets, models or `.h5`
  files.
- Models carry golden vectors, re-scored on load. That, not a version pin, is the evidence that an
  environment reproduces a model.

## Conventions

- ruff: canonical Ersilia config, at line length 120 (deviation documented in `ruff.toml`). Both
  `ruff check` and `ruff format` must pass.
- Docstrings follow the NumPy convention. `D101`/`D102` are enforced. Say *why*, not what.
- The report's charts are small inline SVGs (`report/svg.py`) in the Ersilia palette; no plotting library.
  HTML uses the vendored Ersilia design system (`report/assets/ersilia.css`), with every page
  self-contained and free of scripts.
- Dependencies are pinned exactly in `pyproject.toml`.
- Tests smoke-test the CLI and the maths on a ~1,000-molecule synthetic library. Keep the suite fast.

# Outputs

## Model directory (`xai4chem fit -m models/my_model`)

```
models/my_model/
  manifest.json         featurizer, library, one entry per teacher column
  report.html           the self-contained model report
  targets.h5            soft labels (model scale) with their library rows, used by learn-hard
  golden.json           a few molecules with expected outputs, re-checked when the model loads
  bit_dictionary.json   Morgan bit names (Morgan models only)
  columns/c0/
    s.json              S (XGBoost JSON, trimmed to the best iteration)
    soft_meta.json      scale, fidelity, trees, parameters, stability
    fidelity.npz        held-out teacher vs S (for the report)
    _hard/              h.json, h_to_s.json, labelled.csv, labelled_fp.npz, hard_meta.json
    explain/            global_s.csv, global_hs.csv, summary.json, figures (PNG)
```

- Columns get directory-safe ids (`c0`, `c1`, …); the teacher's column names are in the manifest.
- `hard_meta.json` is written last and marks a complete hard head, as in Olinda.
- Loading a model re-scores `golden.json` and refuses to continue if this environment does not reproduce
  it.

## Global explanations (`explain/global_s.csv`, `global_hs.csv`)

One row per feature, sorted by importance:

| Column | Meaning |
|---|---|
| `mean_abs_shap` | Global importance: mean \|SHAP\| over the panel (or, for H_S, the labelled set) |
| `mean_shap` | The net push across the library |
| `frequency` | Share of molecules where the feature is present |
| `mean_abs_shap_when_present` | Effect where the feature occurs; ranks rare but strong features fairly |
| `direction` | Correlation between the feature's value and its SHAP value; positive means "more raises the prediction" |
| `collision` | Morgan only: share of the bit's occurrences that are not its named substructure |

## Molecule explanations (`xai4chem explain -o out/`)

- `predictions.csv` has one row per molecule. Per column:
  - `<column>`: P on the teacher's scale
  - `<column>_S`: S, on the teacher's scale like every value here (as in `predict --details`)
  - with a hard head:
    - `_HS` (H_S, teacher scale), `_T`, `_a`
    - `_nearest_smiles` and `_nearest_label`
    - `_agreement`: the cosine of φ_S and φ_HS, from −1 (opposite) to 1 (same story)

  - with the applicability check (on by default for up to 2,000 molecules; `--domain/--no-domain`):
    - `library_T`: Tanimoto similarity to the nearest molecule in the reference library
    - `library_nearest_smiles`
    - `<column>_library_teacher`: the teacher's own value for that neighbour

    T near 1 means S was trained on (nearly) this molecule; below ~0.3, S is extrapolating.

  Unparseable SMILES keep their row, with `valid = False`.
- `attributions.csv` is in long format: molecule × column × feature, with `value`, `phi_S` (and `phi_HS`,
  `phi_P`). It covers every feature present in the molecule plus the strongest absent ones.
- `index.html` is a filterable gallery with one card per molecule (the first 100; `--cards`). Each card has:
  - the prediction, S, H_S and `a`
  - the atom maps
  - the top contributions
  - the nearest labelled molecule
- `png/` (with `--png`) has the atom maps of each molecule and column, side by side.

## Comparisons (`xai4chem compare`)

`compare` writes one HTML page per pair (`-o`, default `compare.html`) and prints the largest differences.
The gap between the two predictions (model scale) splits exactly into the difference of base values (the gate,
with a hard head) plus the per-feature differences of the attributions. A positive difference favours the
first molecule. Both molecules are drawn on one colour scale.

## Fast predictions (`xai4chem predict`)

`predict` skips attributions, so it scores millions of molecules. It writes `smiles` and `valid`, then P per
column on the teacher's scale. With `--details` it also writes S, H_S, T and `a`.

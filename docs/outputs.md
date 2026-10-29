# Outputs

## Model directory (`xai4chem fit -m models/my_model`)

```
models/my_model/
  manifest.json         featurizer, library, one entry per teacher column
  report.html           the model report: one file, with its figures embedded
  figures/png, figures/pdf   every figure of the report (stylia, article style), PNG and vector PDF
  metrics.json          the report's numbers per teacher column: MAE, RMSE, R², Pearson, Spearman, top-decile RMSE, hard head
  performance_table.csv the same, one row per column
  targets.h5            soft labels with their library rows (for `explain --domain`)
  golden.json           a few molecules with expected outputs, re-checked when the model loads
  bit_dictionary.json   Morgan bit names (Morgan models only)
  columns/c0/
    s.json              S (XGBoost JSON, trimmed to the best iteration)
    soft_meta.json      fidelity, rows, trees, reweighting, parameters
    fidelity.npz        held-out teacher vs S (for the report)
    _hard/              h.json, h_to_s.json, h_heldout.npz, labelled.csv, labelled_fp.npz, hard_meta.json
    explain/            global_s.csv, global_hs.csv, summary.json, and the data behind the report's charts
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

S is explained on 2,000 molecules of the library panel, H_S on up to 2,000 of its labelled molecules.
`summary.json` holds the fidelity, the top features and, with a hard head, its held-out metric, `a_max`
and *reach*: the share of panel molecules the hard head speaks for (a > 0).

## Molecule explanations (`xai4chem explain -o out/`)

- `predictions.csv` has one row per molecule. Per column:
  - `<column>`: P on the teacher's scale
  - `<column>_S`: S, on the teacher's scale like every value here (as in `predict(details=True)` from Python)
  - with a hard head:
    - `_HS` (H_S, teacher scale), `_T`, `_a`
    - `_nearest_smiles` and `_nearest_label`
    - `_agreement`: the cosine of φ_S and φ_HS, from −1 (opposite) to 1 (same story)

  - with `--domain`, the applicability check:
    - `library_T`: Tanimoto similarity to the nearest molecule in the reference library
    - `library_nearest_smiles`
    - `<column>_library_teacher`: the teacher's own value for that neighbour

    T near 1 means S was trained on (nearly) this molecule; below ~0.3, S is extrapolating.

  Molecules are named from an `id`, `name` or `key` column of the input when there is one, else
  `mol_0000`, `mol_0001`… Unparseable SMILES keep their row, with `valid = False`.
- `attributions.csv` is in long format: molecule × column × feature, with `value`, `phi_S` (and `phi_HS`,
  `phi_P`). It covers every feature present in the molecule plus the strongest absent ones.
- `index.html` is a self-contained gallery: an overview table, then one card per molecule (the first 100).
  Each card has:
  - the prediction, S, H_S and `a`
  - the atom maps
  - the top contributions
  - the nearest labelled molecule

## Two molecules (`xai4chem explain -S A -S B`)

With exactly two molecules, `explain` also splits the gap between their predictions, in the terminal and at
the top of the gallery. The gap (model scale) is exactly the difference of base values (the gate, with a
hard head) plus the per-feature differences of the attributions. A positive difference favours the first
molecule. Both molecules are drawn on one colour scale.

## Fast predictions (`xai4chem predict`)

`predict` skips attributions, so it scores millions of molecules. It writes `smiles` and `valid`, then P per
column on the teacher's scale. From Python, `predict(..., details=True)` also gives S, H_S, T and `a`.

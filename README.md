# XAI4Chem

Explainable AI for chemistry: explain *what a model has learned* by distilling it into an interpretable
surrogate, then attributing the surrogate's predictions to substructures, functional groups or
descriptors.

XAI4Chem follows the logic of [Olinda](https://github.com/ersilia-os/olinda):

1. **Soft labels.** Score a reference library of molecules with any *teacher* model, for example an
   Ersilia Model Hub model or a ZairaChem model.
2. **S, the surrogate.** XGBoost learns to reproduce the teacher on the chosen featurizer. Teacher
   probabilities are learned in logit space.
3. **H, from hard labels (optional).** A model trained on experimental measurements with the same
   featurizer. It is calibrated onto S's scale by an isotonic map (**H_S**) and mixed in only near the
   labelled chemistry: `P = (1 − a)·S + a·H_S`, where `a = a_max · ramp(T)` and T is the 1-NN Tanimoto
   to the labelled set.

For a single molecule `a` is fixed, so the explanations combine exactly. Each molecule gets three views:

- **teacher (S)**
- **experiments (H_S)**
- **combined (P)**

There is also a map of where the experiments disagree with the teacher.

Explanations describe S. They are only as faithful as S is to the teacher, so every report shows that
fidelity.

## Installation

```bash
conda create -n xai4chem -c conda-forge python=3.10 -y
conda activate xai4chem
python -m pip install git+https://github.com/ersilia-os/xai4chem.git
```

## Usage

### 1. Prepare the reference library (once)

Featurizing a large library is the expensive, model-independent step. `setup` stores row-aligned
feature matrices, a panel of molecules for global explanations and (for Morgan) a bit dictionary under
`~/.xai4chem/library/`. Set `XAI4CHEM_HOME` or pass `--library-dir` to change the location.

```bash
xai4chem setup --library-file reference_library.csv --featurizer morgan accfg datamol --n-jobs 8
```

### 2. Get soft labels

Score the library with your teacher and save a CSV with a `smiles` column plus one column per teacher
output. This is the same format Olinda uses.

```bash
xai4chem library -o reference_library_smiles.csv
# ...run the teacher on it → soft_labels.csv
```

### 3. Fit

```bash
xai4chem fit --soft-labels soft_labels.csv --featurizer morgan --model-dir models/my_model
xai4chem fit --soft-labels soft_labels.csv --hard-labels hard_labels.csv --featurizer accfg --model-dir models/my_model
```

- `fit` runs `learn-soft`, then `learn-hard` (when hard labels are given), then `explain-global`.
- Each stage is also a command of its own.
- Hard-label columns must be named like the soft-label columns: 0/1 labels give a classifier, anything
  else a regressor.
- Add `--stability` to retrain S with a second seed and report how stable the global importances are.

### 4. Explain, report, validate

```bash
xai4chem explain  --model-dir models/my_model -i my_molecules.csv -o explanations/ --id-col name
xai4chem report   --model-dir models/my_model -o report/
xai4chem validate --model-dir models/my_model --soft-labels heldout_soft.csv --hard-labels heldout_hard.csv
```

`explain` writes:

- `predictions.csv`: P on the teacher's scale, plus S, H_S, T, a, the nearest labelled analogue and an
  S-vs-H_S agreement score.
- `attributions.csv`: feature-level attributions φ_S, φ_HS and φ_P.
- Per molecule:
  - a waterfall plot
  - atom highlights (teacher, experiments, combined, disagreement)
  - drawings of the top features

## Featurizers

| Featurizer | Features | Atom view | Labels |
|---|---|---|---|
| `morgan` | Morgan counts, radius 3, 2048 bits | yes (occurrence centres) | named by the bit dictionary, e.g. `bit 1683: Cl (78%)`; the percentage shows how often the bit's top substructure is the one that set it, so collisions are visible |
| `accfg` | AccFG functional-group and heterocycle counts | yes (group atoms) | group names |
| `datamol` | datamol physicochemical descriptors | no | descriptor names |

## Python API

```python
from xai4chem import XAIModel

model = XAIModel("models/my_model")
model.predict(["CCO", "c1ccccc1"])
res = model.explain(["CCO"])["activity"]   # S, phi_S, P, phi_P, and with a hard head HS, phi_HS, T, a...
```

## Model directory

```
models/my_model/
  manifest.json  targets.h5  golden.json  bit_dictionary.json
  columns/c0/s.json  soft_meta.json  explain/  _hard/{h.json, h_to_s.json, labelled.csv, hard_meta.json}
```

`golden.json` stores expected outputs for a few molecules. They are re-checked every time the model
loads, so a mismatched environment fails loudly.

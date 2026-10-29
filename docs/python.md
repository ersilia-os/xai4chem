# Python API

Each command is a function in `xai4chem.api`, also importable from `xai4chem`. The functions print
nothing; pass `display=xai4chem.ui.console.ConsoleDisplay()` to see the CLI's progress output.

```python
import xai4chem

smiles = ["CC(=O)Nc1ccc(O)cc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]
xai4chem.setup()                                                  # as `xai4chem setup`
xai4chem.fit("soft.csv", "models/my_model", hard_labels="hard.csv")
out = xai4chem.explain("models/my_model", smiles, "explanations/") # predictions, attributions, gallery
xai4chem.predict("models/my_model", smiles, "predictions.csv")
```

`fit` also takes `seed`, `transform`, `n_estimators` and `panel_rows`, which the CLI leaves at their
defaults.

A model directory is loaded with `XAIModel`. Loading
re-scores the model's golden molecules and refuses to continue if this environment does not reproduce them.

```python
from rdkit import Chem

from xai4chem import XAIModel
from xai4chem.explain.atoms import atom_attributions

model = XAIModel("models/my_model")
model.column_names                       # teacher columns, e.g. ['qed', 'clogp']

smiles = ["CC(=O)Nc1ccc(O)cc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]
model.predict(smiles, details=True)      # DataFrame: P (teacher scale) per column, plus S, H_S, T, a
```

## Explanations

`explain` returns, per column, arrays over molecules, on the model's scale (the teacher's own scale):

```python
res = model.explain(smiles)["qed"]
res["S"], res["phi_S"], res["base_S"]    # surrogate, its TreeSHAP values (n × features), base value
res["P"], res["phi_P"], res["base_P"]    # blended prediction and its exact attribution
res["prob"]                              # P on the teacher's scale
# with a hard head:
res["HS"], res["phi_HS"], res["T"], res["a"], res["nearest"]

assert abs(res["phi_P"][0].sum() + res["base_P"][0] - res["P"][0]) < 1e-4   # exact, per molecule
```

Attributions are dense (`n × features`), so explain hundreds or thousands of molecules at a time. `predict`
skips attributions and handles millions.

Onto atoms:

```python
scores, absent = atom_attributions(model.featurizer, Chem.MolFromSmiles(smiles[0]), res["phi_P"][0])
# scores: one value per atom; absent: the effect of features the molecule does not have
```

Feature names:

```python
model.featurizer.labels()                               # library-wide names (Morgan: bit dictionary)
model.featurizer.molecule_labels(Chem.MolFromSmiles(s)) # Morgan: the environment setting each bit in s
```

The nearest reference-library molecule and the teacher's value for it (applicability domain):

```python
model.library_neighbours(smiles)   # {"T": ..., "smiles": [...], "teacher": {"qed": ...}}
```

## The report

```python
from xai4chem.report.model_report import write_report

write_report("models/my_model", "elsewhere.html")   # `fit` writes models/my_model/report.html
```

## The reference library

```python
from xai4chem import ReferenceLibrary

library = ReferenceLibrary()                    # ~/.xai4chem/library
library.smiles[:3], len(library), library.featurizers
X = library.csr("morgan", rows=[0, 10, 20])     # sparse feature rows
library.bit_dictionary().label(1380)            # 'bit 1380: [c;D3;R] (93%)'
```

# Explainable surrogates of chemistry models

xai4chem explains *what a model has learned*. Score Ersilia's reference library (1.36M molecules) with any
**teacher** model, for example an [Ersilia Model Hub](https://github.com/ersilia-os/ersilia) model or a
ZairaChem model. xai4chem then distills it into a fast XGBoost **surrogate** and attributes every
prediction to atoms, substructures, functional groups or descriptors.

It follows the logic of [Olinda](https://github.com/ersilia-os/olinda), and the same soft-label files work
for both tools:

1. **S** learns the teacher's outputs (soft labels) over the library.
2. **H**, optional, learns your own measurements (hard labels) with the same features.
3. **H_S** is H carried onto S's scale by an isotonic map.
4. The two are blended only near the labelled chemistry: `P = (1 − a)·S + a·H_S`, where
   `a = a_max · ramp(T)` and T is the Tanimoto similarity to the nearest labelled molecule.

For a given molecule `a` is fixed, so the explanations merge exactly: `φ_P = (1 − a)·φ_S + a·φ_HS`. Every
molecule gets four views:

- **teacher**
- **experiments**
- **combined**
- **disagreement**: where the experiments disagree with the teacher

## Installation

```bash
conda create -n xai4chem -c conda-forge python=3.10 -y
conda activate xai4chem
pip install git+https://github.com/ersilia-os/xai4chem.git
```

## Quick start

```bash
xai4chem setup -f morgan -f accfg                # once: library features, panel, Morgan bit names
xai4chem library -o erl0_smiles.csv              # score this file with your teacher → soft.csv
xai4chem fit -s soft.csv -m models/my_model      # S → explanations → HTML report
xai4chem fit -s soft.csv -h measurements.csv -f accfg -m models/my_model_fg   # with a hard head
xai4chem explain -m models/my_model -i molecules.csv -o explanations/ --open
xai4chem predict -m models/my_model -S "CC(=O)Nc1ccc(O)cc1" -o paracetamol.csv     # quick, no file needed
```

What `setup` reuses and computes:

- **Morgan counts** come precomputed from Olinda's `erl0_qmfp_r3.h5`. The copy that `olinda setup`
  puts in `~/.olinda` is reused; otherwise xai4chem tries to download it (2.9 GB). With your own
  molecules instead, use `xai4chem setup --library my_smiles.csv -f all`.
- **AccFG and datamol features** are computed locally, once (about 6 and 13 minutes on 8 cores).

With an Ersilia Model Hub model as the teacher, `ersilia serve <model-id>` and then
`ersilia run -i erl0_smiles.csv -o soft.csv` writes soft labels in a format xai4chem reads directly: the
`key` column is ignored and `input` is taken as the SMILES.

The input files:

- **Soft labels:** a `smiles` (or `input`) column plus one numeric column per teacher output.
- **Hard labels:** columns named like the teacher's outputs. 0/1 labels give a classifier, anything else
  a regressor.

## Commands

| Command | What it does |
|---|---|
| `setup` | Prepare the reference library: feature matrices, explanation panel, Morgan bit dictionary |
| `library` | Write the library SMILES for the teacher to score |
| `fit` | Run learn-soft → [learn-hard] → explain-global → report |
| `explain` | Explain molecules: predictions, attributions, an HTML gallery (and PNGs with `--png`) |
| `predict` | Score molecules fast, without explanations |
| `report` | Write the self-contained HTML model report |
| `info` | Show a model card in the terminal |
| `learn-soft` | Learn S from soft labels |
| `learn-hard` | Learn H from hard labels, calibrate it onto S, set up the gate |
| `explain-global` | Global importances and figures over the library panel |
| `validate` | Score a model on held-out soft and/or hard labels |

Run `xai4chem <command> --help` for every option.

## Featurizers

| `-f` | Features | Explanations |
|---|---|---|
| `morgan` | Morgan counts, radius 3, 2048 bits, clipped at 15 (identical to Olinda's) | atoms; bits named by their most common substructure in the library, with the collision rate |
| `accfg` | AccFG functional-group and heterocycle counts | atoms; functional groups by name |
| `datamol` | datamol physicochemical descriptors | descriptors (no atom view) |

## Python

```python
from xai4chem import XAIModel

model = XAIModel("models/my_model")
model.predict(["CCO", "c1ccccc1O"])                 # P on the teacher's scale
res = model.explain(["c1ccccc1O"])["my_column"]     # S, φ_S, P, φ_P (+ H_S, φ_HS, T, a)
```

On the full Ersilia library a Morgan fit takes about 3 minutes, and scoring all 1.36M molecules takes
about a minute ([benchmarks](docs/benchmarks.md)). How it works, the outputs, and what is precomputed are
described in [docs/](docs/).

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)

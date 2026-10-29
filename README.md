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

## What you get

- **A model report** (`report.html`, written by `fit`):
  - fidelity to the teacher
  - global importances and the top substructures, functional groups or descriptors
  - the hard head's held-out performance, calibration and reach
- **Molecule explanations** (`explain`):
  - atom maps (teacher, experiments, combined, disagreement)
  - top contributions, and the nearest labelled molecule
  - with exactly two molecules, why one scores above the other, as exact per-feature differences
  - all of it as CSV plus a self-contained HTML gallery
- **Fast scoring** (`predict`): about a minute for all 1.36M library molecules.

## Installation

```bash
conda create -n xai4chem -c conda-forge python=3.12 -y
conda activate xai4chem
pip install git+https://github.com/ersilia-os/xai4chem.git
```

## Quick start

```bash
xai4chem setup --export-smiles erl0_smiles.csv      # once; then score this file with your teacher → soft.csv
xai4chem fit -s soft.csv -m models/my_model         # S on descriptors → global explanations → report.html
xai4chem fit -s soft.csv -h measurements.csv -f accfg -m models/my_model_fg   # with a hard head (--task auto|binary|regression)
xai4chem explain -m models/my_model -i molecules.csv                  # → explanations/
xai4chem explain -m models/my_model -S "<smiles A>" -S "<smiles B>"   # two molecules: a matched pair
xai4chem predict -m models/my_model -i molecules.csv -o predictions.csv
```

`setup` downloads Ersilia's reference library with every feature precomputed: SMILES, Morgan and datamol
by default (about 0.5 GB), AccFG with `-f accfg` (60 MB more). With your own molecules instead,
`xai4chem setup --library my_smiles.csv -f all` featurizes them locally, once.

`fit` uses datamol descriptors unless told otherwise (`-f morgan` for substructures and atom maps,
`-f accfg` for functional groups). Check the fidelity in the final panel: explanations are only as good as
the surrogate's fit to the teacher.

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
| `setup` | Prepare the reference library once: features, explanation panel, Morgan bit names (`--export-smiles` writes the SMILES to score) |
| `fit` | Learn S from soft labels and, with `-h`, H from hard labels; explain the library panel; write `report.html` |
| `explain` | Predictions, attributions and an HTML gallery per molecule (`--domain` adds the nearest library molecule) |
| `predict` | Score molecules fast, without explanations |

Run `xai4chem <command> --help` for the options.

## Featurizers

| `-f` | Features | Explanations |
|---|---|---|
| `morgan` | Morgan counts, radius 3, 2048 bits, clipped at 15 (identical to Olinda's) | atoms; per molecule, bits named by the environment present; globally, by their most common substructure in the library, with the collision rate |
| `accfg` | AccFG functional-group and heterocycle counts | atoms; functional groups by name |
| `datamol` | datamol physicochemical descriptors | descriptors (no atom view) |

## Python

```python
import xai4chem

xai4chem.fit("soft.csv", "models/my_model")           # the same steps as the CLI, silently
model = xai4chem.XAIModel("models/my_model")
model.predict(["CCO", "c1ccccc1O"])                    # P on the teacher's scale
res = model.explain(["c1ccccc1O"])["my_column"]        # S, φ_S, P, φ_P (+ H_S, φ_HS, T, a)
```

On the full Ersilia library a Morgan fit takes about 3 minutes, and scoring all 1.36M molecules takes
about a minute ([benchmarks](docs/benchmarks.md)). How it works, the outputs, the Python API and what is
precomputed are described in [docs/](docs/).

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)

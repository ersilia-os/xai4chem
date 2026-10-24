# How xai4chem works

## The idea: explain a surrogate trained on a million soft labels

Explaining a model trained on a few hundred measurements mostly explains the noise in those measurements.
xai4chem instead explains a **global surrogate**:

1. A *teacher* model is scored once over Ersilia's reference library (1,355,109 molecules).
2. A gradient-boosting model, S, learns to reproduce those outputs.
3. With a million training examples, S's attributions are stable, and they describe *what the teacher
   has learned*.

This is a well-established explainable-AI pattern, and it carries three caveats that xai4chem makes visible
rather than hiding:

- **Fidelity.** The explanations describe S. They are only as faithful to the teacher as S is, so every
  report shows S's R² and Spearman against the teacher on held-out library molecules.
- **Vocabulary.** S can only use the features it is given. A teacher may rely on something the featurizer
  cannot express, and the explanations will then point at proxies.
- **Correlation.** SHAP values say how S uses features, not what causes activity. Correlated features
  share credit. `--stability` retrains S with a second seed and reports how stable the global importances
  are.

## The models (Olinda's letters)

| | |
|---|---|
| **S** | XGBoost regressor on the teacher's outputs. Probabilities are learned in logit space, where SHAP values are additive. Early stopping on 10% held-out library molecules; the booster is trimmed to its best iteration |
| **H** | XGBoost model on your measurements, with the same features as S: binary classifier for 0/1 labels, regressor otherwise. Margin space. Quality from 5-fold cross-validation |
| **H_S** | H carried onto S's scale by isotonic regression, fitted on library molecules where both H's score and the teacher's output exist. The direction is learned |
| **T** | Exact Tanimoto similarity (binary Morgan, radius 3, 2048 bits) to the nearest labelled molecule |
| **a** | `a_max · ramp(T)`: 0 below T = 0.4, rising to `a_max` at T = 0.7 (Olinda's constants) |
| **a_max** | Earned from the R² of H_S against the teacher on library molecules held out from the isotonic fit. Capped at 0.66, and 0 below 0.1 (the column then stays soft-only) |
| **P** | `(1 − a)·S + a·H_S` |

Training uses Olinda's tuned defaults:
- 64-bin histogram trees, which is lossless for counts clipped at 15
- depth 8
- learning rate 0.1
- row subsampling 0.8, column subsampling 0.5
- a `min_child_weight` that grows with the data and reaches 200 at about a million rows

Morgan and AccFG counts are passed as sparse matrices whose zeros are *missing*. For a count feature,
"absent" and "missing" are the same split, and training only touches the ~3% of entries that are nonzero.

## Merging soft and hard explanations

For one molecule, `a` is a number, so attributions combine linearly and exactly:

- `φ_S` is TreeSHAP of S. It sums to `S(x) − E[S]`.
- `φ_H` is TreeSHAP of H in margin space. It sums to `h(x) − E[h]`.
- `φ_HS = φ_H · (H_S(x) − H_S(E[h])) / (h(x) − E[h])` carries H onto S's scale.
  - This is the *rescale rule* from DeepLIFT: the secant slope of the isotonic map.
  - A decreasing map flips the signs.
- `φ_P = (1 − a)·φ_S + a·φ_HS`. It sums to `P(x) − [(1 − a)·E[S] + a·H_S(E[h])]`.

The decomposition is exact *given* `a`. The gate itself is not attributed to features. Instead, T, `a` and
the nearest labelled molecule (with its measured value) are reported alongside every explanation.

The explanation gallery shows four atom maps per molecule, all on one colour scale:

- teacher (S)
- experiments (H_S)
- combined (P)
- **disagreement** (H_S − S): the atoms where the measurements tell a different story from the teacher

## Applicability domain

S learned the teacher over the reference library, so it is most trustworthy for molecules that resemble
library molecules. For every explained molecule, `explain` therefore finds the nearest molecule in the
whole library (exact Tanimoto on binary Morgan, computed as library × queries) and reports:
- its similarity
- the teacher's own value for it

A molecule far from everything in the library (T below ~0.3) gets a warning in the gallery. A large gap
between S's prediction and the teacher's value for a close neighbour is a hint to look closer.

## From features to atoms

**Morgan bits.** A bit's attribution is split evenly over its occurrences in the molecule and placed on each
occurrence's centre atom.

**AccFG groups.** A group's attribution is split over its matches, and each match's share is spread over
its atoms.

In both cases the atom scores sum exactly to the attribution of the features *present* in the molecule. The
rest is the effect of *absent* features: TreeSHAP scores them too, but they have no atoms to sit on. It is
reported as a separate number.

**Folded Morgan bits collide.** Per-molecule outputs therefore name each present bit by the environment that
sets it *in that molecule*. For global summaries, `setup` builds a **bit dictionary**:
- every substructure that sets each bit, counted over 100,000 library molecules
- each bit named by its most common substructure and that substructure's share, e.g. `bit 1683: Cl (76%)`
  - single atoms are written as SMARTS with the Morgan invariants: `[cH;D2;R]` is an aromatic CH with
    two neighbours, in a ring
  - when the top substructure covers less than half of a bit's occurrences, the runner-up is named as
    well
- a collision rate: the share of the bit's occurrences that come from other substructures

## What is precomputed, and where

`xai4chem setup` builds everything that does not depend on a model, once, under `~/.xai4chem/library/`
(override with `--library-dir` or `XAI4CHEM_HOME`):

| File | What | Size (Ersilia library) |
|---|---|---|
| `library.json` | Manifest: where the SMILES and every matrix live | — |
| Morgan counts | Olinda's `erl0_qmfp_r3.h5`, used in place (not copied) | 2.9 GB |
| `accfg.h5` | AccFG counts, uint8 | 0.7 GB |
| `datamol.h5` | datamol descriptors, float32 | 0.1 GB |
| `panel.npy` | 50,000 random library rows; global explanations use the first 10,000 by default (`--explain-rows`) | — |
| `bit_dictionary.json` | Morgan bit names and collision rates | ~2 MB |
| `morgan_bits_*.npy` | Binary Morgan fingerprints as CSR index arrays, for the applicability search | 0.2 GB |

These files are model-independent, so they are what an eosvc data folder (or a public bucket, like
Olinda's) should hold. Anything model-specific lives in the model directory. A custom library is a CSV of
SMILES: `xai4chem setup --library my_library.csv -f all`.

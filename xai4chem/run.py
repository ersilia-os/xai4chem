"""The model directory: its manifest, its layout and the per-column files it holds.

    <model-dir>/
      manifest.json         featurizer config, library location, one entry per teacher column
      targets.h5            the soft labels, with their library rows (for the applicability check)
      bit_dictionary.json   copied from the library, so Morgan explanations are self-contained
      golden.json           a few molecules with expected outputs, re-checked when the model loads
      columns/<id>/
        s.json  soft_meta.json  fidelity.npz            S, the surrogate of the teacher
        _hard/  h.json  h_to_s.json  labelled.csv  labelled_fp.npz  hard_meta.json
                                                    H, its isotonic map, and the gate's labelled set
        explain/                            global explanations
      report.html           the model report

Columns get directory-safe ids (``c0``, ``c1``...) because teacher headers are user data; the real name
lives in the manifest. As in Olinda, ``hard_meta.json`` is written last and marks a complete hard head.
"""

import json
from pathlib import Path

SCHEMA = "xai4chem.model.v2"
MANIFEST_NAME = "manifest.json"
TARGETS_NAME = "targets.h5"
GOLDEN_NAME = "golden.json"
BIT_DICTIONARY_NAME = "bit_dictionary.json"
COLUMNS_DIRNAME = "columns"
HARD_DIRNAME = "_hard"
EXPLAIN_DIRNAME = "explain"
S_NAME = "s.json"
SOFT_META_NAME = "soft_meta.json"
FIDELITY_NAME = "fidelity.npz"
H_NAME = "h.json"
H_TO_S_NAME = "h_to_s.json"
LABELLED_NAME = "labelled.csv"
LABELLED_FP_NAME = "labelled_fp.npz"
HARD_META_NAME = "hard_meta.json"
REPORT_NAME = "report.html"
FIGURES_DIRNAME = "figures"
METRICS_NAME = "metrics.json"
PERFORMANCE_NAME = "performance_table.csv"
H_HELDOUT_NAME = "h_heldout.npz"


def column_id(index):
    return f"c{index}"


def column_dir(model_dir, col_id):
    return Path(model_dir) / COLUMNS_DIRNAME / col_id


def hard_dir(model_dir, col_id):
    return column_dir(model_dir, col_id) / HARD_DIRNAME


def has_hard_head(model_dir, col_id):
    return (hard_dir(model_dir, col_id) / HARD_META_NAME).exists()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_manifest(model_dir):
    path = Path(model_dir) / MANIFEST_NAME
    if not path.exists():
        hint = "it does not exist" if not Path(model_dir).exists() else f"it has no {MANIFEST_NAME}"
        raise FileNotFoundError(
            f"{model_dir} is not an xai4chem model ({hint}). Train one with `xai4chem fit -s soft.csv -m {model_dir}`."
        )
    return read_json(path)


def write_manifest(model_dir, manifest):
    manifest = dict(manifest)
    manifest["schema"] = SCHEMA
    write_json(Path(model_dir) / MANIFEST_NAME, manifest)

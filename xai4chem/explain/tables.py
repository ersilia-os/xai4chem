"""Tables built from explanations: predictions, long-format attributions, and two-molecule comparisons."""

import numpy as np
import pandas as pd

TOP_FEATURES = 20
COMPARE_TOP = 12


def cosine(a, b):
    """Cosine similarity of two vectors; NaN if either has no length."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def prediction_rows(cols, res, ids, smiles, valid):
    """One row per molecule; every value on the teacher's scale (the same columns as ``predict --details``)."""
    df = pd.DataFrame({"id": ids, "smiles": smiles, "valid": valid})
    for col in cols:
        r = res[col.name]
        df[col.name] = r["P"]
        df[f"{col.name}_S"] = r["S"]
        if col.hard is not None:
            hh = col.hard
            df[f"{col.name}_HS"] = r["HS"]
            df[f"{col.name}_T"] = r["T"]
            df[f"{col.name}_a"] = r["a"]
            df[f"{col.name}_nearest_smiles"] = [hh.labelled_smiles[j] if j >= 0 else None for j in r["nearest"]]
            df[f"{col.name}_nearest_label"] = [hh.labels[j] if j >= 0 else np.nan for j in r["nearest"]]
            df[f"{col.name}_agreement"] = [cosine(r["phi_S"][i], r["phi_HS"][i]) for i in range(len(ids))]
    values = [c for c in df.columns if c not in ("id", "smiles", "valid")]
    df.loc[~df["valid"], values] = np.nan
    return df


def attribution_rows(cols, res, X, ids, mols, featurizer):
    """Long format: every feature present in a molecule, plus the strongest absent ones."""
    labels, names = featurizer.labels(), featurizer.feature_names
    rows = []
    for col in cols:
        r = res[col.name]
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            phi = r["phi_P"][i]
            local = featurizer.molecule_labels(mol)
            keep = set(np.nonzero(X[i])[0]) | set(np.argsort(-np.abs(phi))[:TOP_FEATURES])
            for f in sorted(keep, key=lambda f: -abs(phi[f])):
                row = {
                    "id": ids[i],
                    "column": col.name,
                    "feature": names[f],
                    "label": local.get(f, labels[f]),
                    "value": float(X[i, f]),
                    "phi_S": float(r["phi_S"][i, f]),
                }
                if col.hard is not None:
                    row["phi_HS"] = float(r["phi_HS"][i, f])
                    row["phi_P"] = float(phi[f])
                rows.append(row)
    return rows


def compare_pair(col, r, X, mols, featurizer):
    """Why molecule 0 scores above or below molecule 1: the gap split into exact per-feature differences.

    ``P(0) − P(1) = [base(0) − base(1)] + Σ_f [φ_P(0)_f − φ_P(1)_f]``. The base values differ only through the
    gate; a remainder appears only when H's attribution could not be rescaled for one of the molecules.
    """
    labels = featurizer.labels()
    local = [featurizer.molecule_labels(m) for m in mols]
    d = r["phi_P"][0] - r["phi_P"][1]
    gap = float(r["P"][0] - r["P"][1])
    gate = float(r["base_P"][0] - r["base_P"][1])
    rows = []
    for f in [f for f in np.argsort(-np.abs(d)) if d[f] != 0][:COMPARE_TOP]:
        rows.append(
            {
                "label": local[0].get(f) or local[1].get(f) or labels[f],
                "a": float(X[0, f]),
                "b": float(X[1, f]),
                "phi_a": float(r["phi_P"][0, f]),
                "phi_b": float(r["phi_P"][1, f]),
                "diff": float(d[f]),
            }
        )
    pred = r["P"]
    return {
        "column": col.name,
        "P": (float(pred[0]), float(pred[1])),
        "gap": gap,
        "gate": gate,
        "residual": gap - gate - float(d.sum()),
        "a": (float(r["a"][0]), float(r["a"][1])) if "a" in r else None,
        "rows": rows,
    }

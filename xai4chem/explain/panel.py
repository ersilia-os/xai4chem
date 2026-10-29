"""Global explanations: which features drive S across the library, and H across its labelled set.

S is explained on the first molecules of the library *panel* (a fixed random subset chosen at setup;
2,000 by default, about 10 s of exact TreeSHAP per column). H is explained on up to 2,000 of its labelled
molecules. Per feature:

- ``mean_abs_shap``: global importance.
- ``mean_shap``: the net push across the library.
- ``frequency``: how often the feature is present (nonzero).
- ``mean_abs_shap_when_present``: the effect where it occurs, which ranks rare-but-strong features fairly.
- ``direction``: correlation between the feature's value and its SHAP value; positive means "more of this
  raises the prediction".

Everything is written to ``columns/<id>/explain/``, from where :mod:`xai4chem.report.model_report` draws.
"""

import time

import numpy as np
import pandas as pd

from xai4chem.data.similarity import nearest
from xai4chem.models.surrogate import contributions, safe_spearman
from xai4chem.progress import SILENT
from xai4chem.run import EXPLAIN_DIRNAME, write_json

PANEL_ROWS = 2000
PLOT_ROWS = 300
PLOT_FEATURES = 12
MAX_LABELLED = 2000
CHUNK = 500
TOP = 25


class Accumulator(object):
    """Running per-feature sums, so statistics never need a dense molecules × features matrix at once."""

    def __init__(self, n_features):
        self.n = 0
        self.sums = {k: np.zeros(n_features) for k in ("x", "xx", "p", "pp", "xp", "abs", "present", "abs_present")}

    def add(self, X, phi):
        """Add a block of molecules: features ``X`` and their attributions ``phi``."""
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype=np.float64)
        present = X > 0
        s = self.sums
        self.n += len(X)
        s["x"] += X.sum(0)
        s["xx"] += (X * X).sum(0)
        s["p"] += phi.sum(0)
        s["pp"] += (phi * phi).sum(0)
        s["xp"] += (X * phi).sum(0)
        s["abs"] += np.abs(phi).sum(0)
        s["present"] += present.sum(0)
        s["abs_present"] += (np.abs(phi) * present).sum(0)

    @property
    def importance(self):
        """Mean |SHAP| per feature."""
        return self.sums["abs"] / max(self.n, 1)

    def table(self, feature_names, labels, collisions=None):
        """The per-feature statistics, most important first."""
        s, n = self.sums, max(self.n, 1)
        cov = s["xp"] / n - (s["x"] / n) * (s["p"] / n)
        vx = s["xx"] / n - (s["x"] / n) ** 2
        vp = s["pp"] / n - (s["p"] / n) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            direction = np.where((vx > 0) & (vp > 0), cov / np.sqrt(vx * vp), 0.0)
            when_present = np.where(s["present"] > 0, s["abs_present"] / s["present"], 0.0)
        df = pd.DataFrame(
            {
                "feature": feature_names,
                "label": labels,
                "mean_abs_shap": s["abs"] / n,
                "mean_shap": s["p"] / n,
                "frequency": s["present"] / n,
                "mean_abs_shap_when_present": when_present,
                "direction": direction,
            }
        )
        if collisions is not None:
            df["collision"] = collisions
        return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def _plot_sample(phi, X, importance, labels):
    """The data behind the per-molecule effects figure: the top features for a few hundred molecules."""
    top = np.argsort(-importance)[:PLOT_FEATURES]
    X = X[:PLOT_ROWS]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return {"labels": np.array([str(labels[i]) for i in top]), "phi": phi[:PLOT_ROWS][:, top], "values": X[:, top]}


def explain_panel(model, library, rows=PANEL_ROWS, display=SILENT):
    """Global explanations for every column of ``model``, over the first ``rows`` panel molecules."""
    featurizer = model.featurizer
    panel = library.panel()[:rows]
    labels, names = featurizer.labels(), featurizer.feature_names
    collisions = None
    if featurizer.name == "morgan" and featurizer.bit_dictionary is not None:
        collisions = [featurizer.bit_dictionary.collision(i) for i in range(len(names))]
    X = (
        library.csr(featurizer.name, panel)
        if featurizer.sparse
        else library.matrix(featurizer.name, panel).astype(np.float32)
    )
    hard = any(c.hard for c in model.columns)
    panel_fps = library.csr("morgan", panel) if hard and library.has("morgan") else None
    fields = ["Top feature"] + (["S~H_S ρ"] if hard else []) + ["Time"]
    summaries = []
    with display.table(model.column_names, "Explaining the library panel", fields) as table:
        for col in model.columns:
            table.start(col.name)
            started = time.time()
            out = col.dir / EXPLAIN_DIRNAME
            out.mkdir(parents=True, exist_ok=True)
            n_lab = min(len(col.hard.labelled_smiles), MAX_LABELLED) if col.hard is not None else 0
            total = len(panel) + n_lab

            def status(done, total=total):
                """Forward the TreeSHAP progress to the live table."""
                table.status("TreeSHAP", done, total)

            acc, first = Accumulator(len(names)), None
            for start in range(0, len(panel), CHUNK):
                Xc = X[start : start + CHUNK]
                phi = contributions(col.s, Xc, model.sparse)[:, :-1]
                acc.add(Xc, phi)
                first = (phi, Xc) if first is None else first
                status(start + Xc.shape[0])
            tab = acc.table(names, labels, collisions)
            tab.to_csv(out / "global_s.csv", index=False)
            np.savez_compressed(out / "plot_s.npz", **_plot_sample(first[0], first[1], acc.importance, labels))
            summary = {
                "name": col.name,
                "panel_size": int(len(panel)),
                "fidelity": col.soft_meta["fidelity"],
                "top_features": tab.head(TOP)[["label", "mean_abs_shap", "direction"]].to_dict("records"),
            }
            cells = {"Top feature": str(tab.iloc[0]["label"])[:40]}
            if col.hard is not None:
                summary["hard"] = _explain_hard(
                    model, col, panel_fps, out, names, labels, collisions, lambda done: status(len(panel) + done)
                )
                cells["S~H_S ρ"] = f"{summary['hard']['importance_spearman_s_vs_hs']:.2f}"
            write_json(out / "summary.json", summary)
            summaries.append(summary)
            table.finish(col.name, Time=f"{time.time() - started:.0f}s", **cells)
    return summaries


def _explain_hard(model, col, panel_fps, out, names, labels, collisions, status):
    """What the experiments say (on the labelled set), and how far the hard head reaches into the panel."""
    hh = col.hard
    lab = hh.labelled_smiles
    if len(lab) > MAX_LABELLED:
        lab = [lab[i] for i in np.sort(np.random.default_rng(0).choice(len(lab), MAX_LABELLED, replace=False))]
    acc_h, acc_s = Accumulator(len(names)), Accumulator(len(names))
    plot, h_values = None, []
    for start in range(0, len(lab), CHUNK):
        chunk = lab[start : start + CHUNK]
        Xh = model.featurize(chunk)
        r = model.explain(chunk, X=Xh, columns=[col.name])[col.name]
        acc_h.add(Xh, r["phi_HS"])
        acc_s.add(Xh, r["phi_S"])
        h_values.append(r["h"])
        if plot is None:
            plot = (r["phi_HS"], Xh)
        status(start + len(chunk))
    tab_h = acc_h.table(names, labels, collisions)
    tab_h.to_csv(out / "global_hs.csv", index=False)
    np.savez_compressed(out / "plot_hs.npz", **_plot_sample(plot[0], plot[1], acc_h.importance, labels))
    used = (acc_s.importance > 0) | (acc_h.importance > 0)
    T_panel = nearest(panel_fps, hh.labelled_fps)[0] if panel_fps is not None else np.array([])
    np.savez_compressed(out / "reach.npz", T=T_panel, h=np.concatenate(h_values))
    return {
        **hh.meta,
        "importance_spearman_s_vs_hs": safe_spearman(acc_s.importance[used], acc_h.importance[used]),
        "reach": float((hh.gate(T_panel) > 0).mean()) if len(T_panel) else float("nan"),
        "top_features": tab_h.head(TOP)[["label", "mean_abs_shap", "direction"]].to_dict("records"),
    }

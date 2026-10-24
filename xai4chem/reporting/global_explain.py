"""Global explanations: which features drive S across the library, and H across its labelled set.

S is explained on the library *panel*, a fixed random subset chosen at setup (50k molecules by default),
in chunks: per-feature statistics are accumulated as running sums, so a dense ``molecules × features``
attribution matrix never exists. Per feature we keep:

- ``mean_abs_shap``: global importance.
- ``mean_shap``: the net push across the library.
- ``frequency``: how often the feature is present (nonzero).
- ``mean_abs_shap_when_present``: the effect where it occurs, which ranks rare-but-strong features fairly.
- ``direction``: correlation between the feature's value and its SHAP value. Positive means "more of
  this raises the prediction".
"""

import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from xai4chem.console import STEP_COLORS, LiveTable, elapsed, rule, set_active_color, summary_panel
from xai4chem.console import path as cpath
from xai4chem.library import ReferenceLibrary
from xai4chem.model import XAIModel
from xai4chem.run import EXPLAIN_DIRNAME, FIDELITY_NAME, write_json
from xai4chem.similarity import nearest
from xai4chem.soft import contributions

CHUNK = 5000
PLOT_ROWS = 2000
TOP = 25


class _Accumulator(object):
    def __init__(self, n_features):
        self.n = 0
        z = lambda: np.zeros(n_features)  # noqa: E731
        self.sx, self.sxx, self.sp, self.spp, self.sxp, self.sabs, self.present, self.abs_present = (
            z() for _ in range(8)
        )

    def add(self, X, phi):
        present = X > 0
        self.n += len(X)
        self.sx += X.sum(0)
        self.sxx += (X * X).sum(0)
        self.sp += phi.sum(0)
        self.spp += (phi * phi).sum(0)
        self.sxp += (X * phi).sum(0)
        self.sabs += np.abs(phi).sum(0)
        self.present += present.sum(0)
        self.abs_present += (np.abs(phi) * present).sum(0)

    def table(self, feature_names, labels, collisions=None):
        n = max(self.n, 1)
        cov = self.sxp / n - (self.sx / n) * (self.sp / n)
        vx = self.sxx / n - (self.sx / n) ** 2
        vp = self.spp / n - (self.sp / n) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            direction = np.where((vx > 0) & (vp > 0), cov / np.sqrt(vx * vp), 0.0)
            when_present = np.where(self.present > 0, self.abs_present / self.present, 0.0)
        df = pd.DataFrame(
            {
                "feature": feature_names,
                "label": labels,
                "mean_abs_shap": self.sabs / n,
                "mean_shap": self.sp / n,
                "frequency": self.present / n,
                "mean_abs_shap_when_present": when_present,
                "direction": direction,
            }
        )
        if collisions is not None:
            df["collision"] = collisions
        return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def draw_global_figures(out, prefix, phi_plot, X_plot, importance, direction, labels):
    from xai4chem.reporting.figures import beeswarm, import_stylia, importance_lollipop

    stylia = import_stylia()
    paths = {}
    _, axs = stylia.create_figure(1, 1)
    importance_lollipop(axs.next(), stylia, labels, importance, direction)
    paths["importance"] = out / f"{prefix}_importance.png"
    stylia.save_figure(str(paths["importance"]))
    _, axs = stylia.create_figure(1, 1)
    beeswarm(axs.next(), stylia, phi_plot, X_plot, labels)
    paths["beeswarm"] = out / f"{prefix}_beeswarm.png"
    stylia.save_figure(str(paths["beeswarm"]))
    import matplotlib.pyplot as plt

    plt.close("all")
    return paths


def explain_global(model_dir, library_dir=None, max_rows=None, seed=42):
    started = time.time()
    color = STEP_COLORS["explain-global"]
    set_active_color(color)
    model = XAIModel(model_dir)
    library = ReferenceLibrary(library_dir or model.manifest["library"]["dir"])
    rows = library.panel()
    if max_rows is not None:
        rows = rows[:max_rows]
    rule(
        "xai4chem · explain-global",
        style=color,
        right=f"{len(model.columns)} column(s) · panel of {len(rows):,} library molecules",
    )
    featurizer = model.featurizer
    labels = featurizer.labels()
    names = featurizer.feature_names
    collisions = None
    if featurizer.name == "morgan" and featurizer.bit_dictionary is not None:
        collisions = [featurizer.bit_dictionary.collision(i) for i in range(len(names))]
    if featurizer.sparse:
        X_panel = library.csr(featurizer.name, rows, verb="reading panel features")
    else:
        X_panel = library.matrix(featurizer.name, rows).astype(np.float32)
    plot_idx = np.sort(np.random.default_rng(seed).permutation(len(rows))[:PLOT_ROWS])
    panel_fps = (
        library.csr("morgan", rows, verb="reading panel fingerprints") if any(c.hard for c in model.columns) else None
    )

    summaries = []
    with LiveTable(
        model.column_names,
        title="Explaining",
        fields=["Top feature", "Agreement S~H_S", "Time"],
        item_label="Column",
        running_verb="explaining",
        color=color,
    ) as table:
        for col in model.columns:
            table.start(col.name)
            col_started = time.time()
            out = col.dir / EXPLAIN_DIRNAME
            out.mkdir(parents=True, exist_ok=True)
            acc = _Accumulator(len(names))
            phi_plot = []
            for start in range(0, len(rows), CHUNK):
                Xc = X_panel[start : start + CHUNK]
                phi = contributions(col.s, Xc, model.sparse)[:, :-1]
                acc.add(_dense(Xc).astype(np.float64), phi)
                sel = plot_idx[(plot_idx >= start) & (plot_idx < start + CHUNK)] - start
                phi_plot.append(phi[sel])
            tab = acc.table(names, labels, collisions)
            tab.to_csv(out / "global_s.csv", index=False)
            importance = acc.sabs / max(acc.n, 1)
            direction = tab.set_index("feature").loc[names, "direction"].to_numpy()
            figs = draw_global_figures(
                out, "s", np.vstack(phi_plot), _dense(X_panel[plot_idx]), importance, direction, labels
            )
            _fidelity_figure(col, out)
            summary = {
                "name": col.name,
                "panel_size": int(len(rows)),
                "base_S": float(contributions(col.s, X_panel[:1], model.sparse)[0, -1]),
                "fidelity": col.soft_meta["fidelity"],
                "fidelity_teacher_scale": col.soft_meta.get("fidelity_teacher_scale"),
                "stability_spearman": col.soft_meta.get("stability_spearman"),
                "top_features": tab.head(TOP)[["label", "mean_abs_shap", "direction"]].to_dict("records"),
                "figures": {k: v.name for k, v in figs.items()},
            }
            agreement = "—"
            if col.hard is not None:
                hh = col.hard
                Xh = model.featurize(hh.labelled_smiles)
                rh = model.explain(hh.labelled_smiles, X=Xh, columns=[col.name])[col.name]
                acc_h = _Accumulator(len(names))
                acc_h.add(Xh.astype(np.float64), rh["phi_HS"])
                tab_h = acc_h.table(names, labels, collisions)
                tab_h.to_csv(out / "global_hs.csv", index=False)
                imp_h = np.abs(rh["phi_HS"]).mean(0)
                dir_h = tab_h.set_index("feature").loc[names, "direction"].to_numpy()
                figs_h = draw_global_figures(
                    out, "hs", rh["phi_HS"][:PLOT_ROWS], Xh[:PLOT_ROWS].astype(np.float32), imp_h, dir_h, labels
                )
                imp_s = np.abs(rh["phi_S"]).mean(0)
                used = (imp_s > 0) | (imp_h > 0)
                rho = float(spearmanr(imp_s[used], imp_h[used]).statistic) if used.sum() > 2 else float("nan")
                T_panel, _ = nearest(panel_fps, hh.labelled_fps)
                _hard_figures(hh, rh["h"], T_panel, out)
                summary["hard"] = {
                    **hh.meta,
                    "importance_spearman_s_vs_hs": rho,
                    "mean_a_panel": float(hh.gate(T_panel).mean()),
                    "frac_panel_gated": float((hh.gate(T_panel) > 0).mean()),
                    "top_features": tab_h.head(TOP)[["label", "mean_abs_shap", "direction"]].to_dict("records"),
                    "figures": {k: v.name for k, v in figs_h.items()},
                }
                agreement = f"ρ {rho:.2f}"
            write_json(out / "summary.json", summary)
            summaries.append(summary)
            table.finish(
                col.name,
                **{
                    "Top feature": tab.iloc[0]["label"][:40],
                    "Agreement S~H_S": agreement,
                    "Time": elapsed(time.time() - col_started),
                },
            )

    rows_out = [(s["name"], f"top: {s['top_features'][0]['label']}") for s in summaries]
    rows_out += [
        ("Outputs", f"[dim]{cpath(model.dir)}/columns/*/explain[/]"),
        ("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]"),
    ]
    summary_panel("xai4chem · explain-global", rows_out, border_style=color, icon="✓")
    return model


def _fidelity_figure(col, out):
    path = col.dir / FIDELITY_NAME
    if not path.exists():
        return
    from xai4chem.reporting.figures import fidelity_hexbin, import_stylia

    stylia = import_stylia()
    data = np.load(path)
    _, axs = stylia.create_figure(1, 1, width=0.5, height=0.5)
    fidelity_hexbin(axs.next(), stylia, data["teacher"], data["surrogate"], col.transform)
    stylia.save_figure(str(out / "s_fidelity.png"))
    import matplotlib.pyplot as plt

    plt.close("all")


def _hard_figures(hh, h_labelled, T_panel, out):
    from xai4chem.reporting.figures import gate_histogram, import_stylia, isotonic_curve

    stylia = import_stylia()
    _, axs = stylia.create_figure(1, 1, width=0.5, height=0.5)
    isotonic_curve(axs.next(), stylia, hh.iso, h_labelled)
    stylia.save_figure(str(out / "hs_calibration.png"))
    _, axs = stylia.create_figure(1, 1)
    gate_histogram(axs.next(), stylia, T_panel, hh.meta["t_lo"], hh.meta["t_hi"])
    stylia.save_figure(str(out / "hs_reach.png"))
    import matplotlib.pyplot as plt

    plt.close("all")

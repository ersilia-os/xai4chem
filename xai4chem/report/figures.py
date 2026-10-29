"""The figures of ``report.html``, drawn with stylia through :mod:`xai4chem.report.style`.

Every plot takes the axis to draw into, so the caller owns the figure: :func:`render` builds it, writes a PNG
and a vector PDF, and closes it (matplotlib keeps figures alive until told otherwise, and a multi-column
report opens dozens). Nothing here computes a metric: values come in already computed, so a number on a figure
and the same number in ``metrics.json`` cannot disagree.
"""

import base64
from pathlib import Path

import numpy as np

from xai4chem.report.style import (
    REPORT_DPI,
    density,
    figure,
    hexcol,
    import_stylia,
    limits,
    reference,
    rgb,
)

MAX_POINTS = 400_000
LABEL_CHARS = 30
DIRECTION_BAND = 0.05  # |correlation| below this is "mixed", as in the page's badges

# The title and one-line caption of every figure, in one table: the page prints the caption and the plot
# draws the title, so a figure cannot end up described two ways.
FIGURES = {
    "correlation": ("Agreement with the teacher", "Teacher value against S on held-out library molecules."),
    "residuals": ("Residual distribution", "How far S lands from the teacher, and in which direction."),
    "residual_structure": (
        "Residual structure",
        "Residual against prediction; a fan or a slope means the error depends on the answer.",
    ),
    "calibration": (
        "Calibration",
        "Mean teacher value against mean prediction, in equal-population bins of the prediction.",
    ),
    "importance": (
        "Global importance",
        "Mean |SHAP| of the top features over the explanation panel; crimson raises the prediction, cobalt lowers it.",
    ),
    "effects": (
        "Feature effects",
        "SHAP value of the top features, one dot per panel molecule, coloured by the feature's value (crimson high).",
    ),
    "hard_importance": (
        "What the experiments say",
        "Mean |SHAP| of H_S on the labelled molecules; crimson raises the prediction, cobalt lowers it.",
    ),
    "hard_effects": (
        "Effects according to the experiments",
        "SHAP value of H_S for the top features on the labelled molecules, coloured by the feature's value.",
    ),
    "hard_roc": ("Hard head · ROC", "True positives against false positives for H on held-out labelled molecules."),
    "hard_agreement": (
        "Hard head · agreement",
        "H's margin against the measurement on held-out labelled molecules.",
    ),
    "isotonic": (
        "H → S map",
        "The isotonic map carrying H's margin onto S's scale; ticks are labelled molecules.",
    ),
    "tanimoto": (
        "Reach of the hard head",
        "Similarity of panel molecules to the labelled set; dashed lines bound the gate's ramp.",
    ),
}

# Footprint on the 3 cm grid, (rows, cols); square data spaces stay square, lists earn the height.
CELLS = {
    "residuals": (2, 3),
    "importance": (4, 4),
    "effects": (4, 4),
    "hard_importance": (4, 4),
    "hard_effects": (4, 4),
    "tanimoto": (2, 3),
}


def caption(key):
    """``(title, caption)`` of a figure."""
    return FIGURES.get(key, (key.replace("_", " "), ""))


def _subsample(arrays, seed=0):
    """Jointly subsample equal-length arrays, so rows stay paired."""
    if len(arrays[0]) <= MAX_POINTS:
        return arrays
    idx = np.random.default_rng(seed).choice(len(arrays[0]), size=MAX_POINTS, replace=False)
    return [a[idx] for a in arrays]


def _short(label):
    label = str(label)
    return label if len(label) <= LABEL_CHARS else label[: LABEL_CHARS - 1] + "…"


# --- S against the teacher ---------------------------------------------------------------------------


def correlation(ax, st, y, p, metrics):
    """Teacher against S as a density surface, with the y = x line."""
    y, p = _subsample([np.asarray(y, dtype=float), np.asarray(p, dtype=float)])
    density(ax, st, y, p, role="teacher")
    lo, hi = limits(ax, y, p)
    reference(ax, st, "diagonal", lo=lo, hi=hi)
    st.label(
        ax,
        xlabel="Teacher value",
        ylabel="Surrogate S",
        title=f"R² = {metrics['r2']:.3f} · ρ = {metrics['spearman']:.3f}",
    )


def residuals(ax, st, y, p):
    """Distribution of S minus the teacher; a narrow peak at zero is the goal."""
    r = np.asarray(p, dtype=float) - np.asarray(y, dtype=float)
    ax.hist(r, bins=60, color=rgb(st, "model"), edgecolor="white", linewidth=0.4)
    reference(ax, st, "vertical", value=0.0)
    st.label(ax, xlabel="S − teacher", ylabel="Molecules", title=f"bias = {r.mean():+.4f} · sd = {r.std():.4f}")


def residual_structure(ax, st, y, p):
    """Residual against the prediction."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    ps, rs = _subsample([p, p - y])
    density(ax, st, ps, rs, role="model")
    reference(ax, st, "horizontal", value=0.0)
    st.label(ax, xlabel="Surrogate S", ylabel="S − teacher", title="")


def calibration(ax, st, y, p, n_bins=20):
    """Mean teacher value in each equal-population bin of the prediction, against the bin's mean prediction."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(p) < 2 * n_bins or np.ptp(p) == 0:
        return False
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, n_bins)
    mp = np.array([p[b].mean() for b in bins])
    my = np.array([y[b].mean() for b in bins])
    ax.plot(mp, my, marker="o", color=rgb(st, "model"))
    lo, hi = limits(ax, mp, my)
    reference(ax, st, "diagonal", lo=lo, hi=hi)
    st.label(ax, xlabel="Mean S", ylabel="Mean teacher", title="")


# --- what S has learned ------------------------------------------------------------------------------


def _direction_color(st, d):
    if d > DIRECTION_BAND:
        return rgb(st, "raises")
    if d < -DIRECTION_BAND:
        return rgb(st, "lowers")
    return rgb(st, "neutral")


def importance(ax, st, labels, values, directions, n=12):
    """Horizontal bars of mean |SHAP| for the ``n`` most important features, coloured by direction."""
    labels, values, directions = list(labels)[:n], np.asarray(values, dtype=float)[:n], np.asarray(directions)[:n]
    if not len(values):
        return False
    ypos = np.arange(len(values))[::-1]
    ax.barh(ypos, values, color=[_direction_color(st, d) for d in directions])
    ax.set_yticks(ypos)
    ax.set_yticklabels([_short(label) for label in labels])
    st.label(ax, xlabel="Mean |SHAP|", ylabel="", title="")


def effects(ax, st, labels, phi, values, seed=0):
    """SHAP value of the top features, a row per feature and a dot per molecule, coloured by feature value."""
    from matplotlib.colors import LinearSegmentedColormap

    phi, values = np.asarray(phi, dtype=float), np.asarray(values, dtype=float)
    if phi.size == 0:
        return False
    cmap = LinearSegmentedColormap.from_list(
        "xai4chem_effects", [hexcol(st, "lowers"), "#DDDDDD", hexcol(st, "raises")]
    )
    rng = np.random.default_rng(seed)
    n_feat = phi.shape[1]
    for k in range(n_feat):
        v = values[:, k]
        span = float(v.max() - v.min())
        level = (v - v.min()) / span if span > 0 else np.full(len(v), 0.5)
        ax.scatter(phi[:, k], n_feat - 1 - k + rng.uniform(-0.3, 0.3, len(v)), c=level, cmap=cmap, vmin=0, vmax=1)
    ax.axvline(0, color=rgb(st, "neutral"), linewidth=1, zorder=0)
    ax.set_yticks(np.arange(n_feat)[::-1])
    ax.set_yticklabels([_short(label) for label in labels])
    st.label(ax, xlabel="SHAP value", ylabel="", title="")


# --- the hard head -----------------------------------------------------------------------------------


def hard_roc(ax, st, y, score, auroc):
    """ROC of H on held-out labelled molecules."""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y, score)
    ax.plot(fpr, tpr, color=rgb(st, "hard"))
    reference(ax, st, "diagonal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    st.label(ax, xlabel="False positive rate", ylabel="True positive rate", title=f"AUROC = {auroc:.3f}")


def hard_agreement(ax, st, y, score, spearman):
    """H's margin against the measurement on held-out labelled molecules."""
    ax.scatter(np.asarray(score, dtype=float), np.asarray(y, dtype=float), color=rgb(st, "hard"))
    st.label(ax, xlabel="H margin", ylabel="Measured", title=f"ρ = {spearman:.3f}")


def isotonic(ax, st, x, y, rug):
    """The isotonic H → S map with a rug of the labelled molecules' H margins."""
    x, y, rug = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(rug, dtype=float)
    ax.plot(x, y, color=rgb(st, "hard"))
    if len(rug):
        ax.plot(rug[:3000], np.full(min(len(rug), 3000), ax.get_ylim()[0]), "|", color=rgb(st, "neutral"), alpha=0.5)
    st.label(ax, xlabel="H margin", ylabel="H_S (S scale)", title="")


def tanimoto(ax, st, T, t_lo, t_hi):
    """Histogram of the panel's similarity to the labelled set, with the gate's ramp marked."""
    T = np.asarray(T, dtype=float)
    if not len(T):
        return False
    ax.hist(T, bins=40, range=(0, 1), color=rgb(st, "tanimoto"), edgecolor="white", linewidth=0.4)
    for v in (t_lo, t_hi):
        reference(ax, st, "vertical", value=v)
    st.label(ax, xlabel="1-NN Tanimoto to the labelled set", ylabel="Panel molecules", title="")


# --- writing -----------------------------------------------------------------------------------------


def render(draw, out_dir, name, key):
    """Draw one figure and write ``png/<name>.png`` and ``pdf/<name>.pdf`` under ``out_dir``.

    Parameters
    ----------
    draw : callable
        Receives ``(ax, stylia)`` and draws. Returning ``False`` declines the figure (no data to show).
    out_dir : str or Path
        The figures directory.
    name : str
        The file stem, unique within the model (it carries the column id).
    key : str
        The figure's kind, a key of :data:`FIGURES`, which sets its title, caption and footprint.

    Returns
    -------
    dict or None
        ``name``, ``key``, ``title``, ``caption``, ``png`` and ``pdf`` (relative to the model directory's
        figures folder) and ``data_uri`` (the PNG, for embedding), or ``None`` if the plot declined.
    """
    import matplotlib.pyplot as plt

    st = import_stylia()
    fig, ax = figure(st, CELLS.get(key, (2, 2)))
    try:
        if draw(ax, st) is False:
            return None
        out_dir = Path(out_dir)
        png, pdf = out_dir / "png" / f"{name}.png", out_dir / "pdf" / f"{name}.pdf"
        png.parent.mkdir(parents=True, exist_ok=True)
        pdf.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        fig.savefig(png, dpi=REPORT_DPI, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
    finally:
        plt.close(fig)
    title, text = caption(key)
    uri = "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()
    return {
        "name": name,
        "key": key,
        "title": title,
        "caption": text,
        "png": f"png/{png.name}",
        "pdf": f"pdf/{pdf.name}",
        "data_uri": uri,
    }

"""Global figures, drawn with stylia so they share the Ersilia palette with the HTML pages.

Every helper takes an ``ax`` and draws into it; the caller owns the figure (stylia's convention). Colour
roles are fixed across figures: crimson raises the prediction, cobalt lowers it, periwinkle is the
model, orchid is the hard head, amber is the gate.
"""

import numpy as np

# Format: slide | Style: ersilia (report figures sit inside an Ersilia-branded page)
FORMAT = "slide"
STYLE = "ersilia"


def import_stylia():
    """Import stylia without letting it wipe matplotlib's font cache.

    stylia 1.0.1 calls ``shutil.rmtree(matplotlib.get_cachedir())`` at import time, which forces
    matplotlib to rebuild its font cache, several seconds, on every run (Olinda hit the same thing).
    The rmtree is suppressed for the duration of the import only.
    """
    import os
    import shutil

    import matplotlib

    matplotlib.use("Agg")
    os.makedirs(matplotlib.get_cachedir(), exist_ok=True)
    original = shutil.rmtree
    shutil.rmtree = lambda *a, **k: None
    try:
        import stylia
    finally:
        shutil.rmtree = original
    stylia.set_format(FORMAT)
    stylia.set_style(STYLE)
    return stylia


def _article(stylia, name):
    return stylia.ArticleColors().get(name)


def fidelity_hexbin(ax, stylia, teacher, surrogate, transform):
    """Teacher vs surrogate on held-out rows, binned (a scatter of 100k points is a blob)."""
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    cmap = LinearSegmentedColormap.from_list("fid", ["#FFFFFF", _article(stylia, "periwinkle")])
    ax.hexbin(teacher, surrogate, gridsize=40, cmap=cmap, mincnt=1, linewidths=0, norm=LogNorm())
    lo = float(min(np.min(teacher), np.min(surrogate)))
    hi = float(max(np.max(teacher), np.max(surrogate)))
    if hi - lo < 1e-9:  # a constant teacher: give the axes some room rather than a singular range
        lo, hi = lo - 0.5, hi + 0.5
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=_article(stylia, "silver"), zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    scale = " (logit)" if transform == "logit" else ""
    stylia.label(ax, xlabel=f"Teacher{scale}", ylabel=f"Surrogate S{scale}", title="Fidelity on held-out rows")


def importance_lollipop(ax, stylia, labels, importance, direction, top=15):
    """Mean |SHAP| of the top features, as dots on hairlines; colour says which way the feature pushes."""
    order = np.argsort(-importance)[:top][::-1]
    y = np.arange(len(order))
    colors = [_article(stylia, "crimson") if direction[i] >= 0 else _article(stylia, "cobalt") for i in order]
    ax.hlines(y, 0, importance[order], color=_article(stylia, "silver"), zorder=1)
    ax.scatter(importance[order], y, color=colors, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([_short(labels[i]) for i in order])
    ax.set_xlim(left=0)
    stylia.label(ax, xlabel="Mean |SHAP|", ylabel="", title="Global importance")


def beeswarm(ax, stylia, phi, X, labels, top=12, seed=0):
    """SHAP value per molecule for the top features, coloured by the feature's value."""
    from matplotlib.colors import LinearSegmentedColormap

    importance = np.abs(phi).mean(0)
    order = np.argsort(-importance)[:top][::-1]
    cmap = LinearSegmentedColormap.from_list(
        "val", [_article(stylia, "cobalt"), "#E6E6EE", _article(stylia, "crimson")]
    )
    rng = np.random.default_rng(seed)
    for row, f in enumerate(order):
        x = X[:, f]
        span = x.max() - x.min()
        c = (x - x.min()) / span if span > 0 else np.full(len(x), 0.5)
        jitter = rng.uniform(-0.3, 0.3, len(x))
        ax.scatter(phi[:, f], row + jitter, c=cmap(c), s=stylia.get_markersize("small"), linewidths=0, rasterized=True)
    ax.axvline(0, color=_article(stylia, "silver"), zorder=0)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([_short(labels[i]) for i in order])
    stylia.label(
        ax, xlabel="SHAP value (low feature value → blue, high → red)", ylabel="", title="Per-molecule effects"
    )


def isotonic_curve(ax, stylia, iso, h_values=None):
    """The isotonic map from H's margin to S's scale, with a rug of the labelled molecules' margins."""
    x, y = np.asarray(iso["x"]), np.asarray(iso["y"])
    ax.plot(x, y, color=_article(stylia, "orchid"))
    if h_values is not None and len(h_values):
        lo = float(np.min(y))
        ax.plot(h_values, np.full(len(h_values), lo), "|", color=_article(stylia, "silver"), alpha=0.6)
    stylia.label(ax, xlabel="H margin", ylabel="H_S (S scale)", title="H → S calibration")


def gate_histogram(ax, stylia, T, t_lo, t_hi):
    ax.hist(T, bins=50, color=_article(stylia, "amber"))
    for t in (t_lo, t_hi):
        ax.axvline(t, color=_article(stylia, "silver"), linestyle="--")
    stylia.label(
        ax, xlabel="1-NN Tanimoto to the labelled set (T)", ylabel="Library molecules", title="Reach of the hard head"
    )


def _short(label, width=38):
    label = str(label)
    return label if len(label) <= width else label[: width - 1] + "…"

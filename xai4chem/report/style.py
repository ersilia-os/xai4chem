"""What an xai4chem figure looks like: the one place that decides, as in Olinda.

Figures are drawn with stylia, in its ``article`` style (the non-branded NPG palette with a neutral
foreground) and its ``print`` format, so a panel can be dropped into a paper; the Ersilia identity belongs to
the page around them. Plots ask for a *meaning* (``rgb(st, "model")``), never a colour, so a quantity keeps
its hue across the whole report. Every helper takes the already-imported stylia module from its caller.
"""

import numpy as np

# Screen resolution, not stylia's 600 dpi: the PNGs are embedded in a page read in a browser, where more
# only costs encoding time and bytes. The PDF next to each PNG stays vector.
REPORT_DPI = 200

# Figures are sized on a 3 cm grid: a footprint of (rows, cols) cells. Six cells span stylia's print width.
CELLS_PER_WIDTH = 6

# Semantic roles -> stylia ArticleColors names (by name, not by palette index, so a plot asking for a
# different number of colours cannot shift the others).
ROLES = {
    "model": "periwinkle",  # S, the surrogate, and the blended prediction
    "teacher": "cobalt",  # the teacher values S is judged against
    "hard": "orchid",  # the hard-label head H and its calibration onto S
    "tanimoto": "amber",  # T, the similarity to the labelled set, and the gate
    "raises": "crimson",  # features that push a prediction up
    "lowers": "cobalt",  # features that push it down
    "neutral": "silver",  # diagonals, chance lines, no-effect baselines
}


def import_stylia():
    """Import stylia in the report's style and format, with a headless matplotlib.

    stylia 1.0.1 runs ``shutil.rmtree(matplotlib.get_cachedir())`` at import. Whichever process imports it
    first deletes that directory, and the next import then fails on a path nobody asked about. Creating the
    directory first turns that into the no-op it was meant to be.
    """
    import os

    import matplotlib

    matplotlib.use("Agg")
    os.makedirs(matplotlib.get_cachedir(), exist_ok=True)
    import stylia

    stylia.set_style("article")
    stylia.set_format("print")
    return stylia


def rgb(st, role):
    """The matplotlib colour for a semantic role (see :data:`ROLES`)."""
    return st.ArticleColors().get(ROLES[role])


def hexcol(st, role):
    """The hex string for a role, for colour maps."""
    from matplotlib.colors import to_hex

    return to_hex(rgb(st, role))


def figure(st, cells=(2, 2)):
    """A one-panel figure of ``cells`` = ``(rows, cols)`` on the 3 cm grid; returns ``(fig, ax)``.

    stylia stamps placeholder axis labels on every axis it hands out, so the axis is taken once and
    cleared at once; the plot then sets its own labels.
    """
    rows, cols = cells
    fig, axs = st.create_figure(1, 1, width=cols / CELLS_PER_WIDTH, height=rows / CELLS_PER_WIDTH)
    ax = axs.next()
    ax.set_xlabel("")
    ax.set_ylabel("")
    return fig, ax


def reference(ax, st, kind, value=0.0, lo=0.0, hi=1.0):
    """A silver dashed baseline behind the data: ``diagonal`` (y = x), ``horizontal`` or ``vertical``."""
    style = {"linestyle": "--", "color": rgb(st, "neutral"), "linewidth": 1, "zorder": 0}
    if kind == "diagonal":
        ax.plot([lo, hi], [lo, hi], **style)
    elif kind == "horizontal":
        ax.axhline(value, **style)
    elif kind == "vertical":
        ax.axvline(value, **style)
    else:
        raise ValueError(f"unknown reference line {kind!r}")


def density(ax, st, x, y, role="teacher", label="Molecules"):
    """Hexbin of ``x`` against ``y`` shaded by count on a log scale, with a colourbar.

    A scatter overplots into a blob long before the structure shows; binning costs the same for ten
    thousand points as for a million. Counts are logarithmic because chemical data is peaked: on a linear
    ramp one bin takes the whole range and washes the others to white.
    """
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    cmap = LinearSegmentedColormap.from_list("xai4chem_density", ["#FFFFFF", hexcol(st, role)])
    hb = ax.hexbin(
        np.asarray(x, dtype=float).ravel(),
        np.asarray(y, dtype=float).ravel(),
        gridsize=45,
        cmap=cmap,
        mincnt=1,
        linewidths=0,
        norm=LogNorm(),
    )
    bar = ax.figure.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label(label)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2)
    return hb


def limits(ax, *values, pad=0.03):
    """Give both axes one shared, slightly padded range and return it.

    A plot with a y = x line is misleading unless both axes share a scale: matplotlib scales each on its
    own, so "above the line" stops meaning what it looks like.
    """
    flat = np.concatenate([np.asarray(v, dtype=float).ravel() for v in values])
    lo, hi = float(flat.min()), float(flat.max())
    margin = (hi - lo) * pad or 0.01
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    return lo - margin, hi + margin

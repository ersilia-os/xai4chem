"""The report's figures as small inline SVG charts, in the Ersilia palette.

Four forms cover the report: a density of teacher against surrogate, a dot plot of per-molecule effects, a
curve with a rug (the H → S calibration) and a histogram (the hard head's reach). Drawing them directly keeps
the report a single light file, with no plotting library to import or font cache to rebuild.
"""

import html

import numpy as np

# Ersilia palette (report/assets/ersilia.css)
BRAND = "#6C5CE7"
CRIMSON = "#E63946"
COBALT = "#457B9D"
AMBER = "#FCBF49"
ORCHID = "#B05CC8"
SILVER = "#A0A0A0"
INK = "#2C3E50"
MUTED = "#6B6675"
BORDER = "#E6E6EE"

FONT = 'font-family="Inter,system-ui,sans-serif" font-size="10"'


def _ticks(lo, hi, n=5):
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    step = 10 ** np.floor(np.log10(raw))
    step *= next(m for m in (1, 2, 5, 10) if raw <= m * step)
    start = np.ceil(lo / step) * step
    return [round(float(t), 10) for t in np.arange(start, hi + step * 1e-9, step)]


def _fmt(t):
    return f"{t:g}" if abs(t) < 1e4 else f"{t:.0e}"


class _Frame(object):
    """A plotting area with linear x/y scales, axes, ticks and labels."""

    def __init__(self, width, height, xlim, ylim, xlabel="", ylabel="", left=46, bottom=34, top=10, right=12):
        self.w, self.h = width, height
        self.x0, self.x1, self.y0, self.y1 = left, width - right, top, height - bottom
        self.xlim, self.ylim = self._pad(xlim), self._pad(ylim)
        self.xlabel, self.ylabel = xlabel, ylabel

    @staticmethod
    def _pad(lim):
        lo, hi = float(lim[0]), float(lim[1])
        if not np.isfinite(lo) or not np.isfinite(hi):
            return (0.0, 1.0)
        return (lo - 0.5, hi + 0.5) if hi - lo < 1e-12 else (lo, hi)

    def x(self, v):
        lo, hi = self.xlim
        return self.x0 + (np.asarray(v, dtype=float) - lo) / (hi - lo) * (self.x1 - self.x0)

    def y(self, v):
        lo, hi = self.ylim
        return self.y1 - (np.asarray(v, dtype=float) - lo) / (hi - lo) * (self.y1 - self.y0)

    def axes(self, yticks=True):
        parts = [
            f'<rect x="{self.x0}" y="{self.y0}" width="{self.x1 - self.x0}" height="{self.y1 - self.y0}" fill="none" stroke="{BORDER}"/>'
        ]
        for t in _ticks(*self.xlim):
            x = float(self.x(t))
            parts.append(f'<line x1="{x:.1f}" y1="{self.y1}" x2="{x:.1f}" y2="{self.y1 + 3}" stroke="{MUTED}"/>')
            parts.append(
                f'<text x="{x:.1f}" y="{self.y1 + 14}" text-anchor="middle" fill="{MUTED}" {FONT}>{_fmt(t)}</text>'
            )
        if yticks:
            for t in _ticks(*self.ylim):
                y = float(self.y(t))
                parts.append(f'<line x1="{self.x0 - 3}" y1="{y:.1f}" x2="{self.x0}" y2="{y:.1f}" stroke="{MUTED}"/>')
                parts.append(
                    f'<text x="{self.x0 - 5}" y="{y + 3:.1f}" text-anchor="end" fill="{MUTED}" {FONT}>{_fmt(t)}</text>'
                )
        if self.xlabel:
            parts.append(
                f'<text x="{(self.x0 + self.x1) / 2:.1f}" y="{self.h - 4}" text-anchor="middle" fill="{INK}" {FONT}>{html.escape(self.xlabel)}</text>'
            )
        if self.ylabel:
            cy = (self.y0 + self.y1) / 2
            parts.append(
                f'<text x="11" y="{cy:.1f}" text-anchor="middle" fill="{INK}" {FONT} transform="rotate(-90 11 {cy:.1f})">{html.escape(self.ylabel)}</text>'
            )
        return parts

    def svg(self, parts):
        body = "".join(parts)
        return f'<svg viewBox="0 0 {self.w} {self.h}" width="100%" xmlns="http://www.w3.org/2000/svg" role="img">{body}</svg>'


def density(x, y, xlabel, ylabel, bins=36):
    """Teacher against surrogate as a binned density (log-shaded), with the identity line."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    f = _Frame(340, 320, (lo, hi), (lo, hi), xlabel, ylabel)
    counts, xe, ye = np.histogram2d(x, y, bins=bins, range=[f.xlim, f.ylim])
    top = np.log1p(counts.max()) or 1.0
    parts = []
    for i, j in zip(*np.nonzero(counts)):
        opacity = 0.12 + 0.88 * np.log1p(counts[i, j]) / top
        xa, xb, ya, yb = f.x(xe[i]), f.x(xe[i + 1]), f.y(ye[j + 1]), f.y(ye[j])
        parts.append(
            f'<rect x="{xa:.1f}" y="{ya:.1f}" width="{xb - xa + 0.3:.1f}" height="{yb - ya + 0.3:.1f}" fill="{BRAND}" fill-opacity="{opacity:.2f}"/>'
        )
    parts.append(
        f'<line x1="{f.x(f.xlim[0]):.1f}" y1="{f.y(f.ylim[0]):.1f}" x2="{f.x(f.xlim[1]):.1f}" y2="{f.y(f.ylim[1]):.1f}" stroke="{SILVER}" stroke-dasharray="4 3"/>'
    )
    return f.svg(f.axes() + parts)


def effects(phi, values, labels, seed=0):
    """Per-molecule effects of the top features: one row per feature, one dot per molecule.

    Dots are coloured by the feature's value in that molecule (cobalt low, crimson high).
    """
    phi, values = np.asarray(phi, dtype=float), np.asarray(values, dtype=float)
    n_feat = phi.shape[1]
    rng = np.random.default_rng(seed)
    lim = float(np.abs(phi).max()) or 1.0
    row_h, left = 22, 190
    f = _Frame(
        560,
        40 + row_h * n_feat,
        (-lim, lim),
        (0, n_feat),
        "SHAP value (cobalt: low feature value, crimson: high)",
        "",
        left=left,
    )
    parts = [f'<line x1="{f.x(0):.1f}" y1="{f.y0}" x2="{f.x(0):.1f}" y2="{f.y1}" stroke="{SILVER}"/>']
    for k in range(n_feat):
        row = n_feat - 1 - k  # most important at the top
        cy = float(f.y(row + 0.5))
        label = str(labels[k])
        label = label if len(label) <= 34 else label[:33] + "…"
        parts.append(
            f'<text x="{left - 6}" y="{cy + 3:.1f}" text-anchor="end" fill="{INK}" {FONT}>{html.escape(label)}</text>'
        )
        v = values[:, k]
        span = v.max() - v.min()
        level = (v - v.min()) / span if span > 0 else np.full(len(v), 0.5)
        jitter = rng.uniform(-0.32, 0.32, len(v)) * row_h
        for p, lv, j in zip(phi[:, k], level, jitter):
            color, opacity = (
                (CRIMSON, 0.25 + 0.7 * (lv - 0.5) * 2) if lv >= 0.5 else (COBALT, 0.25 + 0.7 * (0.5 - lv) * 2)
            )
            parts.append(
                f'<circle cx="{float(f.x(p)):.1f}" cy="{cy + j:.1f}" r="2.1" fill="{color}" fill-opacity="{opacity:.2f}"/>'
            )
    return f.svg(f.axes(yticks=False) + parts)


def curve(x, y, rug, xlabel, ylabel):
    """A monotone curve (the isotonic H → S map), with a rug of the labelled molecules' positions."""
    x, y, rug = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(rug, dtype=float)
    xs = np.concatenate([x, rug]) if len(rug) else x
    f = _Frame(340, 300, (xs.min(), xs.max()), (y.min(), y.max()), xlabel, ylabel)
    points = " ".join(f"{float(f.x(a)):.1f},{float(f.y(b)):.1f}" for a, b in zip(x, y))
    parts = [f'<polyline points="{points}" fill="none" stroke="{ORCHID}" stroke-width="1.6"/>']
    for r in rug[:3000]:
        xr = float(f.x(r))
        parts.append(
            f'<line x1="{xr:.1f}" y1="{f.y1 - 6}" x2="{xr:.1f}" y2="{f.y1}" stroke="{SILVER}" stroke-opacity="0.5"/>'
        )
    return f.svg(f.axes() + parts)


def histogram(values, xlabel, ylabel, vlines=(), bins=40, lim=(0.0, 1.0)):
    """A histogram (the similarity of library molecules to the labelled set), with dashed reference lines."""
    values = np.asarray(values, dtype=float)
    counts, edges = np.histogram(values, bins=bins, range=lim)
    f = _Frame(560, 300, lim, (0, max(1, counts.max())), xlabel, ylabel)
    parts = []
    for c, a, b in zip(counts, edges[:-1], edges[1:]):
        if c:
            xa, xb, yt = float(f.x(a)), float(f.x(b)), float(f.y(c))
            parts.append(
                f'<rect x="{xa:.1f}" y="{yt:.1f}" width="{xb - xa - 0.6:.1f}" height="{f.y1 - yt:.1f}" fill="{AMBER}"/>'
            )
    for v in vlines:
        xv = float(f.x(v))
        parts.append(
            f'<line x1="{xv:.1f}" y1="{f.y0}" x2="{xv:.1f}" y2="{f.y1}" stroke="{SILVER}" stroke-dasharray="4 3"/>'
        )
    return f.svg(f.axes() + parts)

"""Target reweighting for S, ported from Olinda.

A teacher's outputs are often skewed (mostly near zero, a thin tail of actives). Squared error then spends
its effort on the dense bulk and fits the tail poorly, and the tail is usually what a chemist cares about.
So each column's target is weighted by inverse density, but only when its shape calls for it:

- a balanced target is left unweighted;
- an imbalanced but discrete or small one gets coarse equal-width bins;
- an imbalanced continuous one gets a Gaussian-smoothed density (Steininger et al., "Density-based
  weighting for imbalanced regression", Machine Learning 2021), which keeps rare tail values apart from
  the dense flank of a nearby mode.

Weights are at least 1 and at most ``max_weight``. They apply to training and to early stopping together;
the reported fidelity is always computed unweighted.
"""

import numpy as np

N_BINS = 20
CV_THRESHOLD = 0.5
MAX_WEIGHT = 10.0
N_GRID = 512
MIN_N_FOR_KDE = 1000
MIN_UNIQUE_FOR_KDE = 20
MODES = ("auto", "on", "off", "kde", "bins")


def _bin_edges(y, n_bins):
    """Equal-width edges over the 1st to 99th percentile, open at both ends; None for a near-constant y."""
    lo, hi = np.percentile(y, [1, 99])
    if hi - lo < 1e-12:
        return None
    edges = np.linspace(lo, hi, n_bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def detect_imbalance(y, n_bins=N_BINS, cv_threshold=CV_THRESHOLD):
    """Whether a target is imbalanced: the coefficient of variation of its bin counts, or many empty bins.

    Returns
    -------
    tuple
        ``(is_imbalanced, cv)``.
    """
    y = np.asarray(y, dtype=np.float64)
    edges = _bin_edges(y, n_bins)
    if edges is None:
        return False, 0.0
    counts = np.bincount(np.digitize(y, edges[1:-1]), minlength=n_bins).astype(np.float64)
    non_empty = counts[counts > 0]
    cv = float(non_empty.std() / non_empty.mean())
    empty_frac = float(np.sum(counts == 0)) / n_bins
    return bool(cv > cv_threshold or empty_frac > 0.3), cv


def bin_weights(y, n_bins=N_BINS, max_weight=MAX_WEIGHT):
    """Inverse-density weights over equal-width bins: ``(edges, weights)``, one weight per bin."""
    y = np.asarray(y, dtype=np.float64)
    edges = _bin_edges(y, n_bins)
    if edges is None:
        return np.array([-np.inf, np.inf]), np.array([1.0], dtype=np.float32)
    counts = np.bincount(np.digitize(y, edges[1:-1]), minlength=n_bins).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(counts > 0, len(y) / (n_bins * counts), 1.0)
    return edges, np.clip(weights, 1.0, max_weight).astype(np.float32)


def density_weights(y, alpha=1.0, max_weight=MAX_WEIGHT, n_grid=N_GRID):
    """Inverse weights from a Gaussian-smoothed density on a fine grid: ``(edges, weights)``.

    The bandwidth is a robust Silverman estimate, so outliers do not inflate it. ``alpha`` softens the
    weighting (0 is off, 1 is full inverse density).
    """
    y = np.asarray(y, dtype=np.float64)
    lo, hi = float(y.min()), float(y.max())
    if hi - lo < 1e-12:
        return np.array([-np.inf, np.inf]), np.array([1.0], dtype=np.float32)
    q75, q25 = np.percentile(y, [75, 25])
    iqr = float(q75 - q25)
    scale = min(float(y.std()), iqr / 1.349) if iqr > 0 else float(y.std())
    bandwidth = max(0.9 * scale * len(y) ** (-1 / 5), (hi - lo) / n_grid)

    edges = np.linspace(lo, hi, n_grid + 1)
    counts, _ = np.histogram(y, bins=edges)
    dx = (hi - lo) / n_grid
    half = max(int(3 * bandwidth / dx), 1)
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) * dx / bandwidth) ** 2)
    kernel /= kernel.sum()
    density = np.convolve(counts.astype(np.float64), kernel, mode="same")
    density = np.maximum(density, density.max() * 1e-6)  # an empty cell must not divide by zero

    at_y = density[np.clip(np.digitize(y, edges[1:-1]), 0, n_grid - 1)]
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.clip((at_y.mean() / density) ** float(alpha), 1.0, max_weight)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges, weights.astype(np.float32)


def apply_weights(y, edges, weights):
    """The weight of each value of ``y``, from the bin or grid cell it falls in."""
    return weights[np.clip(np.digitize(y, edges[1:-1]), 0, len(weights) - 1)]


def choose_strategy(y, mode="auto"):
    """The strategy for a target, ``none``, ``bins`` or ``kde``, and why.

    ``mode`` is ``auto`` (weight only an imbalanced target), ``on`` (weight regardless), ``off``, ``kde``
    or ``bins`` (force one).
    """
    if mode not in MODES:
        raise ValueError(f"Unknown weighting mode {mode!r}. Choose one of {', '.join(MODES)}.")
    y = np.asarray(y, dtype=np.float64)
    if mode == "off":
        return "none", "reweighting disabled"
    if mode in ("kde", "bins"):
        return mode, f"forced by mode={mode}"
    imbalanced, cv = detect_imbalance(y)
    continuous = len(y) >= MIN_N_FOR_KDE and np.unique(y).size >= MIN_UNIQUE_FOR_KDE
    if not imbalanced and mode == "auto":
        return "none", f"balanced target (cv={cv:.2f})"
    strategy = "kde" if continuous else "bins"
    reason = f"imbalanced target (cv={cv:.2f})" if imbalanced else f"forced on despite a balanced target (cv={cv:.2f})"
    return strategy, reason


def resolve_weights(y, mode="auto"):
    """Weights for a target: ``(edges, weights, info)``; ``edges`` and ``weights`` are None when unweighted."""
    strategy, reason = choose_strategy(y, mode)
    if strategy == "none":
        return None, None, {"strategy": "none", "reason": reason}
    edges, weights = bin_weights(y) if strategy == "bins" else density_weights(y)
    info = {
        "strategy": strategy,
        "reason": reason,
        "max_weight": MAX_WEIGHT,
        "weight_range": [float(weights.min()), float(weights.max())],
    }
    return edges, weights, info

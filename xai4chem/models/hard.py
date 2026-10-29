"""H, H_S and the gate: refining the surrogate with hard (experimental) labels, Olinda's logic.

1. **H** is trained on measurements with the same featurizer as S: a binary classifier for 0/1 labels,
   a regressor otherwise (or whatever ``task`` forces). Its output is taken in margin space, where SHAP values are additive. One
   stratified held-out split (20%) gives an honest metric and the number of trees; H is then refitted on
   every label.
2. **H_S** is H carried onto S's scale by an isotonic map, fitted on soft-labelled molecules where both H's
   score and the teacher's value exist. The direction is learned from the data.
3. **a_max**, the most weight H_S ever gets, is the R² of H_S against the teacher on molecules held out
   from the isotonic fit, capped at 0.66 and set to 0 below 0.1.
4. **The gate** is ``a = a_max · ramp(T)``, where T is the exact 1-NN Tanimoto (binary Morgan) to the
   labelled set, rising from 0 at T = 0.4 to ``a_max`` at T = 0.7.

The prediction is ``P = (1 − a)·S + a·H_S``, on S's scale.
"""

import numpy as np

from xai4chem.models.surrogate import dmatrix, n_threads, predict_margin, safe_spearman

# Olinda's constants (olinda/tanimoto.py): absolute Tanimoto knees, a ceiling that always keeps a third of
# the surrogate, and a dead zone below which the hard head is not mixed in at all
T_LO = 0.4
T_HI = 0.7
A_CEILING = 0.66
A_MIN = 0.1
MIN_HARD_ROWS = 30
MIN_PER_CLASS = 5
ISOTONIC_ROWS = 50_000
VAL_FRAC = 0.2
PARAMS = {"tree_method": "hist", "eta": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.5}


def blend_ceiling(alignment_r2):
    """Olinda's ``_blend_ceiling``: the R² of H_S against the teacher caps the blend."""
    r2 = float(alignment_r2)
    if not np.isfinite(r2):
        return 0.0
    ceiling = min(A_CEILING, max(0.0, r2))
    return float(ceiling) if ceiling >= A_MIN else 0.0


def ramp(T, t_lo=T_LO, t_hi=T_HI):
    return np.clip((np.asarray(T) - t_lo) / (t_hi - t_lo), 0.0, 1.0)


def apply_isotonic(h, iso):
    return np.interp(h, iso["x"], iso["y"])


TASKS = ("auto", "binary", "regression")


def detect_task(y):
    """Binary if every label is 0 or 1, else regression (Olinda's rule)."""
    return "binary" if np.isin(y, [0, 1]).all() else "regression"


def resolve_task(y, task="auto", name="the hard labels"):
    """The task for a hard column: detected, or forced with ``binary`` or ``regression``.

    Forcing ``binary`` on values other than 0 and 1 is refused rather than cast, since casting would floor
    every value below 1 to a negative.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}. Choose one of {', '.join(TASKS)}.")
    if task == "auto":
        return detect_task(y)
    if task == "binary" and not np.isin(y, [0, 1]).all():
        off = sorted(set(np.unique(y).tolist()) - {0.0, 1.0})[:3]
        raise ValueError(
            f"{name} was forced to --task binary but holds non-binary values (e.g. {off}). Casting them would "
            "floor everything below 1.0 to a negative. Threshold the column yourself, or drop --task."
        )
    return task


def enough_labels(y, task):
    if len(y) < MIN_HARD_ROWS:
        return False
    return task != "binary" or min((y == 0).sum(), (y == 1).sum()) >= MIN_PER_CLASS


def train_hard_model(X, y, task, sparse, seed=42):
    """Early-stop on one stratified held-out split, report the metric there, then refit on every label."""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    groups = [np.where(y == c)[0] for c in (0, 1)] if task == "binary" else [np.arange(len(y))]
    val = np.sort(np.concatenate([rng.permutation(g)[: max(1, int(len(g) * VAL_FRAC))] for g in groups]))
    trn = np.setdiff1d(np.arange(len(y)), val)
    objective = (
        {"objective": "binary:logistic", "eval_metric": "logloss"}
        if task == "binary"
        else {"objective": "reg:squarederror", "eval_metric": "rmse"}
    )
    params = {**PARAMS, **objective, "seed": seed, "min_child_weight": 1.0, "nthread": n_threads(len(y))}
    probe = xgb.train(
        params,
        dmatrix(X[trn], sparse, label=y[trn]),
        num_boost_round=2000,
        evals=[(dmatrix(X[val], sparse, label=y[val]), "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    rounds = int(probe.best_iteration) + 1
    pred = predict_margin(probe[:rounds], X[val], sparse)
    if task == "binary":
        heldout = {"auroc": float(roc_auc_score(y[val], pred))}
    else:
        heldout = {"spearman": safe_spearman(y[val], pred)}
    heldout["n"] = int(len(val))
    return xgb.train(params, dmatrix(X, sparse, label=y), num_boost_round=rounds), heldout


def metric_text(heldout):
    return f"AUROC {heldout['auroc']:.3f}" if "auroc" in heldout else f"ρ {heldout['spearman']:.3f}"


def fit_isotonic(h, z):
    """Fit the H → S map on the first half of (h, z), judge it on the second; return the map and its R²."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import r2_score

    half = len(h) // 2
    iso = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(h[:half], z[:half])
    iso_map = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist(), "increasing": bool(iso.increasing_)}
    return iso_map, float(r2_score(z[half:], apply_isotonic(h[half:], iso_map)))

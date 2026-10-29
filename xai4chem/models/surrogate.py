"""S: the surrogate, a gradient-boosted regressor of the teacher's soft labels.

Soft labels can be anything a teacher returns (scores, probabilities, concentrations), so, as in Olinda,
S is fitted to the raw values with a squared-error objective and explained on that same scale. A ``logit``
scale remains available explicitly, for teachers known to output probabilities. Training uses Olinda's tuned defaults:
histogram trees with 64 bins (lossless for counts clipped at 15), depth 8, row and column subsampling, a
``min_child_weight`` that grows with the data, and early stopping, after which the booster is trimmed to
its best iteration.

Count featurizers (Morgan, AccFG) are passed as CSR, whose implicit zeros XGBoost treats as missing: for a
count feature "absent" and "missing" are the same split, and training touches only the ~3% of entries
that are nonzero.
"""

import os

import numpy as np
from scipy.stats import spearmanr

from xai4chem.models.weights import apply_weights, resolve_weights

EPS = 1e-4
MIN_SOFT_ROWS = 100
VAL_FRAC = 0.1
N_ESTIMATORS = 2000
EARLY_STOPPING = 50
PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "max_bin": 64,
    "max_depth": 8,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
}


def n_threads(n_rows):
    """XGBoost threads: all cores for big data, few for small data, where thread overhead dominates."""
    return int(max(1, min(os.cpu_count() or 1, n_rows // 20000 + 1)))


def min_child_weight(n_rows):
    """Olinda uses 200 at ~1.3M rows; scale it down so small datasets can still grow trees."""
    return float(max(1.0, min(200.0, n_rows / 6500)))


# --- scales -----------------------------------------------------------------------------------------


def choose_transform(y, mode="auto"):
    """The scale S learns on: ``auto`` is the raw values, whatever the teacher returns, as in Olinda."""
    return "identity" if mode == "auto" else mode


def forward(y, transform):
    if transform == "logit":
        y = np.clip(y, EPS, 1 - EPS)
        return np.log(y / (1 - y))
    return y


def inverse(z, transform):
    if transform == "logit":
        return 1 / (1 + np.exp(-z))
    return z


# --- matrices and boosters --------------------------------------------------------------------------


def to_model_input(X, sparse):
    """What a booster expects: CSR (zeros = missing) for count featurizers, dense float32 otherwise."""
    import scipy.sparse as sp

    if sparse:
        return X if sp.issparse(X) else sp.csr_matrix(np.asarray(X, dtype=np.float32))
    return X.toarray() if sp.issparse(X) else np.asarray(X, dtype=np.float32)


def dmatrix(X, sparse, label=None):
    import xgboost as xgb

    return xgb.DMatrix(to_model_input(X, sparse), label=label)


def predict_margin(booster, X, sparse):
    return booster.predict(dmatrix(X, sparse), output_margin=True)


def contributions(booster, X, sparse):
    """Exact TreeSHAP values in margin space; the last column is the base value."""
    return booster.predict(dmatrix(X, sparse), pred_contribs=True)


def train_booster(X, y, sparse, seed=42, n_estimators=N_ESTIMATORS, val_frac=VAL_FRAC, on_round=None, reweight="auto"):
    """Fit with early stopping on a random validation split.

    Returns the trimmed booster, the validation rows, the parameters and the reweighting that was used.
    The weights (see :mod:`xai4chem.models.weights`) come from the training rows and apply to training and
    early stopping together. ``on_round(round, total, rmse, best)`` is called after every boosting round.
    """
    import xgboost as xgb

    idx = np.random.default_rng(seed).permutation(len(y))
    n_val = max(1, int(round(len(y) * val_frac)))
    trn, val = np.sort(idx[n_val:]), np.sort(idx[:n_val])
    Xm = to_model_input(X, sparse)
    edges, weights, reweight_info = resolve_weights(y[trn], reweight)
    w_trn = w_val = None
    if weights is not None:
        w_trn, w_val = apply_weights(y[trn], edges, weights), apply_weights(y[val], edges, weights)
    dtrain = xgb.QuantileDMatrix(Xm[trn], label=y[trn], weight=w_trn, max_bin=PARAMS["max_bin"])
    # A plain DMatrix on purpose: XGBoost 2.0 does not cache predictions for a QuantileDMatrix eval set,
    # so each round would re-predict the validation split with every tree (20x slower, measured)
    dval = xgb.DMatrix(Xm[val], label=y[val], weight=w_val)
    params = {**PARAMS, "min_child_weight": min_child_weight(len(trn)), "nthread": n_threads(len(trn)), "seed": seed}
    callbacks = [_RoundCallback(on_round, n_estimators)] if on_round is not None else []
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=False,
        callbacks=callbacks,
    )
    return booster[: int(booster.best_iteration) + 1], val, params, reweight_info


def _RoundCallback(on_round, total):
    import xgboost as xgb

    class RoundCallback(xgb.callback.TrainingCallback):
        def __init__(self):
            self.best = float("inf")

        def after_iteration(self, model, epoch, evals_log):
            rmse = evals_log["val"]["rmse"][-1]
            self.best = min(self.best, rmse)
            on_round(epoch + 1, total, rmse, self.best)
            return False

    return RoundCallback()


# --- metrics ----------------------------------------------------------------------------------------


def safe_spearman(a, b):
    """Spearman's ρ, or NaN (quietly) when undefined: fewer than 3 points or a constant input."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def fidelity(y_true, y_pred):
    """R², Spearman ρ and RMSE of predictions against the teacher, and the RMSE on its top decile.

    The top-decile RMSE (rows where the teacher is at or above its 90th percentile, as in Olinda) shows how
    well the sparse high-value tail is reproduced, which the global R² can hide.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    tail = y_true >= np.quantile(y_true, 0.9) if len(y_true) else np.zeros(0, dtype=bool)
    return {
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "spearman": safe_spearman(y_true, y_pred),
        "rmse": float(np.sqrt(ss_res / max(len(y_true), 1))),
        "top_decile_rmse": float(np.sqrt(((y_true[tail] - y_pred[tail]) ** 2).mean())) if tail.any() else float("nan"),
        "n": int(len(y_true)),
    }


def fit_column(
    X,
    y,
    sparse,
    transform="auto",
    seed=42,
    n_estimators=N_ESTIMATORS,
    val_frac=VAL_FRAC,
    on_round=None,
    reweight="auto",
):
    """Learn S for one teacher column.

    Returns ``None`` when the column has fewer than :data:`MIN_SOFT_ROWS` finite values, else a dict with
    the booster, the scale, the rows used (``mask``), their model-scale targets (``z``), the training
    metadata and the held-out (teacher, surrogate) pairs used for the fidelity figure.
    """
    mask = np.isfinite(y)
    if mask.sum() < MIN_SOFT_ROWS:
        return None
    t = choose_transform(y, transform)
    z = forward(y[mask], t)
    Xm = X[np.where(mask)[0]] if not mask.all() else X
    booster, val, params, reweight_info = train_booster(
        Xm, z, sparse, seed=seed, n_estimators=n_estimators, val_frac=val_frac, on_round=on_round, reweight=reweight
    )
    pred = predict_margin(booster, Xm[val], sparse)
    meta = {
        "transform": t,
        "n_train": int(mask.sum() - len(val)),
        "n_val": int(len(val)),
        "n_trees": int(booster.num_boosted_rounds()),
        "fidelity": fidelity(z[val], pred),
        "fidelity_teacher_scale": fidelity(inverse(z[val], t), inverse(pred, t)),
        "constant": bool(np.ptp(z) == 0),
        "seed": seed,
        "params": params,
        "reweight": reweight_info,
    }
    return {"booster": booster, "transform": t, "mask": mask, "z": z, "meta": meta, "heldout": (z[val], pred)}

"""Optional hyperparameter search for S (``--tune``), in the spirit of Olinda's ``olinda tune``.

A short Optuna search on a subsample of the soft labels picks the learning rate, depth and row/column
subsampling. ``min_child_weight`` is not searched: it is scaled to the size of the full training set (see
:func:`xai4chem.soft.min_child_weight`), so a value tuned on a subsample would not transfer. The search is
deliberately small (20 trials on 100k rows by default): the defaults are already Olinda's tuned ones, and
the point is to adapt to an unusual teacher, not to squeeze the last 0.001 of R².
"""

import time

import numpy as np

from xai4chem.console import live_status, status_line

TUNE_ROWS = 100_000
TRIALS = 20
ROUNDS = 1000


def tune(X, z, sparse, seed=42, trials=TRIALS, rows=TUNE_ROWS):
    """Return the best parameters found, and the validation RMSE they reached."""
    import optuna
    import xgboost as xgb

    from xai4chem.soft import PARAMS, min_child_weight, n_threads, to_model_input

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(z))[: min(rows, len(z))]
    n_val = max(1, len(idx) // 5)
    val, trn = np.sort(idx[:n_val]), np.sort(idx[n_val:])
    Xm = to_model_input(X, sparse)
    dtrain = xgb.QuantileDMatrix(Xm[trn], label=z[trn], max_bin=PARAMS["max_bin"])
    dval = xgb.DMatrix(Xm[val], label=z[val])
    base = {**PARAMS, "min_child_weight": min_child_weight(len(trn)), "nthread": n_threads(len(trn)), "seed": seed}
    started = time.time()
    best = {"rmse": float("inf")}

    def objective(trial):
        params = {
            "eta": trial.suggest_float("eta", 0.03, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }
        booster = xgb.train(
            {**base, **params},
            dtrain,
            num_boost_round=ROUNDS,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        rmse = float(booster.best_score)
        if rmse < best["rmse"]:
            best.update(rmse=rmse, params=params)
        return rmse

    with live_status() as update:

        def report(study, trial):
            update(
                status_line(
                    "tuning",
                    frac=(trial.number + 1) / trials,
                    pairs=[("trial", f"{trial.number + 1}/{trials}"), ("best RMSE", f"{best['rmse']:.4f}")],
                    tail=f"{len(trn):,} rows · {time.time() - started:.0f}s",
                ),
                detail={"trial": f"{trial.number + 1}/{trials}", "best RMSE": f"{best['rmse']:.4f}"},
            )

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=trials, callbacks=[report])
    return best["params"], best["rmse"]

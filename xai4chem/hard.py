"""H, H_S and the gate: refining the surrogate with hard (experimental) labels.

This follows Olinda's logic:

1. **H** is trained on measurements with the same featurizer as S: a binary classifier for 0/1 labels,
   a regressor otherwise. Its output is taken in margin space (log-odds for a classifier), where its
   SHAP values are additive. Its quality is reported from 5-fold cross-validation, and the number of
   trees is the CV average, so the final H is fitted on every label.
2. **H_S** is H carried onto S's scale by an isotonic map. The map is fitted on reference-library rows,
   where both H's score and the teacher's soft label exist, and the direction is learned from the data.
3. **a_max**, the most weight H_S ever gets, is earned from the R² of H_S against the teacher. Unlike
   Olinda, that R² is measured on library rows held out from the isotonic fit.
4. **The gate** is ``a = a_max · ramp(T)``, where T is the *exact* 1-NN Tanimoto (binary Morgan) to the
   labelled set. The labelled fingerprints are kept in the model, so reports can also name the nearest
   labelled analogue.

The prediction is ``P = (1 − a)·S + a·H_S``, on S's (transformed) scale.
"""

import time

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

from xai4chem.console import (
    STEP_COLORS,
    LiveTable,
    echo,
    elapsed,
    rule,
    set_active_color,
    summary_panel,
)
from xai4chem.console import path as cpath
from xai4chem.featurizers import featurizer_from_config
from xai4chem.io import match_hard_columns, read_labels
from xai4chem.library import ReferenceLibrary
from xai4chem.run import (
    H_NAME,
    H_TO_S_NAME,
    HARD_META_NAME,
    LABELLED_FP_NAME,
    LABELLED_NAME,
    TARGETS_NAME,
    hard_dir,
    read_manifest,
    write_json,
)
from xai4chem.similarity import binarize, fingerprints, nearest
from xai4chem.soft import contributions, dmatrix, load_features, n_threads, predict_margin

# Same constants as Olinda (olinda/tanimoto.py): absolute Tanimoto knees, a ceiling that always keeps a
# third of the surrogate, and a dead zone below which the hard head is not mixed in at all.
T_LO = 0.4
T_HI = 0.7
A_CEILING = 0.66
A_MIN = 0.1
MIN_HARD_ROWS = 30
MIN_PER_CLASS = 5
ISOTONIC_ROWS = 200_000
CV_FOLDS = 5
PARAMS = {"tree_method": "hist", "eta": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.5}


def blend_ceiling(alignment_r2):
    """Olinda's ``_blend_ceiling``: R² (not a correlation) of H_S against the teacher caps the blend."""
    r2 = float(alignment_r2)
    if not np.isfinite(r2):
        return 0.0
    ceiling = min(A_CEILING, max(0.0, r2))
    return float(ceiling) if ceiling >= A_MIN else 0.0


def ramp(T, t_lo=T_LO, t_hi=T_HI):
    return np.clip((np.asarray(T) - t_lo) / (t_hi - t_lo), 0.0, 1.0)


def apply_isotonic(h, iso):
    return np.interp(h, iso["x"], iso["y"])


def detect_task(y):
    return "binary" if np.isin(y, [0, 1]).all() else "continuous"


def _objective(task):
    if task == "binary":
        return {"objective": "binary:logistic", "eval_metric": "logloss"}
    return {"objective": "reg:squarederror", "eval_metric": "rmse"}


def _folds(y, task, seed, k=CV_FOLDS):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    groups = [np.where(y == c)[0] for c in (0, 1)] if task == "binary" else [np.arange(len(y))]
    for g in groups:
        g = rng.permutation(g)
        fold[g] = np.arange(len(g)) % k
    return fold


def train_hard_model(X, y, task, sparse, seed):
    """5-fold CV for an honest metric and the tree count, then one fit on every label."""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    params = {**PARAMS, **_objective(task), "seed": seed, "min_child_weight": 1.0, "nthread": n_threads(len(y))}
    fold = _folds(y, task, seed)
    oof = np.zeros(len(y))
    rounds = []
    for k in range(CV_FOLDS):
        trn, val = np.where(fold != k)[0], np.where(fold == k)[0]
        booster = xgb.train(
            params,
            dmatrix(X[trn], sparse, label=y[trn]),
            num_boost_round=2000,
            evals=[(dmatrix(X[val], sparse, label=y[val]), "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        rounds.append(booster.best_iteration + 1)
        oof[val] = predict_margin(booster[: booster.best_iteration + 1], X[val], sparse)
    if task == "binary":
        cv = {"auroc": float(roc_auc_score(y, oof))}
    else:
        cv = {
            "spearman": float(spearmanr(y, oof).statistic),
            "r2": float(1 - ((y - oof) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        }
    cv["folds"] = CV_FOLDS
    n_rounds = int(np.mean(rounds))
    booster = xgb.train(params, dmatrix(X, sparse, label=y), num_boost_round=n_rounds)
    return booster, cv


def _cv_text(cv):
    return f"AUROC {cv['auroc']:.3f}" if "auroc" in cv else f"ρ {cv['spearman']:.3f}"


def learn_hard(hard_labels, model_dir, library_dir=None, seed=42, smiles_col=None, label_cols=None):
    started = time.time()
    color = STEP_COLORS["learn-hard"]
    set_active_color(color)
    manifest = read_manifest(model_dir)
    featurizer = featurizer_from_config(manifest["featurizer"])
    library = ReferenceLibrary(library_dir or manifest["library"]["dir"])
    smiles, Y = read_labels(hard_labels, smiles_col=smiles_col, label_cols=label_cols)
    by_name = {c["name"]: c for c in manifest["columns"]}
    mapping = match_hard_columns(list(by_name), list(Y.columns), strict=False)
    ignored = [c for c in Y.columns if c not in mapping]
    if not mapping:
        raise ValueError(f"No column of {hard_labels} matches a soft-label column ({list(by_name)}).")
    rule("xai4chem · learn-hard", style=color, right=f"{len(mapping)} column(s) · {len(smiles):,} labelled molecules")
    if ignored:
        echo(f"ignoring columns that match no soft-label column: {', '.join(map(str, ignored))}", "warning")
    for hard, soft in mapping.items():
        if hard != soft:
            echo(f"hard column [bold]{hard}[/] refines soft column [bold]{soft}[/]", "info")

    echo(f"featurizing {len(smiles):,} labelled molecules ({featurizer.name})", "run")
    X = featurizer.transform_raw(smiles)
    fps = binarize(X) if featurizer.name == "morgan" else fingerprints(smiles)

    with h5py.File(f"{model_dir}/{TARGETS_NAME}", "r") as f:
        target_rows = f["rows"][:]
        Z = f["Z"][:]
        extra = {}
        if "extra_pos" in f:
            extra = dict(
                zip(
                    f["extra_pos"][:].tolist(),
                    [s.decode() if isinstance(s, bytes) else s for s in f["extra_smiles"][:]],
                )
            )
    col_index = {c["id"]: j for j, c in enumerate(manifest["columns"])}

    panel = library.panel()
    panel_fps = library.csr("morgan", panel, verb="reading panel fingerprints") if library.has("morgan") else None

    results = []
    with LiveTable(
        list(mapping),
        title="Learning H",
        fields=["Task", "Labels", "CV", "R² H_S", "a_max", "Reach", "Time"],
        item_label="Column",
        running_verb="fitting",
        color=color,
    ) as table:
        for hard_name, soft_name in mapping.items():
            table.start(hard_name)
            col_started = time.time()
            col = by_name[soft_name]
            stale = hard_dir(model_dir, col["id"])
            if stale.exists():
                # A head from an earlier learn-hard must not survive, least of all when this run decides
                # the column should stay soft-only
                import shutil

                shutil.rmtree(stale)
            y = Y[hard_name].to_numpy(dtype=np.float64)
            mask = np.isfinite(y)
            y = y[mask]
            idx = np.where(mask)[0]
            Xh, fh, smi = X[idx], fps[idx], [smiles[i] for i in idx]
            task = detect_task(y)
            table.update(hard_name, Task=task, Labels=f"{len(y):,}")
            if len(y) < MIN_HARD_ROWS or (task == "binary" and min((y == 0).sum(), (y == 1).sum()) < MIN_PER_CLASS):
                table.finish(hard_name, ok=False, CV="too few", Time=elapsed(time.time() - col_started))
                echo(f"[{hard_name}] too few hard labels ({len(y)}); the column stays soft-only", "warning")
                continue
            booster, cv = train_hard_model(Xh, y, task, featurizer.sparse, seed)
            table.update(hard_name, CV=_cv_text(cv))

            # Carry H onto S's scale over soft-labelled molecules, half to fit the map, half to judge it
            z = Z[:, col_index[col["id"]]]
            ok = np.where(np.isfinite(z) & ((target_rows >= 0) | np.isin(np.arange(len(z)), list(extra))))[0]
            if len(ok) < 20:
                raise ValueError(
                    f"[{hard_name}] only {len(ok)} soft-labelled molecules: the isotonic map needs the teacher's "
                    "scores (re-run learn-soft with this version of xai4chem if the model predates it)."
                )
            ok = np.random.default_rng(seed).permutation(ok)[:ISOTONIC_ROWS]
            sub_rows = target_rows[ok]
            sub_smiles = [library.smiles[r] if r >= 0 else extra[i] for i, r in zip(ok, sub_rows)]
            L = load_features(sub_smiles, sub_rows, library, featurizer)
            h_lib = predict_margin(booster, L, featurizer.sparse)
            half = len(ok) // 2
            from sklearn.isotonic import IsotonicRegression
            from sklearn.metrics import r2_score

            iso = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(h_lib[:half], z[ok[:half]])
            iso_map = {
                "x": iso.X_thresholds_.tolist(),
                "y": iso.y_thresholds_.tolist(),
                "increasing": bool(iso.increasing_),
            }
            alignment_r2 = float(r2_score(z[ok[half:]], apply_isotonic(h_lib[half:], iso_map)))
            a_max = blend_ceiling(alignment_r2)

            reach = {}
            if panel_fps is not None:
                T_panel, _ = nearest(panel_fps, fh)
                reach = {
                    "frac_panel_above_t_lo": float((T_panel >= T_LO).mean()),
                    "frac_panel_above_t_hi": float((T_panel >= T_HI).mean()),
                    "mean_a_panel": float((a_max * ramp(T_panel)).mean()),
                }

            hdir = hard_dir(model_dir, col["id"])
            hdir.mkdir(parents=True, exist_ok=True)
            booster.save_model(str(hdir / H_NAME))
            write_json(hdir / H_TO_S_NAME, iso_map)
            pd.DataFrame({"smiles": smi, "label": y}).to_csv(hdir / LABELLED_NAME, index=False)
            sp.save_npz(hdir / LABELLED_FP_NAME, sp.csr_matrix(fh))
            base_h = float(contributions(booster, Xh[:1], featurizer.sparse)[0, -1])
            meta = {
                "name": soft_name,
                "hard_column": hard_name,
                "task": task,
                "n_labels": int(len(y)),
                "n_positive": int((y == 1).sum()) if task == "binary" else None,
                "cv": cv,
                "n_trees": int(booster.num_boosted_rounds()),
                "base_h": base_h,
                "isotonic_direction": "increasing" if iso.increasing_ else "decreasing",
                "alignment_r2": alignment_r2,
                "a_max": a_max,
                "t_lo": T_LO,
                "t_hi": T_HI,
                "a_ceiling": A_CEILING,
                "seed": seed,
                **reach,
            }
            write_json(hdir / HARD_META_NAME, meta)
            results.append(meta)
            table.finish(
                hard_name,
                **{
                    "R² H_S": f"{alignment_r2:.3f}",
                    "a_max": f"{a_max:.2f}",
                    "Reach": f"{reach['frac_panel_above_t_lo']:.1%}" if reach else "—",
                    "Time": elapsed(time.time() - col_started),
                },
            )
            if a_max == 0.0:
                echo(
                    f"[{hard_name}] H_S does not reproduce the teacher's scale (R² {alignment_r2:.2f}); "
                    "P stays equal to S, but H's explanations are still reported",
                    "warning",
                )

    from xai4chem.model import write_golden

    write_golden(model_dir)
    rows_out = []
    for m in results:
        line = f"{m['task']} · {m['n_labels']:,} labels · CV {_cv_text(m['cv'])} · a_max {m['a_max']:.2f}"
        if "frac_panel_above_t_lo" in m:
            line += f" · {m['frac_panel_above_t_lo']:.1%} of library within T≥{T_LO}"
        rows_out.append((m["name"], line))
    rows_out += [("Model", f"[dim]{cpath(model_dir)}[/]"), ("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]")]
    summary_panel("xai4chem · learn-hard", rows_out, border_style=color, icon="✓")

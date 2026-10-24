"""S: the surrogate, learned from the teacher's soft labels over the reference library.

Teacher outputs that look like probabilities are moved to logit space before fitting. S, its SHAP values
and the blend with the hard head all live in that space, where contributions are additive; predictions
are mapped back to the teacher's scale only at the end.

Speed comes from three choices:

- **Sparse input.** Morgan and AccFG counts are ~97% zeros. They are handed to XGBoost as CSR, whose
  implicit zeros are *missing*: for a count feature "absent" and "missing" are the same split, and
  training touches only the nonzeros. Dense featurizers (datamol) go in as they are.
- **Olinda's tuned defaults**: histogram trees with 64 bins (lossless for counts clipped at 15), depth
  8, row and column subsampling, and a ``min_child_weight`` that grows with the data.
- **Early stopping**, after which the booster is trimmed to its best iteration, so predicting and
  explaining never evaluate trees that did not help.
"""

import time
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr

from xai4chem import __version__
from xai4chem.console import (
    STEP_COLORS,
    LiveTable,
    console,
    echo,
    elapsed,
    live_status,
    rule,
    set_active_color,
    status_line,
    summary_panel,
    sweep_progress,
)
from xai4chem.console import path as cpath
from xai4chem.io import read_labels
from xai4chem.library import BIT_DICTIONARY_NAME as LIB_BIT_DICTIONARY_NAME
from xai4chem.library import ReferenceLibrary
from xai4chem.log import logger
from xai4chem.run import (
    BIT_DICTIONARY_NAME,
    FIDELITY_NAME,
    S_NAME,
    SOFT_META_NAME,
    TARGETS_NAME,
    column_dir,
    column_id,
    write_json,
    write_manifest,
)

EPS = 1e-4
MIN_SOFT_ROWS = 100
VAL_FRAC = 0.1
STABILITY_ROWS = 3000
FIDELITY_POINTS = 50000
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
    """Threads for XGBoost: all cores for big data, few for small data, where thread overhead dominates."""
    import os

    return int(max(1, min(os.cpu_count() or 1, n_rows // 20000 + 1)))


def min_child_weight(n_rows):
    """Olinda uses 200 at ~1.3M rows; scale it down so small datasets can still grow trees."""
    return float(max(1.0, min(200.0, n_rows / 6500)))


# --- scales -----------------------------------------------------------------------------------------


def choose_transform(y, mode="auto"):
    if mode != "auto":
        return mode
    y = y[np.isfinite(y)]
    in_unit = y.min() >= 0 and y.max() <= 1
    binary = np.isin(y, [0, 1]).all()
    return "logit" if in_unit and not binary else "identity"


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
    """What a booster expects: CSR (zeros = missing) for sparse featurizers, dense float32 otherwise."""
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
    """Exact TreeSHAP values (last column is the base value), in margin space."""
    return booster.predict(dmatrix(X, sparse), pred_contribs=True)


def _split(n, val_frac, seed):
    idx = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])


def train_booster(X, y, sparse, seed, n_estimators=2000, val_frac=VAL_FRAC, params=None, progress=None):
    """Fit with early stopping on a random validation split; return the trimmed booster and the split."""
    import xgboost as xgb

    trn, val = _split(len(y), val_frac, seed)
    Xm = to_model_input(X, sparse)
    dtrain = xgb.QuantileDMatrix(Xm[trn], label=y[trn], max_bin=PARAMS["max_bin"])
    # A plain DMatrix on purpose: XGBoost 2.0 does not cache predictions for a QuantileDMatrix eval set,
    # so every round re-predicted the whole validation split with every tree (20x slower, measured)
    dval = xgb.DMatrix(Xm[val], label=y[val])
    full = {
        **PARAMS,
        "min_child_weight": min_child_weight(len(trn)),
        "nthread": n_threads(len(trn)),
        "seed": seed,
        **(params or {}),
    }
    logger.debug(f"boosting {len(trn):,} train / {len(val):,} validation rows · params {full}")
    callbacks = [_progress_callback(progress, n_estimators)] if progress is not None else []
    booster = xgb.train(
        full,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=False,
        callbacks=callbacks,
    )
    best = int(booster.best_iteration)
    trimmed = booster[: best + 1]
    return trimmed, trn, val


def _progress_callback(update, total):
    """A TrainingCallback that feeds round and validation RMSE to a live status line (~4 Hz)."""
    import xgboost as xgb

    class ProgressCallback(xgb.callback.TrainingCallback):
        def __init__(self):
            self.best = float("inf")
            self.started = time.time()
            self.last = 0.0
            self.last_log = time.time()

        def after_iteration(self, model, epoch, evals_log):
            rmse = evals_log["val"]["rmse"][-1]
            self.best = min(self.best, rmse)
            now = time.time()
            if now - self.last > 0.25:
                self.last = now
                rate = (epoch + 1) / max(now - self.started, 1e-6)
                update(
                    status_line(
                        "boosting",
                        frac=(epoch + 1) / total,
                        pairs=[("round", f"{epoch + 1}/{total}"), ("val RMSE", f"{rmse:.4f}")],
                        tail=f"best {self.best:.4f} · {rate:.0f} rounds/s",
                    ),
                    detail={"round": f"{epoch + 1:,}", "val RMSE": f"{rmse:.4f}", "best": f"{self.best:.4f}"},
                )
            if not console.is_terminal and now - self.last_log > 30:
                # Piped output has no live line; a milestone every 30 s shows a long fit is moving
                self.last_log = now
                echo(f"boosting · round {epoch + 1:,} · val RMSE {rmse:.4f} · best {self.best:.4f}", "info", sub=True)
            return False

    return ProgressCallback()


def fidelity(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "pearson": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan"),
        "spearman": float(spearmanr(y_true, y_pred).statistic) if len(y_true) > 1 else float("nan"),
        "rmse": float(np.sqrt(ss_res / max(len(y_true), 1))),
        "n": int(len(y_true)),
    }


# --- alignment --------------------------------------------------------------------------------------


def align_to_library(smiles, library):
    """Each molecule's library row, or -1. Fast path when the file is in library order (Olinda's layout)."""
    lib = library.smiles
    if len(smiles) == len(lib) and smiles[0] == lib[0] and smiles[-1] == lib[-1] and smiles == lib:
        return np.arange(len(smiles))
    index = library.index
    rows = np.array([index.get(s, -1) for s in smiles])
    missing = np.where(rows < 0)[0]
    if len(missing):
        # Different writers spell the same molecule differently; retry the misses as RDKit canonical SMILES
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        for i in missing:
            mol = Chem.MolFromSmiles(smiles[i])
            if mol is not None:
                rows[i] = index.get(Chem.MolToSmiles(mol), -1)
    return rows


def load_features(smiles, rows, library, featurizer, n_jobs=None):
    """Feature rows for ``smiles``: from the library where possible, featurized otherwise."""
    import scipy.sparse as sp

    found = rows >= 0
    missing = [s for s, f in zip(smiles, found) if not f]
    if featurizer.sparse:
        X_lib = library.csr(featurizer.name, rows[found], verb=f"reading {featurizer.name} features")
    else:
        X_lib = library.matrix(featurizer.name, rows[found]).astype(np.float32)
    if not missing:
        return X_lib
    from xai4chem.library import _featurize_chunk, _imap, default_n_jobs

    n_jobs = n_jobs or default_n_jobs()
    echo(f"featurizing {len(missing):,} molecules that are not in the library · {n_jobs} workers", "run")
    chunks = [missing[i : i + 5000] for i in range(0, len(missing), 5000)]
    cfg = featurizer.to_config()
    parts, done = [], 0
    with sweep_progress("featurizing", len(missing)) as tick:
        for block in _imap(_featurize_chunk, ((cfg, c) for c in chunks), n_jobs if len(missing) > 20000 else 1):
            parts.append(sp.csr_matrix(block, dtype=np.float32) if featurizer.sparse else block)
            done += block.shape[0]
            tick(done)
    X_new = sp.vstack(parts, format="csr") if featurizer.sparse else np.vstack(parts)
    order = np.concatenate([np.where(found)[0], np.where(~found)[0]])
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(len(order))
    if featurizer.sparse:
        return sp.vstack([X_lib, X_new], format="csr")[inverse_order]
    return np.vstack([X_lib, X_new.astype(np.float32)])[inverse_order]


# --- learn-soft -------------------------------------------------------------------------------------


def learn_soft(
    soft_labels,
    model_dir,
    featurizer_name="morgan",
    library_dir=None,
    transform="auto",
    max_rows=None,
    n_estimators=2000,
    val_frac=VAL_FRAC,
    stability=False,
    seed=42,
    smiles_col=None,
    label_cols=None,
    params=None,
):
    started = time.time()
    color = STEP_COLORS["learn-soft"]
    set_active_color(color)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    stale = model_dir / "columns"
    if stale.exists():
        # learn-soft starts a model afresh; column directories of an earlier run would otherwise linger
        import shutil

        shutil.rmtree(stale)
    library = ReferenceLibrary(library_dir)
    featurizer = library.featurizer(featurizer_name)
    smiles, Y = read_labels(soft_labels, smiles_col=smiles_col, label_cols=label_cols)
    rule(
        "xai4chem · learn-soft",
        style=color,
        right=f"{Y.shape[1]} column(s) · {featurizer_name} · library {library.name}",
    )
    echo(f"soft labels · [bold]{len(smiles):,}[/] molecules · columns {', '.join(map(str, Y.columns))}", "run")
    if max_rows is not None and len(smiles) > max_rows:
        keep = np.sort(np.random.default_rng(seed).choice(len(smiles), size=max_rows, replace=False))
        smiles = [smiles[i] for i in keep]
        Y = Y.iloc[keep].reset_index(drop=True)
        echo(f"subsampled to {max_rows:,} rows (--max-rows)", "info", sub=True)
    rows = align_to_library(smiles, library)
    n_found = int((rows >= 0).sum())
    echo(f"{n_found:,}/{len(smiles):,} molecules found in the reference library", "info", sub=True)
    X = load_features(smiles, rows, library, featurizer)

    columns, results = [], []
    Z = np.full(Y.shape, np.nan, dtype=np.float32)
    with LiveTable(
        [str(c) for c in Y.columns],
        title="Learning S",
        fields=["Scale", "Rows", "R²", "ρ", "Trees", "Time"],
        item_label="Column",
        running_verb="boosting",
        color=color,
    ) as table:
        for j, name in enumerate(Y.columns):
            name = str(name)
            col_id = column_id(j)
            table.start(name)
            col_started = time.time()
            y = Y.iloc[:, j].to_numpy(dtype=np.float64)
            mask = np.isfinite(y)
            if mask.sum() < MIN_SOFT_ROWS:
                table.finish(name, ok=False, Rows=f"{int(mask.sum()):,}", Time="skipped")
                echo(f"[{name}] only {int(mask.sum())} finite soft labels; column skipped", "warning")
                continue
            t = choose_transform(y, transform)
            z = forward(y[mask], t)
            Z[mask, j] = z
            Xm = X[np.where(mask)[0]] if not mask.all() else X
            table.update(name, Scale=t, Rows=f"{int(mask.sum()):,}")
            with live_status() as update:
                booster, _, val = train_booster(
                    Xm,
                    z,
                    featurizer.sparse,
                    seed,
                    n_estimators=n_estimators,
                    val_frac=val_frac,
                    params=params,
                    progress=update,
                )
            pred = predict_margin(booster, Xm[val], featurizer.sparse)
            fid = fidelity(z[val], pred)
            meta = {
                "name": name,
                "transform": t,
                "n_train": int(mask.sum() - len(val)),
                "n_val": int(len(val)),
                "n_trees": int(booster.num_boosted_rounds()),
                "fidelity": fid,
                "fidelity_teacher_scale": fidelity(inverse(z[val], t), inverse(pred, t)),
                "seed": seed,
                "params": {
                    **PARAMS,
                    "min_child_weight": min_child_weight(int(mask.sum() - len(val))),
                    **(params or {}),
                },
            }
            if stability:
                table.update(name, Time="stability…")
                with live_status() as update:
                    meta["stability_spearman"] = _stability(
                        Xm, z, featurizer.sparse, booster, seed, n_estimators, val_frac, params, update
                    )
            cdir = column_dir(model_dir, col_id)
            cdir.mkdir(parents=True, exist_ok=True)
            booster.save_model(str(cdir / S_NAME))
            write_json(cdir / SOFT_META_NAME, meta)
            keep = np.random.default_rng(seed).permutation(len(val))[:FIDELITY_POINTS]
            np.savez_compressed(cdir / FIDELITY_NAME, teacher=z[val][keep], surrogate=pred[keep])
            columns.append({"id": col_id, "name": name, "transform": t})
            results.append(meta)
            table.finish(
                name,
                **{
                    "R²": f"{fid['r2']:.3f}",
                    "ρ": f"{fid['spearman']:.3f}",
                    "Trees": f"{meta['n_trees']:,}",
                    "Time": elapsed(time.time() - col_started),
                },
            )

    if not columns:
        raise ValueError(f"No column of {soft_labels} has at least {MIN_SOFT_ROWS} finite soft labels.")
    with h5py.File(model_dir / TARGETS_NAME, "w") as f:
        f.create_dataset("rows", data=rows)
        f.create_dataset("Z", data=Z)
        # Molecules outside the library have no library row; keep their SMILES so learn-hard can featurize them
        extra = np.where(rows < 0)[0]
        f.create_dataset("extra_pos", data=extra)
        f.create_dataset(
            "extra_smiles", data=np.array([smiles[i] for i in extra], dtype=object), dtype=h5py.string_dtype()
        )
        f.attrs["soft_labels"] = str(soft_labels)

    if featurizer_name == "morgan" and (library.dir / LIB_BIT_DICTIONARY_NAME).exists():
        import shutil

        shutil.copy(library.dir / LIB_BIT_DICTIONARY_NAME, model_dir / BIT_DICTIONARY_NAME)

    golden_idx = np.random.default_rng(seed).choice(len(smiles), size=min(8, len(smiles)), replace=False)
    write_manifest(
        model_dir,
        {
            "xai4chem_version": __version__,
            "featurizer": featurizer.to_config(),
            "library": {"dir": str(library.dir), "name": library.name, "n_molecules": len(library)},
            "soft_labels": str(soft_labels),
            "columns": columns,
            "golden_smiles": [smiles[i] for i in sorted(golden_idx)],
        },
    )
    from xai4chem.model import write_golden

    write_golden(model_dir)
    rows_out = [("Featurizer", featurizer.description), ("Molecules", f"{len(smiles):,}")]
    for meta in results:
        line = f"R² {meta['fidelity']['r2']:.3f} · ρ {meta['fidelity']['spearman']:.3f} · {meta['n_trees']:,} trees"
        if "stability_spearman" in meta:
            line += f" · stability {meta['stability_spearman']:.2f}"
        rows_out.append((meta["name"], line))
    rows_out += [("Model", f"[dim]{cpath(model_dir)}[/]"), ("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]")]
    summary_panel("xai4chem · learn-soft", rows_out, border_style=color, icon="✓")
    return model_dir


def _stability(X, z, sparse, booster, seed, n_estimators, val_frac, params, update):
    """Spearman of global importances (mean |SHAP|) between two seeds and splits."""
    other, _, _ = train_booster(
        X, z, sparse, seed + 1, n_estimators=n_estimators, val_frac=val_frac, params=params, progress=update
    )
    idx = np.sort(np.random.default_rng(seed).choice(len(z), size=min(STABILITY_ROWS, len(z)), replace=False))
    a = np.abs(contributions(booster, X[idx], sparse)[:, :-1]).mean(0)
    b = np.abs(contributions(other, X[idx], sparse)[:, :-1]).mean(0)
    used = (a > 0) | (b > 0)
    return float(spearmanr(a[used], b[used]).statistic)

"""The four entry points: :func:`setup`, :func:`fit`, :func:`explain` and :func:`predict`.

Each is plain Python and prints nothing. Progress goes to ``display`` (see :mod:`xai4chem.progress`); the
CLI passes a console display, Python callers get silence by default. Each returns what it produced, so the
CLI can render its final summary and a notebook can keep working with the result.
"""

import shutil
import time
from pathlib import Path

import numpy as np

from xai4chem.progress import SILENT

FIDELITY_POINTS = 5000
GOLDEN_MOLECULES = 8
EXPLAIN_BATCH = 500
MAX_CARDS = 100


def _plural(n, word):
    return f"{n:,} {word}{'' if n == 1 else 's'}"


# --- setup ------------------------------------------------------------------------------------------


def setup(source=None, featurizers=("datamol",), library_dir=None, force=False, export_smiles=None, display=SILENT):
    """Download Ersilia's reference library, or build one from your own SMILES.

    Parameters
    ----------
    source : str or Path, optional
        ``None`` or ``"ersilia"`` for Ersilia's published library; a CSV of SMILES to build your own.
    featurizers : sequence of str
        Featurizers to get (``morgan``, ``accfg``, ``datamol``). Morgan is always included.
    library_dir : str or Path, optional
        Where the library lives (default ``~/.xai4chem/library``).
    force : bool
        Replace whatever library is already there.
    export_smiles : str or Path, optional
        Also write the library's SMILES to this CSV, for a teacher model to score.
    display : xai4chem.progress.Display
        Where progress goes; silent by default, ``ConsoleDisplay`` for the CLI's output.

    Returns
    -------
    ReferenceLibrary
        The library, ready to read.
    """
    from xai4chem.data.build import export_smiles as export
    from xai4chem.data.build import setup_library
    from xai4chem.data.library import ReferenceLibrary

    library = ReferenceLibrary(setup_library(source, featurizers, library_dir, force=force, display=display))
    if export_smiles is not None:
        display.note(f"writing the library's SMILES to {export_smiles}")
        export(library.dir, export_smiles)
    return library


# --- fit --------------------------------------------------------------------------------------------


def fit(
    soft_labels,
    model_dir,
    hard_labels=None,
    featurizer="datamol",
    library_dir=None,
    force=False,
    seed=42,
    task="auto",
    reweight="auto",
    n_estimators=None,
    panel_rows=None,
    report=True,
    display=SILENT,
):
    """Learn S from soft labels, optionally H from hard labels, explain the library panel and write the report.

    Parameters
    ----------
    soft_labels : str or Path
        The teacher's outputs: a ``smiles`` (or ``input``) column plus one numeric column per output.
    model_dir : str or Path
        Where to write the model.
    hard_labels : str or Path, optional
        Measurements, in columns named like the soft-label columns (or as a prefix of them).
    featurizer : str
        ``datamol`` (descriptors), ``morgan`` (substructures) or ``accfg`` (functional groups).
    library_dir : str or Path, optional
        The reference library (default ``~/.xai4chem/library``).
    force : bool
        Overwrite an existing model in ``model_dir``.
    seed : int
        Seed for the validation splits and the golden molecules.
    task : str
        Task of the hard labels: ``auto`` (binary if every label is 0 or 1, else regression), ``binary`` or
        ``regression``. Forcing ``binary`` on other values is refused.
    reweight : str
        Target reweighting for S: ``auto`` (only for an imbalanced target, as in Olinda), ``on``, ``off``,
        ``kde`` or ``bins``.
    n_estimators : int, optional
        Most boosting rounds for S (default 2,000, with early stopping).
    panel_rows : int, optional
        Library panel molecules explained globally (default 2,000).
    report : bool
        Write ``report.html``.
    display : xai4chem.progress.Display
        Where progress goes; silent by default, ``ConsoleDisplay`` for the CLI's output.

    Returns
    -------
    dict
        ``model_dir``; ``columns``, one global summary per column (fidelity, top features and, with a hard
        head, its held-out metric, ``a_max`` and reach); ``report``, its path or None; ``elapsed`` seconds.
    """
    from xai4chem.data.features import align_to_library, load_features
    from xai4chem.data.io import match_hard_columns, read_labels
    from xai4chem.data.library import ReferenceLibrary
    from xai4chem.explain.panel import PANEL_ROWS, explain_panel
    from xai4chem.models.model import XAIModel
    from xai4chem.models.surrogate import MIN_SOFT_ROWS, N_ESTIMATORS
    from xai4chem.run import MANIFEST_NAME

    started = time.time()
    model_dir = Path(model_dir)
    if (model_dir / MANIFEST_NAME).exists():
        if not force:
            raise ValueError(f"{model_dir} already holds a model. Pass --force to overwrite it.")
        shutil.rmtree(model_dir)
    library = ReferenceLibrary(library_dir)
    feat = library.featurizer(featurizer)
    smiles, Y = read_labels(soft_labels)
    mapping, hard = {}, None
    if hard_labels is not None:
        # Fail on a column mismatch now, not after minutes of boosting
        hard = read_labels(hard_labels)
        trainable = [str(c) for c in Y.columns if Y[c].notna().sum() >= MIN_SOFT_ROWS]
        mapping = match_hard_columns(trainable, [str(c) for c in hard[1].columns], strict=False)
        if not mapping:
            raise ValueError(
                f"No column of {hard_labels} ({', '.join(map(str, hard[1].columns))}) matches a soft-label column "
                f"({', '.join(map(str, Y.columns))}). Names must match exactly (case included), or as a prefix."
            )
        from xai4chem.models.hard import check_hard_labels, resolve_task

        for hard_name in mapping:  # refuse a bad --task now, not after minutes of boosting
            y = hard[1][hard_name].to_numpy(dtype=np.float64)
            y = y[np.isfinite(y)]
            check_hard_labels(hard_name, y, resolve_task(y, task, f"hard column '{hard_name}'"))
    total = 5 if hard_labels is not None else 4
    if not report:
        total -= 1

    display.step(1, total, "Reading soft labels")
    from xai4chem.models.surrogate import check_target

    for name in Y.columns:  # refuse an unlearnable column now, before any work (Olinda's checks)
        for warning in check_target(str(name), Y[name].to_numpy(dtype=np.float64)):
            display.note(warning, "warning")
    display.note(f"{_plural(len(smiles), 'molecule')} · {_plural(Y.shape[1], 'column')} · {feat.description}")
    rows = align_to_library(smiles, library)
    display.note(f"{int((rows >= 0).sum()):,} of them in the {library.name} reference library")
    X = load_features(smiles, rows, library, feat, display)

    display.step(2, total, "Learning S, the surrogate of the teacher")
    model_dir.mkdir(parents=True, exist_ok=True)
    Z, columns = _learn_soft(X, Y, feat, model_dir, reweight, seed, n_estimators or N_ESTIMATORS, display)
    _write_model(model_dir, library, feat, soft_labels, smiles, rows, Z, columns, seed)

    if hard_labels is not None:
        display.step(3, total, "Learning H from the hard labels")
        _learn_hard(hard, hard_labels, mapping, columns, X, Z, feat, model_dir, seed, task, display)

    step = 4 if hard_labels is not None else 3
    display.step(step, total, "Explaining the library panel")
    model = XAIModel(model_dir, check_golden=False)
    summaries = explain_panel(model, library, rows=panel_rows or PANEL_ROWS, display=display)

    out = None
    if report:
        from xai4chem.report.model_report import write_report

        display.step(step + 1, total, "Writing the report")
        out = write_report(model_dir)
    return {"model_dir": model_dir, "columns": summaries, "report": out, "elapsed": time.time() - started}


def _learn_soft(X, Y, feat, model_dir, reweight, seed, n_estimators, display):
    """S per teacher column, in a live table. Returns the teacher's values and the manifest's column entries."""
    from xai4chem.models.surrogate import fit_column
    from xai4chem.run import FIDELITY_NAME, S_NAME, SOFT_META_NAME, column_dir, column_id, write_json

    Z = np.full(Y.shape, np.nan, dtype=np.float32)
    columns = []
    names = [str(c) for c in Y.columns]
    with display.table(names, "Learning S", ["Rows", "R²", "ρ", "Trees", "Time"]) as table:
        for j, name in enumerate(names):
            table.start(name)
            col_started = time.time()

            def on_round(done, total, rmse, best, name=name):
                table.status("boosting", done, total, val_RMSE=f"{rmse:.4f}", best=f"{best:.4f}")

            fitted = fit_column(
                X,
                Y.iloc[:, j].to_numpy(dtype=np.float64),
                feat.sparse,
                seed,
                n_estimators,
                on_round=on_round,
                reweight=reweight,
            )
            meta = {"name": name, **fitted["meta"]}
            if meta["reweight"]["strategy"] != "none":
                display.note(f"{name}: {meta['reweight']['strategy']} reweighting ({meta['reweight']['reason']})")
            Z[fitted["mask"], j] = fitted["y"]
            cdir = column_dir(model_dir, column_id(j))
            cdir.mkdir(parents=True, exist_ok=True)
            fitted["booster"].save_model(str(cdir / S_NAME))
            write_json(cdir / SOFT_META_NAME, meta)
            teacher, surrogate = fitted["heldout"]
            keep = np.random.default_rng(seed).permutation(len(teacher))[:FIDELITY_POINTS]
            np.savez_compressed(cdir / FIDELITY_NAME, teacher=teacher[keep], surrogate=surrogate[keep])
            columns.append({"id": column_id(j), "name": name})
            fid = meta["fidelity"]
            table.finish(
                name,
                Rows=f"{meta['n_train'] + meta['n_val']:,}",
                **{"R²": f"{fid['r2']:.3f}", "ρ": f"{fid['spearman']:.3f}"},
                Trees=f"{meta['n_trees']:,}",
                Time=f"{time.time() - col_started:.0f}s",
            )
    return Z, columns


def _write_model(model_dir, library, feat, soft_labels, smiles, rows, Z, columns, seed):
    """The manifest, the targets (for the applicability check), the bit dictionary and the golden vectors."""
    import h5py

    from xai4chem import __version__
    from xai4chem.data.library import BIT_DICTIONARY_NAME as LIB_BIT_DICTIONARY
    from xai4chem.models.model import write_golden
    from xai4chem.run import BIT_DICTIONARY_NAME, TARGETS_NAME, write_manifest

    with h5py.File(model_dir / TARGETS_NAME, "w") as f:
        f.create_dataset("rows", data=rows)
        f.create_dataset("Z", data=Z)
    if feat.name == "morgan" and (library.dir / LIB_BIT_DICTIONARY).exists():
        shutil.copy(library.dir / LIB_BIT_DICTIONARY, model_dir / BIT_DICTIONARY_NAME)
    golden = np.random.default_rng(seed).choice(len(smiles), size=min(GOLDEN_MOLECULES, len(smiles)), replace=False)
    write_manifest(
        model_dir,
        {
            "xai4chem_version": __version__,
            "featurizer": feat.to_config(),
            "library": {"dir": str(library.dir), "name": library.name, "n_molecules": len(library)},
            "soft_labels": str(soft_labels),
            "columns": columns,
            "golden_smiles": [smiles[i] for i in sorted(golden)],
        },
    )
    write_golden(model_dir)


def _hard_inputs(hard, hard_labels, mapping, by_name, feat, display):
    """The matched columns, which labelled SMILES parse, and their features and fingerprints."""
    from xai4chem.data.features import featurize
    from xai4chem.data.similarity import binarize, fingerprints
    from xai4chem.models.model import valid_mask

    smiles, Y = hard
    mapping = {h: s for h, s in mapping.items() if s in by_name}
    ignored = [str(c) for c in Y.columns if str(c) not in mapping]
    if ignored:
        display.note(
            f"ignoring columns of {Path(hard_labels).name} that match no soft-label column: {', '.join(ignored)}",
            "warning",
        )
    for h, s in mapping.items():
        if h != s:
            display.note(f"hard column {h} refines soft column {s}")
    valid = valid_mask(smiles)
    if not valid.all():
        display.note(f"{int((~valid).sum()):,} unparseable SMILES in {Path(hard_labels).name} are left out", "warning")
    Xh_all = featurize(feat, smiles, display)
    return mapping, valid, Xh_all, binarize(Xh_all) if feat.name == "morgan" else fingerprints(smiles)


def _save_hard_head(hdir, booster, iso_map, labelled, fps, meta):
    """Write a hard head; ``hard_meta.json`` goes last, because its presence marks a complete head."""
    import scipy.sparse as sp

    from xai4chem.run import H_NAME, H_TO_S_NAME, HARD_META_NAME, LABELLED_FP_NAME, LABELLED_NAME, write_json

    hdir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(hdir / H_NAME))
    write_json(hdir / H_TO_S_NAME, iso_map)
    labelled.to_csv(hdir / LABELLED_NAME, index=False)
    sp.save_npz(hdir / LABELLED_FP_NAME, sp.csr_matrix(fps))
    write_json(hdir / HARD_META_NAME, meta)


def _learn_hard(hard, hard_labels, mapping, columns, X, Z, feat, model_dir, seed, task_mode, display):
    """H, its isotonic map onto S and the gate's labelled set, per matched column, in a live table."""
    import pandas as pd

    from xai4chem.models import hard as hd
    from xai4chem.models.model import write_golden
    from xai4chem.models.surrogate import contributions, predict_margin
    from xai4chem.run import hard_dir

    smiles, Y = hard
    by_name = {c["name"]: c for c in columns}
    mapping, valid, Xh_all, fps_all = _hard_inputs(hard, hard_labels, mapping, by_name, feat, display)
    with display.table(list(mapping), "Learning H", ["Task", "Labels", "Held-out", "R² H_S", "a_max", "Time"]) as table:
        for hard_name, soft_name in mapping.items():
            table.start(hard_name)
            col_started = time.time()
            col = by_name[soft_name]
            y = Y[hard_name].to_numpy(dtype=np.float64)
            idx = np.where(np.isfinite(y) & valid)[0]
            y = y[idx]
            task = hd.resolve_task(y, task_mode, f"hard column '{hard_name}'")
            table.update(
                hard_name,
                Task=task,
                Labels=f"{len(y):,}" + (f" ({int((y == 1).sum()):,} +)" if task == "binary" else ""),
            )
            for warning in hd.check_hard_labels(hard_name, y, task):
                display.note(warning, "warning")
            Xh, fh = Xh_all[idx], fps_all[idx]
            table.status("fitting H")
            booster, heldout = hd.train_hard_model(Xh, y, task, feat.sparse, seed)
            table.update(hard_name, **{"Held-out": hd.metric_text(heldout)})

            # Carry H onto S's scale over soft-labelled molecules, half to fit the map and half to judge it
            j = int(col["id"][1:])
            ok = np.where(np.isfinite(Z[:, j]))[0]
            ok = np.random.default_rng(seed).permutation(ok)[: hd.ISOTONIC_ROWS]
            table.status("calibrating H → S")
            iso_map, alignment_r2 = hd.fit_isotonic(predict_margin(booster, X[ok], feat.sparse), Z[ok, j])
            a_max = hd.blend_ceiling(alignment_r2)

            meta = {
                "name": soft_name,
                "hard_column": hard_name,
                "task": task,
                "n_labels": int(len(y)),
                "n_positive": int((y == 1).sum()) if task == "binary" else None,
                "heldout": heldout,
                "n_trees": int(booster.num_boosted_rounds()),
                "base_h": float(contributions(booster, Xh[:1], feat.sparse)[0, -1]),
                "isotonic_direction": "increasing" if iso_map["increasing"] else "decreasing",
                "alignment_r2": alignment_r2,
                "a_max": a_max,
                "t_lo": hd.T_LO,
                "t_hi": hd.T_HI,
                "a_ceiling": hd.A_CEILING,
                "seed": seed,
            }
            labelled = pd.DataFrame({"smiles": [smiles[i] for i in idx], "label": y})
            _save_hard_head(hard_dir(model_dir, col["id"]), booster, iso_map, labelled, fh, meta)
            table.finish(
                hard_name,
                **{"R² H_S": f"{alignment_r2:.3f}", "a_max": f"{a_max:.2f}"},
                Time=f"{time.time() - col_started:.0f}s",
            )
            if a_max == 0.0:
                display.note(
                    f"{hard_name}: H_S does not reproduce the teacher's scale (R² {alignment_r2:.2f}), so P stays equal "
                    "to S; H's explanations are still reported",
                    "warning",
                )
    write_golden(model_dir)


# --- explain ----------------------------------------------------------------------------------------


def _load(model):
    from xai4chem.models.model import XAIModel

    return model if isinstance(model, XAIModel) else XAIModel(model)


def explain(model, smiles, output_dir="explanations", ids=None, domain=False, display=SILENT):
    """Predict and explain molecules; write ``predictions.csv``, ``attributions.csv`` and ``index.html``.

    With exactly two valid molecules, the gap between their predictions is also split into exact
    per-feature differences.

    Parameters
    ----------
    model : str, Path or XAIModel
        A model directory written by :func:`fit`, or the model already loaded.
    smiles : sequence of str
        Molecules to explain; unparseable ones keep their row, with empty values.
    output_dir : str or Path
        Where to write the tables and the gallery.
    ids : sequence of str, optional
        A name per molecule (default ``mol_0000``, ``mol_0001``…).
    domain : bool
        Also look up each molecule's nearest library molecule and the teacher's value for it (loads the
        library's fingerprints, a few seconds).
    display : xai4chem.progress.Display
        Where progress goes; silent by default, ``ConsoleDisplay`` for the CLI's output.

    Returns
    -------
    dict
        ``predictions`` and ``attributions`` (DataFrames); ``comparisons`` (one dict per column, or empty);
        ``gallery`` (its path); ``cards`` (molecules drawn); ``domain`` (the median library similarity and the
        count below 0.3, or None); ``elapsed`` seconds.
    """
    import pandas as pd

    from xai4chem.explain.tables import attribution_rows, compare_pair, prediction_rows
    from xai4chem.report import gallery as G

    started = time.time()
    model = _load(model)
    smiles = [str(s) for s in smiles]
    ids = [str(i) for i in ids] if ids is not None else [f"mol_{i:04d}" for i in range(len(smiles))]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cols, feat = model.columns, model.featurizer
    mols = _parse(smiles, ids, display)
    neighbours = None
    if domain:
        display.note("finding each molecule's nearest library molecule")
        neighbours = model.library_neighbours(smiles)
        if neighbours is None:
            display.note(
                "the reference library this model was trained on was not found; the applicability columns are left out",
                "warning",
            )

    pred_parts, long_rows, cards, comparisons = [], [], [], []
    pair = sum(m is not None for m in mols) == 2 and len(smiles) == 2
    with display.loop("explaining", len(smiles)) as tick:
        for start in range(0, len(smiles), EXPLAIN_BATCH):
            stop = min(start + EXPLAIN_BATCH, len(smiles))
            b_smi, b_ids, b_mols = smiles[start:stop], ids[start:stop], mols[start:stop]
            X = model.featurize(b_smi)
            res = model.explain(b_smi, X=X)
            pred_parts.append(prediction_rows(cols, res, b_ids, b_smi, [m is not None for m in b_mols]))
            long_rows.extend(attribution_rows(cols, res, X, b_ids, b_mols, feat))
            for i, mol in enumerate(b_mols):
                if mol is not None and len(cards) < MAX_CARDS:
                    nb = _neighbour(neighbours, start + i)
                    cards.append(G.card(feat, cols, res, i, b_ids[i], b_smi[i], mol, X[i], nb))
            if pair:
                comparisons = [(col, compare_pair(col, res[col.name], X, b_mols, feat), res[col.name]) for col in cols]
            tick(stop)
    pred = pd.concat(pred_parts, ignore_index=True)
    summary = _add_domain(pred, neighbours) if neighbours is not None else None
    attributions = pd.DataFrame(long_rows)
    pred.to_csv(output_dir / "predictions.csv", index=False)
    attributions.to_csv(output_dir / "attributions.csv", index=False)
    sections = [G.comparison(feat, col, c, r, mols, ids) for col, c, r in comparisons]
    gallery = G.write_gallery(output_dir / "index.html", feat, cols, pred, cards, sections)
    return {
        "predictions": pred,
        "attributions": attributions,
        "comparisons": [c for _, c, _ in comparisons],
        "gallery": gallery,
        "cards": len(cards),
        "domain": summary,
        "elapsed": time.time() - started,
    }


def _parse(smiles, ids, display):
    """RDKit molecules (None where unparseable); refuses an input where nothing parses."""
    from rdkit import Chem

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    invalid = [ids[i] for i, m in enumerate(mols) if m is None]
    if len(invalid) == len(smiles):
        raise ValueError("None of the SMILES could be parsed; nothing to explain.")
    if invalid:
        shown = ", ".join(invalid[:5]) + (" …" if len(invalid) > 5 else "")
        display.note(
            f"{_plural(len(invalid), 'unparseable SMILES')} ({shown}) reported with empty predictions", "warning"
        )
    return mols


def _add_domain(pred, neighbours):
    """Add the applicability columns to ``pred``; return the median similarity and the count below 0.3."""
    from xai4chem.report.gallery import DOMAIN_FAR

    pred["library_T"] = neighbours["T"]
    pred["library_nearest_smiles"] = neighbours["smiles"]
    for name, values in neighbours["teacher"].items():
        pred[f"{name}_library_teacher"] = values
    T = pred["library_T"].dropna()
    return {"median_T": float(T.median()), "far": int((T < DOMAIN_FAR).sum())} if len(T) else None


def _neighbour(neighbours, k):
    if neighbours is None or not np.isfinite(neighbours["T"][k]):
        return None
    return {
        "T": float(neighbours["T"][k]),
        "smiles": neighbours["smiles"][k],
        "teacher": {c: float(v[k]) for c, v in neighbours["teacher"].items()},
    }


# --- predict ----------------------------------------------------------------------------------------


def predict(model, smiles, output=None, details=False, display=SILENT):
    """Score molecules without explanations; millions are fine.

    Parameters
    ----------
    model : str, Path or XAIModel
        A model directory written by :func:`fit`, or the model already loaded.
    smiles : sequence of str
        Molecules to score.
    output : str or Path, optional
        A CSV to write the table to.
    details : bool
        Also give S and, with a hard head, H_S, T and ``a`` per column.
    display : xai4chem.progress.Display
        Where progress goes; silent by default, ``ConsoleDisplay`` for the CLI's output.

    Returns
    -------
    pandas.DataFrame
        ``smiles``, ``valid``, then P per column on the teacher's scale (NaN for unparseable SMILES).
    """
    model = _load(model)
    df = model.predict(list(smiles), details=details, display=display)
    if output is not None:
        df.to_csv(output, index=False)
    return df

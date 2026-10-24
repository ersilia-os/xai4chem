"""Per-molecule explanations: tables, an HTML gallery and (optionally) PNG drawings for a list of SMILES.

Outputs, in the output directory:

- ``predictions.csv``: one row per molecule. Per column: the prediction on the teacher's scale, S and
  (with a hard head) H_S, T, a, the nearest labelled molecule and its label, and the agreement between
  the teacher's and the experiments' explanations (cosine of φ_S and φ_HS).
- ``attributions.csv``: long format, one row per molecule × column × feature, for every feature present
  in the molecule plus the strongest absent ones.
- ``index.html``: a self-contained gallery, one card per molecule.
- ``png/`` (with ``png=True``): atom-highlight panels per molecule and column.
"""

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rich.markup import escape

from xai4chem.console import STEP_COLORS, console, echo, elapsed, rule, set_active_color, summary_panel, sweep_progress
from xai4chem.console import path as cpath
from xai4chem.reporting import html as H
from xai4chem.reporting.drawing import atom_attributions, atom_png, atom_svg, prepare
from xai4chem.soft import inverse

TOP_FEATURES = 20
CARD_FEATURES = 7
ABSENT_FEATURES = 3
BATCH = 500
MAX_CARDS = 100
TERMINAL_ROWS = 15
DOMAIN_MAX = 2000
DOMAIN_FAR = 0.3


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))[:80]


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def explain_molecules(
    model,
    smiles,
    output_dir,
    ids=None,
    columns=None,
    png=False,
    gallery=True,
    quiet=False,
    domain=None,
    max_cards=MAX_CARDS,
):
    started = time.time()
    color = STEP_COLORS["explain"]
    set_active_color(color)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    smiles = [str(s) for s in smiles]
    ids = [str(i) for i in ids] if ids is not None else [f"mol_{i:04d}" for i in range(len(smiles))]
    cols = model._select(columns)
    if not quiet:
        rule(
            "xai4chem · explain",
            style=color,
            right=f"{len(smiles):,} molecules · {len(cols)} column(s) · {model.featurizer.name}",
        )
    labels = model.featurizer.labels()
    names = model.featurizer.feature_names
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    invalid = [i for i, m in enumerate(mols) if m is None]
    if invalid:
        bad = ", ".join(escape(ids[i]) for i in invalid[:5]) + (" …" if len(invalid) > 5 else "")
        n = len(invalid)
        echo(f"{n} unparseable SMILES ({bad}) {'is' if n == 1 else 'are'} reported with empty predictions", "warning")

    if domain is None:
        domain = len(smiles) <= DOMAIN_MAX
    neighbours = model.library_neighbours(smiles) if domain else None
    if neighbours is not None:
        for i in invalid:
            neighbours["T"][i] = np.nan

    pred_parts, long_rows, cards = [], [], []
    with sweep_progress("explaining", len(smiles)) as tick:
        for start in range(0, len(smiles), BATCH):
            stop = min(start + BATCH, len(smiles))
            b_smi, b_ids, b_mols = smiles[start:stop], ids[start:stop], mols[start:stop]
            X = model.featurize(b_smi)
            res = model.explain(b_smi, X=X, columns=[c.name for c in cols])
            pred_parts.append(_prediction_rows(cols, res, b_ids, b_smi, b_mols))
            long_rows.extend(_attribution_rows(cols, res, X, b_ids, b_mols, labels, names, model.featurizer))
            for i in range(stop - start):
                if b_mols[i] is None:
                    continue
                if gallery and len(cards) < max_cards:
                    nb = _neighbour(neighbours, start + i)
                    cards.append(_card(model, cols, res, i, b_ids[i], b_smi[i], b_mols[i], X[i], labels, nb))
                if png:
                    _png(model, cols, res, i, b_ids[i], b_mols[i], output_dir / "png")
            tick(stop)
    pred = pd.concat(pred_parts, ignore_index=True)
    if neighbours is not None:
        pred["library_T"] = neighbours["T"]
        pred["library_nearest_smiles"] = neighbours["smiles"]
        for col in cols:
            if col.name in neighbours["teacher"]:
                pred[f"{col.name}_library_teacher"] = neighbours["teacher"][col.name]
        pred.loc[~pred["valid"], ["library_T", "library_nearest_smiles"]] = np.nan
    pred.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(long_rows).to_csv(output_dir / "attributions.csv", index=False)
    if gallery:
        _write_gallery(model, cols, cards, len(smiles), output_dir / "index.html", pred)
        if len(smiles) > max_cards:
            echo(
                f"the gallery shows the first {max_cards} molecules (--cards); the CSV files have all {len(smiles):,}",
                "info",
            )
    if not quiet:
        _terminal_table(cols, pred, long_rows, sparse=model.featurizer.sparse)
        if neighbours is not None and pred["library_T"].notna().any():
            T = pred["library_T"].dropna()
            far = int((T < DOMAIN_FAR).sum())
            note = f" · [yellow]{far} below {DOMAIN_FAR} (S extrapolates)[/]" if far else ""
            echo(f"applicability · median Tanimoto to the nearest library molecule {T.median():.2f}{note}", "info")
        rows_out = [("Molecules", f"{len(smiles):,}"), ("Columns", ", ".join(c.name for c in cols))]
        rows_out.append(("Tables", f"[dim]{cpath(output_dir)}/predictions.csv · attributions.csv[/]"))
        if gallery:
            rows_out.append(("Gallery", f"[dim]{cpath(output_dir / 'index.html')}[/]"))
        rows_out.append(("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]"))
        summary_panel("xai4chem · explain", rows_out, border_style=color, icon="✓")
    return pred


def _prediction_rows(cols, res, ids, smiles, mols):
    df = pd.DataFrame({"id": ids, "smiles": smiles, "valid": [m is not None for m in mols]})
    for col in cols:
        r = res[col.name]
        # Same columns, same (teacher) scale as `predict --details`
        df[col.name] = r["prob"]
        df[f"{col.name}_S"] = inverse(r["S"], col.transform)
        if col.hard is not None:
            hh = col.hard
            df[f"{col.name}_HS"] = inverse(r["HS"], col.transform)
            df[f"{col.name}_T"] = r["T"]
            df[f"{col.name}_a"] = r["a"]
            df[f"{col.name}_nearest_smiles"] = [hh.labelled_smiles[j] if j >= 0 else None for j in r["nearest"]]
            df[f"{col.name}_nearest_label"] = [hh.labels[j] if j >= 0 else np.nan for j in r["nearest"]]
            df[f"{col.name}_agreement"] = [cosine(r["phi_S"][i], r["phi_HS"][i]) for i in range(len(ids))]
    value_cols = [c for c in df.columns if c not in ("id", "smiles", "valid")]
    df.loc[~df["valid"], value_cols] = np.nan
    return df


def _attribution_rows(cols, res, X, ids, mols, labels, names, featurizer):
    rows = []
    for col in cols:
        r = res[col.name]
        for i in range(len(ids)):
            if mols[i] is None:
                continue
            phi = r["phi_P"][i]
            local = featurizer.molecule_labels(mols[i])
            keep = set(np.nonzero(X[i])[0]) | set(np.argsort(-np.abs(phi))[:TOP_FEATURES])
            for f in sorted(keep, key=lambda f: -abs(phi[f])):
                row = {
                    "id": ids[i],
                    "column": col.name,
                    "feature": names[f],
                    "label": local.get(f, labels[f]),
                    "value": float(X[i, f]),
                    "phi_S": float(r["phi_S"][i, f]),
                }
                if col.hard is not None:
                    row["phi_HS"] = float(r["phi_HS"][i, f])
                    row["phi_P"] = float(phi[f])
                rows.append(row)
    return rows


def _views(model, col, r, i, mol):
    """The atom-map panels for one molecule: [(title, scores)], or None for non-atomic featurizers."""
    s = atom_attributions(model.featurizer, mol, r["phi_S"][i])
    if s is None:
        return None
    views = [("Teacher (S)", s[0])]
    if col.hard is not None:
        hs = atom_attributions(model.featurizer, mol, r["phi_HS"][i])[0]
        p = atom_attributions(model.featurizer, mol, r["phi_P"][i])[0]
        views += [("Experiments (H_S)", hs), ("Combined (P)", p), ("Disagreement (H_S − S)", hs - s[0])]
    return views, s[1]


def _neighbour(neighbours, k):
    if neighbours is None or not np.isfinite(neighbours["T"][k]):
        return None
    return {
        "T": float(neighbours["T"][k]),
        "smiles": neighbours["smiles"][k],
        "teacher": {c: float(v[k]) for c, v in neighbours["teacher"].items()},
    }


def _card(model, cols, res, i, mol_id, smi, mol, x_row, labels, nb=None):
    drawn = prepare(mol)
    parts = [
        f'<div class="card molcard" id="m-{_safe(mol_id)}"><div class="head"><b>{H.esc(mol_id)}</b>'
        f'<span class="smi">{H.esc(smi)}</span></div>'
    ]
    for col in cols:
        r = res[col.name]
        hard = col.hard is not None
        stats = [H.stat("Prediction", f"{r['prob'][i]:.3f}", f"{col.name} · teacher scale")]
        stats.append(H.stat("S", f"{inverse(r['S'][i], col.transform):.3f}", "teacher surrogate"))
        if hard:
            stats.append(H.stat("H_S", f"{inverse(r['HS'][i], col.transform):.3f}", "experiments"))
            stats.append(
                H.stat(
                    "a",
                    f"{r['a'][i]:.2f}",
                    f"T = {r['T'][i]:.2f} to nearest labelled",
                    tip="Blend weight of the hard head: a = a_max · ramp(T)",
                )
            )
        parts.append(f'<div class="eyebrow" style="margin-top:12px">{H.esc(col.name)}</div>{H.kpis(stats)}')
        views = _views(model, col, r, i, mol)
        if views is not None:
            panels, unmapped = views
            vmax = max(float(np.max(np.abs(v))) for _, v in panels) or 1.0
            figs = "".join(
                f"<figure>{atom_svg(drawn, v, vmax)}<figcaption>{H.esc(t)}</figcaption></figure>" for t, v in panels
            )
            parts.append(f'<div class="views">{figs}</div>')
            parts.append(
                '<div class="legend"><span style="--c:var(--crimson)">raises the prediction</span><span style="--c:var(--cobalt)">lowers it</span></div>'
            )
        phi = r["phi_P"][i]
        # Present features first: they are what a chemist can act on. Absent ones are listed apart, briefly
        order = np.argsort(-np.abs(phi))
        if model.featurizer.sparse:
            present = [f for f in order if x_row[f] > 0 and phi[f] != 0][:CARD_FEATURES]
            absent = [f for f in order if x_row[f] == 0 and phi[f] != 0][:ABSENT_FEATURES]
        else:
            # Descriptors are values, not presences: a negative logP is not an absent feature
            present = [f for f in order if phi[f] != 0][:CARD_FEATURES]
            absent = []
        scale = max([abs(float(phi[f])) for f in present + absent] or [1.0])
        local = model.featurizer.molecule_labels(mol)
        bars = H.contributions([(local.get(f, labels[f]), float(phi[f])) for f in present], scale=scale)
        if absent and model.featurizer.sparse:
            bars += '<div class="faint" style="font-size:11px;margin:8px 0 4px">Strongest absent features</div>'
            bars += H.contributions([(labels[f], float(phi[f])) for f in absent], scale=scale)
        side = ""
        if hard:
            j = int(r["nearest"][i])
            if j >= 0:
                side = (
                    f'<p class="muted" style="font-size:12px">Nearest labelled molecule (T = {r["T"][i]:.2f}): '
                    f'<span class="smi">{H.esc(col.hard.labelled_smiles[j])}</span> · label {col.hard.labels[j]:g}. '
                    f"Agreement between teacher and experiments: {cosine(r['phi_S'][i], r['phi_HS'][i]):+.2f}.</p>"
                )
        if nb is not None:
            teacher = nb["teacher"].get(col.name, float("nan"))
            said = f" · teacher {teacher:.3f}" if np.isfinite(teacher) else ""
            warn = " S is extrapolating here." if nb["T"] < DOMAIN_FAR else ""
            side = (
                f'<p class="muted" style="font-size:12px">Nearest library molecule (T = {nb["T"]:.2f}): '
                f'<span class="smi">{H.esc(nb["smiles"])}</span>{said}.{warn}</p>'
            ) + side
        note = ""
        if views is not None and abs(views[1]) > 1e-3:
            note = f'<p class="faint" style="font-size:11.5px">Absent features contribute {views[1]:+.3f}; they have no atoms to sit on.</p>'
        parts.append(f'<div class="split"><div>{bars}</div><div>{side}{note}</div></div>')
    parts.append("</div>")
    return "".join(parts)


def _overview(cols, pred, n_cards):
    """One row per molecule with its predictions, linking to its card."""
    headers = ["Molecule"]
    for col in cols:
        headers.append(col.name)
        if col.hard is not None:
            headers += ["a", "Agreement"]
    rows = []
    for k, rec in pred.iterrows():
        link = (
            f'<a href="#m-{_safe(rec["id"])}">{H.esc(rec["id"])}</a>'
            if k < n_cards and rec["valid"]
            else H.esc(rec["id"])
        )
        cells = [link]
        for col in cols:
            v = rec[col.name]
            cells.append("—" if pd.isna(v) else f"{v:.3f}")
            if col.hard is not None:
                a, g = rec[f"{col.name}_a"], rec[f"{col.name}_agreement"]
                cells.append("—" if pd.isna(a) else f"{a:.2f}")
                cells.append("—" if pd.isna(g) else f"{g:+.2f}")
        rows.append(cells)
    numeric = tuple(range(1, len(headers)))
    return H.table(headers, rows, numeric=numeric, table_id="overview", sortable=True)


def _write_gallery(model, cols, cards, n_total, path, pred):
    lede = (
        f"{n_total:,} molecules explained with a {H.esc(model.featurizer.description)} surrogate. "
        "Atom colours and bars are SHAP contributions on the model's scale (logit for probability teachers)."
    )
    body = H.section(
        "Overview", _overview(cols, pred.head(2000), len(cards)), hint="predictions on the teacher's scale"
    )
    body += '<div style="margin-top:22px"><input class="filter" placeholder="Filter cards by id or SMILES" data-target=".molcard"></div>'
    body += "".join(cards)
    page = H.page(
        "xai4chem · explanations",
        "xai4chem",
        "Molecule <em>explanations</em>",
        lede,
        body,
        script=H.FILTER_SCRIPT + H.SORT_SCRIPT,
    )
    path.write_text(page)


def _png(model, cols, res, i, mol_id, mol, out):
    out.mkdir(parents=True, exist_ok=True)
    drawn = prepare(mol)
    for col in cols:
        views = _views(model, col, res[col.name], i, mol)
        if views is None:
            continue
        panels = views[0]
        vmax = max(float(np.max(np.abs(v))) for _, v in panels) or 1.0
        atom_png(drawn, panels, out / f"{_safe(mol_id)}_{_safe(col.name)}.png", vmax)


def _terminal_table(cols, pred, long_rows, sparse=True):
    from rich import box
    from rich.table import Table

    table = Table(box=box.SIMPLE_HEAD, header_style="bold blue", pad_edge=False)
    table.add_column("Molecule", no_wrap=True, min_width=8, max_width=18, overflow="ellipsis")
    for col in cols:
        table.add_column(
            col.name, justify="right", no_wrap=True, min_width=max(6, min(len(col.name), 14)), overflow="ellipsis"
        )
        if col.hard is not None:
            table.add_column("a", justify="right", no_wrap=True, min_width=4)
    table.add_column("Top contributions", overflow="ellipsis", no_wrap=True, ratio=1)
    first = cols[0].name
    by_mol = {}
    for row in long_rows:
        if row["column"] == first:
            by_mol.setdefault(row["id"], []).append(row)
    for _, rec in pred.head(TERMINAL_ROWS).iterrows():
        cells = [escape(str(rec["id"]))]
        for col in cols:
            v = rec[col.name]
            cells.append("—" if pd.isna(v) else f"{v:.3f}")
            if col.hard is not None:
                a = rec[f"{col.name}_a"]
                cells.append("—" if pd.isna(a) else f"{a:.2f}")
        feats = sorted(by_mol.get(rec["id"], []), key=lambda r: -abs(r.get("phi_P", r["phi_S"])))[:3]
        tops = []
        for fr in feats:
            v = fr.get("phi_P", fr["phi_S"])
            absent = "[dim]absent[/] " if sparse and fr["value"] == 0 else ""
            tops.append(f"[{'red' if v > 0 else 'blue'}]{v:+.2f}[/] {absent}{escape(fr['label'][:28])}")
        cells.append("  ".join(tops))
        table.add_row(*cells)
    console.print(table)
    if len(pred) > TERMINAL_ROWS:
        echo(f"showing {TERMINAL_ROWS} of {len(pred):,} molecules", "info")

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

from xai4chem.console import STEP_COLORS, console, echo, elapsed, rule, set_active_color, summary_panel, sweep_progress
from xai4chem.console import path as cpath
from xai4chem.reporting import html as H
from xai4chem.reporting.drawing import atom_attributions, atom_png, atom_svg, prepare

TOP_FEATURES = 20
CARD_FEATURES = 8
BATCH = 500
MAX_CARDS = 300
TERMINAL_ROWS = 15


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))[:80]


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def explain_molecules(model, smiles, output_dir, ids=None, columns=None, png=False, gallery=True, quiet=False):
    started = time.time()
    color = STEP_COLORS["explain"]
    set_active_color(color)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    smiles = [str(s) for s in smiles]
    ids = [str(i) for i in ids] if ids is not None else [f"mol_{i:04d}" for i in range(len(smiles))]
    cols = model._select(columns)
    if not quiet:
        rule("xai4chem · explain", style=color, right=f"{len(smiles):,} molecules · {len(cols)} column(s) · {model.featurizer.name}")
    labels = model.featurizer.labels()
    names = model.featurizer.feature_names
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    invalid = [i for i, m in enumerate(mols) if m is None]
    if invalid:
        echo(f"{len(invalid)} unparseable SMILES are reported with empty predictions", "warning")

    pred_parts, long_rows, cards = [], [], []
    with sweep_progress("explaining", len(smiles)) as tick:
        for start in range(0, len(smiles), BATCH):
            stop = min(start + BATCH, len(smiles))
            b_smi, b_ids, b_mols = smiles[start:stop], ids[start:stop], mols[start:stop]
            X = model.featurize(b_smi)
            res = model.explain(b_smi, X=X, columns=[c.name for c in cols])
            pred_parts.append(_prediction_rows(cols, res, b_ids, b_smi, b_mols))
            long_rows.extend(_attribution_rows(cols, res, X, b_ids, b_mols, labels, names))
            for i in range(stop - start):
                if b_mols[i] is None:
                    continue
                if gallery and len(cards) < MAX_CARDS:
                    cards.append(_card(model, cols, res, i, b_ids[i], b_smi[i], b_mols[i], X[i], labels))
                if png:
                    _png(model, cols, res, i, b_ids[i], b_mols[i], output_dir / "png")
            tick(stop)
    pred = pd.concat(pred_parts, ignore_index=True)
    pred.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(long_rows).to_csv(output_dir / "attributions.csv", index=False)
    if gallery:
        _write_gallery(model, cols, cards, len(smiles), output_dir / "index.html")
        if len(smiles) > MAX_CARDS:
            echo(f"the gallery shows the first {MAX_CARDS} molecules; the CSV files have all {len(smiles):,}", "info")
    if not quiet:
        _terminal_table(cols, pred, long_rows)
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
        df[col.name] = r["prob"]
        df[f"{col.name}_S"] = r["S"]
        if col.hard is not None:
            hh = col.hard
            df[f"{col.name}_HS"] = r["HS"]
            df[f"{col.name}_T"] = r["T"]
            df[f"{col.name}_a"] = r["a"]
            df[f"{col.name}_nearest_smiles"] = [hh.labelled_smiles[j] if j >= 0 else None for j in r["nearest"]]
            df[f"{col.name}_nearest_label"] = [hh.labels[j] if j >= 0 else np.nan for j in r["nearest"]]
            df[f"{col.name}_agreement"] = [cosine(r["phi_S"][i], r["phi_HS"][i]) for i in range(len(ids))]
    value_cols = [c for c in df.columns if c not in ("id", "smiles", "valid")]
    df.loc[~df["valid"], value_cols] = np.nan
    return df


def _attribution_rows(cols, res, X, ids, mols, labels, names):
    rows = []
    for col in cols:
        r = res[col.name]
        for i in range(len(ids)):
            if mols[i] is None:
                continue
            phi = r["phi_P"][i]
            keep = set(np.nonzero(X[i])[0]) | set(np.argsort(-np.abs(phi))[:TOP_FEATURES])
            for f in sorted(keep, key=lambda f: -abs(phi[f])):
                row = {"id": ids[i], "column": col.name, "feature": names[f], "label": labels[f], "value": float(X[i, f]), "phi_S": float(r["phi_S"][i, f])}
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


def _card(model, cols, res, i, mol_id, smi, mol, x_row, labels):
    drawn = prepare(mol)
    parts = [f'<div class="card molcard"><div class="head"><b>{H.esc(mol_id)}</b><span class="smi">{H.esc(smi)}</span></div>']
    for col in cols:
        r = res[col.name]
        hard = col.hard is not None
        stats = [H.stat("Prediction", f"{r['prob'][i]:.3f}", f"{col.name} · teacher scale")]
        stats.append(H.stat("S", f"{r['S'][i]:+.2f}", "teacher surrogate", tip="S on its own scale (logit for probabilities)"))
        if hard:
            stats.append(H.stat("H_S", f"{r['HS'][i]:+.2f}", "experiments, on S's scale"))
            stats.append(H.stat("a", f"{r['a'][i]:.2f}", f"T = {r['T'][i]:.2f} to nearest labelled", tip="Blend weight of the hard head: a = a_max · ramp(T)"))
        parts.append(f'<div class="eyebrow" style="margin-top:12px">{H.esc(col.name)}</div>{H.kpis(stats)}')
        views = _views(model, col, r, i, mol)
        if views is not None:
            panels, unmapped = views
            vmax = max(float(np.max(np.abs(v))) for _, v in panels) or 1.0
            figs = "".join(f"<figure>{atom_svg(drawn, v, vmax)}<figcaption>{H.esc(t)}</figcaption></figure>" for t, v in panels)
            parts.append(f'<div class="views">{figs}</div>')
            parts.append('<div class="legend"><span style="--c:var(--crimson)">raises the prediction</span><span style="--c:var(--cobalt)">lowers it</span></div>')
        phi = r["phi_P"][i]
        top = np.argsort(-np.abs(phi))[:CARD_FEATURES]
        items = [(("" if x_row[f] > 0 else "absent · ") + labels[f], float(phi[f])) for f in top if phi[f] != 0]
        side = ""
        if hard:
            j = int(r["nearest"][i])
            if j >= 0:
                side = (
                    f'<p class="muted" style="font-size:12px">Nearest labelled molecule (T = {r["T"][i]:.2f}): '
                    f'<span class="smi">{H.esc(col.hard.labelled_smiles[j])}</span> · label {col.hard.labels[j]:g}. '
                    f"Agreement between teacher and experiments: {cosine(r['phi_S'][i], r['phi_HS'][i]):+.2f}.</p>"
                )
        note = ""
        if views is not None and abs(views[1]) > 1e-3:
            note = f'<p class="faint" style="font-size:11.5px">Absent features contribute {views[1]:+.3f}; they have no atoms to sit on.</p>'
        parts.append(f'<div class="split"><div>{H.contributions(items)}</div><div>{side}{note}</div></div>')
    parts.append("</div>")
    return "".join(parts)


def _write_gallery(model, cols, cards, n_total, path):
    lede = (
        f"{n_total:,} molecules explained with a {H.esc(model.featurizer.description)} surrogate. "
        "Atom colours and bars are SHAP contributions on the model's scale (logit for probability teachers)."
    )
    body = '<div style="margin-top:16px"><input class="filter" placeholder="Filter by id or SMILES" data-target=".molcard"></div>'
    body += "".join(cards)
    page = H.page(
        "xai4chem · explanations",
        "xai4chem",
        "Molecule <em>explanations</em>",
        lede,
        body,
        script=H.FILTER_SCRIPT,
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


def _terminal_table(cols, pred, long_rows):
    from rich import box
    from rich.table import Table

    table = Table(box=box.SIMPLE_HEAD, header_style="bold blue", pad_edge=False)
    table.add_column("Molecule", no_wrap=True, max_width=18)
    for col in cols:
        table.add_column(col.name, justify="right", no_wrap=True)
        if col.hard is not None:
            table.add_column("a", justify="right", no_wrap=True)
    table.add_column("Top contributions", overflow="ellipsis", no_wrap=True, max_width=70)
    first = cols[0].name
    by_mol = {}
    for row in long_rows:
        if row["column"] == first:
            by_mol.setdefault(row["id"], []).append(row)
    for _, rec in pred.head(TERMINAL_ROWS).iterrows():
        cells = [str(rec["id"])]
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
            tops.append(f"[{'red' if v > 0 else 'blue'}]{v:+.2f}[/] {fr['label'][:28]}")
        cells.append("  ".join(tops))
        table.add_row(*cells)
    console.print(table)
    if len(pred) > TERMINAL_ROWS:
        echo(f"showing {TERMINAL_ROWS} of {len(pred):,} molecules", "info")

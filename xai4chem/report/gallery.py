"""``index.html`` for ``explain``: an overview table, one card per molecule and, for two molecules, a comparison."""

import re

import numpy as np
import pandas as pd

from xai4chem.explain.atoms import atom_attributions
from xai4chem.explain.tables import cosine
from xai4chem.report import html as H
from xai4chem.report.drawing import atom_svg, prepare

CARD_FEATURES = 7
ABSENT_FEATURES = 3
OVERVIEW_ROWS = 2000
DOMAIN_FAR = 0.3

LEGEND = (
    '<div class="legend"><span style="--c:var(--crimson)">raises the prediction</span>'
    '<span style="--c:var(--cobalt)">lowers it</span></div>'
)


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))[:80]


def _views(featurizer, col, r, i, mol):
    """The atom-map panels for one molecule: ``([(title, scores)], unmapped)``, or None for descriptors."""
    s = atom_attributions(featurizer, mol, r["phi_S"][i])
    if s is None:
        return None
    views = [("Teacher (S)", s[0])]
    if col.hard is not None:
        hs = atom_attributions(featurizer, mol, r["phi_HS"][i])[0]
        p = atom_attributions(featurizer, mol, r["phi_P"][i])[0]
        views += [("Experiments (H_S)", hs), ("Combined (P)", p), ("Disagreement (H_S − S)", hs - s[0])]
    return views, s[1]


def _atom_maps(drawn, panels):
    vmax = max(float(np.max(np.abs(v))) for _, v in panels) or 1.0
    figs = "".join(f"<figure>{atom_svg(drawn, v, vmax)}<figcaption>{H.esc(t)}</figcaption></figure>" for t, v in panels)
    return f'<div class="views">{figs}</div>{LEGEND}'


def card(featurizer, cols, res, i, mol_id, smi, mol, x_row, neighbour=None):
    """The card of one molecule: predictions, atom maps and the strongest contributions, per column."""
    labels = featurizer.labels()
    local = featurizer.molecule_labels(mol)
    drawn = prepare(mol)
    parts = [
        f'<div class="card molcard" id="m-{_safe(mol_id)}"><div class="head"><b>{H.esc(mol_id)}</b>'
        f'<span class="smi">{H.esc(smi)}</span></div>'
    ]
    for col in cols:
        r = res[col.name]
        hard = col.hard is not None
        stats = [
            H.stat("Prediction", f"{r['P'][i]:.3f}", f"{col.name} · teacher scale"),
            H.stat("S", f"{r['S'][i]:.3f}", "teacher surrogate"),
        ]
        if hard:
            stats.append(H.stat("H_S", f"{r['HS'][i]:.3f}", "experiments"))
            stats.append(H.stat("a", f"{r['a'][i]:.2f}", f"T = {r['T'][i]:.2f} to nearest labelled"))
        parts.append(f'<div class="eyebrow" style="margin-top:12px">{H.esc(col.name)}</div>{H.kpis(stats)}')
        views = _views(featurizer, col, r, i, mol)
        if views is not None:
            parts.append(_atom_maps(drawn, views[0]))
        phi = r["phi_P"][i]
        order = np.argsort(-np.abs(phi))
        if featurizer.sparse:
            # Present features first: they are what a chemist can act on. Absent ones are listed apart, briefly
            present = [f for f in order if x_row[f] > 0 and phi[f] != 0][:CARD_FEATURES]
            absent = [f for f in order if x_row[f] == 0 and phi[f] != 0][:ABSENT_FEATURES]
        else:
            # Descriptors are values, not presences: a negative logP is not an absent feature
            present, absent = [f for f in order if phi[f] != 0][:CARD_FEATURES], []
        scale = max([abs(float(phi[f])) for f in present + absent] or [1.0])
        bars = H.contributions([(local.get(f, labels[f]), float(phi[f])) for f in present], scale=scale)
        if absent:
            bars += '<div class="faint" style="font-size:11px;margin:8px 0 4px">Strongest absent features</div>'
            bars += H.contributions([(labels[f], float(phi[f])) for f in absent], scale=scale)
        side = ""
        if neighbour is not None:
            teacher = neighbour["teacher"].get(col.name, float("nan"))
            said = f" · teacher {teacher:.3f}" if np.isfinite(teacher) else ""
            warn = " S is extrapolating here." if neighbour["T"] < DOMAIN_FAR else ""
            side += (
                f'<p class="muted" style="font-size:12px">Nearest library molecule (T = {neighbour["T"]:.2f}): '
                f'<span class="smi">{H.esc(neighbour["smiles"])}</span>{said}.{warn}</p>'
            )
        if hard and int(r["nearest"][i]) >= 0:
            j = int(r["nearest"][i])
            side += (
                f'<p class="muted" style="font-size:12px">Nearest labelled molecule (T = {r["T"][i]:.2f}): '
                f'<span class="smi">{H.esc(col.hard.labelled_smiles[j])}</span> · label {col.hard.labels[j]:g}. '
                f"Agreement between teacher and experiments: {cosine(r['phi_S'][i], r['phi_HS'][i]):+.2f}.</p>"
            )
        if views is not None and abs(views[1]) > 1e-3:
            side += f'<p class="faint" style="font-size:11.5px">Absent features contribute {views[1]:+.3f}; they have no atoms to sit on.</p>'
        parts.append(f'<div class="split"><div>{bars}</div><div>{side}</div></div>')
    parts.append("</div>")
    return "".join(parts)


def comparison(featurizer, col, pair, res, mols, ids):
    """Why the first molecule scores above or below the second, as exact attribution differences."""
    a, b = (H.esc(i) for i in ids)
    stats = [
        H.stat(f"Prediction · {a}", f"{pair['P'][0]:.3f}", "teacher scale"),
        H.stat(f"Prediction · {b}", f"{pair['P'][1]:.3f}", "teacher scale"),
        H.stat("Gap", f"{pair['gap']:+.2f}", "model scale", tip="P(first) − P(second) on the model's scale"),
    ]
    if pair["a"] is not None and abs(pair["gate"]) >= 0.005:
        stats.append(H.stat("Gate term", f"{pair['gate']:+.2f}", f"a = {pair['a'][0]:.2f} vs {pair['a'][1]:.2f}"))
    if abs(pair["residual"]) > 1e-3:
        stats.append(H.stat("Unattributed", f"{pair['residual']:+.2f}", "H could not be rescaled for one molecule"))
    parts = [H.kpis(stats)]
    maps = [atom_attributions(featurizer, m, res["phi_P"][k]) for k, m in enumerate(mols)]
    if maps[0] is not None:
        drawn = [prepare(m) for m in mols]
        vmax = max(float(np.max(np.abs(s))) for s, _ in maps) or 1.0
        figs = "".join(
            f"<figure>{atom_svg(drawn[k], maps[k][0], vmax)}<figcaption>{H.esc(ids[k])}</figcaption></figure>"
            for k in range(2)
        )
        parts.append(f'<div class="views">{figs}</div>{LEGEND}')
    rows = pair["rows"]
    bars = H.contributions([(row["label"], row["diff"]) for row in rows])
    cells = [
        [
            H.esc(r["label"]),
            f"{r['a']:g}",
            f"{r['b']:g}",
            f"{r['phi_a']:+.3f}",
            f"{r['phi_b']:+.3f}",
            f"{r['diff']:+.3f}",
        ]
        for r in rows
    ]
    table = H.table(["Feature", a, b, f"φ {a}", f"φ {b}", "Difference"], cells, numeric=(1, 2, 3, 4, 5))
    parts.append(
        f'<div class="split"><div><div class="faint" style="font-size:11px;margin:8px 0 4px">Largest differences '
        f"(positive: favours {a})</div>{bars}</div><div>{table}</div></div>"
    )
    return H.section(
        f"{col.name} · {a} vs {b}", "".join(parts), hint="the gap, split into exact per-feature differences"
    )


def _overview(cols, pred, n_cards):
    headers = ["Molecule"]
    for col in cols:
        headers += [col.name] + (["a", "Agreement"] if col.hard is not None else [])
    rows = []
    for k, rec in pred.head(OVERVIEW_ROWS).iterrows():
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
                cells += ["—" if pd.isna(a) else f"{a:.2f}", "—" if pd.isna(g) else f"{g:+.2f}"]
        rows.append(cells)
    return H.table(headers, rows, numeric=tuple(range(1, len(headers))))


def write_gallery(path, featurizer, cols, pred, cards, comparisons=()):
    """Write the gallery page: comparisons (if any), the overview table, then the cards."""
    lede = (
        f"{len(pred):,} molecule{'' if len(pred) == 1 else 's'} explained with a {H.esc(featurizer.description)} "
        "surrogate. Atom colours and bars are SHAP contributions on the teacher's scale."
    )
    if featurizer.name == "morgan":
        lede += (
            " Morgan bits are named by the environment that sets them in each molecule (“+1” means one more distinct "
            "environment sets the same bit); absent bits keep their library-wide name."
        )
    body = "".join(comparisons)
    hint = "predictions on the teacher's scale" + (f" · first {OVERVIEW_ROWS:,}" if len(pred) > OVERVIEW_ROWS else "")
    body += H.section("Overview", _overview(cols, pred, len(cards)), hint=hint)
    body += "".join(cards)
    path.write_text(H.page("xai4chem · explanations", "xai4chem", "Molecule <em>explanations</em>", lede, body))
    return path

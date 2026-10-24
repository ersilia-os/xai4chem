"""Contrastive explanation of two molecules: why does the model score A above (or below) B?

Attributions are exact and additive, so the gap between two predictions decomposes exactly:

    P(A) − P(B) = [base(A) − base(B)] + Σ_f [φ_P(A)_f − φ_P(B)_f]

The base values differ only through the gate (a depends on each molecule's similarity to the labelled set),
so without a hard head the gap is entirely the sum of per-feature differences. Those differences are what
a medicinal chemist reads off a matched pair: what A has that B lacks, and the reverse.
"""

import time
from pathlib import Path

import numpy as np
from rdkit import Chem
from rich.markup import escape

from xai4chem.console import STEP_COLORS, console, echo, elapsed, rule, set_active_color, summary_panel
from xai4chem.console import path as cpath
from xai4chem.reporting import html as H
from xai4chem.reporting.drawing import atom_attributions, atom_svg, prepare
from xai4chem.soft import inverse

TOP = 12


def _label(featurizer, labels, local_a, local_b, f):
    return local_a.get(f) or local_b.get(f) or labels[f]


def compare_molecules(model, smiles_a, smiles_b, output, columns=None, names=("A", "B"), quiet=False):
    """Write an HTML comparison of two molecules and return ``{column: (P_a, P_b, differences)}``."""
    started = time.time()
    color = STEP_COLORS["explain"]
    set_active_color(color)
    mols = [Chem.MolFromSmiles(s) for s in (smiles_a, smiles_b)]
    for name, smi, mol in zip(names, (smiles_a, smiles_b), mols):
        if mol is None:
            raise ValueError(f"Molecule {name} ({smi}) could not be parsed.")
    cols = model._select(columns)
    if not quiet:
        rule("xai4chem · compare", style=color, right=f"{escape(names[0])} vs {escape(names[1])}")
    X = model.featurize([smiles_a, smiles_b])
    res = model.explain([smiles_a, smiles_b], X=X, columns=[c.name for c in cols])
    labels = model.featurizer.labels()
    local = [model.featurizer.molecule_labels(m) for m in mols]
    drawn = [prepare(m) for m in mols]
    sections, out = [], {}
    for col in cols:
        r = res[col.name]
        d = r["phi_P"][0] - r["phi_P"][1]
        gate_term = float(r["base_P"][0] - r["base_P"][1])
        gap = float(r["P"][0] - r["P"][1])
        order = [f for f in np.argsort(-np.abs(d)) if d[f] != 0][:TOP]
        rows = []
        for f in order:
            rows.append(
                {
                    "label": _label(model.featurizer, labels, local[0], local[1], f),
                    "a": float(X[0, f]),
                    "b": float(X[1, f]),
                    "phi_a": float(r["phi_P"][0, f]),
                    "phi_b": float(r["phi_P"][1, f]),
                    "diff": float(d[f]),
                }
            )
        prob = inverse(r["P"], col.transform)
        out[col.name] = (float(prob[0]), float(prob[1]), rows)
        sections.append(_section(model, col, r, rows, prob, gap, gate_term, drawn, mols, names))
        if not quiet:
            _terminal(col, prob, gap, gate_term, rows, names)
    page = H.page(
        "xai4chem · compare",
        "xai4chem",
        f"{H.esc(names[0])} vs {H.esc(names[1])} · <em>compare</em>",
        f"<span class='smi'>{H.esc(smiles_a)}</span> against <span class='smi'>{H.esc(smiles_b)}</span>. "
        "The gap between the two predictions splits exactly into per-feature differences of their attributions "
        "(model scale; logit for probability teachers).",
        "".join(sections),
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page)
    if not quiet:
        summary_panel(
            "xai4chem · compare",
            [("Page", f"[dim]{cpath(output)}[/]"), ("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]")],
            border_style=color,
            icon="✓",
        )
    return out


def _section(model, col, r, rows, prob, gap, gate_term, drawn, mols, names):
    stats = [
        H.stat(f"Prediction · {names[0]}", f"{prob[0]:.3f}", "teacher scale"),
        H.stat(f"Prediction · {names[1]}", f"{prob[1]:.3f}", "teacher scale"),
        H.stat("Gap", f"{gap:+.2f}", "model scale", tip="P(A) − P(B) on the model's scale"),
    ]
    if col.hard is not None:
        stats.append(H.stat("Gate term", f"{gate_term:+.2f}", f"a = {r['a'][0]:.2f} vs {r['a'][1]:.2f}"))
    parts = [H.kpis(stats)]
    maps = [atom_attributions(model.featurizer, m, r["phi_P"][i]) for i, m in enumerate(mols)]
    if maps[0] is not None:
        vmax = max(float(np.max(np.abs(s))) for s, _ in maps) or 1.0
        figs = "".join(
            f"<figure>{atom_svg(drawn[i], maps[i][0], vmax)}<figcaption>{H.esc(names[i])}</figcaption></figure>"
            for i in range(2)
        )
        parts.append(f'<div class="views">{figs}</div>')
        parts.append(
            '<div class="legend"><span style="--c:var(--crimson)">raises the prediction</span>'
            '<span style="--c:var(--cobalt)">lowers it</span></div>'
        )
    bars = H.contributions([(row["label"], row["diff"]) for row in rows])
    table_rows = [
        [
            H.esc(row["label"]),
            f"{row['a']:g}",
            f"{row['b']:g}",
            f"{row['phi_a']:+.3f}",
            f"{row['phi_b']:+.3f}",
            f"{row['diff']:+.3f}",
        ]
        for row in rows
    ]
    table = H.table(
        ["Feature", names[0], names[1], f"φ {names[0]}", f"φ {names[1]}", "Difference"],
        table_rows,
        numeric=(1, 2, 3, 4, 5),
    )
    parts.append(
        f'<div class="split"><div><div class="faint" style="font-size:11px;margin:8px 0 4px">Largest differences '
        f"(positive: favours {H.esc(names[0])})</div>{bars}</div><div>{table}</div></div>"
    )
    return H.section(col.name, "".join(parts), hint="differences of exact attributions")


def _terminal(col, prob, gap, gate_term, rows, names):
    from rich import box
    from rich.table import Table

    echo(
        f"[bold]{escape(col.name)}[/] · {escape(names[0])} {prob[0]:.3f} vs {escape(names[1])} {prob[1]:.3f} "
        f"· gap {gap:+.2f} (model scale)" + (f" · gate term {gate_term:+.2f}" if abs(gate_term) > 1e-9 else ""),
        "run",
    )
    table = Table(box=box.SIMPLE_HEAD, header_style="bold blue", pad_edge=False)
    table.add_column("Feature", overflow="ellipsis", ratio=1)
    for name in (f"φ {names[0]}", f"φ {names[1]}", "Difference"):
        table.add_column(escape(name), justify="right", no_wrap=True)
    for row in rows[:8]:
        d = row["diff"]
        table.add_row(
            escape(row["label"]),
            f"{row['phi_a']:+.3f}",
            f"{row['phi_b']:+.3f}",
            f"[{'red' if d > 0 else 'blue'}]{d:+.3f}[/]",
        )
    console.print(table)

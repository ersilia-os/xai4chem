"""Terminal tables for what the commands return: the explain overview and the two-molecule comparison."""

import pandas as pd
from rich import box
from rich.markup import escape
from rich.table import Table

from xai4chem.ui.console import console, echo

TERMINAL_ROWS = 15


def _table():
    return Table(box=box.SIMPLE_HEAD, header_style="bold blue", pad_edge=False, expand=True)


def explanations(cols, pred, attributions, sparse=True):
    """One row per molecule: its predictions and the three strongest contributions to the first column."""
    table = _table()
    # Fixed widths for the numbers; the contributions take what is left and are cut, never the predictions
    table.add_column("Molecule", no_wrap=True, width=14, overflow="ellipsis")
    for col in cols:
        table.add_column(
            escape(col.name), justify="right", no_wrap=True, width=min(max(6, len(col.name)), 12), overflow="ellipsis"
        )
        if col.hard is not None:
            table.add_column("a", justify="right", no_wrap=True, width=4)
    table.add_column("Top contributions", overflow="ellipsis", no_wrap=True, ratio=1)
    by_mol = {}
    if len(attributions):
        first = attributions[attributions["column"] == cols[0].name]
        phi = first["phi_P"] if "phi_P" in first else first["phi_S"]
        first = first.assign(phi=phi.to_numpy(), size=phi.abs().to_numpy()).sort_values("size", ascending=False)
        by_mol = {k: g.head(3) for k, g in first.groupby("id", sort=False)}
    for _, rec in pred.head(TERMINAL_ROWS).iterrows():
        cells = [escape(str(rec["id"]))]
        for col in cols:
            v = rec[col.name]
            cells.append("—" if pd.isna(v) else f"{v:.3f}")
            if col.hard is not None:
                a = rec[f"{col.name}_a"]
                cells.append("—" if pd.isna(a) else f"{a:.2f}")
        tops = []
        if rec["id"] in by_mol:
            for _, fr in by_mol[rec["id"]].iterrows():
                absent = "[dim]absent[/] " if sparse and fr["value"] == 0 else ""
                tops.append(
                    f"[{'red' if fr['phi'] > 0 else 'blue'}]{fr['phi']:+.2f}[/] {absent}{escape(str(fr['label'])[:28])}"
                )
        cells.append("  ".join(tops))
        table.add_row(*cells)
    console.print(table)
    if len(pred) > TERMINAL_ROWS:
        echo(
            f"showing {TERMINAL_ROWS} of {len(pred):,} molecules; all of them are in the tables and the gallery", "info"
        )


def comparison(pair, ids):
    """The gap between two molecules and its largest per-feature differences."""
    a, b = (escape(str(i)) for i in ids)
    gate = f" · gate term {pair['gate']:+.2f}" if abs(pair["gate"]) >= 0.005 else ""
    echo(
        f"[bold]{escape(pair['column'])}[/] · {a} {pair['prob'][0]:.3f} vs {b} {pair['prob'][1]:.3f} · "
        f"gap {pair['gap']:+.2f} on the model's scale{gate}",
        "run",
    )
    table = _table()
    table.add_column("Feature", overflow="ellipsis", no_wrap=True, ratio=1)
    for name in (f"φ {a}", f"φ {b}", "Difference"):
        table.add_column(name, justify="right", no_wrap=True, width=max(10, len(name)))
    for row in pair["rows"][:8]:
        d = row["diff"]
        table.add_row(
            escape(row["label"]),
            f"{row['phi_a']:+.3f}",
            f"{row['phi_b']:+.3f}",
            f"[{'red' if d > 0 else 'blue'}]{d:+.3f}[/]",
        )
    console.print(table)

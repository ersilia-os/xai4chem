"""Score a model against held-out soft labels (fidelity to the teacher) and, optionally, hard labels."""

import time

import numpy as np
from scipy.stats import spearmanr

from xai4chem.console import STEP_COLORS, console, echo, elapsed, rule, set_active_color, summary_panel
from xai4chem.console import path as cpath
from xai4chem.io import match_hard_columns, read_labels
from xai4chem.model import XAIModel, valid_mask
from xai4chem.run import write_json
from xai4chem.soft import fidelity, forward


def _hard_metrics(y, pred):
    from sklearn.metrics import average_precision_score, roc_auc_score

    if np.isin(y, [0, 1]).all() and len(np.unique(y)) == 2:
        return {
            "auroc": float(roc_auc_score(y, pred)),
            "auprc": float(average_precision_score(y, pred)),
            "n": int(len(y)),
        }
    return {"spearman": float(spearmanr(y, pred).statistic), "n": int(len(y))}


def validate(model_dir, soft_labels=None, hard_labels=None, output=None):
    from rich import box
    from rich.table import Table

    started = time.time()
    color = STEP_COLORS["validate"]
    set_active_color(color)
    model = XAIModel(model_dir)
    rule("xai4chem · validate", style=color, right=cpath(model_dir))
    report = {}
    table = Table(box=box.SIMPLE_HEAD, header_style=f"bold {color}", pad_edge=False)
    for name, justify in [
        ("Column", "left"),
        ("Labels", "left"),
        ("n", "right"),
        ("Metric", "left"),
        ("S", "right"),
        ("P", "right"),
    ]:
        table.add_column(name, justify=justify, no_wrap=True)
    for kind, path in (("soft", soft_labels), ("hard", hard_labels)):
        if path is None:
            continue
        smiles, Y = read_labels(path)
        if kind == "soft":
            mapping = {c: c for c in Y.columns if c in model.column_names}
        else:
            mapping = match_hard_columns(model.column_names, list(Y.columns), strict=False)
        unknown = [c for c in Y.columns if c not in mapping]
        if unknown:
            echo(f"skipping columns the model does not have: {unknown}", "warning")
        echo(f"scoring {len(smiles):,} molecules from [dim]{cpath(path)}[/]", "run")
        res = model.scores(smiles, columns=sorted(set(mapping.values())))
        valid = valid_mask(smiles)
        if (~valid).any():
            echo(f"{int((~valid).sum()):,} unparseable SMILES are left out of the metrics", "warning")
        for src, name in mapping.items():
            col = model.column(name)
            y = Y[src].to_numpy(dtype=np.float64)
            m = np.isfinite(y) & valid
            entry = report.setdefault(name, {})
            for out in ("S", "P"):
                pred = res[name][out][m]
                entry[f"{kind}_{out}"] = (
                    fidelity(forward(y[m], col.transform), pred) if kind == "soft" else _hard_metrics(y[m], pred)
                )
            keys = (
                ["r2", "spearman"]
                if kind == "soft"
                else (["auroc", "auprc"] if "auroc" in entry[f"{kind}_S"] else ["spearman"])
            )
            for k in keys:
                table.add_row(
                    name, kind, f"{int(m.sum()):,}", k, f"{entry[f'{kind}_S'][k]:.3f}", f"{entry[f'{kind}_P'][k]:.3f}"
                )
    console.print(table)
    if output is not None:
        write_json(output, report)
    rows = [("Columns", ", ".join(report))]
    if output:
        rows.append(("Metrics", f"[dim]{cpath(output)}[/]"))
    rows.append(("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]"))
    summary_panel("xai4chem · validate", rows, border_style=color, icon="✓")
    return report

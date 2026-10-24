"""``xai4chem info``: a model card in the terminal."""

from pathlib import Path

from xai4chem.console import STEP_COLORS, console, detail, echo, rule, set_active_color
from xai4chem.console import path as cpath
from xai4chem.run import EXPLAIN_DIRNAME, SOFT_META_NAME, column_dir, hard_dir, has_hard_head, read_json, read_manifest


def show_info(model_dir):
    from rich import box
    from rich.table import Table

    color = STEP_COLORS["info"]
    set_active_color(color)
    model_dir = Path(model_dir)
    m = read_manifest(model_dir)
    rule(f"xai4chem · {model_dir.name}", style=color, right=cpath(model_dir))
    detail(
        [
            ("featurizer", m["featurizer"]["name"]),
            ("library", f"{m['library']['name']} · {m['library']['n_molecules']:,} molecules"),
            ("soft labels", cpath(m.get("soft_labels", "—"))),
            ("xai4chem", m["xai4chem_version"]),
        ]
    )
    table = Table(box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False, title="Columns", title_justify="left")
    for name, justify in [
        ("Column", "left"),
        ("Scale", "left"),
        ("R²", "right"),
        ("ρ", "right"),
        ("Trees", "right"),
        ("Hard head", "left"),
        ("a_max", "right"),
    ]:
        table.add_column(name, justify=justify, no_wrap=True)
    tops = []
    for col in m["columns"]:
        cdir = column_dir(model_dir, col["id"])
        meta = read_json(cdir / SOFT_META_NAME)
        hard, a_max = "—", "—"
        if has_hard_head(model_dir, col["id"]):
            h = read_json(hard_dir(model_dir, col["id"]) / "hard_meta.json")
            cv = h["cv"]
            metric = f"AUROC {cv['auroc']:.3f}" if "auroc" in cv else f"ρ {cv['spearman']:.3f}"
            hard = f"{h['task']} · {h['n_labels']:,} labels · CV {metric}"
            a_max = f"{h['a_max']:.2f}"
        table.add_row(
            col["name"],
            col["transform"],
            f"{meta['fidelity']['r2']:.3f}",
            f"{meta['fidelity']['spearman']:.3f}",
            f"{meta['n_trees']:,}",
            hard,
            a_max,
        )
        summary = cdir / EXPLAIN_DIRNAME / "summary.json"
        if summary.exists():
            tops.append((col["name"], read_json(summary)["top_features"][:8]))
    console.print()
    console.print(table)
    for name, feats in tops:
        t = Table(
            box=box.SIMPLE_HEAD,
            header_style="bold",
            pad_edge=False,
            title=f"Top features · {name}",
            title_justify="left",
        )
        t.add_column("Feature", overflow="ellipsis", max_width=60)
        t.add_column("Importance", justify="right")
        t.add_column("Direction", justify="left")
        for f in feats:
            d = f["direction"]
            t.add_row(
                f["label"],
                f"{f['mean_abs_shap']:.3f}",
                "[red]raises[/]" if d > 0.05 else "[blue]lowers[/]" if d < -0.05 else "[dim]mixed[/]",
            )
        console.print(t)
    if not tops:
        echo(f"no global explanations yet · run [bold]xai4chem explain-global -m {cpath(model_dir)}[/]", "info")
    report = model_dir / "report.html"
    if report.exists():
        echo(f"report · [dim]{cpath(report)}[/]", "info")

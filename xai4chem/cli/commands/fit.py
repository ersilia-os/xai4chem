import rich_click as click

from xai4chem.cli import options


@click.command("fit")
@click.option(
    "--soft-labels",
    "-s",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="The teacher's outputs over the library: a `smiles` column plus one numeric column per output.",
)
@click.option(
    "--hard-labels",
    "-h",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Optional measurements: a `smiles` column plus columns named like the soft-label columns.",
)
@options.model_dir("Where to write the model.")
@click.option(
    "--featurizer",
    "-f",
    default="datamol",
    show_default=True,
    type=click.Choice(options.FEATURIZERS),
    help="Features the surrogate learns from: descriptors, substructures (atom maps) or functional groups.",
)
@click.option(
    "--hard-columns",
    default=None,
    help="Comma-separated columns of the hard-label file to use (default: every numeric column after `smiles`). "
    "Each must match a soft-label column.",
)
@click.option(
    "--task",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "binary", "regression"]),
    help="Task of the hard labels: binary if every label is 0 or 1, else regression. Forcing binary on other values is refused.",
)
@click.option(
    "--reweight",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "on", "off", "kde", "bins"]),
    help="Inverse-density weights for S. `auto` weights only a skewed target; `on`/`off` force it; `kde`/`bins` pick the method.",
)
@click.option(
    "--n-estimators", default=None, type=click.IntRange(min=1), help="Most boosting rounds for S (default 2000)."
)
@click.option(
    "--panel-rows",
    default=None,
    type=click.IntRange(min=1),
    help="Library molecules explained globally (default 2000).",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    type=int,
    help="Seed for XGBoost, the hard head's split and the golden molecules.",
)
@click.option("--no-report", is_flag=True, help="Skip report.html.")
@options.library_dir()
@click.option("--force", is_flag=True, help="Overwrite an existing model.")
@options.friendly_errors
def fit_cmd(
    soft_labels,
    hard_labels,
    model_dir,
    featurizer,
    hard_columns,
    task,
    reweight,
    n_estimators,
    panel_rows,
    seed,
    no_report,
    library_dir,
    force,
):
    """Learn an explainable surrogate of a teacher model, and refine it with measurements if given.

    Writes the model directory, with global explanations and `report.html`.
    """
    from xai4chem import api
    from xai4chem.models.hard import metric_text

    display, started = options.begin("fit", f"{soft_labels}" + (f" + {hard_labels}" if hard_labels else ""))
    out = api.fit(
        soft_labels,
        model_dir,
        hard_labels,
        featurizer,
        library_dir,
        force=force,
        seed=seed,
        hard_columns=[c.strip() for c in hard_columns.split(",") if c.strip()] if hard_columns else None,
        task=task,
        reweight=reweight,
        n_estimators=n_estimators,
        panel_rows=panel_rows,
        report=not no_report,
        display=display,
    )
    rows = []
    for s in out["columns"]:
        line = f"R² {s['fidelity']['r2']:.3f} · ρ {s['fidelity']['spearman']:.3f}"
        if s.get("hard"):
            h = s["hard"]
            line += f" · H {metric_text(h['heldout'])} · a_max {h['a_max']:.2f} · reach {h['reach']:.0%}"
        rows.append((s["name"], line))
    rows.append(("Model", options.shown(out["model_dir"])))
    if out["report"] is not None:
        rows.append(("Report", options.shown(out["report"])))
    rows.append(("Next", f"xai4chem explain -m {model_dir} -i molecules.csv"))
    options.finish("fit", rows, started)

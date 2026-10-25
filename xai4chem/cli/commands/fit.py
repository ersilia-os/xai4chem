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
    default="morgan",
    show_default=True,
    type=click.Choice(options.FEATURIZERS),
    help="Features the surrogate learns from (must be prepared by `xai4chem setup`).",
)
@options.library_dir()
@click.option("--force", is_flag=True, help="Overwrite an existing model.")
@options.friendly_errors
def fit_cmd(soft_labels, hard_labels, model_dir, featurizer, library_dir, force):
    """Learn an explainable surrogate of a teacher model, and refine it with measurements if given.

    Writes the model directory, with global explanations and `report.html`.
    """
    from xai4chem import api
    from xai4chem.models.hard import metric_text

    display, started = options.begin("fit", f"{soft_labels}" + (f" + {hard_labels}" if hard_labels else ""))
    out = api.fit(soft_labels, model_dir, hard_labels, featurizer, library_dir, force=force, display=display)
    rows = []
    for s in out["columns"]:
        line = f"R² {s['fidelity']['r2']:.3f} · ρ {s['fidelity']['spearman']:.3f}"
        if s.get("hard"):
            h = s["hard"]
            line += f" · H {metric_text(h['heldout'])} · a_max {h['a_max']:.2f} · reach {h['reach']:.0%}"
        rows.append((s["name"], line))
    rows += [("Model", options.shown(out["model_dir"])), ("Report", options.shown(out["report"]))]
    rows.append(("Next", f"xai4chem explain -m {model_dir} -i molecules.csv"))
    options.finish("fit", rows, started)

import rich_click as click

from xai4chem.cli import options


@click.command("explain")
@options.model_dir()
@options.molecules_input("explain")
@click.option(
    "--output-dir",
    "-o",
    default="explanations",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Where to write predictions.csv, attributions.csv and the index.html gallery.",
)
@click.option(
    "--domain",
    is_flag=True,
    help="Also find each molecule's nearest library molecule and the teacher's value for it (loads the library).",
)
@options.friendly_errors
def explain_cmd(model_dir, input_file, smiles_list, output_dir, domain):
    """Predict and explain molecules: which atoms and features push each prediction, and how.

    With a hard head, each molecule gets the teacher's view, the experiments' view, the combined view and
    where they disagree. With exactly two molecules, the gap between them is split feature by feature.
    """
    from xai4chem import api
    from xai4chem.models.model import XAIModel
    from xai4chem.ui import render

    smiles, ids = options.load_molecules(input_file, smiles_list)
    display, started = options.begin("explain", f"{options.plural(len(smiles), 'molecule')} · {model_dir}")
    model = XAIModel(model_dir)
    out = api.explain(model, smiles, output_dir, ids=ids, domain=domain, display=display)
    render.explanations(model.columns, out["predictions"], out["attributions"], sparse=model.sparse)
    for pair in out["comparisons"]:
        render.comparison(pair, out["predictions"]["id"].tolist())
    if out["domain"] is not None:
        d = out["domain"]
        far = f" · {d['far']} below 0.3, where S extrapolates" if d["far"] else ""
        display.note(
            f"median Tanimoto to the nearest library molecule {d['median_T']:.2f}{far}",
            "warning" if d["far"] else "info",
        )
    rows = [
        ("Molecules", f"{len(smiles):,}"),
        ("Columns", ", ".join(model.column_names)),
        ("Tables", options.shown(f"{output_dir}/predictions.csv") + " · attributions.csv"),
        (
            "Gallery",
            options.shown(out["gallery"])
            + (f" · first {out['cards']} molecules" if out["cards"] < len(smiles) else ""),
        ),
    ]
    options.finish("explain", rows, started)

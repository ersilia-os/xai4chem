import rich_click as click

from xai4chem.cli import options


@click.command("predict")
@options.model_dir()
@options.molecules_input("score")
@click.option(
    "--output", "-o", default="predictions.csv", show_default=True, type=click.Path(dir_okay=False), help="Output CSV."
)
@click.option("--details", is_flag=True, help="Also give S and, with a hard head, H_S, T and `a` per column.")
@options.friendly_errors
def predict_cmd(model_dir, input_file, smiles_list, output, details):
    """Score molecules fast, without explanations (millions of molecules are fine)."""
    from xai4chem import api
    from xai4chem.models.model import XAIModel

    smiles, ids = options.load_molecules(input_file, smiles_list)
    display, started = options.begin("predict", f"{options.plural(len(smiles), 'molecule')} · {model_dir}")
    out = api.predict(XAIModel(model_dir), smiles, output, details=details, ids=ids, display=display)
    bad = int((~out["valid"]).sum())
    rows = [
        ("Molecules", f"{len(smiles):,}" + (f" · {bad:,} unparseable" if bad else "")),
        ("Columns", ", ".join(c for c in out.columns if c not in ("id", "smiles", "valid"))),
        ("Output", options.shown(output)),
    ]
    options.finish("predict", rows, started)

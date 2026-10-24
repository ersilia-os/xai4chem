import rich_click as click

from xai4chem.cli import options


@click.command("compare")
@options.model_dir()
@click.option(
    "--smiles", "-S", "smiles", multiple=True, required=True, help="The two molecules to compare (pass -S twice)."
)
@click.option("--names", default=None, help="Comma-separated names for the two molecules (default: A,B).")
@options.columns()
@click.option(
    "--output",
    "-o",
    default="compare.html",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="HTML page to write.",
)
@click.option("--open", "open_browser", is_flag=True, help="Open the page in a browser.")
@options.friendly_errors
def compare_cmd(model_dir, smiles, names, columns, output, open_browser):
    """Explain why the model scores one molecule above another (a contrastive, matched-pair view).

    The gap between the two predictions splits exactly into per-feature differences of their attributions.
    """
    if len(smiles) != 2:
        raise click.ClickException(f"Pass exactly two molecules with -S (got {len(smiles)}).")
    labels = options.split(names) or ["A", "B"]
    if len(labels) != 2:
        raise click.ClickException("--names takes two comma-separated names.")
    from xai4chem.model import XAIModel
    from xai4chem.reporting.compare import compare_molecules

    compare_molecules(XAIModel(model_dir), smiles[0], smiles[1], output, columns=options.split(columns), names=labels)
    if open_browser:
        import webbrowser
        from pathlib import Path

        webbrowser.open(Path(output).resolve().as_uri())

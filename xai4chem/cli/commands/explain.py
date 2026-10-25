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
    help="Where to write predictions, attributions and the gallery.",
)
@options.columns()
@options.smiles_column()
@click.option("--id-column", default=None, help="Column naming each molecule (default: mol_0000, mol_0001, …).")
@click.option("--png", is_flag=True, help="Also write atom-highlight PNGs per molecule.")
@click.option("--gallery/--no-gallery", default=True, show_default=True, help="Write the HTML gallery.")
@click.option(
    "--cards", default=100, show_default=True, type=int, help="Molecules drawn in the gallery (each adds ~90 kB)."
)
@click.option(
    "--domain/--no-domain",
    default=None,
    help="Find each molecule's nearest library molecule and the teacher's value for it (default: on for up to "
    "2,000 molecules; costs ~10 s to load the library fingerprints).",
)
@click.option("--open", "open_browser", is_flag=True, help="Open the gallery in a browser when done.")
@options.friendly_errors
def explain_cmd(
    model_dir,
    input_file,
    smiles_list,
    output_dir,
    columns,
    smiles_column,
    id_column,
    png,
    gallery,
    cards,
    domain,
    open_browser,
):
    """Predict and explain individual molecules: which atoms and features push each prediction, and how.

    With a hard head, every molecule gets three views (teacher, experiments, combined) plus a map of
    where the experiments disagree with the teacher, the blend weight a and the nearest labelled molecule.
    """
    from xai4chem.model import XAIModel
    from xai4chem.reporting.molecules import explain_molecules

    smiles, ids = options.load_molecules(input_file, smiles_list, smiles_column, id_column)
    explain_molecules(
        XAIModel(model_dir),
        smiles,
        output_dir,
        ids=ids,
        columns=options.split(columns),
        png=png,
        gallery=gallery,
        domain=domain,
        max_cards=cards,
    )
    if open_browser and gallery:
        import webbrowser
        from pathlib import Path

        webbrowser.open((Path(output_dir) / "index.html").resolve().as_uri())

import rich_click as click

from xai4chem.cli import options


@click.command("explain")
@options.model_dir()
@click.option(
    "--input",
    "-i",
    "input_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Molecules to explain: a CSV/TSV/Parquet with a SMILES column.",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(file_okay=False),
    help="Where to write predictions, attributions and the gallery.",
)
@options.columns()
@options.smiles_column()
@click.option("--id-column", default=None, help="Column naming each molecule (default: mol_0000, mol_0001, …).")
@click.option("--png", is_flag=True, help="Also write atom-highlight PNGs per molecule.")
@click.option("--gallery/--no-gallery", default=True, show_default=True, help="Write the HTML gallery.")
@click.option("--open", "open_browser", is_flag=True, help="Open the gallery in a browser when done.")
@options.friendly_errors
def explain_cmd(model_dir, input_file, output_dir, columns, smiles_column, id_column, png, gallery, open_browser):
    """Predict and explain individual molecules: which atoms and features push each prediction, and how.

    With a hard head, every molecule gets three views (teacher, experiments, combined) plus a map of
    where the experiments disagree with the teacher, the blend weight a and the nearest labelled molecule.
    """
    from xai4chem.io import read_table
    from xai4chem.io import smiles_column as find_smiles
    from xai4chem.model import XAIModel
    from xai4chem.reporting.molecules import explain_molecules

    df = read_table(input_file)
    smiles = df[find_smiles(df, smiles_column)].astype(str).tolist()
    ids = df[id_column].astype(str).tolist() if id_column else None
    explain_molecules(
        XAIModel(model_dir), smiles, output_dir, ids=ids, columns=options.split(columns), png=png, gallery=gallery
    )
    if open_browser and gallery:
        import webbrowser
        from pathlib import Path

        webbrowser.open((Path(output_dir) / "index.html").resolve().as_uri())

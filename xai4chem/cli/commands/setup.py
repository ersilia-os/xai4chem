import rich_click as click

from xai4chem.cli import options


@click.command("setup")
@click.option(
    "--library",
    "-l",
    "source",
    default=None,
    help="`ersilia` (default) for Ersilia's reference library of 1.36M molecules, downloaded ready-made; "
    "or a CSV of SMILES to build your own library.",
)
@click.option(
    "--featurizer",
    "-f",
    "featurizers",
    multiple=True,
    type=click.Choice(options.FEATURIZERS + ["all"]),
    default=("datamol",),
    show_default=True,
    help="Featurizers to get for the library (repeatable). Morgan is always included.",
)
@options.library_dir()
@click.option(
    "--export-smiles",
    default=None,
    type=click.Path(dir_okay=False),
    help="Also write the library's SMILES to this CSV, for your teacher model to score.",
)
@click.option("--force", is_flag=True, help="Rebuild the library from scratch.")
@options.friendly_errors
def setup_cmd(source, featurizers, library_dir, export_smiles, force):
    """Prepare the reference library: features, explanation panel and Morgan bit names. Run once.

    Ersilia's library is downloaded with every feature precomputed (about 0.5 GB with the default featurizers).
    A library of your own SMILES is featurized locally, once. Re-running only adds what is missing.
    """
    from xai4chem import api

    names = options.FEATURIZERS if "all" in featurizers else list(dict.fromkeys(["morgan", *featurizers]))
    display, started = options.begin("setup", f"{source or 'ersilia'} · {', '.join(names)}")
    library = api.setup(source, names, library_dir, force=force, export_smiles=export_smiles, display=display)
    rows = [
        ("Library", f"{library.name} · {len(library):,} molecules"),
        ("Featurizers", ", ".join(library.featurizers)),
        ("Location", options.shown(library.dir)),
    ]
    if export_smiles:
        rows.append(("SMILES", options.shown(export_smiles)))
        rows.append(("Next", f"score {export_smiles} with your teacher, then xai4chem fit -s soft.csv -m model"))
    else:
        rows.append(("Next", "xai4chem fit -s soft.csv -m model"))
    options.finish("setup", rows, started)

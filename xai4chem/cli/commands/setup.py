import rich_click as click

from xai4chem.cli import options


@click.command("setup")
@click.option(
    "--library",
    "-l",
    "source",
    default=None,
    help="`ersilia` (default) for Ersilia's reference library of 1.36M molecules, reused from Olinda or downloaded; "
    "or a CSV of SMILES to use as your own library.",
)
@click.option(
    "--featurizer",
    "-f",
    "featurizers",
    multiple=True,
    type=click.Choice(options.FEATURIZERS + ["all"]),
    default=("morgan",),
    show_default=True,
    help="Featurizers to precompute over the library (repeatable). Morgan is always included.",
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

    With Ersilia's library the Morgan counts come precomputed from Olinda's `erl0_qmfp_r3.h5`. AccFG and
    datamol features are computed locally, once (tens of minutes). Re-running only adds what is missing.
    """
    from xai4chem import api

    names = options.FEATURIZERS if "all" in featurizers else list(dict.fromkeys(["morgan", *featurizers]))
    display, started = options.begin("setup", f"{source or 'ersilia'} · {', '.join(names)}")
    library = api.setup(source, names, library_dir, force=force, display=display)
    rows = [
        ("Library", f"{library.name} · {len(library):,} molecules"),
        ("Featurizers", ", ".join(library.featurizers)),
        ("Location", options.shown(library.dir)),
    ]
    if export_smiles:
        from xai4chem.data.build import export_smiles as export

        display.note(f"writing the library's SMILES to {export_smiles}")
        export(library.dir, export_smiles)
        rows.append(("SMILES", options.shown(export_smiles)))
        rows.append(("Next", f"score {export_smiles} with your teacher, then xai4chem fit -s soft.csv -m model"))
    else:
        rows.append(("Next", "xai4chem fit -s soft.csv -m model"))
    options.finish("setup", rows, started)

import rich_click as click

from xai4chem.cli import options


@click.command("library")
@click.option("--output", "-o", required=True, type=click.Path(dir_okay=False), help="Where to write the SMILES (CSV).")
@options.library_dir()
@options.friendly_errors
def library_cmd(output, library_dir):
    """Write the reference library's SMILES, in library order, for your teacher to score.

    The teacher's outputs on this file, with a `smiles` column and one column per output, are the soft
    labels `xai4chem fit` learns from. Files made for Olinda work as they are.
    """
    import csv

    from xai4chem.console import STEP_COLORS, rule, set_active_color, success
    from xai4chem.console import path as cpath
    from xai4chem.library import ReferenceLibrary

    set_active_color(STEP_COLORS["library"])
    lib = ReferenceLibrary(library_dir)
    rule("xai4chem · library", right=f"{lib.name} · {len(lib):,} molecules")
    with open(output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles"])
        for s in lib.smiles:
            w.writerow([s])
    success(f"{len(lib):,} SMILES → [dim]{cpath(output)}[/]")

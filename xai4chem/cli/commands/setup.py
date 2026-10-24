import rich_click as click

from xai4chem.cli import options


@click.command("setup")
@click.option(
    "--library",
    "-l",
    "source",
    default="ersilia",
    show_default=True,
    help="`ersilia` for Ersilia's reference library (1.36M molecules, reused from Olinda or downloaded), "
    "or a CSV/TSV of SMILES to use as your own library.",
)
@click.option(
    "--featurizer",
    "-f",
    "featurizers",
    multiple=True,
    type=click.Choice(options.FEATURIZERS + ["all"]),
    default=("morgan",),
    show_default=True,
    help="Featurizers to precompute over the library (repeat the flag for several). Morgan is always included.",
)
@options.library_dir()
@click.option("--n-jobs", "-j", default=None, type=int, help="Worker processes (default: CPUs − 1, at most 8).")
@click.option(
    "--panel-size", default=50000, show_default=True, type=int, help="Library molecules used for global explanations."
)
@click.option(
    "--bit-dictionary-mols",
    default=100000,
    show_default=True,
    type=int,
    help="Library molecules scanned to name Morgan bits.",
)
@click.option("--force", is_flag=True, help="Recompute everything, even what is already there.")
@options.friendly_errors
def setup_cmd(source, featurizers, library_dir, n_jobs, panel_size, bit_dictionary_mols, force):
    """Prepare the reference library: feature matrices, explanation panel and Morgan bit dictionary.

    Run once. With the default Ersilia library, the Morgan counts come precomputed from Olinda's
    `erl0_qmfp_r3.h5` (reused from `~/.olinda` if present, otherwise downloaded, 2.9 GB); AccFG and
    datamol features are computed locally in parallel (~6 and ~15 minutes on 8 cores). Anything already
    present is kept, so re-running only adds what is missing.
    """
    import time

    from xai4chem.console import STEP_COLORS, elapsed, rule, set_active_color, summary_panel
    from xai4chem.console import path as cpath
    from xai4chem.library import ReferenceLibrary, default_library_dir, setup_library

    started = time.time()
    color = STEP_COLORS["setup"]
    set_active_color(color)
    names = options.FEATURIZERS if "all" in featurizers else list(dict.fromkeys(["morgan", *featurizers]))
    rule("xai4chem · setup", style=color, right=f"{source} · {', '.join(names)}")
    out = setup_library(
        source=source,
        featurizers=names,
        library_dir=library_dir,
        n_jobs=n_jobs,
        panel_size=panel_size,
        bit_dictionary_mols=bit_dictionary_mols,
        force=force,
    )
    lib = ReferenceLibrary(out or default_library_dir())
    summary_panel(
        "xai4chem · setup",
        [
            ("Library", f"{lib.name} · {len(lib):,} molecules"),
            ("Featurizers", ", ".join(lib.featurizers)),
            ("Location", f"[dim]{cpath(lib.dir)}[/]"),
            ("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]"),
        ],
        border_style=color,
        icon="✓",
    )

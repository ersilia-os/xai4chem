import rich_click as click

from xai4chem.cli import options


@click.command("explain-global")
@options.model_dir()
@options.library_dir()
@click.option(
    "--max-rows",
    default=10000,
    show_default=True,
    type=int,
    help="Panel molecules to explain (exact TreeSHAP costs a few ms per molecule); 0 uses the whole panel.",
)
@options.friendly_errors
def explain_global_cmd(model_dir, library_dir, max_rows):
    """Explain S across the library panel (and H across its labelled set): importances and figures."""
    from xai4chem.reporting.global_explain import explain_global

    explain_global(model_dir, library_dir=library_dir, max_rows=max_rows)

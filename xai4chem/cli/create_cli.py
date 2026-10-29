"""Assemble the xai4chem CLI: one group, one module per command."""

import rich_click as click

from xai4chem.cli.commands.explain import explain_cmd
from xai4chem.cli.commands.fit import fit_cmd
from xai4chem.cli.commands.predict import predict_cmd
from xai4chem.cli.commands.setup import setup_cmd
from xai4chem.cli.rendering import configure_help

COMMANDS = [setup_cmd, fit_cmd, explain_cmd, predict_cmd]


def _version():
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("xai4chem")
    except PackageNotFoundError:
        return "dev"


@click.group(name="xai4chem")
@click.option("--verbose", "-v", is_flag=True, help="Show debug diagnostics.")
@click.version_option(_version(), "--version", prog_name="xai4chem")
def cli(verbose):
    """[bold]xai4chem[/] · explainable surrogates of chemistry models, learned from soft labels.

    Score Ersilia's reference library with any [italic]teacher[/] model, distill it into a fast XGBoost
    surrogate, optionally refine it with your own measurements, and explain what it has learned: per
    molecule (atoms, substructures, functional groups) and across the library.

    Typical run: [green]xai4chem setup --export-smiles library.csv[/] → score it with your teacher →
    [green]xai4chem fit[/] → [green]xai4chem explain[/].
    """
    if verbose:
        from xai4chem.ui.log import set_verbose

        set_verbose(True)


def create_cli():
    """Configure help rendering, register the commands, and return the group."""
    configure_help()
    for command in COMMANDS:
        cli.add_command(command)
    return cli

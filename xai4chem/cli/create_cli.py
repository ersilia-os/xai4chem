"""Assemble the xai4chem CLI: one group, one module per command."""

import rich_click as click

from xai4chem.cli.commands.explain import explain_cmd
from xai4chem.cli.commands.explain_global import explain_global_cmd
from xai4chem.cli.commands.fit import fit_cmd
from xai4chem.cli.commands.info import info_cmd
from xai4chem.cli.commands.learn_hard import learn_hard_cmd
from xai4chem.cli.commands.learn_soft import learn_soft_cmd
from xai4chem.cli.commands.library import library_cmd
from xai4chem.cli.commands.predict import predict_cmd
from xai4chem.cli.commands.report import report_cmd
from xai4chem.cli.commands.setup import setup_cmd
from xai4chem.cli.commands.validate import validate_cmd
from xai4chem.cli.rendering import COMMAND_PANELS, configure_help

_COMMANDS = {
    "setup": setup_cmd,
    "library": library_cmd,
    "fit": fit_cmd,
    "learn-soft": learn_soft_cmd,
    "learn-hard": learn_hard_cmd,
    "explain-global": explain_global_cmd,
    "explain": explain_cmd,
    "predict": predict_cmd,
    "report": report_cmd,
    "validate": validate_cmd,
    "info": info_cmd,
}


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

    Typical run: [green]xai4chem setup[/] → [green]xai4chem library[/] (score it with your teacher) →
    [green]xai4chem fit[/] → [green]xai4chem explain[/].
    """
    if verbose:
        from xai4chem.log import set_verbose

        set_verbose(True)


def create_cli():
    """Configure help rendering, register every command, and return the group."""
    configure_help()
    for panel in COMMAND_PANELS:
        for name in panel["commands"]:
            cli.add_command(_COMMANDS[name])
    return cli

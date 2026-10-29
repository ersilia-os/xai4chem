"""How the help screens look: rich-click styling, as in Olinda."""

import rich_click as click
import rich_click.rich_click as rc

COMMAND_PANELS = [{"name": "Commands", "commands": ["setup", "fit", "explain", "predict"]}]


def configure_help():
    """Apply the styling and the command panel. Safe to call more than once."""
    click.rich_click.TEXT_MARKUP = "rich"
    rc.TEXT_MARKUP = "rich"
    rc.SHOW_ARGUMENTS = True
    rc.STYLE_OPTION = "bold magenta"
    rc.STYLE_COMMAND = "bold green"
    rc.STYLE_METAVAR = "italic yellow"
    rc.STYLE_SWITCH = "underline cyan"
    rc.STYLE_USAGE = "bold blue"
    rc.STYLE_OPTION_DEFAULT = "dim italic"
    # Keyed by program name; it may be invoked as `xai4chem` or through `python -m`
    rc.COMMAND_GROUPS = {"xai4chem": COMMAND_PANELS, "main": COMMAND_PANELS, "cli": COMMAND_PANELS}

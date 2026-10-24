"""How the help screens look: rich-click styling and the two command panels (as in Olinda)."""

import rich_click as click
import rich_click.rich_click as rc

COMMAND_PANELS = [
    {"name": "Main commands", "commands": ["setup", "fit", "explain", "predict", "report", "info"]},
    {"name": "Pipeline commands", "commands": ["library", "learn-soft", "learn-hard", "explain-global", "validate"]},
]


def configure_help():
    """Apply the styling and the panels. Safe to call more than once."""
    click.rich_click.TEXT_MARKUP = "rich"
    rc.TEXT_MARKUP = "rich"
    rc.SHOW_ARGUMENTS = True
    rc.STYLE_OPTION = "bold magenta"
    rc.STYLE_COMMAND = "bold green"
    rc.STYLE_METAVAR = "italic yellow"
    rc.STYLE_SWITCH = "underline cyan"
    rc.STYLE_USAGE = "bold blue"
    rc.STYLE_OPTION_DEFAULT = "dim italic"
    rc.COMMAND_GROUPS = {"xai4chem": COMMAND_PANELS, "main": COMMAND_PANELS, "cli": COMMAND_PANELS}

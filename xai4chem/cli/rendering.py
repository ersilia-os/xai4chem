"""How the help screens look: rich-click styling, as in Olinda."""

import rich_click as click
import rich_click.rich_click as rc

COMMAND_PANELS = [{"name": "Commands", "commands": ["setup", "fit", "explain", "predict"]}]

# Programme names the group may be invoked as: `xai4chem`, or through `python -m xai4chem`
PROGRAMS = ["xai4chem", "main", "cli"]
FIT_PANELS = [
    {"name": "Inputs and output", "options": ["--soft-labels", "--hard-labels", "--model-dir"]},
    {"name": "Features", "options": ["--featurizer", "--library-dir"]},
    {
        "name": "Advanced",
        "options": [
            "--task",
            "--reweight",
            "--soft-columns",
            "--hard-columns",
            "--max-samples",
            "--n-estimators",
            "--panel-rows",
            "--seed",
        ],
    },
    {"name": "Behaviour", "options": ["--no-report", "--force", "--help"]},
]


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
    rc.COMMAND_GROUPS = {program: COMMAND_PANELS for program in PROGRAMS}
    rc.OPTION_GROUPS = {f"{program} fit": FIT_PANELS for program in PROGRAMS}

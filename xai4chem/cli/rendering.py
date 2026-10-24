"""How the help screens look: rich-click styling and the two command panels (as in Olinda)."""

import rich_click as click
import rich_click.rich_click as rc

COMMAND_PANELS = [
    {"name": "Main commands", "commands": ["setup", "fit", "explain", "compare", "predict", "report", "info"]},
    {"name": "Pipeline commands", "commands": ["library", "learn-soft", "learn-hard", "explain-global", "validate"]},
]


_TRAINING = [
    "--featurizer",
    "--transform",
    "--max-rows",
    "--n-estimators",
    "--val-frac",
    "--tune",
    "--trials",
    "--stability",
    "--seed",
]
_SOFT = ["--soft-labels", "--soft-smiles-column", "--soft-label-columns"]
_HARD = ["--hard-labels", "--hard-smiles-column", "--hard-label-columns"]
OPTION_GROUPS = {
    "fit": [
        {"name": "Inputs", "options": _SOFT + _HARD},
        {
            "name": "Model",
            "options": ["--model-dir", "--library-dir", "--explain-rows", "--report", "--force", "--quick"],
        },
        {"name": "Training", "options": _TRAINING},
    ],
    "learn-soft": [
        {"name": "Inputs", "options": _SOFT + ["--model-dir", "--library-dir"]},
        {"name": "Training", "options": _TRAINING},
    ],
}


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
    # Keyed by command path; the program may be invoked as `xai4chem` or through `python -m`
    rc.OPTION_GROUPS = {
        f"{prog} {cmd}": groups for prog in ("xai4chem", "main", "cli") for cmd, groups in OPTION_GROUPS.items()
    }

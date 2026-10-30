"""The CLI and the Python API do the same: every API parameter has a command option, and the other way round."""

import inspect

import pytest

from xai4chem import api
from xai4chem.cli.create_cli import create_cli

# API parameter -> command option. ``None`` marks a parameter the command fills in itself.
MAP = {
    "setup": {
        "source": "source",
        "featurizers": "featurizers",
        "library_dir": "library_dir",
        "force": "force",
        "export_smiles": "export_smiles",
    },
    "fit": {
        "soft_labels": "soft_labels",
        "model_dir": "model_dir",
        "hard_labels": "hard_labels",
        "featurizer": "featurizer",
        "library_dir": "library_dir",
        "force": "force",
        "seed": "seed",
        "hard_columns": "hard_columns",
        "soft_columns": "soft_columns",
        "max_samples": "max_samples",
        "task": "task",
        "reweight": "reweight",
        "n_estimators": "n_estimators",
        "panel_rows": "panel_rows",
        "report": "no_report",
    },
    "explain": {
        "model": "model_dir",
        "smiles": "smiles_list",
        "output_dir": "output_dir",
        "ids": None,  # read from the input file's id column
        "domain": "domain",
    },
    "predict": {
        "model": "model_dir",
        "smiles": "smiles_list",
        "output": "output",
        "details": "details",
        "ids": None,  # read from the input file's id column
    },
}
NOT_OPTIONS = {"display"}  # the CLI passes its own console


@pytest.mark.parametrize("command", MAP)
def test_every_api_parameter_has_a_command_option(command):
    params = set(inspect.signature(getattr(api, command)).parameters) - NOT_OPTIONS
    assert params == set(MAP[command]), f"api.{command} and the parity map disagree: update both"
    options = {p.name for p in create_cli().commands[command].params}
    for param, option in MAP[command].items():
        assert option is None or option in options, f"`xai4chem {command}` has no option for api.{command}({param}=)"


@pytest.mark.parametrize("command", MAP)
def test_every_command_option_reaches_the_api(command):
    options = {p.name for p in create_cli().commands[command].params}
    covered = {o for o in MAP[command].values() if o} | {"input_file"}  # the table that fills `smiles`
    assert options <= covered, f"`xai4chem {command}` has options the API lacks: {sorted(options - covered)}"

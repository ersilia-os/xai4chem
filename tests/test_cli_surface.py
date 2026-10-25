"""The CLI must stay instant: command modules import nothing heavy at module scope (Olinda's contract)."""

import ast
import subprocess
import sys
from pathlib import Path

from click.testing import CliRunner

CLI_DIR = Path(__file__).resolve().parents[1] / "xai4chem" / "cli"
ALLOWED = ("__future__", "rich_click", "click", "xai4chem.cli")
HEAVY = ("xgboost", "rdkit", "pandas", "numpy", "matplotlib", "scipy", "h5py", "sklearn")


def test_cli_modules_import_only_light_modules():
    stdlib = set(sys.stdlib_module_names)
    offenders = []
    for path in CLI_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                root = name.split(".")[0]
                if root in stdlib or name.startswith(ALLOWED):
                    continue
                offenders.append(f"{path.relative_to(CLI_DIR)}: {name}")
    assert not offenders, offenders


def test_help_does_not_import_the_heavy_stack():
    code = (
        "import sys; from xai4chem.cli.create_cli import create_cli; create_cli(); "
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == ""


def test_help_lists_every_command():
    from xai4chem.cli.create_cli import create_cli
    from xai4chem.cli.rendering import COMMAND_PANELS

    result = CliRunner().invoke(create_cli(), ["--help"])
    assert result.exit_code == 0, result.output
    for panel in COMMAND_PANELS:
        for name in panel["commands"]:
            assert name in result.output


def test_four_commands():
    from xai4chem.cli.create_cli import create_cli

    assert sorted(create_cli().commands) == ["explain", "fit", "predict", "setup"]

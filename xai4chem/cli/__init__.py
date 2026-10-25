"""The ``xai4chem`` command line.

Only click/rich-click and the standard library are imported here and in every command module; the
heavy stack (numpy, xgboost, rdkit) is imported inside command bodies, so ``xai4chem --help``
stays instant. ``tests/test_cli_surface.py`` enforces it.
"""

from xai4chem.cli.create_cli import create_cli


def main():
    create_cli()()


cli = main

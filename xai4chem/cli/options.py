"""Options declared identically by several commands, and the error handling every command shares."""

import functools

import rich_click as click

FEATURIZERS = ["morgan", "accfg", "datamol"]


def model_dir(help_text="Model directory."):
    return click.option("--model-dir", "-m", required=True, type=click.Path(file_okay=False), help=help_text)


def library_dir():
    return click.option(
        "--library-dir",
        default=None,
        type=click.Path(file_okay=False),
        help="Reference library directory (default: the one used at training, else ~/.xai4chem/library).",
    )


def seed():
    return click.option("--seed", default=42, show_default=True, type=int, help="Random seed.")


def columns():
    return click.option("--columns", "-c", default=None, help="Comma-separated model columns to use (default: all).")


def smiles_column(flag="--smiles-column"):
    return click.option(
        flag, default=None, help="Name of the SMILES column (default: `smiles`/`input`, else the first column)."
    )


def split(value):
    if value is None:
        return None
    names = [v.strip() for v in str(value).split(",") if v.strip()]
    return names or None


def friendly_errors(func):
    """Turn the errors a user can fix (bad paths, columns, files) into a clean message, not a traceback."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
            message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
            raise click.ClickException(str(message)) from exc

    return wrapper

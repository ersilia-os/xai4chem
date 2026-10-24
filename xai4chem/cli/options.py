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


def molecules_input(verb):
    """``--input`` (a table) or ``--smiles`` (repeatable), one of the two."""

    def decorator(func):
        func = click.option(
            "--smiles", "-S", "smiles_list", multiple=True, help=f"A SMILES to {verb} (repeatable), instead of --input."
        )(func)
        func = click.option(
            "--input",
            "-i",
            "input_file",
            default=None,
            type=click.Path(exists=True, dir_okay=False),
            help=f"Molecules to {verb}: a CSV/TSV/Parquet with a SMILES column.",
        )(func)
        return func

    return decorator


def load_molecules(input_file, smiles_list, smiles_col=None, id_col=None):
    """(smiles, ids) from ``--input`` or ``--smiles``."""
    if bool(input_file) == bool(smiles_list):
        raise click.ClickException("Pass either --input or --smiles.")
    if smiles_list:
        return list(smiles_list), None
    from xai4chem.io import read_table, smiles_column

    df = read_table(input_file)
    smiles = df[smiles_column(df, smiles_col)].astype(str).tolist()
    if id_col is not None and id_col not in df.columns:
        raise click.ClickException(f"--id-column {id_col!r} not found. Columns: {list(df.columns)}")
    return smiles, (df[id_col].astype(str).tolist() if id_col else None)

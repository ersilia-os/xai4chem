"""What the four commands share: common options, reading molecules, error handling, and the rule and panel."""

import functools
import time

import rich_click as click

FEATURIZERS = ["morgan", "accfg", "datamol"]


def model_dir(help_text="Model directory."):
    """The ``--model-dir`` option shared by the commands that read a model."""
    return click.option("--model-dir", "-m", required=True, type=click.Path(file_okay=False), help=help_text)


def library_dir():
    """The ``--library-dir`` option shared by the commands that read the reference library."""
    return click.option(
        "--library-dir",
        default=None,
        type=click.Path(file_okay=False),
        help="Reference library directory (default: ~/.xai4chem/library).",
    )


def molecules_input(verb):
    """``--input`` (a table) or ``--smiles`` (repeatable), one of the two."""

    def decorator(func):
        """Add the molecule input and output options to a command."""
        func = click.option(
            "--smiles", "-S", "smiles_list", multiple=True, help=f"A SMILES to {verb} (repeatable), instead of --input."
        )(func)
        func = click.option(
            "--input",
            "-i",
            "input_file",
            default=None,
            type=click.Path(exists=True, dir_okay=False),
            help=f"Molecules to {verb}: a CSV/TSV/Parquet with a `smiles` column (and optionally an `id` column).",
        )(func)
        return func

    return decorator


def load_molecules(input_file, smiles_list):
    """``(smiles, ids)`` from ``--input`` or ``--smiles``; ids come from an id/name/key column when there is one."""
    if input_file and smiles_list:
        raise click.ClickException("Pass --input or --smiles, not both.")
    if not input_file and not smiles_list:
        raise click.ClickException("Pass the molecules with --input (a file) or --smiles (repeatable).")
    if smiles_list:
        return list(smiles_list), None
    from xai4chem.data.io import check_smiles_column, id_column, read_table, smiles_column

    df = read_table(input_file)
    sc = smiles_column(df)
    check_smiles_column(df, sc, input_file)
    ic = id_column(df, sc)
    return df[sc].astype(str).tolist(), (df[ic].astype(str).tolist() if ic is not None else None)


def friendly_errors(func):
    """Turn the errors a user can fix (bad paths, columns, files) into a clean message, not a traceback."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Turn expected failures into a one-line click error instead of a traceback."""
        try:
            return func(*args, **kwargs)
        except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
            message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
            raise click.ClickException(str(message)) from exc

    return wrapper


def plural(n, word):
    """``n`` and ``word``, pluralised, for console messages."""
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def shown(path):
    """A path for the summary panel: shortened, escaped, dimmed."""
    from rich.markup import escape

    from xai4chem.ui.console import path as short

    return f"[dim]{escape(short(path))}[/]"


def begin(command, right=None):
    """The command's rule; returns the display its steps report to."""
    from rich.markup import escape

    from xai4chem.ui.console import STEP_COLORS, ConsoleDisplay, rule

    rule(f"xai4chem · {command}", style=STEP_COLORS[command], right=escape(right) if right else None)
    return ConsoleDisplay(STEP_COLORS[command]), time.time()


def finish(command, rows, started):
    """The command's summary panel."""
    from xai4chem.ui.console import STEP_COLORS, elapsed, summary_panel

    rows = list(rows) + [("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]")]
    summary_panel(f"xai4chem · {command}", rows, border_style=STEP_COLORS[command], icon="✓")

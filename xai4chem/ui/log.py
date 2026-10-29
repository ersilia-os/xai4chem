"""The package logger: a loguru singleton on Rich, for diagnostics rather than status.

User-facing output belongs to :mod:`xai4chem.ui.console`: steps, panels, progress, summaries. This is the
other half, silent DEBUG diagnostics and genuine warnings and errors that surface. Both write through the
one shared Console, so they interleave correctly with live progress regions.

The console handler is attached at WARNING, so ``debug`` and ``info`` print nothing unless
``set_verbose(True)`` is called (the CLI's ``--verbose`` flag). Anything a user should always read goes
through :mod:`xai4chem.ui.console`, never through ``logger.info``.
"""

from loguru import logger as _loguru
from rich.logging import RichHandler

from xai4chem.ui.console import console

_handler_id = None


def _attach(level):
    global _handler_id
    if _handler_id is not None:
        _loguru.remove(_handler_id)
    _handler_id = _loguru.add(
        RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            log_time_format="%H:%M:%S",
            show_path=False,
        ),
        format="{message}",
        level=level,
    )


_loguru.remove()
_attach("WARNING")


def set_verbose(verbose=True):
    """Show DEBUG diagnostics on the console (or go back to WARNING and above)."""
    _attach("DEBUG" if verbose else "WARNING")


logger = _loguru

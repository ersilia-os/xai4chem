"""How long-running functions report progress, without depending on any console.

Computation in xai4chem never prints. Functions that take a while accept a ``display`` and call four
methods on it:

- ``step(index, total, title)``: a new stage of a command has started.
- ``note(text, kind)``: a one-line fact worth telling the user (``kind`` is info, warning or success).
- ``loop(verb, total)``: a context manager yielding ``tick(done)`` for a long loop.
- ``table(items, title, fields)``: a context manager yielding a row tracker for per-item work, with
  ``start``, ``update``, ``finish`` and ``status`` (live detail for the running row).

:class:`Display` does nothing, so Python callers get silent functions by default. The CLI passes
:class:`xai4chem.ui.console.ConsoleDisplay`, which renders the same calls in the Olinda look.
"""

from contextlib import contextmanager


class _SilentTable(object):
    def start(self, item):
        """A row begins."""
        pass

    def update(self, item, **values):
        """A row's values change."""
        pass

    def finish(self, item, ok=True, **values):
        """A row ends, successfully or not."""
        pass

    def status(self, verb, done=None, total=None, **pairs):
        """One repainting status line for a long loop."""
        pass


class Display(object):
    """A display that shows nothing (the default for Python callers)."""

    def step(self, index, total, title):
        """A new stage of a command has started."""

    def note(self, text, kind="info"):
        """A one-line message."""

    @contextmanager
    def loop(self, verb, total):
        """Track a long loop; yields ``tick(done)``."""
        yield lambda done: None

    @contextmanager
    def table(self, items, title, fields):
        """Track per-item work; yields an object with start, update, finish and status."""
        yield _SilentTable()


class Nested(Display):
    """A display for work done inside another command's step: its own steps become plain notes.

    Keeps the outer command's ``Step i/N`` numbering intact while a download or build it triggered still
    reports its progress.
    """

    def __init__(self, parent):
        self.parent = parent

    def step(self, index, total, title):
        """The nested stage, as a note under the outer step."""
        self.parent.note(title)

    def note(self, text, kind="info"):
        """Passed on unchanged."""
        self.parent.note(text, kind)

    def loop(self, verb, total):
        """Passed on unchanged."""
        return self.parent.loop(verb, total)

    def table(self, items, title, fields):
        """Passed on unchanged."""
        return self.parent.table(items, title, fields)


SILENT = Display()

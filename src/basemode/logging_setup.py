"""Opt-in file logging for interactive entry points.

A published library must not touch the filesystem or attach handlers just
because it was imported — that's for the process that actually wants
persisted logs (the CLI) to ask for explicitly.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .redaction import redact


class RedactingFormatter(logging.Formatter):
    """Format a record, then take the account identifiers back out.

    Redaction happens after formatting rather than on the message, because
    the identifiers arrive inside a provider exception's own text and reach
    the file through the traceback rather than through the format string.
    """

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def setup_file_logging() -> None:
    """Attach a rotating file handler to the `basemode` logger.

    Creates `$XDG_STATE_HOME/basemode/basemode.log` (or
    `~/.local/state/basemode/basemode.log`) if it doesn't already exist.
    Safe to call more than once — a handler is only attached the first time.
    Provider exceptions reach this file in full, so account identifiers are
    stripped on the way in; see `redaction.py`.
    """
    log_dir = (
        Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
        / "basemode"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "basemode.log"

    handler = RotatingFileHandler(log_path, maxBytes=2 * 1024 * 1024, backupCount=3)
    handler.setFormatter(
        RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    logger = logging.getLogger("basemode")
    logger.setLevel(logging.DEBUG)
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        logger.addHandler(handler)


def setup_verbose_logging() -> None:
    """Show content-free Basemode lifecycle events on stderr.

    Diagnostic warnings and exceptions continue to go to the rotating file
    handler. They are deliberately excluded here: handled provider failures
    are represented by the concise attempt lifecycle record instead of a
    traceback in the interactive stream.
    """
    logger = logging.getLogger("basemode")
    logger.setLevel(logging.INFO)
    if any(getattr(handler, "_basemode_verbose", False) for handler in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    handler.addFilter(lambda record: record.levelno == logging.INFO)
    handler.setFormatter(RedactingFormatter("[basemode] %(message)s"))
    handler._basemode_verbose = True  # type: ignore[attr-defined]
    logger.addHandler(handler)

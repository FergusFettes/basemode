"""Importing basemode must not touch the filesystem or attach a file handler.

A published library shouldn't create state directories or start writing logs
just because someone did `import basemode`; that's opt-in via
`basemode.logging_setup.setup_file_logging()`, which the CLI entry point calls.
"""

import logging
import os
import subprocess
import sys
from logging.handlers import RotatingFileHandler


def test_import_does_not_create_state_dir(tmp_path) -> None:
    state_home = tmp_path / "state"
    assert not state_home.exists()

    env = {**os.environ, "XDG_STATE_HOME": str(state_home)}
    result = subprocess.run(
        [sys.executable, "-c", "import basemode"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert not (state_home / "basemode").exists()


def test_import_attaches_only_a_null_handler() -> None:
    # Run in a subprocess: popping "basemode" from sys.modules and reimporting
    # in-process would orphan submodule attributes (basemode.transport,
    # basemode.verify, ...) that other already-imported modules set on the
    # original package object, breaking unrelated tests' monkeypatching.
    script = (
        "import logging\n"
        "from logging.handlers import RotatingFileHandler\n"
        "import basemode\n"
        "logger = logging.getLogger('basemode')\n"
        "assert not any(isinstance(h, RotatingFileHandler) for h in logger.handlers)\n"
        "assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_setup_file_logging_attaches_rotating_handler(tmp_path, monkeypatch) -> None:
    from basemode.logging_setup import setup_file_logging

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    logger = logging.getLogger("basemode")
    for handler in list(logger.handlers):
        if isinstance(handler, RotatingFileHandler):
            logger.removeHandler(handler)

    setup_file_logging()

    assert (tmp_path / "basemode" / "basemode.log").parent.exists()
    assert any(isinstance(h, RotatingFileHandler) for h in logger.handlers)

    # Calling it again must not attach a second handler.
    handler_count = len(
        [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
    )
    setup_file_logging()
    assert (
        len([h for h in logger.handlers if isinstance(h, RotatingFileHandler)])
        == handler_count
    )

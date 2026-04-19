import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _setup_logging() -> None:
    log_dir = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "basemode"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "basemode.log"

    handler = RotatingFileHandler(log_path, maxBytes=2 * 1024 * 1024, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    root = logging.getLogger("basemode")
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        root.addHandler(handler)


_setup_logging()

__all__ = [
    "GenerationParams",
    "GenerationStore",
    "Node",
    "branch_text",
    "continue_text",
    "default_db_path",
    "detect_strategy",
]


def __getattr__(name: str):
    if name in {"branch_text", "continue_text"}:
        from .continue_ import branch_text, continue_text

        return {"branch_text": branch_text, "continue_text": continue_text}[name]
    if name == "detect_strategy":
        from .detect import detect_strategy

        return detect_strategy
    if name == "GenerationParams":
        from .params import GenerationParams

        return GenerationParams
    if name in {"GenerationStore", "Node", "default_db_path"}:
        from .store import GenerationStore, Node, default_db_path

        return {
            "GenerationStore": GenerationStore,
            "Node": Node,
            "default_db_path": default_db_path,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

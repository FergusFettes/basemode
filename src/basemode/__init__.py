import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _setup_logging() -> None:
    log_dir = (
        Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
        / "basemode"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "basemode.log"

    handler = RotatingFileHandler(log_path, maxBytes=2 * 1024 * 1024, backupCount=3)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    root = logging.getLogger("basemode")
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        root.addHandler(handler)


_setup_logging()

__all__ = [
    "GenerationParams",
    "branch_text",
    "build_model_picker_state",
    "continue_text",
    "detect_strategy",
    "list_model_picker_entries",
]


def __getattr__(name: str):
    if name in {"branch_text", "continue_text"}:
        from .continue_ import branch_text, continue_text

        return {"branch_text": branch_text, "continue_text": continue_text}[name]
    if name == "detect_strategy":
        from .detect import detect_strategy

        return detect_strategy
    if name in {"list_model_picker_entries", "build_model_picker_state"}:
        from .models import build_model_picker_state, list_model_picker_entries

        return {
            "list_model_picker_entries": list_model_picker_entries,
            "build_model_picker_state": build_model_picker_state,
        }[name]
    if name == "GenerationParams":
        from .params import GenerationParams

        return GenerationParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

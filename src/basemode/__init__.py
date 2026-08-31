import importlib
import logging

logging.getLogger("basemode").addHandler(logging.NullHandler())

#: Public name -> submodule it is lazily imported from. Keeping this as data
#: (rather than a chain of if/elif branches) means the exported surface and
#: its import graph can't drift apart, and `__all__` is derived from it below.
_LAZY: dict[str, str] = {
    "branch_text": ".continue_",
    "continue_text": ".continue_",
    "detect_strategy": ".detect",
    "select_strategy": ".detect",
    "StrategyChoice": ".detect",
    "score_continuation": ".scoring",
    "ContinuationScore": ".scoring",
    "bench_model": ".bench",
    "list_model_picker_entries": ".models",
    "build_model_picker_state": ".models",
    "GenerationParams": ".params",
    "EmptyCompletionError": ".exceptions",
    "ObservationContext": ".observations",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __package__)
    return getattr(module, name)

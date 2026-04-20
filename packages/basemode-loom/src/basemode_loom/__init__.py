"""Persistent branching exploration for basemode."""

__all__ = [
    "GenerationStore",
    "LoomSession",
    "Node",
    "default_db_path",
]


def __getattr__(name: str):
    if name in {"GenerationStore", "Node", "default_db_path"}:
        from .store import GenerationStore, Node, default_db_path

        return {
            "GenerationStore": GenerationStore,
            "Node": Node,
            "default_db_path": default_db_path,
        }[name]
    if name == "LoomSession":
        from .session import LoomSession

        return LoomSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

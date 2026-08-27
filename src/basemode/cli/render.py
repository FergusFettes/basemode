"""Shared presentation helpers used across CLI commands."""

from rich.console import Console

from ..keys import RATING_DOWN, RATING_UP

console = Console()

_BRANCH_COLORS = ["green", "blue", "yellow", "magenta", "cyan"]

_SCORE_COLORS = ((0.75, "green"), (0.4, "yellow"))

_STRATEGY_SOURCES = {
    "user": "pinned locally (basemode bench --save)",
    "registry": "verified models registry",
    "heuristic": "model-name heuristic",
    "explicit": "passed explicitly",
}

_RATING_MARKS = {RATING_UP: "[green]+[/green]", RATING_DOWN: "[red]-[/red]"}
_RATING_WORDS_BY_VALUE = {RATING_UP: "thumbs up", RATING_DOWN: "thumbs down"}
_RATING_WORDS = {
    "up": RATING_UP,
    "+": RATING_UP,
    "down": RATING_DOWN,
    "-": RATING_DOWN,
    "clear": None,
    "none": None,
}


def _preview(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _format_float(value: float) -> str:
    return f"{value:.2f}"


def _score_color(score: float) -> str:
    for threshold, color in _SCORE_COLORS:
        if score >= threshold:
            return color
    return "red"


def _health_summary(observed: dict) -> str:
    rate = observed["failure_rate"]
    attempts = observed["attempts"]
    if not rate:
        return f"{attempts} attempts, no failures"
    return (
        f"{attempts} attempts, {observed['failures']} failed ({rate:.0%}); "
        f"last {observed['last_category']} at {observed['last_failure_at']}"
    )

from __future__ import annotations

from typing import ClassVar

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from ...stats import LoomStats


class StatsScreen(ModalScreen[None]):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close", show=False),
        Binding("?", "close", "Close", show=False),
    ]

    def __init__(self, stats: LoomStats) -> None:
        super().__init__()
        self._stats = stats

    def compose(self) -> ComposeResult:
        with Vertical(id="stats-panel"):
            yield Static("Tree Stats", id="stats-title")
            with VerticalScroll(id="stats-scroll"):
                yield Static(_stats_renderable(self._stats), id="stats-content")
            yield Static("Esc/q/? close", id="stats-help")

    def action_close(self) -> None:
        self.dismiss(None)


def _stats_renderable(stats: LoomStats) -> Group:
    summary = Table("Metric", "Value", show_header=False)
    summary.add_row("Root", stats.root_id)
    summary.add_row("Total nodes", str(stats.total_nodes))
    summary.add_row("Generated nodes", str(stats.generated_nodes))
    summary.add_row("Expanded nodes", str(stats.expanded_nodes))
    summary.add_row("Leaf nodes", str(stats.leaf_nodes))
    summary.add_row("Max depth", str(stats.max_depth))
    if stats.path is not None:
        summary.add_row("Path depth", str(stats.path.depth))
        summary.add_row("Path generated nodes", str(stats.path.generated_nodes))

    models = Table(
        "Model",
        "Nodes",
        "Expanded",
        "Marked",
        "Hidden",
        "Expand",
        "Mark",
        "Hide",
        "NPDS",
        "Win",
        "DS",
        "DDS",
        header_style="bold",
    )
    for model in stats.model_stats:
        models.add_row(
            model.model,
            str(model.nodes),
            str(model.expanded),
            str(model.bookmarked),
            str(model.hidden),
            _fmt(model.expansion_rate),
            _fmt(model.bookmark_rate),
            _fmt(model.hidden_rate),
            _fmt(model.normalized_peer_descendant_score.mean),
            _fmt(model.batch_win_rate.mean),
            _fmt(model.descendant_score.mean),
            _fmt(model.discounted_descendant_score.mean),
        )

    renderables = [
        summary,
        Text(""),
        Text("Model Stats", style="bold"),
        models,
    ]

    if stats.path and stats.path.models:
        path = Table("Path model", "Count", header_style="bold")
        for model, count in stats.path.models.items():
            path.add_row(model, str(count))
        renderables.extend([Text(""), Text("Path Stats", style="bold"), path])

    return Group(*renderables)


def _fmt(value: float) -> str:
    return f"{value:.2f}"

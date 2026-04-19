from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Footer

from ...store import GenerationStore
from ..widgets.tree_picker import TreePickerView


class TreePickerScreen(ModalScreen[str | None]):
    """Full-screen tree browser. Returns selected root_id or None on cancel."""

    BINDINGS = [
        Binding("j", "move_down", "Next", show=False),
        Binding("k", "move_up", "Prev", show=False),
        Binding("tab", "select", "Open"),
        Binding("enter", "select", "Open"),
        Binding("escape", "cancel", "Back"),
        Binding("q", "cancel", "Back", show=False),
    ]

    def __init__(self, store: GenerationStore, current_root_id: str) -> None:
        super().__init__()
        self._store = store
        self._current_root_id = current_root_id

    def compose(self) -> ComposeResult:
        yield TreePickerView(id="tree-list")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(TreePickerView).load(self._store, self._current_root_id)

    def action_move_down(self) -> None:
        self.query_one(TreePickerView).move(+1)

    def action_move_up(self) -> None:
        self.query_one(TreePickerView).move(-1)

    def action_select(self) -> None:
        root_id = self.query_one(TreePickerView).selected_root_id()
        self.dismiss(root_id)

    def action_cancel(self) -> None:
        self.dismiss(None)

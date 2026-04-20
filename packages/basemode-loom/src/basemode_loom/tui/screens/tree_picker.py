from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Footer

from ...store import GenerationStore
from ..widgets.tree_picker import TreePickerView


class TreePickerScreen(ModalScreen[str | None]):
    """Full-screen tree browser. Returns selected root_id or None on cancel."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("j", "move_down", "Next", show=False),
        Binding("k", "move_up", "Prev", show=False),
        Binding("tab", "select", "Open"),
        Binding("enter", "select", "Open"),
        Binding("d", "delete_selected", "Delete"),
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

    def action_delete_selected(self) -> None:
        view = self.query_one(TreePickerView)
        root_id = view.selected_root_id()
        if root_id is None:
            return

        remaining_root_ids = [
            candidate for candidate in view.root_ids() if candidate != root_id
        ]
        if root_id == self._current_root_id and not remaining_root_ids:
            self.notify(
                "Cannot delete the only open tree", severity="warning", timeout=2
            )
            return

        from ..screens.confirm import ConfirmScreen

        root = self._store.root(root_id)
        name = root.metadata.get("name") or root.id[:8]

        def after_confirm(confirmed: bool) -> None:
            if confirmed:
                self._delete_root(root_id, remaining_root_ids)

        self.app.push_screen(
            ConfirmScreen("Delete tree?", f"{name} ({root.id[:8]})"),
            after_confirm,
        )

    def _delete_root(self, root_id: str, remaining_root_ids: list[str]) -> None:
        view = self.query_one(TreePickerView)
        replacement_root_id = (
            remaining_root_ids[0] if root_id == self._current_root_id else None
        )
        self._store.delete_tree(root_id)

        if replacement_root_id is not None:
            self.dismiss(replacement_root_id)
            return

        view.load(self._store, self._current_root_id)
        self.notify("Tree deleted", timeout=2)

    def action_cancel(self) -> None:
        self.dismiss(None)

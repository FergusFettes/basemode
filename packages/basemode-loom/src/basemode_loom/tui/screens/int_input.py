from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label


class IntInputScreen(ModalScreen[int | None]):
    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "dismiss_none", "Cancel")]

    def __init__(self, label: str, current: int) -> None:
        super().__init__()
        self._label = label
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"{self._label} (current: {self._current})")
            yield Input(str(self._current), type="integer", id="value")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        try:
            self.dismiss(int(event.value))
        except ValueError:
            self.dismiss(None)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

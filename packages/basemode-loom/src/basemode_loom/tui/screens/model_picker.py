from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static


def _fuzzy_match(query: str, text: str) -> bool:
    if not query:
        return True
    it = iter(text.lower())
    return all(c in it for c in query.lower())


class ModelPickerScreen(ModalScreen[str | None]):
    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "dismiss_none", "Cancel")]

    def __init__(self, current: str) -> None:
        super().__init__()
        self._current = current
        self._all_models: list[str] = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Model  j/k=move  Enter=pick  Esc=cancel")
            yield Input(placeholder="filter...", id="search")
            yield OptionList(id="model-list")

    def on_mount(self) -> None:
        from basemode.models import list_models

        models = list_models(available_only=True) or [self._current]
        if self._current in models:
            models.insert(0, models.pop(models.index(self._current)))
        self._all_models = models
        self._update_list("")
        self.query_one("#search", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._update_list(event.value)

    def _update_list(self, query: str) -> None:
        filtered = [m for m in self._all_models if _fuzzy_match(query, m)]
        opt = self.query_one(OptionList)
        opt.clear_options()
        for m in filtered:
            opt.add_option(m)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option.prompt))

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

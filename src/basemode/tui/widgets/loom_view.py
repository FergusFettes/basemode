from __future__ import annotations

from rich.style import Style
from rich.text import Text
from textual import events
from textual.widget import Widget

from ...display import build_loom_display
from ...session import SessionState

_STYLES = {
    "normal": Style(),
    "bold": Style(bold=True),
    "dim": Style(dim=True),
}


class LoomView(Widget):
    """Renders the loom tree: parent text, selected child (bold), siblings (dim),
    and the continuation path below the selection."""

    DEFAULT_CSS = """
    LoomView {
        height: 1fr;
        overflow-y: auto;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state: SessionState | None = None

    def update_state(self, state: SessionState) -> None:
        self._state = state
        self.refresh()

    def render(self) -> Text:
        if self._state is None:
            return Text("")
        width = self.size.width or 80
        lines = build_loom_display(self._state, width)
        result = Text(no_wrap=True, overflow="fold")
        for line in lines:
            result.append(line.text + "\n", style=_STYLES[line.style])
        return result

    def on_resize(self, event: events.Resize) -> None:
        self.refresh()

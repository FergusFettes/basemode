from __future__ import annotations

from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from ...display import build_loom_display
from ...session import SessionState

_STYLES = {
    "normal": Style(),
    "bold": Style(bold=True),
    "dim": Style(dim=True),
}


class LoomView(VerticalScroll):
    """Renders the loom tree: parent text, selected child (bold), siblings (dim),
    and the continuation path below the selection."""

    DEFAULT_CSS = """
    LoomView {
        height: 1fr;
    }
    LoomView Static {
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="loom-content")

    def update_state(self, state: SessionState) -> None:
        self._state = state
        self._render_state()

    def _render_state(self) -> None:
        state = getattr(self, "_state", None)
        if state is None:
            return
        width = self._content_width()
        lines = build_loom_display(state, width)
        result = Text(no_wrap=True, overflow="fold")
        for line in lines:
            result.append(line.text + "\n", style=_STYLES[line.style])
        self.query_one("#loom-content", Static).update(result)
        self.scroll_end(animate=False)

    def _content_width(self) -> int:
        width = self.scrollable_content_region.width
        if width >= 20:
            return width
        width = self.scrollable_size.width
        if width >= 20:
            return width
        try:
            app_width = self.app.size.width
        except Exception:
            app_width = 0
        return app_width if app_width >= 20 else 80

    def on_resize(self, event: events.Resize) -> None:
        self._render_state()

from __future__ import annotations

from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from ...display import build_stream_display

_STYLES = {
    "normal": Style(),
    "bold": Style(bold=True),
    "dim": Style(dim=True),
}


class StreamView(VerticalScroll):
    """Renders live token output for one or more parallel branches."""

    DEFAULT_CSS = """
    StreamView {
        height: 1fr;
    }
    StreamView Static {
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="stream-content")

    def reset(self, n_branches: int, prefix: str) -> None:
        self._n = n_branches
        self._prefix = prefix
        self._buffers: list[list[str]] = [[] for _ in range(n_branches)]
        self._render_content()

    def add_token(self, branch_idx: int, token: str) -> None:
        if branch_idx < len(self._buffers):
            self._buffers[branch_idx].append(token)
        self._render_content()

    def _render_content(self) -> None:
        width = self._content_width()
        n = getattr(self, "_n", 1)
        prefix = getattr(self, "_prefix", "")
        buffers = getattr(self, "_buffers", [[]])
        lines = build_stream_display(prefix, buffers, width)
        result = Text(no_wrap=True, overflow="fold")
        for line in lines:
            result.append(line.text + "\n", style=_STYLES[line.style])
        chars = sum(len(t) for t in buffers[0]) if buffers else 0
        status = (
            f"Generating {n} branches\u2026  {chars} chars  Esc=cancel"
            if n > 1
            else f"Generating\u2026  {chars} chars  Esc=cancel"
        )
        result.append(f"\n{status}", style=Style(dim=True))
        self.query_one("#stream-content", Static).update(result)
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
        self._render_content()

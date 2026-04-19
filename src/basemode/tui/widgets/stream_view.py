from __future__ import annotations

from rich.style import Style
from rich.text import Text
from textual import events
from textual.widget import Widget

from ...display import build_stream_display

_STYLES = {
    "normal": Style(),
    "bold": Style(bold=True),
    "dim": Style(dim=True),
}


class StreamView(Widget):
    """Renders live token output for one or more parallel branches."""

    DEFAULT_CSS = """
    StreamView {
        height: 1fr;
        overflow-y: auto;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._n = 1
        self._prefix = ""
        self._buffers: list[list[str]] = [[]]

    def reset(self, n_branches: int, prefix: str) -> None:
        self._n = n_branches
        self._prefix = prefix
        self._buffers = [[] for _ in range(n_branches)]
        self.refresh()

    def add_token(self, branch_idx: int, token: str) -> None:
        if branch_idx < len(self._buffers):
            self._buffers[branch_idx].append(token)
        self.refresh()

    def render(self) -> Text:
        width = self.size.width or 80
        lines = build_stream_display(self._prefix, self._buffers, width)
        result = Text(no_wrap=True, overflow="fold")
        for line in lines:
            result.append(line.text + "\n", style=_STYLES[line.style])
        chars = sum(len(t) for t in self._buffers[0]) if self._buffers else 0
        if self._n == 1:
            status = f"Generating\u2026  {chars} chars  Esc=cancel"
        else:
            status = f"Generating {self._n} branches\u2026  {chars} chars  Esc=cancel"
        result.append(f"\n{status}", style=Style(dim=True))
        return result

    def on_resize(self, event: events.Resize) -> None:
        self.refresh()

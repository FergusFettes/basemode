from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import ContentSwitcher, Footer

from ...session import (
    GenerationCancelled,
    GenerationComplete,
    GenerationError,
    LoomSession,
    TokenReceived,
)
from ..widgets.loom_view import LoomView
from ..widgets.stream_view import StreamView


class LoomScreen(Screen):
    BINDINGS = [
        Binding("h", "nav_parent", "Parent", show=False),
        Binding("l", "nav_child", "Child", show=False),
        Binding("j", "nav_next", "Next", show=False),
        Binding("k", "nav_prev", "Prev", show=False),
        Binding("space", "generate", "Generate"),
        Binding("e", "edit", "Edit"),
        Binding("c", "edit_context", "Context", show=False),
        Binding("m", "pick_model", "Model"),
        Binding("w", "tokens_up", "+tok", show=False),
        Binding("s", "tokens_down", "-tok", show=False),
        Binding("t", "set_tokens", "Tokens"),
        Binding("a", "branches_down", "-n", show=False),
        Binding("d", "branches_up", "+n", show=False),
        Binding("q", "quit", "Quit"),
        Binding("escape", "cancel_or_quit", "Cancel", show=False),
    ]

    def __init__(self, session: LoomSession) -> None:
        super().__init__()
        self.session = session
        self._generating = False

    def compose(self) -> ComposeResult:
        with ContentSwitcher(initial="loom"):
            yield LoomView(id="loom")
            yield StreamView(id="stream")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(LoomView).update_state(self.session.get_state())
        self._update_subtitle()

    def _update_subtitle(self) -> None:
        s = self.session
        short_model = s.model.split("/")[-1]
        self.sub_title = (
            f"{s._current_id[:8]}  {short_model}"
            f"  tok:{s.max_tokens}  n:{s.n_branches}"
            "  hjkl=nav  spc=gen  e=edit  c=ctx  m=model  w/s=±tok  a/d=±n"
        )

    def _refresh(self) -> None:
        self.query_one(LoomView).update_state(self.session.get_state())
        self._update_subtitle()

    # --- Navigation ---

    def action_nav_child(self) -> None:
        self.session.navigate_child()
        self._refresh()

    def action_nav_parent(self) -> None:
        self.session.navigate_parent()
        self._refresh()

    def action_nav_next(self) -> None:
        self.session.select_sibling(+1)
        self._refresh()

    def action_nav_prev(self) -> None:
        self.session.select_sibling(-1)
        self._refresh()

    # --- Params ---

    def action_tokens_up(self) -> None:
        self.session.set_max_tokens(self.session.max_tokens + 50)
        self._update_subtitle()

    def action_tokens_down(self) -> None:
        self.session.set_max_tokens(self.session.max_tokens - 50)
        self._update_subtitle()

    def action_branches_up(self) -> None:
        self.session.set_n_branches(self.session.n_branches + 1)
        self._update_subtitle()

    def action_branches_down(self) -> None:
        self.session.set_n_branches(self.session.n_branches - 1)
        self._update_subtitle()

    async def action_set_tokens(self) -> None:
        from ..screens.int_input import IntInputScreen

        result = await self.app.push_screen_wait(
            IntInputScreen("Max tokens", self.session.max_tokens)
        )
        if result is not None and result > 0:
            self.session.set_max_tokens(result)
            self._update_subtitle()

    async def action_pick_model(self) -> None:
        from ..screens.model_picker import ModelPickerScreen

        result = await self.app.push_screen_wait(ModelPickerScreen(self.session.model))
        if result is not None:
            self.session.set_model(result)
            self._update_subtitle()

    # --- Edit / context ---

    async def action_edit(self) -> None:
        state = self.session.get_state()
        original = state.full_text
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(original)
            tmpfile = f.name
        with self.app.suspend():
            subprocess.run([os.environ.get("EDITOR", "vim"), tmpfile])
        edited = Path(tmpfile).read_text().rstrip("\n")
        Path(tmpfile).unlink(missing_ok=True)
        self.session.apply_edit(original, edited)
        self._refresh()

    async def action_edit_context(self) -> None:
        state = self.session.get_state()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(state.context)
            tmpfile = f.name
        with self.app.suspend():
            subprocess.run([os.environ.get("EDITOR", "vim"), tmpfile])
        new_context = Path(tmpfile).read_text().rstrip("\n")
        Path(tmpfile).unlink(missing_ok=True)
        self.session.update_context(new_context)

    # --- Quit / cancel ---

    def action_cancel_or_quit(self) -> None:
        if self._generating:
            self.session.cancel()
        else:
            self.app.exit()

    def action_quit(self) -> None:
        self.app.exit()

    # --- Generation ---

    @work(exclusive=True)
    async def action_generate(self) -> None:
        state = self.session.get_state()
        stream_view = self.query_one(StreamView)
        stream_view.reset(self.session.n_branches, state.full_text)
        self.query_one(ContentSwitcher).current = "stream"
        self._generating = True

        try:
            async for event in self.session.generate():
                match event:
                    case TokenReceived(branch_idx=idx, token=tok):
                        stream_view.add_token(idx, tok)
                    case GenerationComplete():
                        pass
                    case GenerationCancelled():
                        pass
                    case GenerationError(error=exc):
                        self.notify(str(exc), severity="error")
        finally:
            self._generating = False
            self.query_one(ContentSwitcher).current = "loom"
            self._refresh()

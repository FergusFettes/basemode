from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import ClassVar

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import ContentSwitcher, Static

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
    BINDINGS: ClassVar[list[Binding]] = [
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
        Binding("v", "toggle_tree_view", "Tree"),
        Binding("n", "toggle_model_names", "Names", show=False),
        Binding("H", "toggle_hoist", "Hoist", show=False),
        Binding("b", "toggle_bookmark", "Bookmark", show=False),
        Binding("B", "next_bookmark", "Next mark", show=False),
        Binding("tab", "open_picker", "Trees"),
        Binding("?", "open_stats", "Stats"),
        Binding("q", "quit", "Quit"),
        Binding("escape", "cancel_or_quit", "Cancel", show=False),
    ]

    def __init__(self, session: LoomSession) -> None:
        super().__init__()
        self.session = session
        self._generating = False

    def compose(self) -> ComposeResult:
        with ContentSwitcher(initial="loom", id="loom-switcher"):
            yield LoomView(id="loom")
            yield StreamView(id="stream")
        yield Static("", id="status-bar")

    def on_mount(self) -> None:
        self.query_one(LoomView).update_state(self.session.get_state())
        self._update_subtitle()

    def _update_subtitle(self) -> None:
        s = self.session
        short_model = s.model.split("/")[-1]
        info = (
            f"{s._current_id[:8]} {short_model}  "
            f"tokens:{s.max_tokens} branches:{s.n_branches}  "
            f"view:{s.view_mode}{' hoist' if s._hoisted_id else ''} "
            f"names:{'on' if s.show_model_names else 'off'}  "
            "hjkl nav  space gen  e edit  v view  b mark  tab trees  ? stats  q quit"
        )
        self.sub_title = info
        self.query_one("#status-bar", Static).update(info)

    def _refresh(self) -> None:
        self.query_one(LoomView).update_state(self.session.get_state())
        self._update_subtitle()

    # --- Navigation ---

    def action_nav_child(self) -> None:
        state = self.session.get_state()
        if not state.children:
            self.notify(
                "No continuations yet \u2014 press space to generate", timeout=2
            )
            return
        self.session.navigate_child()
        self._refresh()

    def action_nav_parent(self) -> None:
        state = self.session.get_state()
        if state.current_node.parent_id is None:
            self.notify("Already at root", timeout=1)
            return
        self.session.navigate_parent()
        self._refresh()

    def action_nav_next(self) -> None:
        state = self.session.get_state()
        if not state.children:
            return
        self.session.select_sibling(+1)
        self._refresh()

    def action_nav_prev(self) -> None:
        state = self.session.get_state()
        if not state.children:
            return
        self.session.select_sibling(-1)
        self._refresh()

    def action_toggle_tree_view(self) -> None:
        self.session.toggle_tree_view()
        self._refresh()

    def action_toggle_model_names(self) -> None:
        self.session.toggle_model_names()
        self._refresh()

    def action_toggle_hoist(self) -> None:
        self.session.toggle_hoist()
        self._refresh()

    def action_toggle_bookmark(self) -> None:
        bookmarked = self.session.toggle_bookmark()
        self.notify("Bookmarked" if bookmarked else "Bookmark removed", timeout=1)
        self._refresh()

    def action_next_bookmark(self) -> None:
        before = self.session.get_state().current_node_id
        self.session.next_bookmark()
        after = self.session.get_state().current_node_id
        if after == before:
            self.notify("No bookmarks", timeout=1)
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

    def action_set_tokens(self) -> None:
        from ..screens.int_input import IntInputScreen

        def apply(result: int | None) -> None:
            if result is not None and result > 0:
                self.session.set_max_tokens(result)
                self._update_subtitle()

        self.app.push_screen(
            IntInputScreen("Max tokens", self.session.max_tokens), apply
        )

    def action_pick_model(self) -> None:
        from ..screens.model_picker import ModelPickerScreen

        def apply(result: str | None) -> None:
            if result is not None:
                self.session.set_model(result)
                self._update_subtitle()

        self.app.push_screen(ModelPickerScreen(self.session.model), apply)

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

    # --- Tree picker ---

    def action_open_picker(self) -> None:
        if self._generating:
            return
        from ..screens.tree_picker import TreePickerScreen

        state = self.session.get_state()

        def on_selected(root_id: str | None) -> None:
            if root_id is None or root_id == state.root_id:
                return
            if self.session.store.get(state.root_id) is not None:
                self.session.save()
            self.session = LoomSession(self.session.store, root_id)
            self.app.session = self.session
            self._refresh()

        self.app.push_screen(
            TreePickerScreen(self.session.store, state.root_id), on_selected
        )

    # --- Stats ---

    def action_open_stats(self) -> None:
        if self._generating:
            return
        from ...stats import analyze_tree
        from ..screens.stats import StatsScreen

        state = self.session.get_state()
        stats = analyze_tree(
            self.session.store,
            state.root_id,
            path_node_id=state.current_node_id,
        )
        self.app.push_screen(StatsScreen(stats))

    # --- Quit / cancel ---

    def action_cancel_or_quit(self) -> None:
        if self._generating:
            self.session.cancel()
        else:
            self.app.exit(message=self._quit_message())

    def action_quit(self) -> None:
        self.app.exit(message=self._quit_message())

    def _quit_message(self) -> str:
        state = self.session.get_state()
        root = self.session.store.root(state.root_id)
        name = root.metadata.get("name") or root.id[:8]
        return f"Quit tree: {name} ({root.id})\nRejoin: basemode-loom view {root.id}"

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

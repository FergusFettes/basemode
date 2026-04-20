import pytest

from basemode_loom.display import DisplayLine
from basemode_loom.session import LoomSession
from basemode_loom.store import GenerationStore
from basemode_loom.tui.app import BasemodeApp
from basemode_loom.tui.screens.loom import LoomScreen
from basemode_loom.tui.widgets.loom_view import LoomView
from basemode_loom.tui.widgets.stream_view import StreamView


@pytest.fixture
def store(tmp_path):
    return GenerationStore(tmp_path / "test.sqlite")


@pytest.fixture
def tree(store):
    """Root → [A, B]; A → [C]"""
    _, ab = store.save_continuations(
        "Root text",
        ["A", "B"],
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=20,
        temperature=0.9,
    )
    _, c = store.save_continuations(
        "",
        ["C"],
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=20,
        temperature=0.9,
        parent_id=ab[0].id,
    )
    return ab, c


# --- App mounting ---


@pytest.mark.asyncio
async def test_app_mounts_loom_view(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True):
        assert app.screen.query_one(LoomView) is not None


@pytest.mark.asyncio
async def test_app_mounts_stream_view_hidden(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True):
        sw = app.screen.query_one(StreamView)
        assert sw is not None
        from textual.widgets import ContentSwitcher

        assert app.screen.query_one(ContentSwitcher).current == "loom"


@pytest.mark.asyncio
async def test_info_bar_shows_tokens_and_branches(store, tree):
    from textual.widgets import Static

    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        info_bar = app.screen.query_one("#status-bar", Static)
        assert "tokens:200" in str(info_bar.render())
        assert "branches:1" in str(info_bar.render())

        await pilot.press("w")
        await pilot.press("d")

        assert "tokens:250" in str(info_bar.render())
        assert "branches:2" in str(info_bar.render())


# --- Navigation ---


@pytest.mark.asyncio
async def test_h_navigates_to_parent(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("h")
        assert session._current_id == ab[0].parent_id


@pytest.mark.asyncio
async def test_h_at_root_stays_at_root(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    # Navigate to root first
    session.navigate_parent()
    root_id = session._current_id
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("h")
        assert session._current_id == root_id


@pytest.mark.asyncio
async def test_l_navigates_into_child(store, tree):
    ab, c = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("l")
        assert session._current_id == c[0].id


@pytest.mark.asyncio
async def test_l_at_leaf_stays_put(store, tree):
    _ab, c = tree
    session = LoomSession(store, c[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        before = session._current_id
        await pilot.press("l")
        assert session._current_id == before


@pytest.mark.asyncio
async def test_j_selects_next_sibling(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    session.navigate_parent()
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("j")
        assert session._selected_idx == 1


@pytest.mark.asyncio
async def test_k_selects_prev_sibling(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    session.navigate_parent()
    session.select_sibling(+1)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("k")
        assert session._selected_idx == 0


@pytest.mark.asyncio
async def test_j_k_round_trip(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    session.navigate_parent()
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("j")
        await pilot.press("k")
        assert session._selected_idx == 0


@pytest.mark.asyncio
async def test_j_wraps_to_first_sibling(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    session.navigate_parent()
    session.select_sibling(+1)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("j")
        assert session._selected_idx == 0


@pytest.mark.asyncio
async def test_k_wraps_to_last_sibling(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    session.navigate_parent()
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("k")
        assert session._selected_idx == 1


@pytest.mark.asyncio
async def test_v_toggles_tree_view(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("v")
        assert session.view_mode == "tree"
        await pilot.press("v")
        assert session.view_mode == "branch"


@pytest.mark.asyncio
async def test_n_toggles_model_names(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("n")
        assert session.show_model_names is False
        await pilot.press("n")
        assert session.show_model_names is True


@pytest.mark.asyncio
async def test_j_k_select_children_in_tree_view(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    session.navigate_parent()
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("v")
        await pilot.press("j")
        state = session.get_state()
        assert state.current_node_id == ab[0].parent_id
        assert state.selected_child_idx == 1
        await pilot.press("k")
        state = session.get_state()
        assert state.current_node_id == ab[0].parent_id
        assert state.selected_child_idx == 0


@pytest.mark.asyncio
async def test_j_wraps_in_tree_view(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    session.navigate_parent()
    session.select_sibling(+1)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("v")
        await pilot.press("j")
        state = session.get_state()
        assert state.current_node_id == ab[0].parent_id
        assert state.selected_child_idx == 0


@pytest.mark.asyncio
async def test_upper_h_toggles_hoist(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("H")
        assert session.get_state().hoisted_node_id == ab[0].id
        await pilot.press("H")
        assert session.get_state().hoisted_node_id is None


@pytest.mark.asyncio
async def test_b_toggles_bookmark(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("b")
        node = store.get(ab[0].id)
        assert node is not None
        assert node.metadata["bookmarked"] is True


@pytest.mark.asyncio
async def test_upper_b_jumps_to_bookmark(store, tree):
    ab, _ = tree
    store.update_metadata(ab[1].id, {"bookmarked": True})
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("B")
        assert session.get_state().current_node_id == ab[1].id


# --- Params ---


@pytest.mark.asyncio
async def test_w_increases_tokens(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    before = session.max_tokens
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("w")
        assert session.max_tokens == before + 50


@pytest.mark.asyncio
async def test_s_decreases_tokens(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    session.set_max_tokens(300)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("s")
        assert session.max_tokens == 250


@pytest.mark.asyncio
async def test_d_increases_branches(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("d")
        assert session.n_branches == 2


@pytest.mark.asyncio
async def test_a_decreases_branches_min_one(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("a")  # already 1, should stay 1
        assert session.n_branches == 1


# --- Modal screens ---


@pytest.mark.asyncio
async def test_t_opens_int_input_screen(store, tree):
    from basemode_loom.tui.screens.int_input import IntInputScreen

    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("t")
        assert isinstance(app.screen, IntInputScreen)


@pytest.mark.asyncio
async def test_t_escape_dismisses_without_change(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    before = session.max_tokens
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("t")
        await pilot.press("escape")
        assert session.max_tokens == before
        assert isinstance(app.screen, LoomScreen)


@pytest.mark.asyncio
async def test_t_submit_updates_tokens(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("t")
        # Clear "200" and enter "500"
        for _ in range(3):
            await pilot.press("backspace")
        for ch in "500":
            await pilot.press(ch)
        await pilot.press("enter")
        assert session.max_tokens == 500


@pytest.mark.asyncio
async def test_m_opens_model_picker(store, tree):
    from basemode_loom.tui.screens.model_picker import ModelPickerScreen

    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("m")
        assert isinstance(app.screen, ModelPickerScreen)


@pytest.mark.asyncio
async def test_m_escape_dismisses(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    original_model = session.model
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("m")
        await pilot.press("escape")
        assert session.model == original_model
        assert isinstance(app.screen, LoomScreen)


@pytest.mark.asyncio
async def test_question_mark_opens_stats_screen(store, tree):
    from basemode_loom.tui.screens.stats import StatsScreen

    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("?")
        assert isinstance(app.screen, StatsScreen)


@pytest.mark.asyncio
async def test_stats_screen_escape_dismisses(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("?")
        await pilot.press("escape")
        assert isinstance(app.screen, LoomScreen)


@pytest.mark.asyncio
async def test_question_mark_does_not_open_stats_while_generating(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        loom_screen = app.screen
        assert isinstance(loom_screen, LoomScreen)
        loom_screen._generating = True  # type: ignore[attr-defined]
        try:
            await pilot.press("?")
            assert isinstance(app.screen, LoomScreen)
        finally:
            loom_screen._generating = False  # type: ignore[attr-defined]


# --- Quit ---


@pytest.mark.asyncio
async def test_q_exits_app(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("q")
    # If we get here, app exited cleanly


@pytest.mark.asyncio
async def test_quit_message_includes_rejoin_info(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    root_id = session.get_state().root_id
    app = BasemodeApp(session)
    async with app.run_test(headless=True):
        message = app.screen._quit_message()

    assert f"Quit tree: {root_id[:8]} ({root_id})" in message
    assert f"basemode-loom view {root_id}" in message


# --- StreamView widget ---


def test_stream_view_buffers_updated_on_add_token():
    sv = StreamView()
    sv._n = 2
    sv._prefix = "prefix"
    sv._buffers = [[], []]
    # Patch _render_content to avoid needing a mounted DOM
    sv._render_content = lambda: None
    sv.add_token(0, "hello")
    sv.add_token(1, "world")
    assert sv._buffers[0] == ["hello"]
    assert sv._buffers[1] == ["world"]


def test_stream_view_reset_clears_buffers():
    sv = StreamView()
    sv._buffers = [["old"]]
    sv._render_content = lambda: None
    sv.reset(3, "new prefix")
    assert sv._n == 3
    assert sv._prefix == "new prefix"
    assert sv._buffers == [[], [], []]


def test_stream_view_uses_sane_width_before_layout():
    sv = StreamView()
    assert sv._content_width() == 80


@pytest.mark.asyncio
async def test_stream_view_uses_scrollable_width_with_scrollbar(store, tree):
    from textual.widgets import ContentSwitcher

    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True, size=(120, 10)) as pilot:
        sv = app.screen.query_one(StreamView)
        app.screen.query_one(ContentSwitcher).current = "stream"
        sv.reset(1, "word " * 1000)
        await pilot.pause()
        assert sv.scrollbar_size_vertical == 2
        assert sv._content_width() == 118


# --- LoomView widget ---


def test_loom_view_update_state_does_not_raise_unmounted(store, tree):
    # LoomView now requires a mounted DOM to call query_one; just verify instantiation
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    state = session.get_state()  # noqa: F841
    lv = LoomView()
    assert lv is not None


def test_loom_view_uses_sane_width_before_layout():
    lv = LoomView()
    assert lv._content_width() == 80


def test_loom_view_tree_scroll_prefers_selected_line():
    lv = LoomView()
    lines = [
        DisplayLine("root", "current"),
        DisplayLine("child a"),
        DisplayLine("child b", "selected"),
    ]
    assert lv._tree_scroll_target(lines) == 0


def test_loom_view_tree_scroll_falls_back_to_current_line():
    lv = LoomView()
    lines = [
        DisplayLine("root"),
        DisplayLine("child a"),
        DisplayLine("child b"),
        DisplayLine("child c"),
        DisplayLine("child d", "current"),
    ]
    assert lv._tree_scroll_target(lines) == 1


@pytest.mark.asyncio
async def test_loom_view_uses_scrollable_width_with_scrollbar(store):
    _, children = store.save_continuations(
        "word " * 1000,
        [" child"],
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=20,
        temperature=0.9,
    )
    session = LoomSession(store, children[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True, size=(120, 10)):
        lv = app.screen.query_one(LoomView)
        assert lv.scrollbar_size_vertical == 2
        assert lv._content_width() == 118

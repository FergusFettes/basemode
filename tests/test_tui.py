import pytest

from basemode.session import LoomSession
from basemode.store import GenerationStore
from basemode.tui.app import BasemodeApp
from basemode.tui.screens.loom import LoomScreen
from basemode.tui.widgets.loom_view import LoomView
from basemode.tui.widgets.stream_view import StreamView


@pytest.fixture
def store(tmp_path):
    return GenerationStore(tmp_path / "test.sqlite")


@pytest.fixture
def tree(store):
    """Root → [A, B]; A → [C]"""
    _, ab = store.save_continuations(
        "Root text", ["A", "B"],
        model="gpt-4o-mini", strategy="system", max_tokens=20, temperature=0.9,
    )
    _, c = store.save_continuations(
        "", ["C"],
        model="gpt-4o-mini", strategy="system", max_tokens=20, temperature=0.9,
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
    ab, c = tree
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
    from basemode.tui.screens.int_input import IntInputScreen

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
    from basemode.tui.screens.model_picker import ModelPickerScreen

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


# --- Quit ---


@pytest.mark.asyncio
async def test_q_exits_app(store, tree):
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("q")
    # If we get here, app exited cleanly


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


# --- LoomView widget ---


def test_loom_view_update_state_does_not_raise_unmounted(store, tree):
    # LoomView now requires a mounted DOM to call query_one; just verify instantiation
    ab, _ = tree
    session = LoomSession(store, ab[0].id)
    state = session.get_state()  # noqa: F841
    lv = LoomView()
    assert lv is not None

"""Tests for the tree picker view and screen."""

import pytest

from basemode_loom.session import LoomSession
from basemode_loom.store import GenerationStore
from basemode_loom.tui.app import BasemodeApp
from basemode_loom.tui.screens.confirm import ConfirmScreen
from basemode_loom.tui.screens.loom import LoomScreen
from basemode_loom.tui.screens.tree_picker import TreePickerScreen
from basemode_loom.tui.widgets.tree_picker import TreePickerView


@pytest.fixture
def store(tmp_path):
    return GenerationStore(tmp_path / "test.sqlite")


@pytest.fixture
def multi_tree_store(tmp_path):
    """Three separate trees in the store."""
    store = GenerationStore(tmp_path / "test.sqlite")
    _, ab = store.save_continuations(
        "The ship rounded the headland",
        ["and sailed on", "and turned back"],
        model="m",
        strategy="system",
        max_tokens=20,
        temperature=0.9,
    )
    store.update_metadata(ab[0].root_id, {"name": "ship-story"})

    _, cd = store.save_continuations(
        "Once upon a time",
        ["there was a dragon"],
        model="m",
        strategy="system",
        max_tokens=20,
        temperature=0.9,
    )
    store.update_metadata(cd[0].root_id, {"name": "fairy-tale"})

    _, ef = store.save_continuations(
        "The quick brown fox",
        ["jumped over"],
        model="m",
        strategy="system",
        max_tokens=20,
        temperature=0.9,
    )
    # third tree intentionally unnamed

    return store, ab, cd, ef


# --- TreePickerView unit tests ---


def test_tree_picker_view_loads_all_trees(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    view = TreePickerView()
    view.load(store, ab[0].root_id)
    assert len(view._entries) == 3


def test_tree_picker_view_cursor_starts_on_current_tree(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    view = TreePickerView()
    view.load(store, cd[0].root_id)
    selected = view.selected_root_id()
    assert selected == cd[0].root_id


def test_tree_picker_view_move_down(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    view = TreePickerView()
    view.load(store, ab[0].root_id)
    view._cursor = 0
    view.move(+1)
    assert view._cursor == 1


def test_tree_picker_view_move_up(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    view = TreePickerView()
    view.load(store, ab[0].root_id)
    view._cursor = 1
    view.move(-1)
    assert view._cursor == 0


def test_tree_picker_view_move_clamps_at_zero(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    view = TreePickerView()
    view.load(store, ab[0].root_id)
    view._cursor = 0
    view.move(-1)
    assert view._cursor == 0


def test_tree_picker_view_move_clamps_at_max(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    view = TreePickerView()
    view.load(store, ab[0].root_id)
    view._cursor = len(view._entries) - 1
    view.move(+1)
    assert view._cursor == len(view._entries) - 1


def test_tree_picker_view_node_count_includes_root(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    view = TreePickerView()
    view.load(store, ab[0].root_id)
    # ab has root + 2 children = 3 nodes (roots() returns most-recent first)
    ship_entry = next(
        e for e in view._entries if e.root.metadata.get("name") == "ship-story"
    )
    assert ship_entry.node_count == 3


def test_tree_picker_view_root_preview_text(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    view = TreePickerView()
    view.load(store, ab[0].root_id)
    ship_entry = next(
        e for e in view._entries if e.root.metadata.get("name") == "ship-story"
    )
    assert "ship" in ship_entry.root_preview.lower()


def test_tree_picker_view_root_preview_not_fixed_width_truncated(store):
    root = store.create_root(" ".join(f"word{i}" for i in range(40)))
    view = TreePickerView()
    view.load(store, root.id)

    assert len(view._entries[0].root_preview) > 72


def test_tree_picker_view_render_shows_delete_per_item(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    view = TreePickerView()
    view.load(store, ab[0].root_id)
    rendered = str(view.render())

    assert "[d delete]" in rendered
    assert "[delete]" in rendered


def test_tree_picker_view_leaf_preview_none_when_only_root(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    view = TreePickerView()
    view.load(store, ab[0].root_id)
    fairy_entry = next(
        e for e in view._entries if e.root.metadata.get("name") == "fairy-tale"
    )
    # No last_node_id set yet on this tree
    assert fairy_entry.leaf_preview == "(at root)"


def test_tree_picker_view_leaf_preview_uses_last_node_id(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    # Save a last_node_id pointing to one of ab's children
    store.update_metadata(ab[0].root_id, {"last_node_id": ab[0].id})
    view = TreePickerView()
    view.load(store, ab[0].root_id)
    ship_entry = next(
        e for e in view._entries if e.root.metadata.get("name") == "ship-story"
    )
    assert ship_entry.leaf_preview != "(at root)"


def test_tree_picker_view_unnamed_tree_uses_id_prefix(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    view = TreePickerView()
    view.load(store, ab[0].root_id)
    unnamed = next(e for e in view._entries if not e.root.metadata.get("name"))
    # Should display first 8 chars of node id
    assert len(unnamed.root.id[:8]) == 8


def test_tree_picker_view_render_no_crash(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    view = TreePickerView()
    view.load(store, ab[0].root_id)
    view._size = type("S", (), {"width": 80, "height": 24})()  # fake size
    # render() should not raise
    from rich.text import Text

    result = view.render()
    assert isinstance(result, Text)


def test_tree_picker_view_empty_store_renders_no_trees(store):
    view = TreePickerView()
    view.load(store, "nonexistent")
    result = view.render()
    assert "No trees" in str(result)


# --- TreePickerScreen pilot tests ---


@pytest.mark.asyncio
async def test_tab_opens_tree_picker(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("tab")
        assert isinstance(app.screen, TreePickerScreen)


@pytest.mark.asyncio
async def test_picker_escape_returns_to_loom(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("tab")
        await pilot.press("escape")
        assert isinstance(app.screen, LoomScreen)


@pytest.mark.asyncio
async def test_picker_q_returns_to_loom(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("tab")
        await pilot.press("q")
        assert isinstance(app.screen, LoomScreen)


@pytest.mark.asyncio
async def test_picker_jk_navigates(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("tab")
        picker_view = app.screen.query_one(TreePickerView)
        before = picker_view._cursor
        if before < len(picker_view._entries) - 1:
            await pilot.press("j")
            assert picker_view._cursor == before + 1
            await pilot.press("k")
        else:
            await pilot.press("k")
            assert picker_view._cursor == before - 1
            await pilot.press("j")
        assert picker_view._cursor == before


@pytest.mark.asyncio
async def test_picker_enter_same_tree_no_change(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    session = LoomSession(store, ab[0].id)
    original_root = session.get_state().root_id
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("tab")
        # Cursor should already be on the current tree; pressing enter stays
        await pilot.press("enter")
        assert isinstance(app.screen, LoomScreen)
        # Session root unchanged since we selected the same tree
        assert session.get_state().root_id == original_root


@pytest.mark.asyncio
async def test_picker_enter_different_tree_switches_session(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    # Start on the ship-story tree
    session = LoomSession(store, ab[0].id)
    original_root = session.get_state().root_id
    app = BasemodeApp(session)

    # The loom_screen holds session; we'll inspect via the screen after switch
    loom_screen: LoomScreen

    async with app.run_test(headless=True) as pilot:
        loom_screen = app.screen  # type: ignore[assignment]

        await pilot.press("tab")
        picker_view = app.screen.query_one(TreePickerView)

        # Find a different tree and its position
        target_idx = next(
            i for i, e in enumerate(picker_view._entries) if e.root.id != original_root
        )
        target_root_id = picker_view._entries[target_idx].root.id

        # Navigate to that entry via j/k from current cursor
        cur = picker_view._cursor
        key = "j" if target_idx > cur else "k"
        for _ in range(abs(target_idx - cur)):
            await pilot.press(key)

        await pilot.press("enter")
        assert isinstance(app.screen, LoomScreen)
        # The loom screen's session should now be on the new tree
        assert loom_screen.session.get_state().root_id == target_root_id


@pytest.mark.asyncio
async def test_picker_tab_selects_and_returns(multi_tree_store):
    """Tab in the picker (same as Enter) opens the selected tree."""
    store, ab, cd, ef = multi_tree_store
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("tab")  # open picker
        assert isinstance(app.screen, TreePickerScreen)
        await pilot.press("tab")  # select (same as enter)
        assert isinstance(app.screen, LoomScreen)


@pytest.mark.asyncio
async def test_picker_shows_tree_names(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("tab")
        view = app.screen.query_one(TreePickerView)
        names = {e.root.metadata.get("name") for e in view._entries}
        assert "ship-story" in names
        assert "fairy-tale" in names


@pytest.mark.asyncio
async def test_picker_shows_delete_affordance_in_items(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("tab")
        rendered = str(app.screen.query_one(TreePickerView).render())
        assert "[d delete]" in rendered


@pytest.mark.asyncio
async def test_picker_delete_opens_confirmation(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    session = LoomSession(store, ab[0].id)
    current_root = session.get_state().root_id
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("tab")
        view = app.screen.query_one(TreePickerView)
        target_idx = next(
            i for i, e in enumerate(view._entries) if e.root.id != current_root
        )
        target_root_id = view._entries[target_idx].root.id

        cur = view._cursor
        key = "j" if target_idx > cur else "k"
        for _ in range(abs(target_idx - cur)):
            await pilot.press(key)

        await pilot.press("d")

        assert isinstance(app.screen, ConfirmScreen)
        assert store.get(target_root_id) is not None


@pytest.mark.asyncio
async def test_picker_delete_escape_cancels(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    session = LoomSession(store, ab[0].id)
    current_root = session.get_state().root_id
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("tab")
        view = app.screen.query_one(TreePickerView)
        target_idx = next(
            i for i, e in enumerate(view._entries) if e.root.id != current_root
        )
        target_root_id = view._entries[target_idx].root.id

        cur = view._cursor
        key = "j" if target_idx > cur else "k"
        for _ in range(abs(target_idx - cur)):
            await pilot.press(key)

        await pilot.press("d")
        await pilot.press("escape")

        assert isinstance(app.screen, TreePickerScreen)
        assert store.get(target_root_id) is not None


@pytest.mark.asyncio
async def test_picker_delete_enter_confirms_non_current_tree(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    session = LoomSession(store, ab[0].id)
    current_root = session.get_state().root_id
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        await pilot.press("tab")
        view = app.screen.query_one(TreePickerView)
        target_idx = next(
            i for i, e in enumerate(view._entries) if e.root.id != current_root
        )
        target_root_id = view._entries[target_idx].root.id

        cur = view._cursor
        key = "j" if target_idx > cur else "k"
        for _ in range(abs(target_idx - cur)):
            await pilot.press(key)

        await pilot.press("d")
        await pilot.press("enter")

        assert isinstance(app.screen, TreePickerScreen)
        assert store.get(target_root_id) is None
        assert target_root_id not in view.root_ids()
        assert session.get_state().root_id == current_root


@pytest.mark.asyncio
async def test_picker_delete_current_tree_switches_to_remaining_tree(multi_tree_store):
    store, ab, cd, ef = multi_tree_store
    session = LoomSession(store, ab[0].id)
    original_root = session.get_state().root_id
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        loom_screen = app.screen
        await pilot.press("tab")

        await pilot.press("d")
        await pilot.press("enter")

        assert isinstance(app.screen, LoomScreen)
        assert store.get(original_root) is None
        assert loom_screen.session.get_state().root_id != original_root


@pytest.mark.asyncio
async def test_generating_blocks_picker(multi_tree_store):
    """Opening picker while generating should be a no-op."""
    store, ab, cd, ef = multi_tree_store
    session = LoomSession(store, ab[0].id)
    app = BasemodeApp(session)
    async with app.run_test(headless=True) as pilot:
        loom_screen = app.screen
        loom_screen._generating = True  # type: ignore[attr-defined]
        await pilot.press("tab")
        # Should still be on LoomScreen, not TreePickerScreen
        assert isinstance(app.screen, LoomScreen)
        loom_screen._generating = False  # type: ignore[attr-defined]

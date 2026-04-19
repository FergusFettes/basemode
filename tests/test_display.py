import pytest

from basemode.display import (
    DisplayLine,
    build_loom_display,
    build_stream_display,
    word_wrap_inline,
    wrap_text,
)
from basemode.session import SessionState
from basemode.store import Node


def _node(id, parent_id=None, root_id=None, text=""):
    return Node(
        id=id,
        parent_id=parent_id,
        root_id=root_id or id,
        text=text,
        model=None,
        strategy=None,
        max_tokens=None,
        temperature=None,
        branch_index=None,
        created_at="2024-01-01T00:00:00Z",
        metadata={},
    )


def _state(
    full_text="Hello",
    children=None,
    selected_idx=0,
    continuation_text="",
    descendant_counts=None,
):
    node = _node("root", text=full_text)
    return SessionState(
        current_node_id="root",
        current_node=node,
        full_text=full_text,
        children=children or [],
        selected_child_idx=selected_idx,
        descendant_counts=descendant_counts or {},
        continuation_text=continuation_text,
        model="gpt-4o-mini",
        max_tokens=200,
        temperature=0.9,
        n_branches=1,
        context="",
        root_id="root",
    )


# --- wrap_text ---


def test_wrap_text_single_line():
    assert wrap_text("hello", 80) == ["hello"]


def test_wrap_text_wraps_long_line():
    lines = wrap_text("one two three", 7)
    assert len(lines) > 1
    assert all(len(l) <= 7 for l in lines)


def test_wrap_text_preserves_blank_lines():
    lines = wrap_text("a\n\nb", 80)
    assert lines == ["a", "", "b"]


def test_wrap_text_leading_blank():
    lines = wrap_text("\nfoo", 80)
    assert lines[0] == ""
    assert "foo" in lines


def test_wrap_text_empty_returns_one_empty():
    assert wrap_text("", 80) == [""]


def test_wrap_text_only_newline():
    lines = wrap_text("\n", 80)
    assert "" in lines


# --- word_wrap_inline ---


def test_word_wrap_inline_short_fits_on_first_line():
    assert word_wrap_inline("hi", 20, 80) == ["hi"]


def test_word_wrap_inline_uses_first_width_then_full():
    # "one two" with first_width=4 should split: "one" on first, "two" on second (full_width=20)
    lines = word_wrap_inline("one two", 4, 20)
    assert lines[0] == "one"
    assert lines[1] == "two"


def test_word_wrap_inline_empty_returns_one_empty():
    assert word_wrap_inline("", 20, 80) == [""]


def test_word_wrap_inline_long_word_truncates_to_first_width():
    lines = word_wrap_inline("toolongword rest", 5, 20)
    assert len(lines[0]) <= 5


def test_word_wrap_inline_subsequent_lines_use_full_width():
    lines = word_wrap_inline("a b c d e f g h i j", 3, 80)
    # Second line onwards should not be constrained to first_width
    if len(lines) > 1:
        assert len(lines[1]) <= 80


# --- build_loom_display ---


def test_build_loom_display_no_children_all_normal():
    state = _state(full_text="Hello world", children=[])
    lines = build_loom_display(state, 80)
    assert lines
    assert all(l.style == "normal" for l in lines)
    text = "".join(l.text for l in lines)
    assert "Hello world" in text


def test_build_loom_display_selected_child_is_bold():
    child = _node("c1", parent_id="root", text=" continuation")
    state = _state(full_text="Hello", children=[child], selected_idx=0)
    lines = build_loom_display(state, 80)
    bold = [l for l in lines if l.style == "bold"]
    assert len(bold) > 0
    assert any("continuation" in l.text for l in bold)


def test_build_loom_display_other_siblings_are_dim():
    c1 = _node("c1", parent_id="root", text=" one")
    c2 = _node("c2", parent_id="root", text=" two")
    state = _state(full_text="Hello", children=[c1, c2], selected_idx=0)
    lines = build_loom_display(state, 80)
    dim = [l for l in lines if l.style == "dim"]
    assert len(dim) > 0
    assert any("two" in l.text for l in dim)
    assert not any("one" in l.text for l in dim)


def test_build_loom_display_arrow_present():
    child = _node("c1", parent_id="root", text=" next")
    state = _state(full_text="Start", children=[child])
    lines = build_loom_display(state, 80)
    assert any(" -> " in l.text for l in lines)


def test_build_loom_display_continuation_appended_as_normal():
    child = _node("c1", parent_id="root", text=" A")
    state = _state(full_text="Hello", children=[child], continuation_text=" deeper text")
    lines = build_loom_display(state, 80)
    normal_with_deeper = [l for l in lines if l.style == "normal" and "deeper" in l.text]
    assert len(normal_with_deeper) > 0


def test_build_loom_display_descendant_count_in_marker():
    child = _node("c1", parent_id="root", text=" A")
    state = _state(
        full_text="X",
        children=[child],
        selected_idx=0,
        descendant_counts={"c1": 5},
    )
    lines = build_loom_display(state, 80)
    text = "".join(l.text for l in lines)
    assert "(5)" in text


def test_build_loom_display_wraps_to_width():
    long_text = "word " * 30
    child = _node("c1", parent_id="root", text=long_text)
    state = _state(full_text="Hello", children=[child])
    lines = build_loom_display(state, 40)
    assert all(len(l.text) <= 45 for l in lines)  # some tolerance for indent+arrow


def test_build_loom_display_narrow_terminal_falls_back():
    # When terminal is very narrow, last_line should be pushed to its own line
    child = _node("c1", parent_id="root", text=" x")
    very_long_parent = "A" * 75
    state = _state(full_text=very_long_parent, children=[child])
    lines = build_loom_display(state, 80)
    assert lines  # should not crash


# --- build_stream_display ---


def test_build_stream_display_single_branch_bold():
    buffers = [["hello", " world"]]
    lines = build_stream_display("Prefix", buffers, 80)
    bold = [l for l in lines if l.style == "bold"]
    assert any("hello" in l.text for l in bold)


def test_build_stream_display_second_branch_dim():
    buffers = [["first"], ["second"]]
    lines = build_stream_display("Prefix", buffers, 80)
    dim = [l for l in lines if l.style == "dim"]
    assert any("second" in l.text for l in dim)


def test_build_stream_display_arrow_present():
    buffers = [["x"]]
    lines = build_stream_display("prefix text", buffers, 80)
    assert any(" -> " in l.text for l in lines)


def test_build_stream_display_empty_buffers_no_crash():
    lines = build_stream_display("", [], 80)
    assert isinstance(lines, list)


def test_build_stream_display_parent_lines_present():
    buffers = [["cont"]]
    lines = build_stream_display("Parent text", buffers, 80)
    all_text = "".join(l.text for l in lines)
    assert "Parent" in all_text


def test_build_stream_display_cursor_glyph_in_active():
    buffers = [["token"]]
    lines = build_stream_display("X", buffers, 80)
    all_text = "".join(l.text for l in lines if l.style == "bold")
    assert "▋" in all_text

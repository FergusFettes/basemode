"""Pure display-building functions shared across UI layers (TUI, web).

`build_loom_display` and `build_stream_display` take clean data types and
return `list[DisplayLine]`. Each UI layer (Textual TUI, future web backend)
converts DisplayLine to its own rendering primitives independently.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .session import SessionState
    from .store import Node

ARROW = " -> "
TREE_MID = "├─ "
TREE_LAST = "└─ "
TREE_PIPE = "│  "
TREE_BLANK = "   "


@dataclass(frozen=True)
class DisplaySpan:
    start: int
    end: int
    style: Literal["model"]


@dataclass(frozen=True)
class DisplayLine:
    text: str
    style: Literal["normal", "bold", "dim", "path", "current", "selected"] = "normal"
    spans: tuple[DisplaySpan, ...] = ()


def wrap_text(text: str, width: int) -> list[str]:
    """Word-wrap plain text to width, preserving blank lines."""
    lines: list[str] = []
    for segment in text.split("\n"):
        if segment:
            lines.extend(textwrap.wrap(segment, width) or [""])
        else:
            lines.append("")
    return lines or [""]


def word_wrap_inline(text: str, first_width: int, full_width: int) -> list[str]:
    """Word-wrap with a narrower first-line width (used after an inline prefix)."""
    text = text.rstrip("\n")
    if not text:
        return [""]
    words = text.split()
    lines: list[str] = []
    current_line = ""
    current_width = first_width

    for word in words:
        if not current_line:
            current_line = word[:current_width]
            if len(word) > current_width:
                lines.append(current_line)
                current_line = ""
                current_width = full_width
        elif len(current_line) + 1 + len(word) <= current_width:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word[:full_width]
            current_width = full_width

    if current_line:
        lines.append(current_line)
    return lines or [""]


def build_loom_display(state: SessionState, width: int) -> list[DisplayLine]:
    """Build display lines for the loom tree view.

    Layout:
      [parent text lines except last]
      [last parent line]->[selected child ... (bold)]
                        ->[sibling 1 (dim)]
                        ->[sibling 2 (dim)]
      [continuation text from selected child's subtree (normal)]
    """
    parent_lines = wrap_text(state.full_text, width)

    if not state.children:
        return [DisplayLine(line) for line in parent_lines]

    lines: list[DisplayLine] = [DisplayLine(line) for line in parent_lines[:-1]]
    last_line = parent_lines[-1]

    if width - len(last_line) - len(ARROW) < 10:
        lines.append(DisplayLine(last_line))
        last_line = ""

    lines += _render_siblings(state, last_line, width)
    return lines


def build_tree_display(state: SessionState, width: int) -> list[DisplayLine]:
    """Build a compact full-tree display for the current root or hoisted subtree."""
    nodes = state.tree_nodes or []
    if not nodes:
        return [DisplayLine("(empty tree)", "dim")]

    by_id = {node.id: node for node in nodes}
    root_id = state.hoisted_node_id or state.root_id
    root = by_id.get(root_id) or by_id.get(state.root_id) or nodes[0]

    children_by_parent: dict[str | None, list[Node]] = {}
    for node in nodes:
        children_by_parent.setdefault(node.parent_id, []).append(node)
    for children in children_by_parent.values():
        children.sort(
            key=lambda n: (
                n.branch_index is None,
                n.branch_index if n.branch_index is not None else 0,
                n.created_at,
                n.id,
            )
        )

    path = _path_ids(by_id, state.current_node_id)
    selected_child_id = (
        state.children[state.selected_child_idx].id if state.children else None
    )
    descendant_counts = _descendant_counts(children_by_parent, root.id)

    lines: list[DisplayLine] = []
    if state.hoisted_node_id:
        lines.append(DisplayLine(f"[hoist] {root_label(root, width - 8)}", "dim"))

    def visit(node: Node, prefix: str, is_last: bool, is_root: bool = False) -> None:
        connector = "" if is_root else (TREE_LAST if is_last else TREE_MID)
        line_prefix = prefix + connector
        label = _tree_node_label(
            node,
            current=node.id == state.current_node_id,
            selected=node.id == selected_child_id,
            bookmarked=bool(node.metadata.get("bookmarked")),
            descendants=descendant_counts.get(node.id, 0),
            show_model=state.show_model_names,
            width=max(10, width - len(line_prefix)),
        )
        if node.id == state.current_node_id:
            style: Literal["normal", "bold", "dim", "path", "current", "selected"] = (
                "current"
            )
        elif node.id in path:
            style = "path"
        elif node.id == selected_child_id:
            style = "selected"
        else:
            style = "normal"
        text = (line_prefix + label.text)[:width]
        spans = tuple(
            DisplaySpan(
                span.start + len(line_prefix),
                min(span.end + len(line_prefix), width),
                span.style,
            )
            for span in label.spans
            if span.start + len(line_prefix) < width
        )
        lines.append(DisplayLine(text, style, spans))

        child_prefix = (
            prefix if is_root else prefix + (TREE_BLANK if is_last else TREE_PIPE)
        )
        children = children_by_parent.get(node.id, [])
        for index, child in enumerate(children):
            visit(child, child_prefix, index == len(children) - 1)

    visit(root, "", True, is_root=True)
    return lines


def root_label(node: Node, width: int) -> str:
    return _flatten_preview(node.text, max(10, width))


def _path_ids(by_id: dict[str, Node], current_id: str) -> set[str]:
    path: set[str] = set()
    node = by_id.get(current_id)
    while node is not None:
        path.add(node.id)
        node = by_id.get(node.parent_id) if node.parent_id else None
    return path


def _descendant_counts(
    children_by_parent: dict[str | None, list[Node]],
    root_id: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}

    def count(node_id: str) -> int:
        total = 0
        for child in children_by_parent.get(node_id, []):
            total += 1 + count(child.id)
        counts[node_id] = total
        return total

    count(root_id)
    return counts


def _tree_node_label(
    node: Node,
    *,
    current: bool,
    selected: bool,
    bookmarked: bool,
    descendants: int,
    show_model: bool,
    width: int,
) -> DisplayLine:
    cursor = ">" if current else "*" if selected else " "
    bookmark = "b" if bookmarked else " "
    count = f" +{descendants}" if descendants else ""
    model = _short_model_name(node.model) if show_model and node.model else ""
    model_prefix = f"{model} " if model else ""
    fixed = f"{cursor}{bookmark} {model_prefix}{count} "
    text = fixed + _flatten_preview(node.text, width - len(fixed))
    spans: tuple[DisplaySpan, ...] = ()
    if model:
        start = len(f"{cursor}{bookmark} ")
        spans = (DisplaySpan(start, start + len(model), "model"),)
    return DisplayLine(text, "normal", spans)


def _short_model_name(model: str | None) -> str:
    if not model:
        return ""
    return model.split("/")[-1]


def _flatten_preview(text: str, width: int) -> str:
    text = " ".join(text.split())
    if width <= 1:
        return ""
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)].rstrip() + "…"


def _render_siblings(
    state: SessionState, last_line: str, width: int
) -> list[DisplayLine]:
    children = state.children
    selected_idx = state.selected_child_idx
    counts = state.descendant_counts

    indent = len(last_line)
    available = width - indent - len(ARROW)
    if available < 10:
        indent = 0
        available = width - len(ARROW)
        last_line = ""

    def marker(node: Node) -> str:
        c = counts.get(node.id, 0)
        return f" ({c})" if c > 0 else ""

    lines: list[DisplayLine] = []
    selected = children[selected_idx]

    sel_seg = selected.text + marker(selected)
    sel_lines = word_wrap_inline(sel_seg, available, width)
    row_prefix = (last_line + ARROW) if last_line else ARROW
    lines.append(DisplayLine(row_prefix + sel_lines[0], "bold"))
    for sl in sel_lines[1:]:
        lines.append(DisplayLine(sl, "bold"))

    for i, child in enumerate(children):
        if i == selected_idx:
            continue
        sib_seg = child.text + marker(child)
        sib_lines = word_wrap_inline(sib_seg, available, width)
        for j, sl in enumerate(sib_lines):
            lines.append(
                DisplayLine(" " * indent + ARROW + sl if j == 0 else sl, "dim")
            )

    if state.continuation_text:
        for line in wrap_text(state.continuation_text, width):
            lines.append(DisplayLine(line))

    return lines


def build_stream_display(
    prefix: str, buffers: list[list[str]], width: int
) -> list[DisplayLine]:
    """Build display lines for the streaming generation view."""
    prefix_lines = wrap_text(prefix, width) if prefix else [""]
    last_line = prefix_lines[-1]
    indent = len(last_line)
    available = width - indent - len(ARROW)
    if available < 10:
        indent = 0
        available = width - len(ARROW)
        last_line = ""

    lines: list[DisplayLine] = [DisplayLine(line) for line in prefix_lines[:-1]]

    for i, buf in enumerate(buffers):
        segment = "".join(buf) + "▋"
        seg_lines = word_wrap_inline(segment, available, width)
        if i == 0:
            row_prefix = (last_line + ARROW) if last_line else ARROW
            lines.append(DisplayLine(row_prefix + seg_lines[0], "bold"))
        else:
            lines.append(DisplayLine(" " * indent + ARROW + seg_lines[0], "dim"))
        for sl in seg_lines[1:]:
            lines.append(DisplayLine(sl, "bold" if i == 0 else "dim"))

    if not buffers:
        lines.append(DisplayLine(last_line))

    return lines

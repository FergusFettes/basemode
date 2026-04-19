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


@dataclass(frozen=True)
class DisplayLine:
    text: str
    style: Literal["normal", "bold", "dim"] = "normal"


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


def _render_siblings(state: SessionState, last_line: str, width: int) -> list[DisplayLine]:
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
            lines.append(DisplayLine(" " * indent + ARROW + sl if j == 0 else sl, "dim"))

    if state.continuation_text:
        for line in wrap_text(state.continuation_text, width):
            lines.append(DisplayLine(line))

    return lines


def build_stream_display(prefix: str, buffers: list[list[str]], width: int) -> list[DisplayLine]:
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

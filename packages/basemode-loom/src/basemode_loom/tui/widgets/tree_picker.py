from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.style import Style
from rich.text import Text
from textual import events
from textual.widget import Widget

if TYPE_CHECKING:
    from basemode_loom.store import GenerationStore, Node

_CURSOR = "▸"
_ACTIVE = "●"  # marks the currently open tree
_LEAF_PREFIX = "  └ "
_NONE_LABEL = "(at root)"


@dataclass
class _TreeEntry:
    root: Node
    node_count: int
    root_preview: str  # flattened first-paragraph text
    leaf_preview: str  # text of the checked-out leaf node


class TreePickerView(Widget):
    """Scrollable list of all trees with stats and checked-out node preview."""

    DEFAULT_CSS = """
    TreePickerView {
        height: 1fr;
        overflow-y: auto;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._entries: list[_TreeEntry] = []
        self._cursor: int = 0
        self._current_root_id: str = ""

    def load(self, store: GenerationStore, current_root_id: str) -> None:
        """Populate the list from the store. Call once after mounting."""
        self._current_root_id = current_root_id
        roots = store.roots()
        counts = store.descendant_counts([r.id for r in roots]) if roots else {}

        entries: list[_TreeEntry] = []
        for root in roots:
            # +1 to count the root node itself
            node_count = counts.get(root.id, 0) + 1

            root_preview = self._flatten(root.text)

            last_id = root.metadata.get("last_node_id")
            if last_id and last_id != root.id:
                node = store.get(last_id)
                leaf_preview = self._flatten(node.text) if node else _NONE_LABEL
            else:
                leaf_preview = _NONE_LABEL

            entries.append(_TreeEntry(root, node_count, root_preview, leaf_preview))

        self._entries = entries
        self._cursor = 0
        for i, e in enumerate(entries):
            if e.root.id == current_root_id:
                self._cursor = i
                break
        self.refresh()

    def move(self, delta: int) -> None:
        self._cursor = max(0, min(self._cursor + delta, len(self._entries) - 1))
        self.refresh()

    def selected_root_id(self) -> str | None:
        if self._entries:
            return self._entries[self._cursor].root.id
        return None

    def root_ids(self) -> list[str]:
        return [entry.root.id for entry in self._entries]

    def tree_count(self) -> int:
        return len(self._entries)

    def render(self) -> Text:
        if not self._entries:
            return Text(
                "No trees found. Run: basemode-loom run <text>", style=Style(dim=True)
            )

        width = self.size.width or 80
        result = Text(no_wrap=True, overflow="fold")

        for i, entry in enumerate(self._entries):
            selected = i == self._cursor
            is_open = entry.root.id == self._current_root_id

            name = entry.root.metadata.get("name") or entry.root.id[:8]
            count_str = f"{entry.node_count} node{'s' if entry.node_count != 1 else ''}"
            delete_str = "d delete" if selected else "delete"
            meta_str = f"{count_str}  [{delete_str}]"

            cursor_char = _CURSOR if selected else " "
            open_char = _ACTIVE if is_open else " "
            prefix = f"{cursor_char}{open_char} "

            # Header: [cursor][open] name .... N nodes
            header_left = f"{prefix}{name}"
            gap = max(2, width - len(header_left) - len(meta_str))
            header = header_left + " " * gap + meta_str

            if selected:
                hdr_style = Style(bold=True)
                body_style = Style(bold=True)
            elif is_open:
                hdr_style = Style(italic=True)
                body_style = Style(dim=True)
            else:
                hdr_style = Style()
                body_style = Style(dim=True)

            result.append(header[:width] + "\n", style=hdr_style)

            indent = "   "
            preview_width = width - len(indent) - 1
            result.append(
                indent + entry.root_preview[:preview_width] + "\n", style=body_style
            )

            leaf_line = _LEAF_PREFIX + entry.leaf_preview
            result.append(leaf_line[:width] + "\n", style=body_style)

            result.append("\n")  # blank line between entries

        return result

    def on_resize(self, event: events.Resize) -> None:
        self.refresh()

    @staticmethod
    def _flatten(text: str) -> str:
        return " ".join(text.split())

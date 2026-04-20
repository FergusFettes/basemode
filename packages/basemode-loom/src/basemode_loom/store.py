"""SQLite persistence for loom-style continuation trees."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def default_db_path() -> Path:
    """Return the default generation database path."""
    if path := os.environ.get("BASEMODE_DB"):
        return Path(path).expanduser()
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data_home / "basemode" / "generations.sqlite"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class Node:
    id: str
    parent_id: str | None
    root_id: str
    text: str
    model: str | None
    strategy: str | None
    max_tokens: int | None
    temperature: float | None
    branch_index: int | None
    created_at: str
    metadata: dict[str, Any]


class AmbiguousNodeReference(ValueError):
    """Raised when a partial node reference matches more than one node."""

    def __init__(self, reference: str, matches: list[str]) -> None:
        self.reference = reference
        self.matches = matches
        super().__init__(
            f"ambiguous node reference {reference!r}: matches {', '.join(matches)}"
        )


class GenerationStore:
    """Persistent node store.

    A root node contains the user-provided starting text. Each generated
    continuation is a child node containing only the added segment. Full text is
    reconstructed by walking ancestors from the selected node back to its root.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.db_path = (
            Path(path).expanduser() if path is not None else default_db_path()
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with closing(self.connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    parent_id TEXT REFERENCES nodes(id) ON DELETE CASCADE,
                    root_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    model TEXT,
                    strategy TEXT,
                    max_tokens INTEGER,
                    temperature REAL,
                    branch_index INTEGER,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_parent_created ON nodes(parent_id, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_root_created ON nodes(root_id, created_at)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute("PRAGMA user_version = 1")

    def create_root(self, text: str, *, metadata: dict[str, Any] | None = None) -> Node:
        node_id = uuid.uuid4().hex
        node = Node(
            id=node_id,
            parent_id=None,
            root_id=node_id,
            text=text,
            model=None,
            strategy=None,
            max_tokens=None,
            temperature=None,
            branch_index=None,
            created_at=_now(),
            metadata=metadata or {},
        )
        self._insert(node)
        return node

    def add_child(
        self,
        parent_id: str,
        text: str,
        *,
        model: str,
        strategy: str,
        max_tokens: int,
        temperature: float,
        branch_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Node:
        parent = self.get(parent_id)
        if parent is None:
            raise KeyError(f"unknown parent node: {parent_id}")
        node = Node(
            id=uuid.uuid4().hex,
            parent_id=parent.id,
            root_id=parent.root_id,
            text=text,
            model=model,
            strategy=strategy,
            max_tokens=max_tokens,
            temperature=temperature,
            branch_index=branch_index,
            created_at=_now(),
            metadata=metadata or {},
        )
        self._insert(node)
        return node

    def save_continuations(
        self,
        prefix: str,
        continuations: list[str],
        *,
        model: str,
        strategy: str,
        max_tokens: int,
        temperature: float,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Node, list[Node]]:
        """Persist one generation fanout and return its parent plus children."""
        parent = self.get(parent_id) if parent_id else self.create_root(prefix)
        if parent is None:
            raise KeyError(f"unknown parent node: {parent_id}")
        children = [
            self.add_child(
                parent.id,
                text,
                model=model,
                strategy=strategy,
                max_tokens=max_tokens,
                temperature=temperature,
                branch_index=i,
                metadata=metadata,
            )
            for i, text in enumerate(continuations)
        ]
        return parent, children

    def resolve_node_id(self, reference: str) -> str | None:
        """Resolve a full id or unique id substring to a canonical node id."""
        if not reference:
            return None
        with closing(self.connect()) as conn:
            exact = conn.execute(
                "SELECT id FROM nodes WHERE id = ?",
                (reference,),
            ).fetchone()
            if exact:
                return str(exact["id"])

            rows = conn.execute(
                "SELECT id FROM nodes WHERE id LIKE ? ORDER BY created_at DESC, id DESC",
                (f"{reference}%",),
            ).fetchall()

        matches = [str(row["id"]) for row in rows]
        if not matches:
            return None
        if len(matches) > 1:
            raise AmbiguousNodeReference(reference, matches)
        return matches[0]

    def get(self, node_id: str) -> Node | None:
        resolved = self.resolve_node_id(node_id)
        if resolved is None:
            return None
        with closing(self.connect()) as conn:
            row = conn.execute(
                "SELECT * FROM nodes WHERE id = ?", (resolved,)
            ).fetchone()
        return self._node(row) if row else None

    def root(self, node_id: str) -> Node:
        node = self.get(node_id)
        if node is None:
            raise KeyError(f"unknown node: {node_id}")
        root = self.get(node.root_id)
        if root is None:
            raise KeyError(f"unknown root node: {node.root_id}")
        return root

    def update_metadata(self, node_id: str, metadata: dict[str, Any]) -> Node:
        node = self.get(node_id)
        if node is None:
            raise KeyError(f"unknown node: {node_id}")
        merged = {**node.metadata, **metadata}
        with closing(self.connect()) as conn, conn:
            conn.execute(
                "UPDATE nodes SET metadata_json = ? WHERE id = ?",
                (json.dumps(merged, sort_keys=True), node_id),
            )
        updated = self.get(node_id)
        assert updated is not None
        return updated

    def children(self, node_id: str) -> list[Node]:
        resolved = self.resolve_node_id(node_id)
        if resolved is None:
            raise KeyError(f"unknown node: {node_id}")
        with closing(self.connect()) as conn:
            rows = conn.execute(
                """
                SELECT * FROM nodes
                WHERE parent_id = ?
                ORDER BY branch_index IS NULL, branch_index, created_at, id
                """,
                (resolved,),
            ).fetchall()
        return [self._node(row) for row in rows]

    def tree(self, root_id: str) -> list[Node]:
        """Return all nodes in the tree rooted at root_id, ordered by creation time."""
        resolved = self.resolve_node_id(root_id)
        if resolved is None:
            return []
        with closing(self.connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM nodes WHERE root_id = ? ORDER BY created_at, id",
                (resolved,),
            ).fetchall()
        return [self._node(row) for row in rows]

    def find_root_by_text(self, text: str) -> Node | None:
        """Return the root node whose text exactly matches, or None."""
        with closing(self.connect()) as conn:
            row = conn.execute(
                "SELECT * FROM nodes WHERE parent_id IS NULL AND text = ?",
                (text,),
            ).fetchone()
        return self._node(row) if row else None

    def import_nodes(self, nodes: list[Node]) -> int:
        """Insert nodes in topological order, skipping existing ids. Returns count inserted."""
        # Topological sort: parents before children
        by_id = {n.id: n for n in nodes}
        ordered: list[Node] = []
        seen: set[str] = set()

        def visit(node: Node) -> None:
            if node.id in seen:
                return
            if node.parent_id and node.parent_id in by_id:
                visit(by_id[node.parent_id])
            seen.add(node.id)
            ordered.append(node)

        for n in nodes:
            visit(n)

        inserted = 0
        with closing(self.connect()) as conn, conn:
            for node in ordered:
                result = conn.execute(
                    """
                    INSERT OR IGNORE INTO nodes (
                        id, parent_id, root_id, text, model, strategy, max_tokens,
                        temperature, branch_index, created_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node.id,
                        node.parent_id,
                        node.root_id,
                        node.text,
                        node.model,
                        node.strategy,
                        node.max_tokens,
                        node.temperature,
                        node.branch_index,
                        node.created_at,
                        json.dumps(node.metadata, sort_keys=True),
                    ),
                )
                inserted += result.rowcount
        return inserted

    def roots(self) -> list[Node]:
        with closing(self.connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM nodes WHERE parent_id IS NULL ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [self._node(row) for row in rows]

    def delete_tree(self, root_id: str) -> int:
        """Delete a root and every node in its tree. Returns deleted node count."""
        root = self.root(root_id)
        nodes = self.tree(root.id)
        node_ids = [node.id for node in nodes]
        if not node_ids:
            return 0

        placeholders = ",".join("?" * len(node_ids))
        checked_out_keys = [f"checked_out:{node_id}" for node_id in node_ids]
        key_placeholders = ",".join("?" * len(checked_out_keys))

        with closing(self.connect()) as conn, conn:
            result = conn.execute("DELETE FROM nodes WHERE id = ?", (root.id,))
            conn.execute(
                f"DELETE FROM state WHERE value IN ({placeholders})",
                node_ids,
            )
            conn.execute(
                f"DELETE FROM state WHERE key IN ({key_placeholders})",
                checked_out_keys,
            )
        return result.rowcount + len(node_ids) - 1

    def descendant_count(self, node_id: str) -> int:
        """Return the total number of descendants (not including the node itself)."""
        resolved = self.resolve_node_id(node_id)
        if resolved is None:
            return 0
        with closing(self.connect()) as conn:
            row = conn.execute(
                """
                WITH RECURSIVE desc(id) AS (
                    SELECT id FROM nodes WHERE parent_id = ?
                    UNION ALL
                    SELECT n.id FROM nodes n JOIN desc d ON n.parent_id = d.id
                )
                SELECT COUNT(*) FROM desc
                """,
                (resolved,),
            ).fetchone()
        return int(row[0]) if row else 0

    def descendant_counts(self, node_ids: list[str]) -> dict[str, int]:
        """Return descendant counts for multiple nodes in a single query."""
        if not node_ids:
            return {}
        placeholders = ",".join("?" * len(node_ids))
        with closing(self.connect()) as conn:
            rows = conn.execute(
                f"""
                WITH RECURSIVE desc(id, root_id) AS (
                    SELECT id, parent_id FROM nodes WHERE parent_id IN ({placeholders})
                    UNION ALL
                    SELECT n.id, d.root_id FROM nodes n JOIN desc d ON n.parent_id = d.id
                )
                SELECT root_id, COUNT(*) FROM desc GROUP BY root_id
                """,
                node_ids,
            ).fetchall()
        counts = {nid: 0 for nid in node_ids}
        for row in rows:
            if row[0] in counts:
                counts[row[0]] = int(row[1])
        return counts

    def recent(self, limit: int = 20) -> list[Node]:
        with closing(self.connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM nodes ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._node(row) for row in rows]

    def set_state(self, key: str, value: str) -> None:
        with closing(self.connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO state (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    def get_state(self, key: str) -> str | None:
        with closing(self.connect()) as conn:
            row = conn.execute(
                "SELECT value FROM state WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else str(row["value"])

    def set_checked_out_child(self, parent_id: str, child_id: str) -> None:
        self.set_state(f"checked_out:{parent_id}", child_id)

    def get_checked_out_child_id(self, parent_id: str) -> str | None:
        return self.get_state(f"checked_out:{parent_id}")

    def set_active_node(self, node_id: str) -> None:
        self.set_state("active_node_id", node_id)

    def get_active_node_id(self) -> str | None:
        return self.get_state("active_node_id")

    def get_active_node(self) -> Node | None:
        active_node_id = self.get_active_node_id()
        return None if active_node_id is None else self.get(active_node_id)

    def select_branch(self, node_id: str, branch_index: int) -> Node:
        children = self.children(node_id)
        if branch_index < 1:
            raise ValueError("branch index must be >= 1")
        if branch_index > len(children):
            raise IndexError(
                f"node {node_id!r} has only {len(children)} child branch(es)"
            )
        return children[branch_index - 1]

    def lineage(self, node_id: str) -> list[Node]:
        nodes: list[Node] = []
        node = self.get(node_id)
        while node is not None:
            nodes.append(node)
            node = self.get(node.parent_id) if node.parent_id else None
        nodes.reverse()
        if not nodes:
            raise KeyError(f"unknown node: {node_id}")
        return nodes

    def full_text(self, node_id: str) -> str:
        return "".join(node.text for node in self.lineage(node_id))

    def _insert(self, node: Node) -> None:
        with closing(self.connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO nodes (
                    id, parent_id, root_id, text, model, strategy, max_tokens,
                    temperature, branch_index, created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node.id,
                    node.parent_id,
                    node.root_id,
                    node.text,
                    node.model,
                    node.strategy,
                    node.max_tokens,
                    node.temperature,
                    node.branch_index,
                    node.created_at,
                    json.dumps(node.metadata, sort_keys=True),
                ),
            )

    @staticmethod
    def _node(row: sqlite3.Row) -> Node:
        return Node(
            id=row["id"],
            parent_id=row["parent_id"],
            root_id=row["root_id"],
            text=row["text"],
            model=row["model"],
            strategy=row["strategy"],
            max_tokens=row["max_tokens"],
            temperature=row["temperature"],
            branch_index=row["branch_index"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata_json"]),
        )

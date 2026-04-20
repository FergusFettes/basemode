"""Load loom-like trees into a common analysis shape.

Older loom tools stored nearly the same tree with small schema differences:
different parent field names, patch field names, id types, and model/type labels.
This module keeps those quirks out of the stats code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .store import GenerationStore, Node


@dataclass(frozen=True)
class AnalysisNode:
    id: str
    parent_id: str | None
    root_id: str
    text: str = ""
    model: str | None = None
    created_at: str | None = None
    branch_index: int | None = None
    generation_id: str | None = None
    bookmarked: bool = False
    hidden: bool = False
    rating: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisTree:
    root_id: str
    nodes: list[AnalysisNode]
    source_format: str = "unknown"


def tree_from_store(store: GenerationStore, root_id: str) -> AnalysisTree:
    root = store.root(root_id)
    nodes = [_from_store_node(node) for node in store.tree(root.id)]
    return AnalysisTree(root_id=root.id, nodes=nodes, source_format="basemode-store")


def load_loom_tree(path: str | Path) -> AnalysisTree:
    data = json.loads(Path(path).read_text())
    return parse_loom_tree(data)


def parse_loom_tree(data: dict[str, Any]) -> AnalysisTree:
    if _is_basemode_export(data):
        return _parse_basemode_export(data)
    if "loomTree" in data and isinstance(data["loomTree"], dict):
        node_store = data["loomTree"].get("nodeStore")
        if isinstance(node_store, dict):
            return _parse_legacy_mapping(node_store, source_format="minihf")
    if "nodes" in data:
        nodes = data["nodes"]
        if isinstance(nodes, list):
            return _parse_bonsai(nodes)
        if isinstance(nodes, dict):
            return _parse_legacy_mapping(nodes, source_format="tinyloom")
    raise ValueError("unknown loom tree format")


def _is_basemode_export(data: dict[str, Any]) -> bool:
    nodes = data.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return False
    first = nodes[0]
    return isinstance(first, dict) and "parent_id" in first and "root_id" in first


def _parse_basemode_export(data: dict[str, Any]) -> AnalysisTree:
    raw_nodes = data["nodes"]
    nodes = [
        AnalysisNode(
            id=str(raw["id"]),
            parent_id=_optional_str(raw.get("parent_id")),
            root_id=str(raw["root_id"]),
            text=str(raw.get("text", "")),
            model=_optional_str(raw.get("model")),
            created_at=_optional_str(raw.get("created_at")),
            branch_index=raw.get("branch_index"),
            generation_id=_optional_str(
                raw.get("generation_id") or raw.get("metadata", {}).get("generation_id")
            ),
            bookmarked=bool(raw.get("metadata", {}).get("bookmarked")),
            hidden=bool(raw.get("metadata", {}).get("hidden")),
            rating=_numeric(raw.get("metadata", {}).get("rating")),
            metadata=dict(raw.get("metadata", {})),
        )
        for raw in raw_nodes
    ]
    root_id = next(
        (node.id for node in nodes if node.parent_id is None), nodes[0].root_id
    )
    return AnalysisTree(root_id=root_id, nodes=nodes, source_format="basemode-json")


def _parse_legacy_mapping(
    raw_nodes: dict[str, dict[str, Any]], *, source_format: str
) -> AnalysisTree:
    nodes: list[AnalysisNode] = []
    root_id: str | None = None
    branch_counts: dict[str, int] = {}

    for key, raw in raw_nodes.items():
        node_id = str(raw.get("id", key))
        parent_id = _optional_str(raw.get("parent"))
        if parent_id is None:
            root_id = node_id
        branch_index = None
        if parent_id is not None:
            branch_index = branch_counts.get(parent_id, 0)
            branch_counts[parent_id] = branch_index + 1
        metadata = _metadata_from_raw(raw)
        nodes.append(
            AnalysisNode(
                id=node_id,
                parent_id=parent_id,
                root_id="",  # filled below once root is known
                text=_legacy_text(raw),
                model=_legacy_model(raw),
                created_at=_optional_str(raw.get("timestamp")),
                branch_index=branch_index,
                generation_id=_optional_str(raw.get("generation_id")),
                bookmarked=bool(raw.get("bookmarked")),
                hidden=bool(raw.get("hidden")),
                rating=_numeric(raw.get("rating")),
                metadata=metadata,
            )
        )

    if root_id is None:
        root_id = nodes[0].id if nodes else ""
    fixed = [AnalysisNode(**{**node.__dict__, "root_id": root_id}) for node in nodes]
    return AnalysisTree(root_id=root_id, nodes=fixed, source_format=source_format)


def _parse_bonsai(raw_nodes: list[dict[str, Any]]) -> AnalysisTree:
    nodes: list[AnalysisNode] = []
    root_id: str | None = None
    branch_counts: dict[str, int] = {}
    for raw in raw_nodes:
        parent_ids = raw.get("parentIds") or []
        parent_id = _optional_str(parent_ids[0]) if parent_ids else None
        node_id = str(raw["id"])
        if parent_id is None:
            root_id = node_id
        branch_index = None
        if parent_id is not None:
            branch_index = branch_counts.get(parent_id, 0)
            branch_counts[parent_id] = branch_index + 1
        nodes.append(
            AnalysisNode(
                id=node_id,
                parent_id=parent_id,
                root_id="",  # filled below once root is known
                text=str(raw.get("text", "")),
                model=_optional_str(raw.get("model") or raw.get("type")),
                created_at=_optional_str(raw.get("createdAt")),
                branch_index=branch_index,
                generation_id=_optional_str(raw.get("generationId")),
                bookmarked=bool(raw.get("bookmarked")),
                hidden=bool(raw.get("hidden")),
                rating=_numeric(raw.get("rating")),
                metadata=_metadata_from_raw(raw),
            )
        )
    if root_id is None:
        root_id = nodes[0].id if nodes else ""
    fixed = [AnalysisNode(**{**node.__dict__, "root_id": root_id}) for node in nodes]
    return AnalysisTree(root_id=root_id, nodes=fixed, source_format="bonsai")


def _from_store_node(node: Node) -> AnalysisNode:
    return AnalysisNode(
        id=node.id,
        parent_id=node.parent_id,
        root_id=node.root_id,
        text=node.text,
        model=node.model,
        created_at=node.created_at,
        branch_index=node.branch_index,
        generation_id=_optional_str(node.metadata.get("generation_id")),
        bookmarked=bool(node.metadata.get("bookmarked")),
        hidden=bool(node.metadata.get("hidden")),
        rating=_numeric(node.metadata.get("rating")),
        metadata=node.metadata,
    )


def _legacy_text(raw: dict[str, Any]) -> str:
    if "text" in raw:
        return str(raw.get("text") or "")
    patches = raw.get("patches") or raw.get("patch") or []
    if not patches:
        return ""
    pieces = []
    for patch in patches:
        for op, text in patch.get("diffs", []):
            if op == 1:
                pieces.append(str(text))
    return "".join(pieces)


def _legacy_model(raw: dict[str, Any]) -> str | None:
    value = raw.get("model") or raw.get("type")
    if value in {None, "root"}:
        return None
    return str(value)


def _metadata_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    metadata = (
        dict(raw.get("metadata", {})) if isinstance(raw.get("metadata"), dict) else {}
    )
    for key in ("type", "cache", "score", "_summary"):
        if key in raw:
            metadata[key] = raw[key]
    return metadata


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

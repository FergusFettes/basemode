"""Quantitative summaries for loom trees.

The metrics here intentionally use only durable facts in the current store:
tree shape, node lineage, and generation metadata. A node is treated as useful
evidence when the user expanded it, because expansion is the revealed preference
the current UI records reliably.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .loom_formats import AnalysisNode, AnalysisTree, tree_from_store
from .store import GenerationStore


@dataclass(frozen=True)
class NodeScores:
    node_id: str
    parent_id: str | None
    generation_id: str | None
    model: str | None
    depth: int
    children: int
    expanded: bool
    bookmarked: bool
    hidden: bool
    rating: float | None
    direct_utility: float
    descendant_score: float
    discounted_descendant_score: float
    normalized_peer_descendant_score: float | None
    batch_win: float | None


@dataclass(frozen=True)
class MetricSummary:
    count: int
    mean: float
    stdev: float


@dataclass(frozen=True)
class ModelStats:
    model: str
    nodes: int
    expanded: int
    bookmarked: int
    hidden: int
    expansion_rate: float
    bookmark_rate: float
    hidden_rate: float
    descendant_score: MetricSummary
    discounted_descendant_score: MetricSummary
    normalized_peer_descendant_score: MetricSummary
    batch_win_rate: MetricSummary


@dataclass(frozen=True)
class PathStats:
    node_id: str
    depth: int
    models: dict[str, int]
    generated_nodes: int


@dataclass(frozen=True)
class LoomStats:
    root_id: str
    total_nodes: int
    generated_nodes: int
    leaf_nodes: int
    expanded_nodes: int
    max_depth: int
    model_counts: dict[str, int]
    path: PathStats | None
    node_scores: list[NodeScores]
    model_stats: list[ModelStats]

    def as_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "total_nodes": self.total_nodes,
            "generated_nodes": self.generated_nodes,
            "leaf_nodes": self.leaf_nodes,
            "expanded_nodes": self.expanded_nodes,
            "max_depth": self.max_depth,
            "model_counts": self.model_counts,
            "path": _dataclass_dict(self.path),
            "node_scores": [_dataclass_dict(score) for score in self.node_scores],
            "model_stats": [_dataclass_dict(stats) for stats in self.model_stats],
        }


def analyze_tree(
    store: GenerationStore,
    root_id: str,
    *,
    path_node_id: str | None = None,
) -> LoomStats:
    """Compute tree, path, node, and model stats for a loom tree."""
    tree = tree_from_store(store, root_id)
    return analyze_analysis_tree(tree, path_node_id=path_node_id)


def analyze_analysis_tree(
    tree: AnalysisTree,
    *,
    path_node_id: str | None = None,
) -> LoomStats:
    """Compute stats for any loom-like tree normalized by loom_formats."""
    nodes = tree.nodes
    by_id = {node.id: node for node in nodes}
    children_by_parent: dict[str, list[AnalysisNode]] = defaultdict(list)
    for node in nodes:
        if node.parent_id is not None:
            children_by_parent[node.parent_id].append(node)

    root = by_id.get(tree.root_id) or next(
        (n for n in nodes if n.parent_id is None), None
    )
    if root is None:
        return LoomStats(
            root_id=tree.root_id,
            total_nodes=0,
            generated_nodes=0,
            leaf_nodes=0,
            expanded_nodes=0,
            max_depth=0,
            model_counts={},
            path=None,
            node_scores=[],
            model_stats=[],
        )

    depth_by_id: dict[str, int] = {root.id: 0}
    stack = [root]
    while stack:
        node = stack.pop()
        for child in children_by_parent.get(node.id, []):
            depth_by_id[child.id] = depth_by_id[node.id] + 1
            stack.append(child)

    descendant_score: dict[str, float] = {}
    discounted_score: dict[str, float] = {}

    def base_score(node: AnalysisNode) -> float:
        if node.hidden:
            return 0.0
        score = 1.0 if children_by_parent.get(node.id) else 0.0
        score += 1.0 if node.bookmarked else 0.0
        if node.rating is not None and node.rating > 0:
            score += node.rating
        return score

    def visit(node: AnalysisNode, distance: int = 0) -> tuple[float, float]:
        ds = base_score(node)
        discount = 1.0 if distance == 0 else math.log2(distance + 2)
        dds = base_score(node) / discount
        for child in children_by_parent.get(node.id, []):
            child_ds, child_dds = visit(child, distance + 1)
            ds += child_ds
            dds += child_dds
        descendant_score[node.id] = ds
        discounted_score[node.id] = dds
        return ds, dds

    visit(root)

    peer_score: dict[str, float] = {}
    batch_win: dict[str, float] = {}
    for siblings in _choice_sets(children_by_parent).values():
        sibling_total = sum(descendant_score[child.id] for child in siblings)
        if sibling_total > 0:
            for child in siblings:
                peer_score[child.id] = (
                    descendant_score[child.id] / sibling_total * len(siblings)
                )
        max_score = max(descendant_score[child.id] for child in siblings)
        winners = [
            child
            for child in siblings
            if descendant_score[child.id] == max_score and max_score > 0
        ]
        for child in siblings:
            batch_win[child.id] = 1 / len(winners) if child in winners else 0.0

    node_scores = [
        NodeScores(
            node_id=node.id,
            parent_id=node.parent_id,
            generation_id=node.generation_id,
            model=node.model,
            depth=depth_by_id.get(node.id, 0),
            children=len(children_by_parent.get(node.id, [])),
            expanded=bool(children_by_parent.get(node.id)),
            bookmarked=node.bookmarked,
            hidden=node.hidden,
            rating=node.rating,
            direct_utility=base_score(node),
            descendant_score=descendant_score[node.id],
            discounted_descendant_score=discounted_score[node.id],
            normalized_peer_descendant_score=peer_score.get(node.id),
            batch_win=batch_win.get(node.id),
        )
        for node in nodes
    ]

    model_counts = Counter(node.model for node in nodes if node.model)
    path = _path_stats_from_tree(by_id, path_node_id) if path_node_id else None
    model_stats = _model_stats(node_scores)

    return LoomStats(
        root_id=tree.root_id,
        total_nodes=len(nodes),
        generated_nodes=sum(1 for node in nodes if node.model),
        leaf_nodes=sum(1 for node in nodes if not children_by_parent.get(node.id)),
        expanded_nodes=sum(1 for node in nodes if children_by_parent.get(node.id)),
        max_depth=max(depth_by_id.values(), default=0),
        model_counts=dict(sorted(model_counts.items())),
        path=path,
        node_scores=node_scores,
        model_stats=model_stats,
    )


def _path_stats(store: GenerationStore, node_id: str) -> PathStats:
    lineage = store.lineage(node_id)
    models = Counter(node.model for node in lineage if node.model)
    return PathStats(
        node_id=lineage[-1].id,
        depth=len(lineage) - 1,
        models=dict(sorted(models.items())),
        generated_nodes=sum(1 for node in lineage if node.model),
    )


def _path_stats_from_tree(
    by_id: dict[str, AnalysisNode],
    node_id: str,
) -> PathStats | None:
    node = by_id.get(node_id)
    if node is None:
        return None
    lineage: list[AnalysisNode] = []
    while node is not None:
        lineage.append(node)
        node = by_id.get(node.parent_id) if node.parent_id else None
    lineage.reverse()
    models = Counter(node.model for node in lineage if node.model)
    return PathStats(
        node_id=lineage[-1].id,
        depth=len(lineage) - 1,
        models=dict(sorted(models.items())),
        generated_nodes=sum(1 for node in lineage if node.model),
    )


def _model_stats(node_scores: list[NodeScores]) -> list[ModelStats]:
    by_model: dict[str, list[NodeScores]] = defaultdict(list)
    for score in node_scores:
        if score.model:
            by_model[score.model].append(score)

    result = []
    for model, scores in sorted(by_model.items()):
        peer_values = [
            score.normalized_peer_descendant_score
            for score in scores
            if score.normalized_peer_descendant_score is not None
        ]
        result.append(
            ModelStats(
                model=model,
                nodes=len(scores),
                expanded=sum(1 for score in scores if score.expanded),
                bookmarked=sum(1 for score in scores if score.bookmarked),
                hidden=sum(1 for score in scores if score.hidden),
                expansion_rate=sum(1 for score in scores if score.expanded)
                / len(scores),
                bookmark_rate=sum(1 for score in scores if score.bookmarked)
                / len(scores),
                hidden_rate=sum(1 for score in scores if score.hidden) / len(scores),
                descendant_score=_summary([score.descendant_score for score in scores]),
                discounted_descendant_score=_summary(
                    [score.discounted_descendant_score for score in scores]
                ),
                normalized_peer_descendant_score=_summary(peer_values),
                batch_win_rate=_summary([score.batch_win for score in scores]),
            )
        )
    result.sort(
        key=lambda stats: (
            stats.normalized_peer_descendant_score.mean,
            stats.descendant_score.mean,
        ),
        reverse=True,
    )
    return result


def _choice_sets(
    children_by_parent: dict[str, list[AnalysisNode]],
) -> dict[str, list[AnalysisNode]]:
    sets: dict[str, list[AnalysisNode]] = {}
    for parent_id, children in children_by_parent.items():
        by_generation: dict[str, list[AnalysisNode]] = defaultdict(list)
        unbatched = []
        for child in children:
            if child.generation_id:
                by_generation[f"{parent_id}:{child.generation_id}"].append(child)
            else:
                unbatched.append(child)
        sets.update(by_generation)
        if unbatched:
            sets[f"{parent_id}:children"] = unbatched
    return {key: value for key, value in sets.items() if len(value) > 1}


def _summary(values: list[float | None]) -> MetricSummary:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return MetricSummary(count=0, mean=0.0, stdev=0.0)
    return MetricSummary(
        count=len(clean),
        mean=sum(clean) / len(clean),
        stdev=statistics.stdev(clean) if len(clean) > 1 else 0.0,
    )


def _dataclass_dict(value: Any) -> Any:
    if value is None:
        return None
    if not hasattr(value, "__dataclass_fields__"):
        return value
    return {
        field: _dataclass_dict(getattr(value, field))
        for field in value.__dataclass_fields__
    }

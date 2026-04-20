import pytest

from basemode_loom.stats import analyze_tree
from basemode_loom.store import GenerationStore


@pytest.fixture
def preference_tree(tmp_path):
    """Root -> [A, B]; A -> [A1, A2]; A1 -> [A1a]."""
    store = GenerationStore(tmp_path / "stats.sqlite")
    root = store.create_root("Root")
    a = store.add_child(
        root.id,
        " A",
        model="model-a",
        strategy="system",
        max_tokens=10,
        temperature=0.9,
        branch_index=0,
    )
    b = store.add_child(
        root.id,
        " B",
        model="model-b",
        strategy="system",
        max_tokens=10,
        temperature=0.9,
        branch_index=1,
    )
    a1 = store.add_child(
        a.id,
        " A1",
        model="model-a",
        strategy="system",
        max_tokens=10,
        temperature=0.9,
        branch_index=0,
    )
    a2 = store.add_child(
        a.id,
        " A2",
        model="model-b",
        strategy="system",
        max_tokens=10,
        temperature=0.9,
        branch_index=1,
    )
    a1a = store.add_child(
        a1.id,
        " A1a",
        model="model-a",
        strategy="system",
        max_tokens=10,
        temperature=0.9,
        branch_index=0,
    )
    return store, root, a, b, a1, a2, a1a


def test_analyze_tree_counts_shape_and_models(preference_tree) -> None:
    store, root, *_ = preference_tree

    stats = analyze_tree(store, root.id)

    assert stats.total_nodes == 6
    assert stats.generated_nodes == 5
    assert stats.leaf_nodes == 3
    assert stats.expanded_nodes == 3
    assert stats.max_depth == 3
    assert stats.model_counts == {"model-a": 3, "model-b": 2}


def test_analyze_tree_scores_expansion_as_revealed_preference(preference_tree) -> None:
    store, root, a, b, a1, a2, a1a = preference_tree

    stats = analyze_tree(store, root.id)
    by_id = {score.node_id: score for score in stats.node_scores}

    assert by_id[a.id].descendant_score == 2
    assert by_id[b.id].descendant_score == 0
    assert by_id[a1.id].descendant_score == 1
    assert by_id[a2.id].descendant_score == 0
    assert by_id[a1a.id].descendant_score == 0

    assert by_id[a.id].normalized_peer_descendant_score == 2
    assert by_id[b.id].normalized_peer_descendant_score == 0
    assert by_id[a1.id].normalized_peer_descendant_score == 2
    assert by_id[a2.id].normalized_peer_descendant_score == 0


def test_analyze_tree_model_stats_are_averaged_by_model(preference_tree) -> None:
    store, root, *_ = preference_tree

    stats = analyze_tree(store, root.id)
    by_model = {model.model: model for model in stats.model_stats}

    assert by_model["model-a"].nodes == 3
    assert by_model["model-a"].expanded == 2
    assert by_model["model-a"].expansion_rate == pytest.approx(2 / 3)
    assert by_model["model-a"].batch_win_rate.mean == pytest.approx(1)
    assert by_model["model-a"].descendant_score.mean == pytest.approx(1)
    assert by_model["model-b"].descendant_score.mean == pytest.approx(0)


def test_analyze_tree_path_stats(preference_tree) -> None:
    store, root, _a, _b, _a1, _a2, a1a = preference_tree

    stats = analyze_tree(store, root.id, path_node_id=a1a.id)

    assert stats.path is not None
    assert stats.path.node_id == a1a.id
    assert stats.path.depth == 3
    assert stats.path.generated_nodes == 3
    assert stats.path.models == {"model-a": 3}


def test_analyze_tree_as_dict_is_json_ready(preference_tree) -> None:
    store, root, *_ = preference_tree

    data = analyze_tree(store, root.id).as_dict()

    assert data["root_id"] == root.id
    assert data["model_stats"][0]["descendant_score"]["mean"] >= 0

import json

from basemode_loom.loom_formats import parse_loom_tree


def test_parse_basemode_export() -> None:
    tree = parse_loom_tree(
        {
            "version": 1,
            "nodes": [
                {
                    "id": "root",
                    "parent_id": None,
                    "root_id": "root",
                    "text": "Seed",
                    "model": None,
                    "strategy": None,
                    "max_tokens": None,
                    "temperature": None,
                    "branch_index": None,
                    "created_at": "now",
                    "metadata": {},
                },
                {
                    "id": "child",
                    "parent_id": "root",
                    "root_id": "root",
                    "text": " alpha",
                    "model": "model-a",
                    "strategy": "system",
                    "max_tokens": 20,
                    "temperature": 0.9,
                    "branch_index": 0,
                    "created_at": "later",
                    "metadata": {"bookmarked": True, "generation_id": "g1"},
                },
            ],
        }
    )

    assert tree.source_format == "basemode-json"
    assert tree.root_id == "root"
    assert tree.nodes[1].bookmarked is True
    assert tree.nodes[1].generation_id == "g1"


def test_parse_tinyloom_mapping() -> None:
    tree = parse_loom_tree(
        {
            "root": 1,
            "nodes": {
                "1": {
                    "id": 1,
                    "parent": None,
                    "type": "root",
                    "patches": None,
                    "timestamp": 1,
                    "children": [2],
                },
                "2": {
                    "id": 2,
                    "parent": 1,
                    "type": "model-a",
                    "patches": [{"diffs": [[0, "Seed"], [1, " alpha"]]}],
                    "timestamp": 2,
                    "children": [],
                    "bookmarked": True,
                    "hidden": False,
                },
            },
        }
    )

    assert tree.source_format == "tinyloom"
    assert tree.root_id == "1"
    assert tree.nodes[1].parent_id == "1"
    assert tree.nodes[1].model == "model-a"
    assert tree.nodes[1].text == " alpha"


def test_parse_minihf_node_store() -> None:
    tree = parse_loom_tree(
        {
            "loomTree": {
                "nodeStore": {
                    "1": {
                        "id": "1",
                        "parent": None,
                        "type": "root",
                        "patch": None,
                        "timestamp": 1000,
                    },
                    "2": {
                        "id": "2",
                        "parent": "1",
                        "type": "model-a",
                        "patch": [{"diffs": [[1, "hello"]]}],
                        "timestamp": 2000,
                    },
                }
            }
        }
    )

    assert tree.source_format == "minihf"
    assert tree.root_id == "1"
    assert tree.nodes[1].text == "hello"


def test_parse_bonsai_list() -> None:
    tree = parse_loom_tree(
        {
            "nodes": [
                {
                    "id": "root",
                    "parentIds": [],
                    "text": "Seed",
                    "createdAt": "now",
                },
                {
                    "id": "child",
                    "parentIds": ["root"],
                    "text": "Seed alpha",
                    "createdAt": "later",
                    "model": "model-a",
                    "hidden": True,
                },
            ]
        }
    )

    assert tree.source_format == "bonsai"
    assert tree.root_id == "root"
    assert tree.nodes[1].parent_id == "root"
    assert tree.nodes[1].hidden is True


def test_parse_unknown_format_raises() -> None:
    try:
        parse_loom_tree({"nope": []})
    except ValueError as exc:
        assert "unknown loom tree format" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_formats_are_json_serializable() -> None:
    tree = parse_loom_tree({"nodes": [{"id": "root", "parentIds": [], "text": ""}]})

    assert json.dumps({"root_id": tree.root_id})

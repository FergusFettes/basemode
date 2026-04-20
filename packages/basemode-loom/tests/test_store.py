import pytest

from basemode_loom.store import AmbiguousNodeReference, GenerationStore, Node


def test_save_continuations_creates_root_and_branch_children(tmp_path) -> None:
    store = GenerationStore(tmp_path / "generations.sqlite")

    parent, children = store.save_continuations(
        "The ship rounded",
        [" the headland", " into fog"],
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=20,
        temperature=0.9,
    )

    assert parent.parent_id is None
    assert parent.root_id == parent.id
    assert [child.parent_id for child in children] == [parent.id, parent.id]
    assert [child.branch_index for child in children] == [0, 1]
    assert store.full_text(children[0].id) == "The ship rounded the headland"
    assert store.full_text(children[1].id) == "The ship rounded into fog"


def test_save_continuations_can_continue_from_existing_node(tmp_path) -> None:
    store = GenerationStore(tmp_path / "generations.sqlite")
    _, first_children = store.save_continuations(
        "A",
        ["B"],
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=5,
        temperature=0.7,
    )

    parent, next_children = store.save_continuations(
        "ignored when parent_id is present",
        ["C", "D"],
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=5,
        temperature=0.7,
        parent_id=first_children[0].id,
    )

    assert parent.id == first_children[0].id
    assert [child.root_id for child in next_children] == [
        parent.root_id,
        parent.root_id,
    ]
    assert store.full_text(next_children[0].id) == "ABC"
    assert store.full_text(next_children[1].id) == "ABD"


def test_children_are_returned_in_branch_order(tmp_path) -> None:
    store = GenerationStore(tmp_path / "generations.sqlite")
    parent = store.create_root("root")
    second = store.add_child(
        parent.id,
        " second",
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=5,
        temperature=0.7,
        branch_index=1,
    )
    first = store.add_child(
        parent.id,
        " first",
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=5,
        temperature=0.7,
        branch_index=0,
    )

    assert [node.id for node in store.children(parent.id)] == [first.id, second.id]


def test_unknown_parent_raises(tmp_path) -> None:
    store = GenerationStore(tmp_path / "generations.sqlite")

    with pytest.raises(KeyError):
        store.add_child(
            "missing",
            "text",
            model="gpt-4o-mini",
            strategy="system",
            max_tokens=5,
            temperature=0.7,
        )


def test_update_metadata_merges_existing_values(tmp_path) -> None:
    store = GenerationStore(tmp_path / "generations.sqlite")
    root = store.create_root("root", metadata={"source": "test"})

    updated = store.update_metadata(root.id, {"name": "this-is-the-topic"})

    assert updated.metadata == {"source": "test", "name": "this-is-the-topic"}
    assert store.get(root.id).metadata["name"] == "this-is-the-topic"


def test_get_resolves_unique_id_substrings(tmp_path) -> None:
    store = GenerationStore(tmp_path / "generations.sqlite")
    root = store.create_root("root")
    child = store.add_child(
        root.id,
        " child",
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=5,
        temperature=0.7,
    )

    assert store.get(child.id[:12]).id == child.id
    assert store.full_text(child.id[:12]) == "root child"


def test_children_resolves_unique_id_substrings(tmp_path) -> None:
    store = GenerationStore(tmp_path / "generations.sqlite")
    parent = store.create_root("root")
    child = store.add_child(
        parent.id,
        " child",
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=5,
        temperature=0.7,
    )

    assert store.children(parent.id[:10]) == [child]


def test_ambiguous_node_prefix_raises(tmp_path) -> None:
    store = GenerationStore(tmp_path / "generations.sqlite")
    _insert_test_node(store, "shared-aaa-one")
    _insert_test_node(store, "shared-bbb-two")

    with pytest.raises(AmbiguousNodeReference) as exc:
        store.get("shared")

    assert exc.value.matches == ["shared-bbb-two", "shared-aaa-one"]


def test_active_node_state_persists(tmp_path) -> None:
    path = tmp_path / "generations.sqlite"
    store = GenerationStore(path)
    root = store.create_root("root")

    store.set_active_node(root.id)

    reopened = GenerationStore(path)
    assert reopened.get_active_node_id() == root.id
    assert reopened.get_active_node().id == root.id


def test_delete_tree_removes_nodes_and_related_state(tmp_path) -> None:
    store = GenerationStore(tmp_path / "generations.sqlite")
    root = store.create_root("root")
    child = store.add_child(
        root.id,
        " child",
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=5,
        temperature=0.7,
    )
    grandchild = store.add_child(
        child.id,
        " grandchild",
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=5,
        temperature=0.7,
    )
    other_root = store.create_root("other")
    store.set_active_node(grandchild.id)
    store.set_checked_out_child(root.id, child.id)

    deleted = store.delete_tree(root.id)

    assert deleted == 3
    assert store.get(root.id) is None
    assert store.get(child.id) is None
    assert store.get(grandchild.id) is None
    assert store.roots() == [other_root]
    assert store.get_active_node_id() is None
    assert store.get_checked_out_child_id(root.id) is None


def test_select_branch_is_one_based(tmp_path) -> None:
    store = GenerationStore(tmp_path / "generations.sqlite")
    parent = store.create_root("root")
    first = store.add_child(
        parent.id,
        " first",
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=5,
        temperature=0.7,
        branch_index=0,
    )
    second = store.add_child(
        parent.id,
        " second",
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=5,
        temperature=0.7,
        branch_index=1,
    )

    assert store.select_branch(parent.id, 1) == first
    assert store.select_branch(parent.id, 2) == second


def test_select_branch_out_of_range_raises(tmp_path) -> None:
    store = GenerationStore(tmp_path / "generations.sqlite")
    parent = store.create_root("root")

    with pytest.raises(IndexError):
        store.select_branch(parent.id, 1)


def _insert_test_node(store: GenerationStore, node_id: str) -> None:
    store._insert(
        Node(
            id=node_id,
            parent_id=None,
            root_id=node_id,
            text=node_id,
            model=None,
            strategy=None,
            max_tokens=None,
            temperature=None,
            branch_index=None,
            created_at=node_id,
            metadata={},
        )
    )

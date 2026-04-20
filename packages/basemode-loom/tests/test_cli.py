from typer.testing import CliRunner

from basemode_loom.cli import app
from basemode_loom.store import GenerationStore

runner = CliRunner()


def test_top_level_help_includes_loom() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "loom" in result.output


def test_loom_help_lists_stateful_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "continue" in result.output
    assert "active" in result.output
    assert "nodes" in result.output


def test_loom_continue_and_branch_selection(tmp_path, monkeypatch) -> None:
    db = tmp_path / "generations.sqlite"

    async def fake_stream_one(*args, **kwargs):
        return " gamma"

    monkeypatch.setattr("basemode_loom.cli._stream_one", fake_stream_one)
    monkeypatch.setattr("basemode_loom.cli.generate_name", lambda text: None)
    monkeypatch.setattr("basemode_loom.cli.should_name", lambda text: False)

    store = GenerationStore(db)
    parent, _ = store.save_continuations(
        "Seed",
        [" alpha", " beta"],
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=20,
        temperature=0.9,
    )
    store.set_active_node(parent.id)

    second = runner.invoke(app, ["continue", "-b", "2", "--db", str(db)])
    assert second.exit_code == 0, second.output

    store = GenerationStore(db)
    active = store.get_active_node()
    assert active is not None
    assert store.full_text(active.id).endswith("beta gamma")


def test_loom_select_marks_active(tmp_path) -> None:
    db = tmp_path / "generations.sqlite"

    store = GenerationStore(db)
    parent, children = store.save_continuations(
        "Seed",
        [" alpha", " beta"],
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=20,
        temperature=0.9,
    )
    store.set_active_node(parent.id)
    child = children[0]

    select = runner.invoke(app, ["select", child.id[:10], "--db", str(db)])
    assert select.exit_code == 0, select.output

    nodes = runner.invoke(app, ["nodes", "--db", str(db)])
    assert nodes.exit_code == 0, nodes.output
    assert "*" in nodes.output

    active_output = runner.invoke(app, ["active", "--db", str(db)])
    assert active_output.exit_code == 0, active_output.output
    assert child.id in active_output.output

    show = runner.invoke(app, ["show", child.id[:10], "--segment", "--db", str(db)])
    assert show.exit_code == 0, show.output
    assert " alpha" in show.output

    children = runner.invoke(app, ["children", parent.id[:10], "--db", str(db)])
    assert children.exit_code == 0, children.output
    assert "alpha" in children.output
    assert "beta" in children.output

    active = GenerationStore(db).get_active_node()
    assert active is not None
    assert active.id == child.id


def test_loom_export_md_prints_checked_out_path(tmp_path) -> None:
    db = tmp_path / "generations.sqlite"
    store = GenerationStore(db)
    parent, children = store.save_continuations(
        "Seed",
        [" alpha", " beta"],
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=20,
        temperature=0.9,
    )
    grandchild = store.add_child(
        children[1].id,
        " gamma",
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=20,
        temperature=0.9,
    )
    store.update_metadata(parent.id, {"last_node_id": grandchild.id})
    store.set_active_node(parent.id)

    result = runner.invoke(app, ["export", "--to", "md", "--db", str(db)])

    assert result.exit_code == 0, result.output
    assert result.output == "Seed beta gamma\n"


def test_loom_export_md_file_uses_extension(tmp_path) -> None:
    db = tmp_path / "generations.sqlite"
    out = tmp_path / "checked-out.md"
    store = GenerationStore(db)
    _parent, children = store.save_continuations(
        "Seed",
        [" alpha"],
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=20,
        temperature=0.9,
    )
    store.set_active_node(children[0].id)

    result = runner.invoke(app, ["export", "--to", str(out), "--db", str(db)])

    assert result.exit_code == 0, result.output
    assert out.read_text() == "Seed alpha\n"


def test_loom_stats_prints_tree_and_model_stats(tmp_path) -> None:
    db = tmp_path / "generations.sqlite"
    store = GenerationStore(db)
    parent, children = store.save_continuations(
        "Seed",
        [" alpha", " beta"],
        model="model-a",
        strategy="system",
        max_tokens=20,
        temperature=0.9,
    )
    store.save_continuations(
        "",
        [" gamma"],
        model="model-b",
        strategy="system",
        max_tokens=20,
        temperature=0.9,
        parent_id=children[0].id,
    )
    store.set_active_node(children[0].id)

    result = runner.invoke(app, ["stats", "--db", str(db)])

    assert result.exit_code == 0, result.output
    assert "Total nodes" in result.output
    assert "model-a" in result.output
    assert "Path model" in result.output
    assert parent.id in result.output


def test_loom_stats_json(tmp_path) -> None:
    db = tmp_path / "generations.sqlite"
    store = GenerationStore(db)
    parent, children = store.save_continuations(
        "Seed",
        [" alpha"],
        model="model-a",
        strategy="system",
        max_tokens=20,
        temperature=0.9,
    )
    store.set_active_node(children[0].id)

    result = runner.invoke(app, ["stats", parent.id, "--json", "--db", str(db)])

    assert result.exit_code == 0, result.output
    assert '"total_nodes": 2' in result.output
    assert '"model": "model-a"' in result.output


def test_loom_stats_can_analyze_json_file(tmp_path) -> None:
    path = tmp_path / "tinyloom.json"
    path.write_text(
        """
        {
          "root": 1,
          "nodes": {
            "1": {"id": 1, "parent": null, "type": "root", "timestamp": 1},
            "2": {
              "id": 2,
              "parent": 1,
              "type": "model-a",
              "timestamp": 2,
              "patches": [{"diffs": [[1, " alpha"]]}],
              "bookmarked": true
            }
          }
        }
        """
    )

    result = runner.invoke(app, ["stats", "--file", str(path), "--json"])

    assert result.exit_code == 0, result.output
    assert '"source_format"' not in result.output
    assert '"model": "model-a"' in result.output
    assert '"bookmarked": true' in result.output

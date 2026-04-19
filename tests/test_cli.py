from typer.testing import CliRunner

from basemode.cli import app
from basemode.store import GenerationStore

runner = CliRunner()


def test_top_level_help_includes_loom() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "loom" in result.output


def test_loom_help_lists_stateful_commands() -> None:
    result = runner.invoke(app, ["loom", "--help"])

    assert result.exit_code == 0
    assert "continue" in result.output
    assert "active" in result.output
    assert "nodes" in result.output


def test_loom_continue_and_branch_selection(tmp_path, monkeypatch) -> None:
    db = tmp_path / "generations.sqlite"

    async def fake_stream_one(*args, **kwargs):
        return " gamma"

    monkeypatch.setattr("basemode.cli._stream_one", fake_stream_one)
    monkeypatch.setattr("basemode.cli.generate_name", lambda text: None)
    monkeypatch.setattr("basemode.cli.should_name", lambda text: False)

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

    second = runner.invoke(app, ["loom", "continue", "-b", "2", "--db", str(db)])
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

    select = runner.invoke(app, ["loom", "select", child.id[:10], "--db", str(db)])
    assert select.exit_code == 0, select.output

    nodes = runner.invoke(app, ["loom", "nodes", "--db", str(db)])
    assert nodes.exit_code == 0, nodes.output
    assert "*" in nodes.output

    active_output = runner.invoke(app, ["loom", "active", "--db", str(db)])
    assert active_output.exit_code == 0, active_output.output
    assert child.id in active_output.output

    show = runner.invoke(app, ["loom", "show", child.id[:10], "--segment", "--db", str(db)])
    assert show.exit_code == 0, show.output
    assert " alpha" in show.output

    children = runner.invoke(app, ["loom", "children", parent.id[:10], "--db", str(db)])
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

    result = runner.invoke(app, ["loom", "export", "--to", "md", "--db", str(db)])

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

    result = runner.invoke(app, ["loom", "export", "--to", str(out), "--db", str(db)])

    assert result.exit_code == 0, result.output
    assert out.read_text() == "Seed alpha\n"

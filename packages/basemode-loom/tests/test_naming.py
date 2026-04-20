from types import SimpleNamespace

from basemode_loom.cli import _maybe_name_tree
from basemode_loom.naming import choose_title_model, generate_name, should_name, slugify
from basemode_loom.store import GenerationStore


def test_choose_title_model_prefers_openai_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("basemode_loom.naming.get_key", lambda provider: None)

    assert choose_title_model() == "gpt-4o-mini"


def test_choose_title_model_uses_anthropic_when_openai_missing(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr("basemode_loom.naming.get_key", lambda provider: None)

    assert choose_title_model() == "anthropic/claude-3-haiku-20240307"


def test_slugify_keeps_short_readable_slug() -> None:
    assert slugify('"The Topic: A Strange Story!"') == "the-topic-a-strange-story"


def test_should_name_respects_threshold() -> None:
    assert should_name("word " * 600, threshold=100)
    assert not should_name("short text", threshold=100)


def test_generate_name_sanitizes_model_output(monkeypatch) -> None:
    def fake_completion(**kwargs):
        assert kwargs["model"] == "gpt-4o-mini"
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=" The Topic: A Strange Story! ")
                )
            ]
        )

    monkeypatch.setattr("basemode_loom.naming.litellm.completion", fake_completion)

    assert (
        generate_name("long text", model="gpt-4o-mini") == "the-topic-a-strange-story"
    )


def test_maybe_name_tree_updates_root_metadata(tmp_path, monkeypatch) -> None:
    store = GenerationStore(tmp_path / "generations.sqlite")
    _, children = store.save_continuations(
        "root ",
        ["continuation"],
        model="gpt-4o-mini",
        strategy="system",
        max_tokens=5,
        temperature=0.7,
    )
    monkeypatch.setattr("basemode_loom.cli.should_name", lambda text: True)
    monkeypatch.setattr(
        "basemode_loom.cli.generate_name", lambda text: "this-is-the-topic"
    )

    _maybe_name_tree(store, children)

    root = store.root(children[0].id)
    assert root.metadata["name"] == "this-is-the-topic"
    assert root.metadata["named_from"] == children[0].id

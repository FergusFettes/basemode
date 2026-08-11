import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from basemode import server  # noqa: E402


async def _fake_continue_text(prefix, model=None, **kwargs):
    for token in [" hello", " world"]:
        yield token


async def _fake_branch_text(prefix, model=None, *, n=1, **kwargs):
    for idx in range(n):
        yield idx, f" branch{idx}"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "continue_text", _fake_continue_text)
    monkeypatch.setattr(server, "branch_text", _fake_branch_text)
    monkeypatch.setattr(server, "get_default_model", lambda: "gpt-4o-mini")
    return TestClient(server.app)


def test_completions_single(client) -> None:
    response = client.post(
        "/v1/completions",
        json={"model": "gpt-4o-mini", "prompt": "Once upon a time", "max_tokens": 10},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "gpt-4o-mini"
    assert len(body["choices"]) == 1
    assert body["choices"][0]["text"] == " hello world"
    assert body["choices"][0]["index"] == 0


def test_completions_echo_prepends_prompt(client) -> None:
    response = client.post(
        "/v1/completions",
        json={"prompt": "Once upon a time", "echo": True},
    )

    assert response.status_code == 200
    text = response.json()["choices"][0]["text"]
    assert text == "Once upon a time hello world"


def test_completions_branches(client) -> None:
    response = client.post(
        "/v1/completions",
        json={"prompt": "The ship rounded the headland", "n": 3},
    )

    assert response.status_code == 200
    choices = response.json()["choices"]
    assert [c["index"] for c in choices] == [0, 1, 2]
    assert [c["text"] for c in choices] == [" branch0", " branch1", " branch2"]


def test_completions_uses_default_model_when_unset(client) -> None:
    response = client.post("/v1/completions", json={"prompt": "hi"})

    assert response.status_code == 200
    assert response.json()["model"] == "gpt-4o-mini"


def test_completions_prompt_list_is_concatenated(client) -> None:
    response = client.post(
        "/v1/completions",
        json={"prompt": ["part one ", "part two"], "echo": True},
    )

    assert response.status_code == 200
    text = response.json()["choices"][0]["text"]
    assert text.startswith("part one part two")


def test_completions_error_returns_502(client, monkeypatch) -> None:
    async def _boom(prefix, model=None, **kwargs):
        raise RuntimeError("provider exploded")
        yield  # pragma: no cover - unreachable, keeps this an async generator

    monkeypatch.setattr(server, "continue_text", _boom)

    response = client.post("/v1/completions", json={"prompt": "hi"})

    assert response.status_code == 502
    assert "provider exploded" in response.json()["detail"]


def test_list_models(client) -> None:
    response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "gpt-4o-mini"

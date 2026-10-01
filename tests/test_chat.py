from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from basemode import chat as chat_module
from basemode.chat import (
    _cut_at,
    build_messages,
    chat_text,
    has_chat_endpoint,
    transcript_prompt,
)
from basemode.cli import app
from basemode.exceptions import EmptyCompletionError
from basemode.transport import set_transport

runner = CliRunner()


def _chat_chunk(content: str | None, finish: str | None = None) -> Any:
    delta = SimpleNamespace(content=content)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=None
    )


def _text_chunk(text: str) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(text=text, finish_reason=None)], usage=None
    )


async def _aiter(items):
    for item in items:
        yield item


class FakeTransport:
    def __init__(self, chat=(), text=()):
        self.chat = list(chat)
        self.text = list(text)
        self.requests: list[dict] = []

    async def chat_completion(self, **request):
        self.requests.append(request)
        return _aiter(self.chat)

    async def text_completion(self, **request):
        self.requests.append(request)
        return _aiter(self.text)


@pytest.fixture
def transport():
    holder: dict[str, FakeTransport] = {}

    def install(fake: FakeTransport) -> FakeTransport:
        holder["previous"] = set_transport(fake)
        return fake

    yield install
    if "previous" in holder:
        set_transport(holder["previous"])


async def _collect(stream) -> str:
    return "".join([token async for token in stream])


def test_build_messages_wraps_a_string_and_prepends_system() -> None:
    assert build_messages("hi", "be terse") == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]


def test_build_messages_keeps_an_existing_system_message() -> None:
    messages = [
        {"role": "system", "content": "mine"},
        {"role": "user", "content": "hi"},
    ]
    assert build_messages(messages, "ignored") == messages


def test_build_messages_rejects_an_empty_conversation() -> None:
    with pytest.raises(ValueError):
        build_messages([])


def test_transcript_prompt_ends_on_an_open_assistant_turn() -> None:
    prompt = transcript_prompt(
        [
            {"role": "system", "content": "A helpful exchange."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4."},
            {"role": "user", "content": "And 3+3?"},
        ]
    )
    assert prompt == (
        "A helpful exchange.\n\nUser: What is 2+2?\nAssistant: 4.\n"
        "User: And 3+3?\nAssistant:"
    )


@pytest.mark.asyncio
async def test_cut_at_stops_on_a_marker_split_across_chunks() -> None:
    chunks = ["Paris is", " the capital.\nUs", "er: next question"]
    assert await _collect(_cut_at(_aiter(chunks), "\nUser:")) == (
        "Paris is the capital."
    )


@pytest.mark.asyncio
async def test_cut_at_releases_a_held_back_near_miss() -> None:
    chunks = ["line one\nU", "nder the sea"]
    assert await _collect(_cut_at(_aiter(chunks), "\nUser:")) == (
        "line one\nUnder the sea"
    )


@pytest.mark.asyncio
async def test_chat_text_sends_messages_straight_to_the_chat_endpoint(
    transport,
) -> None:
    fake = transport(FakeTransport(chat=[_chat_chunk("Par"), _chat_chunk("is.")]))

    answer = await _collect(
        chat_text("Capital of France?", "openai/gpt-4o-mini", system="be terse")
    )

    assert answer == "Paris."
    request = fake.requests[0]
    assert request["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "Capital of France?"},
    ]
    # No continuation coercion anywhere in the request.
    assert "continuation" not in str(request["messages"]).lower()
    assert request["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_chat_text_raises_on_an_empty_answer(transport) -> None:
    transport(FakeTransport(chat=[_chat_chunk(None, finish="length")]))

    with pytest.raises(EmptyCompletionError) as excinfo:
        await _collect(chat_text("hi", "openai/gpt-4o-mini"))

    assert excinfo.value.strategy == "chat"


@pytest.mark.asyncio
async def test_chat_text_answers_base_models_through_a_transcript(
    transport,
) -> None:
    fake = transport(
        FakeTransport(text=[_text_chunk(" Paris."), _text_chunk("\nUser: and")])
    )

    answer = await _collect(chat_text("Capital of France?", "openai/davinci-002"))

    assert answer == "Paris."
    request = fake.requests[0]
    assert request["prompt"].endswith("User: Capital of France?\nAssistant:")
    assert request["stop"] == ["\nUser:"]


@pytest.mark.parametrize(
    ("model", "chats"),
    [
        ("openai/davinci-002", False),
        ("openai/gpt-3.5-turbo-instruct", False),
        ("groq/openai/gpt-oss-120b", True),
        ("anthropic/claude-haiku-4-5", True),
        ("somewhere/never-heard-of-it", True),
    ],
)
def test_has_chat_endpoint_only_excludes_known_completion_models(
    model: str, chats: bool
) -> None:
    assert has_chat_endpoint(model) is chats


@pytest.mark.asyncio
async def test_chat_text_transcript_flag_overrides_detection(transport) -> None:
    fake = transport(FakeTransport(text=[_text_chunk(" Hello.")]))

    answer = await _collect(
        chat_text("hi", "deepinfra/meta-llama/Llama-base", transcript=True)
    )

    assert answer == "Hello."
    assert fake.requests[0]["prompt"] == "User: hi\nAssistant:"


def test_ask_streams_an_answer_and_prepends_piped_stdin(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_chat_text(messages, model, **kwargs):
        seen["messages"] = messages
        seen["system"] = kwargs.get("system")
        kwargs["on_usage"]([{"prompt_tokens": 3, "completion_tokens": 2}])
        for token in ["[bold]Paris", "[/bold]."]:
            yield token

    monkeypatch.setattr(chat_module, "chat_text", fake_chat_text)
    result = runner.invoke(
        app,
        ["ask", "what city?", "-s", "be terse", "--show-usage"],
        input="some notes\n",
    )

    assert result.exit_code == 0
    # Model output is written verbatim, never interpreted as rich markup.
    assert "[bold]Paris[/bold]." in result.output
    assert "Prompt tokens" in result.output
    assert seen["messages"] == [{"role": "user", "content": "some notes\n\nwhat city?"}]
    assert seen["system"] == "be terse"


def test_ask_reports_a_provider_failure_on_one_line(monkeypatch) -> None:
    async def failing_chat_text(messages, model, **kwargs):
        raise RuntimeError("provider exploded\nbad key")
        yield ""

    monkeypatch.setattr(chat_module, "chat_text", failing_chat_text)

    result = runner.invoke(app, ["ask", "hi"])

    assert result.exit_code == 1
    assert "error: RuntimeError: bad key" in result.output
    assert "Traceback" not in result.output

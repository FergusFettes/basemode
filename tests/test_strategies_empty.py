"""Unit tests: strategies surface finish_reason when a stream yields no tokens."""

from types import SimpleNamespace

import pytest

from basemode.exceptions import EmptyCompletionError
from basemode.params import GenerationParams
from basemode.strategies.completion import CompletionStrategy
from basemode.strategies.prefill import PrefillStrategy
from basemode.strategies.system import SystemPromptStrategy


def _chat_chunk(content: str | None, finish_reason: str | None = None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content), finish_reason=finish_reason
            )
        ]
    )


def _text_chunk(text: str | None, finish_reason: str | None = None):
    return SimpleNamespace(
        choices=[SimpleNamespace(text=text, finish_reason=finish_reason)]
    )


async def _achunks(chunks):
    for chunk in chunks:
        yield chunk


async def test_system_strategy_raises_with_finish_reason(monkeypatch):
    async def fake_acompletion(**kwargs):
        return _achunks([_chat_chunk(None, "content_filter")])

    monkeypatch.setattr("basemode.transport.litellm.acompletion", fake_acompletion)

    strat = SystemPromptStrategy()
    params = GenerationParams(model="gpt-4o-mini")
    with pytest.raises(EmptyCompletionError) as exc_info:
        async for _ in strat.stream("hello", params):
            pass

    err = exc_info.value
    assert err.finish_reason == "content_filter"
    assert err.model == "gpt-4o-mini"
    assert err.strategy == "system"
    assert "content_filter" in str(err)


async def test_prefill_strategy_raises_with_finish_reason(monkeypatch):
    async def fake_acompletion(**kwargs):
        return _achunks([_chat_chunk(None, "length")])

    monkeypatch.setattr("basemode.transport.litellm.acompletion", fake_acompletion)

    strat = PrefillStrategy()
    params = GenerationParams(model="anthropic/claude-haiku-4-5-20251001")
    with pytest.raises(EmptyCompletionError) as exc_info:
        async for _ in strat.stream("hello", params):
            pass

    assert exc_info.value.finish_reason == "length"


async def test_completion_strategy_raises_without_finish_reason(monkeypatch):
    async def fake_atext_completion(**kwargs):
        return _achunks([_text_chunk(None), _text_chunk("")])

    monkeypatch.setattr(
        "basemode.transport.litellm.atext_completion", fake_atext_completion
    )

    strat = CompletionStrategy()
    params = GenerationParams(model="davinci-002")
    with pytest.raises(EmptyCompletionError) as exc_info:
        async for _ in strat.stream("hello", params):
            pass

    assert exc_info.value.finish_reason is None


async def test_strategy_yields_normally_when_content_present(monkeypatch):
    async def fake_acompletion(**kwargs):
        return _achunks([_chat_chunk("hi"), _chat_chunk(None, "stop")])

    monkeypatch.setattr("basemode.transport.litellm.acompletion", fake_acompletion)

    strat = SystemPromptStrategy()
    params = GenerationParams(model="gpt-4o-mini")
    tokens = [tok async for tok in strat.stream("hello", params)]
    assert "".join(tokens).strip() == "hi"

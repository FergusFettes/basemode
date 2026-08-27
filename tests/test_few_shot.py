"""Unit tests for FewShotStrategy: message construction, healing, empty-completion."""

from types import SimpleNamespace

import pytest

from basemode import usage_capture
from basemode.exceptions import EmptyCompletionError
from basemode.params import GenerationParams
from basemode.strategies.few_shot import _SYSTEM_PROMPT, FewShotStrategy


def _chat_chunk(content: str | None, finish_reason: str | None = None, usage=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content), finish_reason=finish_reason
            )
        ],
        usage=usage,
    )


async def _achunks(chunks):
    for chunk in chunks:
        yield chunk


async def test_few_shot_builds_system_and_user_messages(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _achunks([_chat_chunk("hi", "stop")])

    monkeypatch.setattr("basemode.transport.litellm.acompletion", fake_acompletion)

    strat = FewShotStrategy()
    params = GenerationParams(model="gpt-4o-mini")
    tokens = [tok async for tok in strat.stream("The cat sat", params)]

    assert "".join(tokens).strip() == "hi"
    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == _SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    # normalize_prefix strips trailing whitespace then adds exactly one space
    assert messages[1]["content"] == "The cat sat "
    assert captured["model"] == "gpt-4o-mini"
    assert captured["stream"] is True


async def test_few_shot_appends_context_to_system_prompt(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _achunks([_chat_chunk("hi", "stop")])

    monkeypatch.setattr("basemode.transport.litellm.acompletion", fake_acompletion)

    strat = FewShotStrategy()
    params = GenerationParams(model="gpt-4o-mini", context="Story so far: a ship.")
    async for _ in strat.stream("The cat sat", params):
        pass

    system = captured["messages"][0]["content"]
    assert system.startswith(_SYSTEM_PROMPT)
    assert "<CONTEXT>\nStory so far: a ship.\n</CONTEXT>" in system


async def test_few_shot_injects_leading_space_on_first_token(monkeypatch):
    async def fake_acompletion(**kwargs):
        return _achunks([_chat_chunk("world"), _chat_chunk(None, "stop")])

    monkeypatch.setattr("basemode.transport.litellm.acompletion", fake_acompletion)

    strat = FewShotStrategy()
    # prefix doesn't end in whitespace -> needs_leading_space should fire
    params = GenerationParams(model="gpt-4o-mini")
    tokens = [tok async for tok in strat.stream("hello", params)]
    assert tokens[0] == " world"
    assert "".join(tokens) == " world"


async def test_few_shot_no_leading_space_when_token_already_has_one(monkeypatch):
    async def fake_acompletion(**kwargs):
        return _achunks([_chat_chunk(" world"), _chat_chunk(None, "stop")])

    monkeypatch.setattr("basemode.transport.litellm.acompletion", fake_acompletion)

    strat = FewShotStrategy()
    params = GenerationParams(model="gpt-4o-mini")
    tokens = [tok async for tok in strat.stream("hello", params)]
    assert tokens[0] == " world"
    assert tokens.count(" world") == 1  # not double-spaced


async def test_few_shot_skips_chunks_with_no_choices(monkeypatch):
    async def fake_acompletion(**kwargs):
        return _achunks(
            [
                SimpleNamespace(choices=[], usage=None),
                _chat_chunk("hi", "stop"),
            ]
        )

    monkeypatch.setattr("basemode.transport.litellm.acompletion", fake_acompletion)

    strat = FewShotStrategy()
    params = GenerationParams(model="gpt-4o-mini")
    tokens = [tok async for tok in strat.stream("hello world", params)]
    assert "".join(tokens) == " hi"


async def test_few_shot_raises_empty_completion_with_finish_reason(monkeypatch):
    async def fake_acompletion(**kwargs):
        return _achunks([_chat_chunk(None, "length")])

    monkeypatch.setattr("basemode.transport.litellm.acompletion", fake_acompletion)

    strat = FewShotStrategy()
    params = GenerationParams(model="gpt-4o-mini")
    with pytest.raises(EmptyCompletionError) as exc_info:
        async for _ in strat.stream("hello", params):
            pass

    err = exc_info.value
    assert err.model == "gpt-4o-mini"
    assert err.strategy == "few_shot"
    assert err.finish_reason == "length"


async def test_few_shot_records_usage_when_present(monkeypatch):
    usage_capture.begin_capture()

    async def fake_acompletion(**kwargs):
        return _achunks(
            [
                _chat_chunk(
                    "hi", "stop", usage={"prompt_tokens": 5, "completion_tokens": 1}
                )
            ]
        )

    monkeypatch.setattr("basemode.transport.litellm.acompletion", fake_acompletion)

    strat = FewShotStrategy()
    params = GenerationParams(model="gpt-4o-mini")
    async for _ in strat.stream("hello", params):
        pass

    events = usage_capture.collect()
    assert events == [{"prompt_tokens": 5, "completion_tokens": 1}]

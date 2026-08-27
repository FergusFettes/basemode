"""Unit tests for FIMStrategy: prompt construction, streaming, empty-completion."""

from types import SimpleNamespace

import pytest

from basemode import usage_capture
from basemode.exceptions import EmptyCompletionError
from basemode.params import GenerationParams
from basemode.strategies.fim import _FIM_FORMATS, FIMStrategy, _fim_prompt

_DEEPSEEK_PRE, _DEEPSEEK_SUF, _DEEPSEEK_MID = _FIM_FORMATS["deepseek"]


def _text_chunk(text: str | None, finish_reason: str | None = None, usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(text=text, finish_reason=finish_reason)],
        usage=usage,
    )


async def _achunks(chunks):
    for chunk in chunks:
        yield chunk


def test_fim_prompt_deepseek():
    prompt = _fim_prompt("def foo():", "deepseek-coder")
    assert prompt == f"{_DEEPSEEK_PRE}def foo():{_DEEPSEEK_SUF}{_DEEPSEEK_MID}"


def test_fim_prompt_starcoder():
    prompt = _fim_prompt("def foo():", "starcoder2-15b")
    assert prompt == "<fim_prefix>def foo():<fim_suffix><fim_middle>"


def test_fim_prompt_codellama():
    prompt = _fim_prompt("def foo():", "codellama-13b")
    assert prompt == "▁<PRE>def foo():▁<SUF>▁<MID>"


def test_fim_prompt_unknown_model_falls_back_to_starcoder_format():
    prompt = _fim_prompt("hello", "gpt-4o-mini")
    assert prompt == "<fim_prefix>hello<fim_suffix><fim_middle>"


def test_fim_prompt_matching_is_case_insensitive():
    prompt = _fim_prompt("x", "DeepSeek-Coder-V2")
    assert prompt == f"{_DEEPSEEK_PRE}x{_DEEPSEEK_SUF}{_DEEPSEEK_MID}"


async def test_fim_strategy_sends_fim_formatted_prompt(monkeypatch):
    captured = {}

    async def fake_atext_completion(**kwargs):
        captured.update(kwargs)
        return _achunks([_text_chunk("hello", "stop")])

    monkeypatch.setattr(
        "basemode.transport.litellm.atext_completion", fake_atext_completion
    )

    strat = FIMStrategy()
    params = GenerationParams(model="deepseek-coder", max_tokens=64, temperature=0.5)
    tokens = [tok async for tok in strat.stream("def foo():", params)]

    assert "".join(tokens) == "hello"
    assert captured["model"] == "deepseek-coder"
    assert (
        captured["prompt"] == f"{_DEEPSEEK_PRE}def foo():{_DEEPSEEK_SUF}{_DEEPSEEK_MID}"
    )
    assert captured["max_tokens"] == 64
    assert captured["temperature"] == 0.5
    assert captured["stream"] is True


async def test_fim_strategy_passes_extra_kwargs(monkeypatch):
    captured = {}

    async def fake_atext_completion(**kwargs):
        captured.update(kwargs)
        return _achunks([_text_chunk("x", "stop")])

    monkeypatch.setattr(
        "basemode.transport.litellm.atext_completion", fake_atext_completion
    )

    strat = FIMStrategy()
    params = GenerationParams(model="starcoder2", extra={"top_p": 0.1})
    async for _ in strat.stream("abc", params):
        pass

    assert captured["top_p"] == 0.1


async def test_fim_strategy_concatenates_multiple_chunks(monkeypatch):
    async def fake_atext_completion(**kwargs):
        return _achunks(
            [_text_chunk("foo"), _text_chunk("bar"), _text_chunk(None, "stop")]
        )

    monkeypatch.setattr(
        "basemode.transport.litellm.atext_completion", fake_atext_completion
    )

    strat = FIMStrategy()
    params = GenerationParams(model="starcoder2")
    tokens = [tok async for tok in strat.stream("abc", params)]
    assert "".join(tokens) == "foobar"


async def test_fim_strategy_skips_chunks_with_no_choices(monkeypatch):
    async def fake_atext_completion(**kwargs):
        return _achunks(
            [SimpleNamespace(choices=[], usage=None), _text_chunk("hi", "stop")]
        )

    monkeypatch.setattr(
        "basemode.transport.litellm.atext_completion", fake_atext_completion
    )

    strat = FIMStrategy()
    params = GenerationParams(model="starcoder2")
    tokens = [tok async for tok in strat.stream("abc", params)]
    assert "".join(tokens) == "hi"


async def test_fim_strategy_raises_empty_completion_with_finish_reason(monkeypatch):
    async def fake_atext_completion(**kwargs):
        return _achunks([_text_chunk(None, "content_filter")])

    monkeypatch.setattr(
        "basemode.transport.litellm.atext_completion", fake_atext_completion
    )

    strat = FIMStrategy()
    params = GenerationParams(model="starcoder2")
    with pytest.raises(EmptyCompletionError) as exc_info:
        async for _ in strat.stream("abc", params):
            pass

    err = exc_info.value
    assert err.model == "starcoder2"
    assert err.strategy == "fim"
    assert err.finish_reason == "content_filter"


async def test_fim_strategy_records_usage_when_present(monkeypatch):
    usage_capture.begin_capture()

    async def fake_atext_completion(**kwargs):
        return _achunks(
            [
                _text_chunk(
                    "hi", "stop", usage={"prompt_tokens": 3, "completion_tokens": 2}
                )
            ]
        )

    monkeypatch.setattr(
        "basemode.transport.litellm.atext_completion", fake_atext_completion
    )

    strat = FIMStrategy()
    params = GenerationParams(model="starcoder2")
    async for _ in strat.stream("abc", params):
        pass

    events = usage_capture.collect()
    assert events == [{"prompt_tokens": 3, "completion_tokens": 2}]

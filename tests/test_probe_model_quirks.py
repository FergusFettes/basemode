import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import probe_model_quirks as pmq  # noqa: E402


def test_temperature_error_regex_matches_known_provider_wordings() -> None:
    assert pmq._TEMPERATURE_ERROR_RE.search(
        "`temperature` is deprecated for this model."
    )
    assert pmq._TEMPERATURE_ERROR_RE.search(
        "Unsupported value: 'temperature' does not support 0.3 with this model."
    )
    assert not pmq._TEMPERATURE_ERROR_RE.search("rate limit exceeded")


def test_prefill_error_regex_matches_known_provider_wordings() -> None:
    assert pmq._PREFILL_ERROR_RE.search(
        "This model does not support assistant message prefill."
    )
    assert not pmq._PREFILL_ERROR_RE.search("rate limit exceeded")


def test_compact_scalar_arrays_collapses_multiline_string_arrays() -> None:
    rendered = '{\n  "quirks": [\n    "no_temperature",\n    "no_prefill"\n  ]\n}'
    assert pmq._compact_scalar_arrays(rendered) == (
        '{\n  "quirks": ["no_temperature", "no_prefill"]\n}'
    )


def test_provider_splits_on_first_slash() -> None:
    assert pmq._provider("anthropic/claude-sonnet-5") == "anthropic"
    assert pmq._provider("openrouter/moonshotai/kimi-k2.6") == "openrouter"
    assert pmq._provider("standalone") == "unknown"


async def test_probe_model_adds_reasoning_budget_quirk_on_empty_baseline(
    monkeypatch,
) -> None:
    """A model whose normal-budget baseline comes back empty, but which
    produces clean output once given a much larger budget, should be tagged
    with `reasoning_budget` rather than silently left broken."""

    async def fake_collect_normal(model: str) -> str:
        return ""  # starved by hidden reasoning tokens at the normal budget

    async def fake_collect_forced(model: str, **kwargs) -> str:
        if kwargs.get("forced_max_tokens") == pmq.REASONING_PROBE_MAX_TOKENS:
            return " a clean continuation"
        return ""

    monkeypatch.setattr(pmq, "_collect_normal", fake_collect_normal)
    monkeypatch.setattr(pmq, "_collect_forced", fake_collect_forced)

    changes = await pmq._probe_model({"model": "anthropic/claude-some-new-reasoner"})

    assert len(changes) == 1
    assert changes[0].quirk == "reasoning_budget"
    assert changes[0].action == "added"


async def test_probe_model_skips_reasoning_budget_retry_when_already_tagged(
    monkeypatch,
) -> None:
    async def fake_collect_normal(model: str) -> str:
        return ""

    calls: list[dict] = []

    async def fake_collect_forced(model: str, **kwargs) -> str:
        calls.append(kwargs)
        return ""

    monkeypatch.setattr(pmq, "_collect_normal", fake_collect_normal)
    monkeypatch.setattr(pmq, "_collect_forced", fake_collect_forced)

    changes = await pmq._probe_model(
        {
            "model": "anthropic/claude-some-new-reasoner",
            "quirks": ["reasoning_budget"],
        }
    )

    assert changes == []
    assert calls == []  # no reasoning-budget retry attempted; already tagged

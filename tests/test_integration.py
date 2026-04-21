"""Integration tests — hit real APIs. Not run by default.

Run with:  pytest -m integration
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pytest

from basemode import branch_text, continue_text
from basemode.cli import _usage_prompt
from basemode.settings import settings
from basemode.usage import estimate_usage

pytestmark = pytest.mark.integration

SHORT = 10  # keep costs low, just verify continuation fires and is clean

BAD_STARTS = ("sure", "of course", "certainly", "i'll", "i will", "here", "this")

HEALTH_REPORT_PATH = Path("dist/integration/provider_health.json")
_HEALTH_ROWS: list[dict] = []


async def collect(gen) -> str:
    return "".join([t async for t in gen])


def assert_clean(result: str) -> None:
    assert len(result) > 0, "empty continuation"
    assert not result.lstrip().lower().startswith(BAD_STARTS), (
        f"chatbot preamble detected: {result!r}"
    )


def required_provider(model: str) -> str | None:
    if "/" in model:
        return model.split("/", 1)[0]
    lower = model.lower()
    if lower.startswith(("gpt-", "o1", "o3", "o4", "davinci", "babbage")):
        return "openai"
    if lower.startswith("claude"):
        return "anthropic"
    return None


def require_provider_access(model: str) -> str:
    provider = required_provider(model)
    if provider and provider not in settings.available_providers:
        pytest.skip(f"missing API key for provider={provider} (model={model})")
    return provider or "unknown"


@pytest.fixture(scope="module", autouse=True)
def write_provider_health_report() -> None:
    yield
    HEALTH_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "rows": _HEALTH_ROWS,
    }
    HEALTH_REPORT_PATH.write_text(json.dumps(payload, indent=2) + "\n")


# ── single-model smoke tests ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model,strategy",
    [
        # OpenAI
        ("gpt-5.4-mini", None),
        ("gpt-4o-mini", None),
        ("gpt-4o-mini", "few_shot"),
        # Anthropic
        ("anthropic/claude-opus-4-7", None),  # system (prefill dropped)
        ("anthropic/claude-haiku-4-5-20251001", None),  # prefill
        # Groq
        ("groq/llama-3.3-70b-versatile", None),
        # Together
        ("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo", None),
        # Gemini (thinking model)
        ("gemini/gemini-2.5-flash", "system"),
        # OpenRouter
        ("openrouter/openai/gpt-4o-mini", None),
        ("openrouter/moonshotai/kimi-k2.6", None),
        ("openrouter/moonshotai/kimi-k2.5", None),  # thinking model
        ("openrouter/deepseek/deepseek-v3.2", None),
        ("openrouter/meta-llama/llama-4-maverick", None),
    ],
)
async def test_continue_text(prefix: str, model: str, strategy: str | None) -> None:
    provider = require_provider_access(model)
    started = perf_counter()
    status = "ok"
    error: str | None = None
    result = ""
    usage = None

    try:
        result = await collect(
            continue_text(prefix, model, max_tokens=SHORT, strategy=strategy)
        )
        assert_clean(result)
        prompt, messages = _usage_prompt(model, prefix, strategy)
        usage = estimate_usage(
            model,
            prompt,
            result,
            prompt_messages=messages,
            prompt_requests=1,
        )
    except Exception as exc:
        status = "error"
        error = str(exc)
        raise
    finally:
        elapsed = perf_counter() - started
        _HEALTH_ROWS.append(
            {
                "provider": provider,
                "model": model,
                "strategy": strategy or "auto",
                "status": status,
                "latency_s": round(elapsed, 3),
                "output_chars": len(result),
                "chars_per_s": round((len(result) / elapsed), 2) if elapsed else None,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
                "estimated_cost_usd": usage.cost_usd if usage else None,
                "pricing_available": usage.pricing_available if usage else None,
                "error": error,
            }
        )


# ── space boundary ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", ["gpt-5.4-mini", "groq/llama-3.3-70b-versatile"])
async def test_no_smashed_words(model: str) -> None:
    """First continuation token must not smash into the last word of the prefix."""
    require_provider_access(model)
    prefix = "The ship rounded the headland and"
    result = await collect(continue_text(prefix, model, max_tokens=SHORT))
    assert result, "empty continuation"
    # The raw result either starts with whitespace (prefill/completion strategy adds it)
    # or the system strategy consumed the trailing space and result starts cleanly.
    # Either way: prefix[-1] is 'd' + result[0] must not BOTH be word chars.
    smashed = prefix[-1].isalpha() and result[0].isalpha()
    assert not smashed, f"word boundary broken: {prefix!r} + {result!r}"


# ── multi-branch ──────────────────────────────────────────────────────────────


async def test_branch_text_produces_n_branches(prefix: str) -> None:
    require_provider_access("gpt-5.4-mini")
    n = 3
    buffers: dict[int, list[str]] = {}
    async for idx, token in branch_text(prefix, "gpt-5.4-mini", n=n, max_tokens=SHORT):
        buffers.setdefault(idx, []).append(token)
    assert len(buffers) == n
    assert all(len("".join(v)) > 0 for v in buffers.values())


async def test_branch_text_branches_differ(prefix: str) -> None:
    require_provider_access("gpt-5.4-mini")
    n = 3
    buffers: dict[int, list[str]] = {}
    async for idx, token in branch_text(
        prefix, "gpt-5.4-mini", n=n, max_tokens=SHORT, temperature=1.0
    ):
        buffers.setdefault(idx, []).append(token)
    results = ["".join(v) for v in buffers.values()]
    assert len(set(results)) > 1, "all branches identical — temperature not working"


# ── poetry / difficult prompts ────────────────────────────────────────────────


async def test_poetry_no_commentary(prefix: str) -> None:
    require_provider_access("gpt-4o-mini")
    poem_prefix = "the rain falls like static\nbetween stations, the city\nblurs into"
    result = await collect(continue_text(poem_prefix, "gpt-4o-mini", max_tokens=SHORT))
    assert_clean(result)
    assert "Here is" not in result
    assert "continuation" not in result.lower()

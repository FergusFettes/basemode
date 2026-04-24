"""Integration tests — hit real APIs. Not run by default.

Run with:  pytest -m integration
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pytest

from basemode import branch_text, continue_text
from basemode.cli import _usage_prompt
from basemode.settings import settings
from basemode.usage import UsageEstimate, estimate_usage

pytestmark = pytest.mark.integration

SHORT = 10  # smoke/depth tests
TOKEN_CAP = 24  # explicit token-limit probes

BAD_STARTS = ("sure", "of course", "certainly", "i'll", "i will", "here", "this")

HEALTH_REPORT_PATH = Path("dist/integration/provider_health.json")
_HEALTH_ROWS: list[dict] = []


# Broad provider smoke coverage + key behavior families
CORE_SMOKE_CASES: list[tuple[str, str | None]] = [
    ("openai/gpt-4o-mini", None),
    ("openai/gpt-5.4-mini", None),
    ("anthropic/claude-opus-4-7", None),  # no-prefill/system
    ("anthropic/claude-haiku-4-5-20251001", None),  # prefill-compatible
    ("gemini/gemini-2.5-flash", "system"),  # thinking model
    ("moonshot/kimi-k2.5", None),  # native moonshot
    ("openrouter/moonshotai/kimi-k2.6", None),  # openrouter moonshot
    ("openrouter/moonshotai/kimi-k2.5", None),  # openrouter moonshot
    ("openrouter/deepseek/deepseek-v3.2", None),  # openrouter deepseek
    ("openrouter/minimax/minimax-m2.5", None),  # openrouter minimax
    ("zai/glm-5", None),
]

# Older + newer families for regression depth across providers.
PROVIDER_DEPTH_CASES: list[tuple[str, str | None]] = [
    # Anthropic older/newer coverage
    ("anthropic/claude-sonnet-4-20250514", None),
    ("anthropic/claude-sonnet-4-5-20250929", None),
    ("anthropic/claude-sonnet-4-6", None),
    ("anthropic/claude-opus-4-6", None),
    # OpenAI family
    ("openai/gpt-4o-mini", "few_shot"),
    # Gemini depth
    ("gemini/gemini-2.5-pro", "system"),
    ("gemini/gemma-4-26b-a4b-it", "system"),
    # Moonshot depth
    ("moonshot/kimi-k2-0905-preview", None),
    ("openrouter/deepseek/deepseek-r1", None),
    ("openrouter/deepseek/deepseek-chat-v3-0324", None),
    ("openrouter/minimax/minimax-m2", None),
    # ZAI depth
    ("zai/glm-4.7", None),
]

# Focused cap checks for models where caller max_tokens should be preserved.
TOKEN_LIMIT_GUARDRAILS: list[tuple[str, str | None, int, int]] = [
    ("openai/gpt-5.4-mini", None, TOKEN_CAP, 8),
    ("openai/gpt-4o-mini", None, TOKEN_CAP, 8),
    ("anthropic/claude-opus-4-7", None, TOKEN_CAP, 8),
    ("anthropic/claude-sonnet-4-6", None, TOKEN_CAP, 8),
    ("gemini/gemma-4-26b-a4b-it", "system", TOKEN_CAP, 12),
    ("openrouter/moonshotai/kimi-k2.6", None, TOKEN_CAP, 12),
    ("openrouter/deepseek/deepseek-v3.2", None, TOKEN_CAP, 12),
    ("openrouter/minimax/minimax-m2.5", None, TOKEN_CAP, 12),
    ("zai/glm-5", None, TOKEN_CAP, 12),
]


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


def _append_health_row(
    *,
    test_kind: str,
    provider: str,
    model: str,
    strategy: str | None,
    max_tokens: int,
    status: str,
    elapsed: float,
    result: str,
    usage,
    error: str | None,
    token_limit_soft_cap: int | None = None,
) -> None:
    completion_tokens = usage.completion_tokens if usage else None
    over_by = None
    if completion_tokens is not None and token_limit_soft_cap is not None:
        over_by = max(0, completion_tokens - token_limit_soft_cap)
    _HEALTH_ROWS.append(
        {
            "test_kind": test_kind,
            "provider": provider,
            "model": model,
            "strategy": strategy or "auto",
            "requested_max_tokens": max_tokens,
            "token_limit_soft_cap": token_limit_soft_cap,
            "token_limit_excess_tokens": over_by,
            "status": status,
            "latency_s": round(elapsed, 3),
            "output_chars": len(result),
            "chars_per_s": round((len(result) / elapsed), 2) if elapsed else None,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": completion_tokens,
            "total_tokens": usage.total_tokens if usage else None,
            "estimated_cost_usd": usage.cost_usd if usage else None,
            "pricing_available": usage.pricing_available if usage else None,
            "error": error,
        }
    )


async def _run_probe(
    *,
    prefix: str,
    model: str,
    strategy: str | None,
    max_tokens: int,
    test_kind: str,
    assert_fn: Callable[[str, UsageEstimate], None] | None = None,
    token_limit_soft_cap: int | None = None,
) -> tuple[str, object]:
    provider = require_provider_access(model)
    started = perf_counter()
    status = "ok"
    error: str | None = None
    result = ""
    usage = None
    try:
        result = await collect(
            continue_text(prefix, model, max_tokens=max_tokens, strategy=strategy)
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
        if assert_fn:
            assert_fn(result, usage)
        return result, usage
    except Exception as exc:
        status = "error"
        error = str(exc)
        raise
    finally:
        elapsed = perf_counter() - started
        _append_health_row(
            test_kind=test_kind,
            provider=provider,
            model=model,
            strategy=strategy,
            max_tokens=max_tokens,
            status=status,
            elapsed=elapsed,
            result=result,
            usage=usage,
            error=error,
            token_limit_soft_cap=token_limit_soft_cap,
        )


@pytest.fixture(scope="module", autouse=True)
def write_provider_health_report() -> None:
    yield
    HEALTH_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_cost = sum((row.get("estimated_cost_usd") or 0.0) for row in _HEALTH_ROWS)
    priced = sum(1 for row in _HEALTH_ROWS if row.get("pricing_available"))
    errors = sum(1 for row in _HEALTH_ROWS if row.get("status") == "error")
    summary = {
        "rows_total": len(_HEALTH_ROWS),
        "rows_with_pricing": priced,
        "rows_without_pricing": len(_HEALTH_ROWS) - priced,
        "rows_with_errors": errors,
        "estimated_total_cost_usd": round(total_cost, 8),
    }
    payload = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "summary": summary,
        "rows": _HEALTH_ROWS,
    }
    HEALTH_REPORT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        "[integration-cost] estimated_total_cost_usd="
        f"{summary['estimated_total_cost_usd']:.8f}"
    )


# ── tier 1: broad smoke ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model,strategy",
    CORE_SMOKE_CASES,
)
async def test_continue_text_core_smoke(
    prefix: str, model: str, strategy: str | None
) -> None:
    await _run_probe(
        prefix=prefix,
        model=model,
        strategy=strategy,
        max_tokens=SHORT,
        test_kind="core_smoke",
    )


# ── tier 2: provider depth (older + newer model families) ────────────────────


@pytest.mark.parametrize("model,strategy", PROVIDER_DEPTH_CASES)
async def test_continue_text_provider_depth(
    prefix: str, model: str, strategy: str | None
) -> None:
    await _run_probe(
        prefix=prefix,
        model=model,
        strategy=strategy,
        max_tokens=SHORT,
        test_kind="provider_depth",
    )


# ── tier 3: token-limit guardrails ────────────────────────────────────────────


@pytest.mark.parametrize(
    "model,strategy,max_tokens,grace_tokens",
    TOKEN_LIMIT_GUARDRAILS,
)
async def test_token_limit_guardrails(
    prefix: str, model: str, strategy: str | None, max_tokens: int, grace_tokens: int
) -> None:
    soft_cap = max_tokens + grace_tokens

    def _assert_within_cap(_result: str, usage: UsageEstimate) -> None:
        completion_tokens = usage.completion_tokens
        assert completion_tokens <= soft_cap, (
            f"visible completion tokens exceeded soft cap: "
            f"model={model} requested={max_tokens} soft_cap={soft_cap} "
            f"completion_tokens={completion_tokens}"
        )

    await _run_probe(
        prefix=prefix,
        model=model,
        strategy=strategy,
        max_tokens=max_tokens,
        test_kind="token_limit_guardrail",
        assert_fn=_assert_within_cap,
        token_limit_soft_cap=soft_cap,
    )


# ── space boundary ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", ["openai/gpt-5.4-mini", "anthropic/claude-opus-4-7"])
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
    require_provider_access("openai/gpt-4o-mini")
    poem_prefix = "the rain falls like static\nbetween stations, the city\nblurs into"
    result = await collect(
        continue_text(poem_prefix, "openai/gpt-4o-mini", max_tokens=SHORT)
    )
    assert_clean(result)
    assert "Here is" not in result
    assert "continuation" not in result.lower()

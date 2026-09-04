import pytest

from basemode.live_models import cached_price_per_million
from basemode.usage import (
    estimate_usage,
    format_per_million,
    format_usd,
    get_price_info,
)


def test_get_price_info_known_model() -> None:
    info = get_price_info("gpt-4o-mini")

    assert info.model == "openai/gpt-4o-mini"
    assert info.pricing_available
    assert info.input_cost_per_token is not None
    assert info.output_cost_per_token is not None


def test_get_price_info_normalizes_alias() -> None:
    info = get_price_info("gemma-4")

    assert info.model == "gemini/gemma-4-26b-a4b-it"


def test_get_price_info_unknown_pricing_model() -> None:
    # Deliberately not a real model: litellm's pricing map gains entries over
    # time, so a name it could plausibly learn makes this test drift.
    info = get_price_info("basemode-nonexistent-test-model")

    assert not info.pricing_available
    assert info.input_cost_per_token is None
    assert info.output_cost_per_token is None


def test_price_falls_back_to_the_provider_catalog(monkeypatch) -> None:
    """litellm's costed list never covers a reseller's whole catalog.

    OpenRouter publishes its own prices and the packaged catalog carries
    them, so a model litellm has never costed is still plannable.
    """
    import basemode.usage as usage

    monkeypatch.setattr(usage, "_model_info", lambda model: {})
    monkeypatch.setattr(usage, "cached_price_per_million", lambda model: (0.08, 0.28))

    info = get_price_info("openrouter/acme/reseller-only")

    assert info.pricing_available
    assert info.price_source == "provider_catalog"
    assert info.input_cost_per_token == 0.08 / 1_000_000
    assert info.output_cost_per_token == 0.28 / 1_000_000


def test_litellm_pricing_wins_over_the_catalog(monkeypatch) -> None:
    import basemode.usage as usage

    monkeypatch.setattr(
        usage,
        "cached_price_per_million",
        lambda model: pytest.fail("catalog consulted despite a litellm price"),
    )

    assert get_price_info("gpt-4o-mini").price_source == "litellm"


def test_router_sentinel_price_is_not_a_price() -> None:
    """OpenRouter answers -1 where the price depends on what it routes to.

    Read literally it made a sweep's cost ceiling come out below zero.
    """
    input_per_m, output_per_m = cached_price_per_million("openrouter/openrouter/auto")

    assert (input_per_m, output_per_m) == (None, None)


def test_catalog_lookup_ignores_the_provider_s_own_casing(monkeypatch) -> None:
    """Providers publish their own casing; basemode normalizes to lowercase.

    Together serves `meta-llama/Llama-3.3-70B-Instruct-Turbo`, so an
    exact-key lookup missed most of its catalog.
    """
    import basemode.live_models as live_models

    monkeypatch.setattr(
        live_models,
        "_cached_catalog",
        lambda: {
            "acme": {
                "models": {
                    "Vendor/Mixed-Case-7B": {
                        "input_price_per_m": 1.0,
                        "output_price_per_m": 2.0,
                    }
                }
            }
        },
    )
    live_models._cached_models_by_lower_id.cache_clear()
    try:
        assert live_models.cached_price_per_million("acme/vendor/mixed-case-7b") == (
            1.0,
            2.0,
        )
    finally:
        live_models._cached_models_by_lower_id.cache_clear()


def test_estimate_usage_known_model_has_cost() -> None:
    usage = estimate_usage("gpt-4o-mini", "hello", "world")

    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert usage.cost_usd is not None


def test_estimate_usage_can_count_prompt_messages() -> None:
    usage = estimate_usage(
        "gpt-4o-mini",
        "ignored",
        "world",
        prompt_messages=[
            {"role": "system", "content": "You continue text."},
            {"role": "user", "content": "hello"},
        ],
    )

    assert usage.prompt_tokens > 1


def test_estimate_usage_multiplies_prompt_requests() -> None:
    single = estimate_usage("gpt-4o-mini", "hello", "world")
    multi = estimate_usage("gpt-4o-mini", "hello", "world", prompt_requests=3)

    assert multi.prompt_tokens == single.prompt_tokens * 3
    assert multi.completion_tokens == single.completion_tokens


def test_format_usd() -> None:
    assert format_usd(None) == "unavailable"
    assert format_usd(0.00000012) == "$0.00000012"
    assert format_usd(0.0012) == "$0.001200"
    assert format_usd(1.2) == "$1.2000"


def test_format_per_million() -> None:
    assert format_per_million(None) == "unavailable"
    assert format_per_million(0.00000125) == "$1.25/1M"

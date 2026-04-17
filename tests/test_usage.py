from basemode.usage import estimate_usage, format_per_million, format_usd, get_price_info


def test_get_price_info_known_model() -> None:
    info = get_price_info("gpt-4o-mini")

    assert info.model == "gpt-4o-mini"
    assert info.pricing_available
    assert info.input_cost_per_token is not None
    assert info.output_cost_per_token is not None


def test_get_price_info_unknown_pricing_model() -> None:
    info = get_price_info("gemma-4")

    assert info.model == "gemini/gemma-4-26b-a4b-it"
    assert not info.pricing_available


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

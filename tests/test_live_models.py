from basemode.live_models import (
    PROVIDER_ENDPOINTS,
    _parse_gemini,
    _parse_openai_style,
    _price_deepinfra,
    _price_novita,
    _price_openrouter,
    _price_together,
)


def test_novita_headers_carry_a_user_agent() -> None:
    # Novita's Cloudflare edge 403s the default urllib User-Agent (error 1010).
    headers = PROVIDER_ENDPOINTS["novita"].headers("secret")
    assert headers["Authorization"] == "Bearer secret"
    assert "User-Agent" in headers


def test_openrouter_architecture_and_parameters_are_preserved() -> None:
    [model] = _parse_openai_style(
        {
            "data": [
                {
                    "id": "vendor/vision-chat",
                    "created": 1_700_000_000,
                    "architecture": {
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["text"],
                    },
                    "supported_parameters": ["temperature", "tools"],
                    "type": "chat",
                }
            ]
        }
    )
    assert model.input_modalities == ("text", "image")
    assert model.output_modalities == ("text",)
    assert model.supported_parameters == ("temperature", "tools")
    assert model.provider_type == "chat"


def test_together_pricing_is_dollars_per_million_tokens_as_is() -> None:
    [model] = _parse_openai_style(
        {"data": [{"id": "moonshotai/Kimi-K3", "pricing": {"input": 3, "output": 15}}]},
        _price_together,
    )
    assert model.input_price_per_m == 3.0
    assert model.output_price_per_m == 15.0


def test_deepinfra_pricing_is_read_from_nested_metadata() -> None:
    [model] = _parse_openai_style(
        {
            "data": [
                {
                    "id": "google/gemma-4-31B-it-Ultra",
                    "metadata": {
                        "pricing": {"input_tokens": 0.27, "output_tokens": 0.76}
                    },
                }
            ]
        },
        _price_deepinfra,
    )
    assert model.input_price_per_m == 0.27
    assert model.output_price_per_m == 0.76


def test_novita_pricing_uses_the_decimal_dollar_field() -> None:
    [model] = _parse_openai_style(
        {
            "data": [
                {
                    "id": "zai-org/glm-5.3",
                    "pricing": {
                        "prompt": {"price_per_m_decimal": "1.4"},
                        "completion": {"price_per_m_decimal": "4.4"},
                    },
                }
            ]
        },
        _price_novita,
    )
    assert model.input_price_per_m == 1.4
    assert model.output_price_per_m == 4.4


def test_openrouter_pricing_is_converted_from_per_token_to_per_million() -> None:
    [model] = _parse_openai_style(
        {
            "data": [
                {
                    "id": "anthropic/claude-sonnet-5",
                    "pricing": {"prompt": "0.000002", "completion": "0.00001"},
                }
            ]
        },
        _price_openrouter,
    )
    assert model.input_price_per_m == 2.0
    assert model.output_price_per_m == 10.0


def test_missing_pricing_leaves_prices_none() -> None:
    [model] = _parse_openai_style({"data": [{"id": "vendor/model"}]}, _price_together)
    assert model.input_price_per_m is None
    assert model.output_price_per_m is None


def test_gemini_generation_methods_are_preserved() -> None:
    [model] = _parse_gemini(
        {
            "models": [
                {
                    "name": "models/gemini-test",
                    "supportedGenerationMethods": ["generateContent", "countTokens"],
                }
            ]
        }
    )
    assert model.supported_methods == ("generateContent", "countTokens")

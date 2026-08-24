from basemode.live_models import _parse_gemini, _parse_openai_style


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

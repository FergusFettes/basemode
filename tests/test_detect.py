import pytest

from basemode.detect import detect_strategy, normalize_model
from basemode.strategies import (
    CompletionStrategy,
    FIMStrategy,
    PrefillStrategy,
    SystemPromptStrategy,
)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-3-5-sonnet-latest", PrefillStrategy),
        ("claude-3-opus-20240229", PrefillStrategy),
        ("gpt-3.5-turbo-instruct", CompletionStrategy),
        ("davinci-002", CompletionStrategy),
        ("text-davinci-003", CompletionStrategy),
        ("deepseek-coder-33b", FIMStrategy),
        ("starcoder2-15b", FIMStrategy),
        ("codellama-13b", FIMStrategy),
        ("gpt-4o", SystemPromptStrategy),
        ("gpt-4o-mini", SystemPromptStrategy),
        ("mistral-large-latest", SystemPromptStrategy),
        ("gemini/gemini-1.5-pro", SystemPromptStrategy),
        ("groq/llama3-70b-8192", SystemPromptStrategy),
    ],
)
def test_auto_detect(model: str, expected: type) -> None:
    assert isinstance(detect_strategy(model), expected)


def test_override_valid() -> None:
    strat = detect_strategy("gpt-4o", override="prefill")
    assert isinstance(strat, PrefillStrategy)


def test_override_invalid() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        detect_strategy("gpt-4o", override="nonexistent")


def test_detect_returns_new_instance_each_call() -> None:
    a = detect_strategy("gpt-4o")
    b = detect_strategy("gpt-4o")
    assert a is not b


def test_normalize_kimi_defaults_to_moonshot() -> None:
    assert normalize_model("kimi-k2.5") == "moonshot/kimi-k2.5"


def test_normalize_kimi_k2_alias_uses_known_good_non_thinking_model() -> None:
    assert normalize_model("kimi-k2") == "moonshot/kimi-k2-0905-preview"


def test_normalize_kimi_thinking_defaults_to_moonshot() -> None:
    assert normalize_model("kimi-k2-thinking") == "moonshot/kimi-k2-thinking"


def test_normalize_gemma_defaults_to_gemini() -> None:
    assert normalize_model("gemma-3-27b-it") == "gemini/gemma-3-27b-it"


def test_normalize_gemma_4_aliases() -> None:
    assert normalize_model("gemma-4") == "gemini/gemma-4-26b-a4b-it"
    assert normalize_model("gemma-4-26b") == "gemini/gemma-4-26b-a4b-it"
    assert normalize_model("gemma-4-31b") == "gemini/gemma-4-31b-it"


def test_normalize_anthropic_aliases_before_litellm_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_probe(model: str) -> None:
        raise AssertionError(f"unexpected LiteLLM provider probe for {model}")

    monkeypatch.setattr("basemode.detect.get_llm_provider", fail_probe)

    assert normalize_model("opus-4.7") == "anthropic/claude-opus-4-7"
    assert normalize_model("sonnet-4.6") == "anthropic/claude-sonnet-4-6"


def test_normalize_kimi_does_not_probe_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_probe(model: str) -> None:
        raise AssertionError(f"unexpected LiteLLM provider probe for {model}")

    monkeypatch.setattr("basemode.detect.get_llm_provider", fail_probe)

    assert normalize_model("kimi-k2-thinking") == "moonshot/kimi-k2-thinking"

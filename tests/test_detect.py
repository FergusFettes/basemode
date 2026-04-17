import pytest

from basemode.detect import detect_strategy
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

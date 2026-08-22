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


def test_normalize_glm_defaults_to_zai() -> None:
    assert normalize_model("glm-4.7") == "zai/glm-4.7"
    assert normalize_model("glm-5") == "zai/glm-5"


def test_normalize_gemma_4_aliases() -> None:
    assert normalize_model("gemma-4") == "gemini/gemma-4-26b-a4b-it"
    assert normalize_model("gemma-4-26b") == "gemini/gemma-4-26b-a4b-it"
    assert normalize_model("gemma-4-31b") == "gemini/gemma-4-31b-it"


def test_normalize_anthropic_aliases_before_litellm_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_normalize_glm_does_not_probe_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_probe(model: str) -> None:
        raise AssertionError(f"unexpected LiteLLM provider probe for {model}")

    monkeypatch.setattr("basemode.detect.get_llm_provider", fail_probe)

    assert normalize_model("glm-4.7") == "zai/glm-4.7"


def test_registry_prompt_method_beats_the_name_heuristic() -> None:
    """kimi-k3 is registered as `prefill`; the heuristic alone would say `system`."""
    from basemode.detect import _heuristic_strategy, select_strategy

    choice = select_strategy("moonshot/kimi-k3")

    assert (choice.name, choice.source) == ("prefill", "registry")
    assert _heuristic_strategy("moonshot/kimi-k3") == "system"


def test_unregistered_model_falls_back_to_the_heuristic() -> None:
    from basemode.detect import select_strategy

    choice = select_strategy("openai/some-unreleased-model")

    assert (choice.name, choice.source) == ("system", "heuristic")


def test_user_pin_beats_the_registry() -> None:
    from basemode.detect import select_strategy
    from basemode.keys import set_strategy_override

    set_strategy_override("moonshot/kimi-k3", "few_shot")
    choice = select_strategy("moonshot/kimi-k3")

    assert (choice.name, choice.source) == ("few_shot", "user")


def test_explicit_argument_beats_a_user_pin() -> None:
    from basemode.detect import select_strategy
    from basemode.keys import set_strategy_override

    set_strategy_override("openai/gpt-4o", "few_shot")
    choice = select_strategy("openai/gpt-4o", "system")

    assert (choice.name, choice.source) == ("system", "explicit")


def test_user_pin_can_be_ignored_for_generated_data() -> None:
    from basemode.detect import select_strategy
    from basemode.keys import set_strategy_override

    set_strategy_override("openai/gpt-4o", "few_shot")
    choice = select_strategy("openai/gpt-4o", allow_user_override=False)

    assert choice.source != "user"


def test_stale_prefill_pin_yields_to_a_no_prefill_quirk() -> None:
    """A pinned strategy the model has since stopped accepting must not stick."""
    from basemode.detect import select_strategy
    from basemode.keys import set_strategy_override

    set_strategy_override("anthropic/claude-opus-5", "prefill")
    choice = select_strategy("anthropic/claude-opus-5")

    assert choice.name != "prefill"
    assert choice.source != "user"


def test_unknown_pinned_strategy_is_ignored_rather_than_raising() -> None:
    from basemode.detect import select_strategy
    from basemode.keys import set_strategy_override

    set_strategy_override("openai/gpt-4o", "telepathy")

    assert select_strategy("openai/gpt-4o").name == "system"


def test_registry_and_published_table_agree_on_every_verified_model() -> None:
    """The README/docs table publishes what basemode will actually do."""
    import json
    from importlib import resources

    from basemode.detect import normalize_model, select_strategy

    rows = json.loads(
        resources.files("basemode")
        .joinpath("data", "verified_models_details.json")
        .read_text()
    )["rows"]

    drift = [
        (row["model"], row["prompt_method"])
        for row in rows
        if row.get("prompt_method")
        and select_strategy(
            normalize_model(row["model"]), allow_user_override=False
        ).name
        != row["prompt_method"]
    ]

    assert drift == []

import pytest

from basemode.identity import canonical_creator, canonical_id, creator_of, model_stem


@pytest.mark.parametrize(
    ("wire_id", "expected"),
    [
        # A first-party provider publishes no creator because from where it
        # stands it is obvious.
        ("anthropic/claude-opus-5", "anthropic/anthropic/claude-opus-5"),
        ("openai/gpt-5.5", "openai/openai/gpt-5.5"),
        ("xai/grok-4", "xai/xai/grok-4"),
        # The route is the product name, not the creator.
        ("gemini/gemini-3-flash-preview", "gemini/google/gemini-3-flash-preview"),
        # A reseller that already publishes a creator keeps its structure.
        ("deepinfra/anthropic/claude-opus-5", "deepinfra/anthropic/claude-opus-5"),
        # Resellers disagree about spelling; the alias table settles it.
        ("deepinfra/deepseek-ai/deepseek-v3.2", "deepinfra/deepseek/deepseek-v3.2"),
        ("novita/deepseek/deepseek-v3.2", "novita/deepseek/deepseek-v3.2"),
        ("openrouter/z-ai/glm-5", "openrouter/zai/glm-5"),
        ("deepinfra/zai-org/glm-5", "deepinfra/zai/glm-5"),
        ("openrouter/meta-llama/llama-4-scout", "openrouter/meta/llama-4-scout"),
        # A reseller serving other people's models under a bare, hyphenated
        # name still gets a creator.
        ("cerebras/zai-glm-4.6", "cerebras/zai/zai-glm-4.6"),
        ("cerebras/gpt-oss-120b", "cerebras/openai/gpt-oss-120b"),
        ("groq/gemma-7b-it", "groq/google/gemma-7b-it"),
        # The provider's own capitalization does not survive.
        (
            "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "together_ai/meta/llama-3.3-70b-instruct-turbo",
        ),
    ],
)
def test_canonical_ids(wire_id: str, expected: str) -> None:
    assert canonical_id(wire_id) == expected


def test_an_openrouter_floating_alias_keeps_its_creator() -> None:
    """`~` marks a floating alias, not a different organisation."""
    assert (
        canonical_id("openrouter/~anthropic/claude-opus-latest")
        == "openrouter/anthropic/claude-opus-latest"
    )


def test_a_variant_suffix_is_part_of_the_model() -> None:
    """`:free` and `:batch` are separate endpoints with separate pricing."""
    assert (
        canonical_id("openrouter/z-ai/glm-5.3-flash:batch")
        == "openrouter/zai/glm-5.3-flash:batch"
    )


def test_an_unidentifiable_codename_is_not_guessed() -> None:
    """Better an honest `unknown` than a creator invented from a name."""
    assert canonical_id("novita/elephant") == "novita/unknown/elephant"


def test_a_model_with_no_route_keeps_one() -> None:
    assert canonical_id("gpt-5.5").startswith("unknown/")


def test_creator_aliases_are_idempotent() -> None:
    """Folding a name that is already canonical must not move it again."""
    for variant in ("deepseek-ai", "z-ai", "minimaxai", "meta-llama", "moonshotai"):
        once = canonical_creator(variant)
        assert canonical_creator(once) == once


def test_the_same_model_on_two_providers_shares_creator_and_stem() -> None:
    """The point of the exercise: comparable identities across resellers."""
    a = canonical_id("deepinfra/deepseek-ai/deepseek-v3.2")
    b = canonical_id("novita/deepseek/deepseek-v3.2")

    assert a != b  # different providers stay distinct endpoints
    assert a.split("/")[1:] == b.split("/")[1:]


def test_stem_drops_route_and_creator() -> None:
    assert model_stem("together_ai/qwen/Qwen3-32B") == "qwen3-32b"
    assert model_stem("openai/gpt-5.5") == "gpt-5.5"


def test_creator_of_is_lowercase() -> None:
    assert creator_of("Deepinfra/Qwen/Qwen3-32B") == "qwen"

"""Integration tests — hit real APIs. Not run by default.

Run with:  pytest -m integration
"""
import pytest

from basemode import branch_text, continue_text

pytestmark = pytest.mark.integration

SHORT = 10  # keep costs low, just verify continuation fires and is clean

BAD_STARTS = ("sure", "of course", "certainly", "i'll", "i will", "here", "this")


async def collect(gen) -> str:
    return "".join([t async for t in gen])


def assert_clean(result: str) -> None:
    assert len(result) > 0, "empty continuation"
    assert not result.lstrip().lower().startswith(BAD_STARTS), (
        f"chatbot preamble detected: {result!r}"
    )


# ── single-model smoke tests ──────────────────────────────────────────────────

@pytest.mark.parametrize("model,strategy", [
    # OpenAI
    ("gpt-5.4-mini",                                         None),
    ("gpt-4o-mini",                                          None),
    ("gpt-4o-mini",                                          "few_shot"),
    # Anthropic
    ("anthropic/claude-opus-4-7",                            None),   # system (prefill dropped)
    ("anthropic/claude-haiku-4-5-20251001",                  None),   # prefill
    ("anthropic/claude-3-haiku-20240307",                    "prefill"),
    # Groq
    ("groq/llama-3.3-70b-versatile",                         None),
    # Together
    ("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",  None),
    # Gemini (thinking model)
    ("gemini/gemini-2.5-flash",                              "system"),
    # OpenRouter
    ("openrouter/openai/gpt-4o-mini",                        None),
    ("openrouter/moonshotai/kimi-k2.5",                      None),   # thinking model
    ("openrouter/deepseek/deepseek-v3.2",                    None),
    ("openrouter/meta-llama/llama-4-maverick",               None),
])
async def test_continue_text(prefix: str, model: str, strategy: str | None) -> None:
    result = await collect(continue_text(prefix, model, max_tokens=SHORT, strategy=strategy))
    assert_clean(result)


# ── space boundary ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model", ["gpt-5.4-mini", "groq/llama-3.3-70b-versatile"])
async def test_no_smashed_words(model: str) -> None:
    """First continuation token must not smash into the last word of the prefix."""
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
    n = 3
    buffers: dict[int, list[str]] = {}
    async for idx, token in branch_text(prefix, "gpt-5.4-mini", n=n, max_tokens=SHORT):
        buffers.setdefault(idx, []).append(token)
    assert len(buffers) == n
    assert all(len("".join(v)) > 0 for v in buffers.values())


async def test_branch_text_branches_differ(prefix: str) -> None:
    n = 3
    buffers: dict[int, list[str]] = {}
    async for idx, token in branch_text(prefix, "gpt-5.4-mini", n=n, max_tokens=SHORT, temperature=1.0):
        buffers.setdefault(idx, []).append(token)
    results = ["".join(v) for v in buffers.values()]
    assert len(set(results)) > 1, "all branches identical — temperature not working"


# ── poetry / difficult prompts ────────────────────────────────────────────────

async def test_poetry_no_commentary(prefix: str) -> None:
    poem_prefix = "the rain falls like static\nbetween stations, the city\nblurs into"
    result = await collect(continue_text(poem_prefix, "gpt-4o-mini", max_tokens=SHORT))
    assert_clean(result)
    assert "Here is" not in result
    assert "continuation" not in result.lower()

"""Integration tests — require real API keys. Run with: pytest -m integration"""

import pytest

from basemode import branch_text, continue_text

pytestmark = pytest.mark.integration


async def _collect(gen) -> str:
    tokens = []
    async for token in gen:
        tokens.append(token)
    return "".join(tokens)


@pytest.mark.parametrize(
    "model,strategy",
    [
        ("gpt-4o-mini", None),
        ("gpt-4o-mini", "system"),
        ("gpt-4o-mini", "few_shot"),
        ("anthropic/claude-3-haiku-20240307", None),
        ("anthropic/claude-3-haiku-20240307", "prefill"),
    ],
)
async def test_continue_text(prefix: str, model: str, strategy: str | None) -> None:
    result = await _collect(
        continue_text(prefix, model, max_tokens=50, strategy=strategy)
    )
    assert len(result) > 0
    assert (
        not result.lstrip()
        .lower()
        .startswith(("sure", "of course", "certainly", "i'll"))
    )


async def test_branch_text(prefix: str) -> None:
    n = 3
    buffers: dict[int, list[str]] = {}
    async for idx, token in branch_text(prefix, "gpt-4o-mini", n=n, max_tokens=50):
        buffers.setdefault(idx, []).append(token)

    assert len(buffers) == n
    results = ["".join(v) for v in buffers.values()]
    assert all(len(r) > 0 for r in results)
    # branches should differ (with high probability at temp=0.9)
    assert len(set(results)) > 1

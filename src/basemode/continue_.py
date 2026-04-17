import asyncio
from collections.abc import AsyncGenerator

from .detect import detect_strategy
from .params import GenerationParams


async def continue_text(
    prefix: str,
    model: str = "gpt-4o-mini",
    *,
    max_tokens: int = 200,
    temperature: float = 0.9,
    strategy: str | None = None,
    **extra,
) -> AsyncGenerator[str, None]:
    """Stream a single continuation."""
    params = GenerationParams(model=model, max_tokens=max_tokens, temperature=temperature, extra=extra)
    strat = detect_strategy(model, strategy)
    async for token in strat.stream(prefix, params):
        yield token


async def branch_text(
    prefix: str,
    model: str = "gpt-4o-mini",
    *,
    n: int = 4,
    max_tokens: int = 200,
    temperature: float = 0.9,
    strategy: str | None = None,
    **extra,
) -> AsyncGenerator[tuple[int, str], None]:
    """Stream n parallel continuations as (branch_idx, token) tuples."""
    params = GenerationParams(model=model, max_tokens=max_tokens, temperature=temperature, extra=extra)
    strat = detect_strategy(model, strategy)

    queue: asyncio.Queue[tuple[int, str] | None] = asyncio.Queue()

    async def run_branch(idx: int) -> None:
        async for token in strat.stream(prefix, params):
            await queue.put((idx, token))
        await queue.put(None)

    tasks = [asyncio.create_task(run_branch(i)) for i in range(n)]
    done = 0
    while done < n:
        item = await queue.get()
        if item is None:
            done += 1
        else:
            yield item

    await asyncio.gather(*tasks)

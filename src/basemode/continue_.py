import asyncio
import logging
from collections.abc import AsyncGenerator

import litellm

from .detect import detect_strategy, normalize_model
from .healing import (
    normalize_stream_newlines,
    rewind_prefix_to_word_boundary,
    strip_rewind_overlap,
)
from .keys import load_into_environ
from .params import GenerationParams

log = logging.getLogger(__name__)


async def continue_text(
    prefix: str,
    model: str = "gpt-4o-mini",
    *,
    max_tokens: int = 200,
    temperature: float = 0.9,
    context: str = "",
    strategy: str | None = None,
    rewind: bool = False,
    **extra,
) -> AsyncGenerator[str, None]:
    """Stream a single continuation."""
    litellm.suppress_debug_info = True
    load_into_environ()
    model = normalize_model(model)
    params = GenerationParams(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        context=context,
        extra=extra,
    )
    strat = detect_strategy(model, strategy)
    log.debug(
        "continue_text: model=%s strategy=%s max_tokens=%d context_len=%d prefix_len=%d",
        model,
        strat.name,
        max_tokens,
        len(context),
        len(prefix),
    )
    token_count = 0
    generation_prefix, rewind_fragment = _generation_prefix(prefix, strat.name, rewind)
    try:
        raw_tokens = strat.stream(generation_prefix, params)
        healed_tokens = strip_rewind_overlap(raw_tokens, rewind_fragment)
        async for token in normalize_stream_newlines(prefix, healed_tokens):
            token_count += 1
            yield token
    except Exception:
        log.exception("continue_text: stream error after %d tokens", token_count)
        raise
    log.debug("continue_text: done, %d tokens", token_count)


async def branch_text(
    prefix: str,
    model: str = "gpt-4o-mini",
    *,
    n: int = 4,
    max_tokens: int = 200,
    temperature: float = 0.9,
    strategy: str | None = None,
    rewind: bool = False,
    **extra,
) -> AsyncGenerator[tuple[int, str], None]:
    """Stream n parallel continuations as (branch_idx, token) tuples."""
    litellm.suppress_debug_info = True
    load_into_environ()
    model = normalize_model(model)
    params = GenerationParams(
        model=model, max_tokens=max_tokens, temperature=temperature, extra=extra
    )
    strat = detect_strategy(model, strategy)

    queue: asyncio.Queue[tuple[int, str] | None] = asyncio.Queue()
    generation_prefix, rewind_fragment = _generation_prefix(prefix, strat.name, rewind)

    async def run_branch(idx: int) -> None:
        raw_tokens = strat.stream(generation_prefix, params)
        healed_tokens = strip_rewind_overlap(raw_tokens, rewind_fragment)
        async for token in normalize_stream_newlines(prefix, healed_tokens):
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


def _generation_prefix(
    prefix: str, strategy_name: str, rewind: bool
) -> tuple[str, str]:
    if rewind and strategy_name in {"system", "few_shot"}:
        return rewind_prefix_to_word_boundary(prefix)
    return prefix, ""

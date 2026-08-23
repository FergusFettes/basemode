import asyncio
import logging
from collections.abc import AsyncGenerator, Callable

import litellm

from . import usage_capture
from .detect import detect_strategy, normalize_model
from .healing import (
    normalize_stream_newlines,
    probe_rewind_overlap,
    rewind_prefix_to_word_boundary,
)
from .health import EMPTY_RESPONSE, classify_error, error_details, record_outcome
from .keys import load_into_environ
from .params import GenerationParams

log = logging.getLogger(__name__)

#: How much of a continuation's opening is enough to diagnose its boundary.
#: Token sizes vary wildly between providers, so this is a character budget
#: rather than a token count.
RAW_HEAD_CHARS = 32


async def continue_text(
    prefix: str,
    model: str = "gpt-4o-mini",
    *,
    max_tokens: int = 200,
    temperature: float = 0.9,
    context: str = "",
    strategy: str | None = None,
    rewind: bool = False,
    strict_max_tokens: bool = False,
    record_health: bool = True,
    on_raw_head: Callable[[str], None] | None = None,
    raw_head_chars: int = RAW_HEAD_CHARS,
    on_usage: Callable[[list[dict]], None] | None = None,
    **extra,
) -> AsyncGenerator[str, None]:
    """Stream a single continuation.

    The outcome is recorded against the model (see :mod:`basemode.health`)
    unless `record_health=False` — which a caller that classifies failures
    itself should pass, so one attempt is not counted twice.

    `on_raw_head` is called once with the opening `raw_head_chars` characters
    as the strategy produced them, before stream normalization touches the
    text. That is the only place the two halves of a boundary defect can be
    told apart: whether a missing space was never emitted, or was emitted and
    then repaired away. A caller that stores it can diagnose the seam long
    after the stream is gone. It is a callback rather than a return value
    because branches run concurrently and each needs its own sink.

    `on_usage` is called once, after the stream ends (successfully or not),
    with every provider-reported usage payload seen along the way (see
    `usage_capture.py`) — normally one, but two when the rewind-retry path
    below aborts a first request and reissues a second, since both were
    billed. Empty when the provider never returned usage (see
    `strategies/compat.py` for which providers were asked), in which case a
    caller should fall back to `usage.estimate_usage`.
    """
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
    usage_capture.begin_capture()
    generation_prefix, rewind_fragment = _generation_prefix(prefix, strat.name, rewind)
    try:
        raw_tokens = _stream_tokens(
            strat, prefix, generation_prefix, rewind_fragment, params
        )
        if on_raw_head is not None:
            raw_tokens = _capture_head(raw_tokens, on_raw_head, raw_head_chars)
        stream = normalize_stream_newlines(prefix, raw_tokens)
        if strict_max_tokens:
            stream = _enforce_visible_token_cap(stream, model, max_tokens)
        async for token in stream:
            token_count += 1
            yield token
    except (GeneratorExit, asyncio.CancelledError):
        # The consumer walked away. Tokens already delivered still count as
        # the model having worked; an immediate cancel is not a verdict.
        if record_health and token_count:
            record_outcome(model, ok=True)
        _report_usage(on_usage)
        raise
    except Exception as exc:
        if record_health:
            category, status = classify_error(exc)
            error_code, error_param = error_details(exc)
            record_outcome(
                model,
                ok=False,
                category=category,
                status=status,
                error_code=error_code,
                error_param=error_param,
            )
        log.exception("continue_text: stream error after %d tokens", token_count)
        _report_usage(on_usage)
        raise
    if record_health:
        record_outcome(
            model,
            ok=bool(token_count),
            category=None if token_count else EMPTY_RESPONSE,
        )
    _report_usage(on_usage)
    log.debug("continue_text: done, %d tokens", token_count)


def _report_usage(on_usage: Callable[[list[dict]], None] | None) -> None:
    if on_usage is None:
        return
    try:
        on_usage(usage_capture.collect())
    except Exception:
        log.warning("on_usage sink raised; ignoring", exc_info=True)


async def branch_text(
    prefix: str,
    model: str = "gpt-4o-mini",
    *,
    n: int = 4,
    max_tokens: int = 200,
    temperature: float = 0.9,
    strategy: str | None = None,
    rewind: bool = False,
    strict_max_tokens: bool = False,
    record_health: bool = True,
    on_usage: Callable[[int, list[dict]], None] | None = None,
    **extra,
) -> AsyncGenerator[tuple[int, str], None]:
    """Stream n parallel continuations as (branch_idx, token) tuples.

    Each branch is recorded as its own attempt (see :mod:`basemode.health`)
    unless `record_health=False`.

    `on_usage`, if given, is called once per branch with its index and every
    provider-reported usage payload seen for it (see `continue_text` for
    details — same semantics, per branch instead of once).
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    litellm.suppress_debug_info = True
    load_into_environ()
    model = normalize_model(model)
    params = GenerationParams(
        model=model, max_tokens=max_tokens, temperature=temperature, extra=extra
    )
    strat = detect_strategy(model, strategy)

    queue: asyncio.Queue[tuple[int, str] | BaseException | None] = asyncio.Queue()
    generation_prefix, rewind_fragment = _generation_prefix(prefix, strat.name, rewind)

    async def run_branch(idx: int) -> None:
        token_count = 0
        usage_capture.begin_capture()
        try:
            raw_tokens = _stream_tokens(
                strat, prefix, generation_prefix, rewind_fragment, params
            )
            stream = normalize_stream_newlines(prefix, raw_tokens)
            if strict_max_tokens:
                stream = _enforce_visible_token_cap(stream, model, max_tokens)
            async for token in stream:
                token_count += 1
                await queue.put((idx, token))
        except asyncio.CancelledError:
            if record_health and token_count:
                record_outcome(model, ok=True)
            raise
        except Exception as exc:
            if record_health:
                category, status = classify_error(exc)
                error_code, error_param = error_details(exc)
                record_outcome(
                    model,
                    ok=False,
                    category=category,
                    status=status,
                    error_code=error_code,
                    error_param=error_param,
                )
            await queue.put(exc)
        else:
            if record_health:
                record_outcome(
                    model,
                    ok=bool(token_count),
                    category=None if token_count else EMPTY_RESPONSE,
                )
        finally:
            if on_usage is not None:
                try:
                    on_usage(idx, usage_capture.collect())
                except Exception:
                    log.warning("on_usage sink raised; ignoring", exc_info=True)
            await queue.put(None)

    tasks = [asyncio.create_task(run_branch(i)) for i in range(n)]
    try:
        done = 0
        while done < n:
            item = await queue.get()
            if item is None:
                done += 1
            elif isinstance(item, BaseException):
                raise item
            else:
                yield item
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _capture_head(
    tokens: AsyncGenerator[str, None],
    sink: Callable[[str], None],
    limit: int,
) -> AsyncGenerator[str, None]:
    """Pass tokens through, reporting the opening `limit` characters once.

    The sink is called as soon as enough text has arrived, and again at the
    end for a stream that finished shorter than that — but never twice, so a
    caller can treat it as a single assignment. A sink that raises is ignored:
    a diagnostic must not be able to break the generation it describes.
    """
    head: list[str] = []
    length = 0
    reported = False

    def report() -> None:
        nonlocal reported
        if reported:
            return
        reported = True
        try:
            sink("".join(head)[:limit])
        except Exception:
            log.warning("on_raw_head sink raised; ignoring", exc_info=True)

    try:
        async for token in tokens:
            if not reported:
                head.append(token)
                length += len(token)
                if length >= limit:
                    report()
            yield token
    finally:
        report()


async def _stream_tokens(
    strat, prefix: str, generation_prefix: str, fragment: str, params
) -> AsyncGenerator[str, None]:
    """Stream a continuation, undoing a rewind that the model did not take up.

    A rewound request is only usable if the model re-emits the fragment we held
    back; otherwise its text was written to follow a prefix the reader never
    sees, and pasting it on would read as a non-sequitur rather than a spacing
    glitch. Nothing has been yielded at that point, so the request is simply
    reissued with the full prefix.
    """
    if not fragment:
        async for token in strat.stream(prefix, params):
            yield token
        return

    stream = strat.stream(generation_prefix, params)
    matched, head = await probe_rewind_overlap(stream, fragment)
    if matched:
        log.debug("_stream_tokens: rewind of %r matched", fragment)
        if head:
            yield head
        async for token in stream:
            yield token
        return

    log.debug("_stream_tokens: rewind of %r not taken up, retrying", fragment)
    await stream.aclose()
    async for token in strat.stream(prefix, params):
        yield token


def _generation_prefix(
    prefix: str, strategy_name: str, rewind: bool
) -> tuple[str, str]:
    if rewind and strategy_name in {"system", "few_shot"}:
        return rewind_prefix_to_word_boundary(prefix)
    return prefix, ""


def _count_tokens_safe(model: str, text: str) -> int:
    try:
        return litellm.token_counter(model=model, text=text)
    except Exception:
        return max(1, len(text) // 4)


def _clip_chunk_to_token_cap(
    *,
    model: str,
    emitted: str,
    chunk: str,
    cap: int,
) -> str:
    if cap <= 0 or not chunk:
        return ""
    if _count_tokens_safe(model, emitted + chunk) <= cap:
        return chunk

    lo = 0
    hi = len(chunk)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _count_tokens_safe(model, emitted + chunk[:mid]) <= cap:
            lo = mid
        else:
            hi = mid - 1
    return chunk[:lo]


async def _enforce_visible_token_cap(
    tokens: AsyncGenerator[str, None],
    model: str,
    cap: int,
) -> AsyncGenerator[str, None]:
    emitted = ""
    async for chunk in tokens:
        clipped = _clip_chunk_to_token_cap(
            model=model, emitted=emitted, chunk=chunk, cap=cap
        )
        if clipped:
            emitted += clipped
            yield clipped
        if clipped != chunk:
            break

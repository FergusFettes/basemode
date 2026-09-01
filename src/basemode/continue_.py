import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from contextvars import ContextVar
from dataclasses import replace
from typing import TYPE_CHECKING

import litellm

from . import usage_capture
from .detect import detect_strategy, normalize_model, select_strategy
from .exceptions import EmptyCompletionError
from .healing import (
    normalize_stream_newlines,
    probe_rewind_overlap,
    rewind_prefix_to_word_boundary,
)
from .keys import load_into_environ
from .observations import ObservationContext, Operation, observe_operation
from .params import GenerationParams

if TYPE_CHECKING:
    from .strategies.base import ContinuationStrategy

log = logging.getLogger(__name__)

#: How much of a continuation's opening is enough to diagnose its boundary.
#: Token sizes vary wildly between providers, so this is a character budget
#: rather than a token count.
RAW_HEAD_CHARS = 32

_current_operation: ContextVar[Operation | None] = ContextVar(
    "basemode_observation_operation", default=None
)
_current_attempt_kind: ContextVar[str] = ContextVar(
    "basemode_observation_attempt_kind", default="initial"
)


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
    observation: ObservationContext | None = None,
    retry_empty_completion: bool = True,
    on_raw_head: Callable[[str], None] | None = None,
    raw_head_chars: int = RAW_HEAD_CHARS,
    on_usage: Callable[[list[dict]], None] | None = None,
    _observation_operation: Operation | None = None,
    _observation_attempt_kind: str = "initial",
    _finalize_observation: bool = True,
    **extra,
) -> AsyncGenerator[str, None]:
    """Stream a single continuation.

    `observation` identifies the calling application without attaching prompt
    or response content. Recording is best-effort and cannot break generation.
    `record_health` is retained for source compatibility but no longer permits
    supported provider calls to bypass the unified observation ledger; use the
    global recording opt-out instead.

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
    choice = select_strategy(model, strategy)
    strat = detect_strategy(model, strategy)
    operation = _observation_operation or observe_operation(
        model, strat.name, choice.source, observation
    )
    log.debug(
        "continue_text: model=%s strategy=%s max_tokens=%d context_len=%d prefix_len=%d",
        model,
        strat.name,
        max_tokens,
        len(context),
        len(prefix),
    )
    usage_capture.begin_capture()
    generation_prefix, rewind_fragment = _generation_prefix(prefix, strat.name, rewind)
    token_counter = [0]
    stream = _stream_with_empty_retry(
        strat,
        model,
        prefix,
        generation_prefix,
        rewind_fragment,
        params,
        retry_empty_completion=retry_empty_completion,
        strict_max_tokens=strict_max_tokens,
        max_tokens=max_tokens,
        on_raw_head=on_raw_head,
        raw_head_chars=raw_head_chars,
        token_counter=token_counter,
        operation=operation,
        initial_attempt_kind=_observation_attempt_kind,
    )
    try:
        async for token in stream:
            yield token
    except (GeneratorExit, asyncio.CancelledError):
        # The consumer walked away. Tokens already delivered still count as
        # the model having worked; an immediate cancel is not a verdict.
        if _finalize_observation:
            operation.finish(
                "cancelled" if token_counter[0] else "inconclusive",
                returned_content=bool(token_counter[0]),
            )
        _report_usage(on_usage)
        raise
    except Exception:
        if _finalize_observation:
            operation.finish("failure", returned_content=bool(token_counter[0]))
        log.exception("continue_text: stream error after %d tokens", token_counter[0])
        _report_usage(on_usage)
        raise
    if _finalize_observation:
        operation.finish(
            "success" if token_counter[0] else "failure",
            returned_content=bool(token_counter[0]),
        )
    _report_usage(on_usage)
    log.debug("continue_text: done, %d tokens", token_counter[0])


def _report_usage(on_usage: Callable[[list[dict]], None] | None) -> None:
    if on_usage is None:
        return
    try:
        on_usage(usage_capture.collect())
    except Exception:
        log.warning("on_usage sink raised; ignoring", exc_info=True)


async def _stream_with_empty_retry(
    strat: "ContinuationStrategy",
    model: str,
    prefix: str,
    generation_prefix: str,
    fragment: str,
    params: GenerationParams,
    *,
    retry_empty_completion: bool,
    strict_max_tokens: bool,
    max_tokens: int,
    on_raw_head: Callable[[str], None] | None,
    raw_head_chars: int,
    token_counter: list[int],
    operation: Operation,
    initial_attempt_kind: str = "initial",
) -> AsyncGenerator[str, None]:
    """Stream a continuation, retrying once on an empty completion.

    `token_counter` is a single-element list the caller owns, so it can keep
    reading the running token count after this generator has finished (or
    been cancelled) — an async generator can't return a value alongside what
    it yields.
    """
    attempt_params = params
    retried_reasoning_off = False
    attempt_kind = initial_attempt_kind
    while True:
        try:
            raw_tokens = _with_observation_context(
                _stream_tokens(
                    strat,
                    prefix,
                    generation_prefix,
                    fragment,
                    attempt_params,
                ),
                operation,
                attempt_kind,
            )
            if on_raw_head is not None:
                raw_tokens = _capture_head(raw_tokens, on_raw_head, raw_head_chars)
            stream = normalize_stream_newlines(prefix, raw_tokens)
            if strict_max_tokens:
                stream = _enforce_visible_token_cap(stream, model, max_tokens)
            async for token in stream:
                token_counter[0] += 1
                yield token
            return
        except EmptyCompletionError:
            # OpenRouter often proxies a model whose reasoning is on by
            # default and consumes the whole visible token budget, so
            # nothing comes out (`finish_reason="length"`) even though
            # the request itself was fine. Retry once with reasoning
            # switched off before giving up — a model that requires
            # reasoning just rejects this immediately as a 400, which
            # `except Exception` in the caller still handles.
            eligible = (
                retry_empty_completion
                and not retried_reasoning_off
                and token_counter[0] == 0
                and model.lower().startswith("openrouter/")
                and "reasoning" not in attempt_params.extra
            )
            if not eligible:
                raise
            retried_reasoning_off = True
            attempt_kind = "reasoning_off"
            attempt_params = replace(
                attempt_params,
                extra={**attempt_params.extra, "reasoning": {"enabled": False}},
            )
            log.debug(
                "%s produced nothing, retrying with reasoning off",
                model,
            )


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
    observation: ObservationContext | None = None,
    retry_empty_completion: bool = True,
    on_usage: Callable[[int, list[dict]], None] | None = None,
    **extra,
) -> AsyncGenerator[tuple[int, str], None]:
    """Stream n parallel continuations as (branch_idx, token) tuples.

    Each branch creates an independent logical observation operation.
    `record_health` is retained for source compatibility but does not disable
    the unified ledger; use the global recording opt-out instead.

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
    choice = select_strategy(model, strategy)
    strat = detect_strategy(model, strategy)

    queue: asyncio.Queue[tuple[int, str] | BaseException | None] = asyncio.Queue()
    generation_prefix, rewind_fragment = _generation_prefix(prefix, strat.name, rewind)

    async def run_branch(idx: int) -> None:
        operation = observe_operation(model, strat.name, choice.source, observation)
        usage_capture.begin_capture()
        token_counter = [0]
        stream = _stream_with_empty_retry(
            strat,
            model,
            prefix,
            generation_prefix,
            rewind_fragment,
            params,
            retry_empty_completion=retry_empty_completion,
            strict_max_tokens=strict_max_tokens,
            max_tokens=max_tokens,
            on_raw_head=None,
            raw_head_chars=RAW_HEAD_CHARS,
            token_counter=token_counter,
            operation=operation,
            initial_attempt_kind="initial",
        )
        try:
            async for token in stream:
                await queue.put((idx, token))
        except asyncio.CancelledError:
            operation.finish(
                "cancelled" if token_counter[0] else "inconclusive",
                returned_content=bool(token_counter[0]),
            )
            raise
        except Exception as exc:
            operation.finish("failure", returned_content=bool(token_counter[0]))
            await queue.put(exc)
        else:
            operation.finish(
                "success" if token_counter[0] else "failure",
                returned_content=bool(token_counter[0]),
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
    strat: "ContinuationStrategy",
    prefix: str,
    generation_prefix: str,
    fragment: str,
    params: GenerationParams,
) -> AsyncGenerator[str, None]:
    """Stream a continuation, undoing a rewind that the model did not take up.

    A rewound request is only usable if the model re-emits the fragment we held
    back; otherwise its text was written to follow a prefix the reader never
    sees, and pasting it on would read as a non-sequitur rather than a spacing
    glitch. Nothing has been yielded at that point, so the request is simply
    reissued with the full prefix.
    """
    if not fragment:
        async for token in _observe_current_attempt(strat.stream(prefix, params)):
            yield token
        return

    stream = _observe_current_attempt(strat.stream(generation_prefix, params))
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
    async for token in _observe_current_attempt(
        strat.stream(prefix, params), "rewind_retry"
    ):
        yield token


async def _with_observation_context(
    stream: AsyncGenerator[str, None], operation: Operation, kind: str
) -> AsyncGenerator[str, None]:
    operation_token = _current_operation.set(operation)
    kind_token = _current_attempt_kind.set(kind)
    try:
        async for token in stream:
            yield token
    finally:
        _current_attempt_kind.reset(kind_token)
        _current_operation.reset(operation_token)


async def _observe_current_attempt(
    stream: AsyncGenerator[str, None], kind: str | None = None
) -> AsyncGenerator[str, None]:
    operation = _current_operation.get()
    if operation is None:
        async for token in stream:
            yield token
        return
    async for token in _observe_attempt(
        stream, operation, kind or _current_attempt_kind.get()
    ):
        yield token


async def _observe_attempt(
    stream: AsyncGenerator[str, None],
    operation: Operation,
    kind: str,
) -> AsyncGenerator[str, None]:
    """Finalize exactly one attempt around one strategy/provider invocation."""
    attempt = operation.begin_attempt(kind)
    usage_offset = usage_capture.mark()
    try:
        async for token in stream:
            attempt.saw_content(token)
            yield token
    except (GeneratorExit, asyncio.CancelledError) as exc:
        attempt.finish(
            "cancelled", exc, usage_events=usage_capture.collect_since(usage_offset)
        )
        raise
    except Exception as exc:
        attempt.finish(
            "failure", exc, usage_events=usage_capture.collect_since(usage_offset)
        )
        raise
    else:
        attempt.finish(
            "success" if attempt.returned_content else "failure",
            usage_events=usage_capture.collect_since(usage_offset),
        )


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

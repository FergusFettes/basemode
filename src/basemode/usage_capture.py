"""Capture provider-reported usage off a streaming response, per call.

Strategies request real usage on the stream (`stream_options={"include_usage":
True}` in `strategies/compat.py`) and call `record` here as soon as a chunk
carries it. `continue_.py` resets the capture with `begin_capture` before
issuing a request and reads it back with `collect` once the stream ends, so a
caller (basemode-loom) can prefer the provider's own token counts over the
local tokenizer estimate in `usage.py` — which can't see hidden reasoning
tokens or a request that was aborted mid-stream (the rewind-retry path in
`continue_.py`).

A `ContextVar` rather than a return value because `ContinuationStrategy.stream`
is a plain `AsyncGenerator[str, None]` shared across six implementations, and
usage arrives as an attribute on a chunk the generator doesn't otherwise keep.
Each `begin_capture` call sets a fresh list, so concurrent branches (each its
own `asyncio.Task`, hence its own copy of the context) never share state.
"""

from contextvars import ContextVar

_events: ContextVar[list[dict] | None] = ContextVar(
    "_stream_usage_events", default=None
)


def begin_capture() -> None:
    """Start a fresh capture in the current task's context."""
    _events.set([])


def record(usage: object | None) -> None:
    """Record one chunk's usage payload, if any, converted to a plain dict."""
    if usage is None:
        return
    events = _events.get()
    if events is None:
        return
    as_dict = _to_dict(usage)
    if as_dict:
        events.append(as_dict)


def collect() -> list[dict]:
    """Return the usage payloads recorded since the last `begin_capture`."""
    events = _events.get()
    return list(events) if events else []


def _to_dict(usage: object) -> dict:
    if isinstance(usage, dict):
        return usage
    for attr in ("model_dump", "dict"):
        fn = getattr(usage, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    try:
        return dict(usage)  # type: ignore[call-overload]
    except Exception:
        pass
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            result[key] = value
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        reasoning = getattr(details, "reasoning_tokens", None)
        if reasoning is not None:
            result["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return result

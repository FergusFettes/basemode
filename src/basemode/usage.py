from dataclasses import dataclass

import litellm

from .detect import normalize_model
from .live_models import cached_price_per_million


@dataclass(frozen=True)
class PriceInfo:
    model: str
    provider: str | None
    input_cost_per_token: float | None
    output_cost_per_token: float | None
    cache_read_input_token_cost: float | None
    output_cost_per_reasoning_token: float | None
    max_input_tokens: int | None
    max_output_tokens: int | None
    supports_reasoning: bool | None
    pricing_available: bool
    #: Where the per-token prices came from: litellm's costed model list, or
    #: the provider's own catalog (`provider_catalog`). None when unpriced.
    price_source: str | None = None


@dataclass(frozen=True)
class UsageEstimate:
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float | None
    pricing_available: bool
    # True when token counts came from a local tokenizer run over the
    # (possibly healed) text rather than the provider's own reported usage.
    is_estimate: bool = True


def get_price_info(model: str) -> PriceInfo:
    resolved = normalize_model(model)
    info = _model_info(resolved)
    input_cost = info.get("input_cost_per_token")
    output_cost = info.get("output_cost_per_token")
    source = "litellm" if input_cost is not None and output_cost is not None else None
    if source is None:
        # litellm's costed list is community-maintained and never covers a
        # reseller's whole catalog, but that reseller publishes its own prices
        # and the packaged catalog already carries them. Unknown pricing is
        # treated as "may cost anything" by every budget in the codebase, so
        # falling back here is the difference between a sweep being plannable
        # and not.
        cached_input, cached_output = cached_price_per_million(resolved)
        if cached_input is not None and cached_output is not None:
            input_cost = cached_input / 1_000_000
            output_cost = cached_output / 1_000_000
            source = "provider_catalog"
    return PriceInfo(
        model=resolved,
        provider=info.get("litellm_provider"),
        input_cost_per_token=input_cost,
        output_cost_per_token=output_cost,
        cache_read_input_token_cost=info.get("cache_read_input_token_cost"),
        output_cost_per_reasoning_token=info.get("output_cost_per_reasoning_token"),
        max_input_tokens=info.get("max_input_tokens"),
        max_output_tokens=info.get("max_output_tokens"),
        supports_reasoning=info.get("supports_reasoning"),
        pricing_available=source is not None,
        price_source=source,
    )


def estimate_usage(
    model: str,
    prompt: str,
    completion: str,
    *,
    prompt_messages: list[dict] | None = None,
    prompt_requests: int = 1,
) -> UsageEstimate:
    resolved = normalize_model(model)
    price = get_price_info(resolved)
    prompt_tokens_per_request = (
        _count_message_tokens(resolved, prompt_messages)
        if prompt_messages
        else _count_tokens(resolved, prompt)
    )
    prompt_tokens = prompt_tokens_per_request * prompt_requests
    completion_tokens = _count_tokens(resolved, completion)
    cost = None
    if price.pricing_available:
        cost = prompt_tokens * (
            price.input_cost_per_token or 0.0
        ) + completion_tokens * (price.output_cost_per_token or 0.0)
    return UsageEstimate(
        model=resolved,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost_usd=cost,
        pricing_available=price.pricing_available,
    )


def usage_from_events(model: str, events: list[dict]) -> UsageEstimate | None:
    """Sum provider-reported usage payloads captured off a stream.

    `events` holds one entry per completed (or aborted-but-billed) request —
    see `usage_capture.py`. Multiple entries happen when the rewind-retry
    path in `continue_.py` issues a second request after an aborted first
    one; both were billed, so both are summed. Returns None when no event
    carried usable token counts, so the caller can fall back to
    `estimate_usage`.
    """
    resolved = normalize_model(model)
    prompt_tokens = 0
    completion_tokens = 0
    reasoning_tokens = 0
    saw_usage = False
    for event in events:
        p = event.get("prompt_tokens")
        c = event.get("completion_tokens")
        if p is None and c is None:
            continue
        saw_usage = True
        prompt_tokens += int(p or 0)
        completion_tokens += int(c or 0)
        details = event.get("completion_tokens_details") or {}
        if isinstance(details, dict):
            reasoning_tokens += int(details.get("reasoning_tokens") or 0)
    if not saw_usage:
        return None

    price = get_price_info(resolved)
    cost = None
    if price.pricing_available:
        visible_completion_tokens = max(completion_tokens - reasoning_tokens, 0)
        reasoning_rate = (
            price.output_cost_per_reasoning_token
            if price.output_cost_per_reasoning_token is not None
            else price.output_cost_per_token
        )
        cost = (
            prompt_tokens * (price.input_cost_per_token or 0.0)
            + visible_completion_tokens * (price.output_cost_per_token or 0.0)
            + reasoning_tokens * (reasoning_rate or 0.0)
        )
    return UsageEstimate(
        model=resolved,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost_usd=cost,
        pricing_available=price.pricing_available,
        is_estimate=False,
    )


def format_usd(cost: float | None) -> str:
    if cost is None:
        return "unavailable"
    if cost < 0.0001:
        return f"${cost:.8f}"
    if cost < 0.01:
        return f"${cost:.6f}"
    return f"${cost:.4f}"


def format_per_million(cost_per_token: float | None) -> str:
    if cost_per_token is None:
        return "unavailable"
    return f"${cost_per_token * 1_000_000:.2f}/1M"


def _model_info(model: str) -> dict:
    try:
        return dict(litellm.get_model_info(model))
    except Exception:
        stem = model.split("/", 1)[-1]
        return dict(litellm.model_cost.get(model) or litellm.model_cost.get(stem) or {})


def _count_tokens(model: str, text: str) -> int:
    try:
        return litellm.token_counter(model=model, text=text)
    except Exception:
        return max(1, len(text) // 4)


def _count_message_tokens(model: str, messages: list[dict]) -> int:
    try:
        return litellm.token_counter(model=model, messages=messages)
    except Exception:
        return _count_tokens(
            model, "\n".join(str(m.get("content", "")) for m in messages)
        )

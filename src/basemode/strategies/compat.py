"""Compatibility helpers for model-specific API quirks."""

# Models that have deprecated the temperature parameter.
_NO_TEMPERATURE_MODELS = {
    "claude-opus-4-7",
}

# Models that no longer support assistant prefill (must use system strategy instead).
NO_PREFILL_MODELS = {
    "claude-opus-4-7",
}

# Thinking models: consume token budget on internal reasoning before output.
# Without a large token budget the visible output is empty or truncated.
# Key = substring to match in model name (after last /), value = (budget, min_output)
_THINKING_MODELS: dict[str, tuple[int, int]] = {
    "gemini-2.5-flash": (1024, 512),
    "gemini-2.5-flash-lite": (512, 256),
    "gemini-2.5-pro": (2048, 512),
    "kimi-k2.5": (4096, 512),   # Kimi K2.5 uses a large reasoning budget
    "kimi-k2-thinking": (4096, 512),
}


def _model_stem(model: str) -> str:
    return model.lower().split("/")[-1]


def no_temperature(model: str) -> bool:
    return _model_stem(model) in _NO_TEMPERATURE_MODELS


def thinking_kwargs(model: str, max_tokens: int) -> dict:
    stem = _model_stem(model)
    via_openrouter = model.lower().startswith("openrouter/")
    for fragment, (budget, min_out) in _THINKING_MODELS.items():
        if fragment in stem:
            adjusted = max(max_tokens, budget + min_out)
            if via_openrouter:
                # OpenRouter proxied models use extra_body for thinking config
                return {
                    "max_tokens": adjusted,
                    "extra_body": {"thinking": {"budget_tokens": budget}},
                }
            return {
                "thinking": {"type": "enabled", "budget_tokens": budget},
                "max_tokens": adjusted,
            }
    return {}


def build_kwargs(params: "GenerationParams") -> dict:  # type: ignore[name-defined]
    """Build litellm kwargs with model-specific compatibility applied."""
    kwargs: dict = {"max_tokens": params.max_tokens}
    if not no_temperature(params.model):
        kwargs["temperature"] = params.temperature
    kwargs.update(thinking_kwargs(params.model, params.max_tokens))
    kwargs.update(params.extra)
    return kwargs

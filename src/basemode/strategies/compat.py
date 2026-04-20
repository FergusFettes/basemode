"""Compatibility helpers for model-specific API quirks."""

import re

from ..params import GenerationParams

# Models with an exact-match temperature lock.
_NO_TEMPERATURE_MODELS = {
    "claude-opus-4-7",
    "kimi-k2.5",
}

# Model stems that only accept temperature=1 (probed 2026-04-18):
#   - gpt-5 / gpt-5-mini / gpt-5-nano / gpt-5-codex  (but NOT gpt-5.1, 5.4, etc.)
#   - o-series reasoning models: o1, o3, o3-mini, o4-mini, ...
_NO_TEMPERATURE_PATTERNS = [
    re.compile(r"^gpt-5(-[a-z]+)?$"),
    re.compile(r"^o\d+(-[a-z]+)?$"),
]

# Models that no longer support assistant prefill (must use system strategy instead).
# Verified 2026-04-17 via API probe: opus 4.7, sonnet 4.6, opus 4.6 reject prefill;
# the 4.5 and 4.0/4.1 families still accept it.
NO_PREFILL_MODELS = {
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
}

# Known live Anthropic model IDs (from /v1/models, 2026-04-17).
# Used for best-effort alias resolution — typing `claude-3-haiku` expands to
# `claude-3-haiku-20240307` when exactly one known model contains that substring.
KNOWN_ANTHROPIC_MODELS = {
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-haiku-20240307",
}

# Thinking models: consume token budget on internal reasoning before output.
# Without a large token budget the visible output is empty or truncated.
# Key = substring to match in model name (after last /), value = (budget, min_output)
_THINKING_MODELS: dict[str, tuple[int, int]] = {
    "gemini-2.5-flash": (1024, 512),
    "gemini-2.5-flash-lite": (512, 256),
    "gemini-2.5-pro": (2048, 512),
    "kimi-k2.5": (4096, 512),  # Kimi K2.5 uses a large reasoning budget
    "kimi-k2-thinking": (4096, 512),
}

_GEMINI_THINKING_LEVEL_MODELS: dict[str, tuple[int, int]] = {
    "gemma-4-26b-a4b-it": (4096, 512),
    "gemma-4-31b-it": (1024, 512),
}

_ZAI_DISABLE_THINKING_PREFIXES = (
    "glm-4.5",
    "glm-4.6",
    "glm-4.7",
    "glm-5",
)


def _model_stem(model: str) -> str:
    return model.lower().split("/")[-1]


def no_temperature(model: str) -> bool:
    stem = _model_stem(model)
    if stem in _NO_TEMPERATURE_MODELS:
        return True
    return any(p.match(stem) for p in _NO_TEMPERATURE_PATTERNS)


def thinking_kwargs(model: str, max_tokens: int) -> dict:
    stem = _model_stem(model)
    lower_model = model.lower()
    via_openrouter = lower_model.startswith("openrouter/")
    via_moonshot = lower_model.startswith("moonshot/")
    via_gemini = lower_model.startswith("gemini/")
    via_zai = lower_model.startswith("zai/")
    if via_zai and stem.startswith(_ZAI_DISABLE_THINKING_PREFIXES):
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    for fragment, (budget, min_out) in _GEMINI_THINKING_LEVEL_MODELS.items():
        if via_gemini and fragment in stem:
            return {
                "max_tokens": max(max_tokens, budget + min_out),
                "extra_body": {
                    "generationConfig": {"thinkingConfig": {"thinkingLevel": "high"}}
                },
            }
    for fragment, (budget, min_out) in _THINKING_MODELS.items():
        if fragment in stem:
            adjusted = max(max_tokens, budget + min_out)
            if via_moonshot:
                return {"max_tokens": adjusted}
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


def build_kwargs(params: GenerationParams) -> dict:
    """Build litellm kwargs with model-specific compatibility applied."""
    kwargs: dict = {"max_tokens": params.max_tokens}
    if not no_temperature(params.model):
        kwargs["temperature"] = params.temperature
    kwargs.update(thinking_kwargs(params.model, params.max_tokens))
    kwargs.update(params.extra)
    return kwargs

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

from .strategies import (
    REGISTRY,
    CompletionStrategy,
    ContinuationStrategy,
    FIMStrategy,
    PrefillStrategy,
    SystemPromptStrategy,
)
from .strategies.compat import KNOWN_ANTHROPIC_MODELS, NO_PREFILL_MODELS

# Models that use the native completions API
_COMPLETION_MODELS = {
    "gpt-3.5-turbo-instruct",
    "davinci-002",
    "babbage-002",
    "gpt-oss-120b",
    "gpt-oss-20b",
}
_COMPLETION_SUBSTRINGS = ["text-davinci", "text-curie", "text-babbage", "text-ada"]

# Models where FIM is the right move
_FIM_SUBSTRINGS = ["deepseek-coder", "starcoder", "codellama", "fim"]

# Exact aliases for useful shorthand or models newer than LiteLLM's baked-in
# list. Values are fully-qualified LiteLLM model IDs.
_MODEL_ALIASES = {
    "kimi-k2": "moonshot/kimi-k2-0905-preview",
    "moonshot/kimi-k2": "moonshot/kimi-k2-0905-preview",
    "gemma-4": "gemini/gemma-4-26b-a4b-it",
    "gemini/gemma-4": "gemini/gemma-4-26b-a4b-it",
    "gemma-4-26b": "gemini/gemma-4-26b-a4b-it",
    "gemini/gemma-4-26b": "gemini/gemma-4-26b-a4b-it",
    "gemma-4-31b": "gemini/gemma-4-31b-it",
    "gemini/gemma-4-31b": "gemini/gemma-4-31b-it",
}

# Provider prefix to add when litellm can't auto-detect from model name alone
_PREFIX_MAP = {
    "claude": "anthropic",
    "opus": "anthropic",
    "sonnet": "anthropic",
    "haiku": "anthropic",
    "gemini": "gemini",
    "gemma": "gemini",
    "glm": "zai",
    "command": "cohere",
    "grok": "xai",
    "kimi": "moonshot",
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
}


def _fix_anthropic_dots(name: str) -> str:
    """Anthropic model IDs use dashes between version numbers (4-6, not 4.6)."""
    return name.replace(".", "-")


def _resolve_anthropic_alias(name: str) -> str:
    """Best-effort: if `name` uniquely matches one known Anthropic model, expand it.

    Lets users type `claude-3-haiku` → `claude-3-haiku-20240307`,
    `sonnet-4-5` → `claude-sonnet-4-5-20250929`, etc. Ambiguous or
    unmatched inputs pass through unchanged (provider will 404 if invalid).
    """
    if name in KNOWN_ANTHROPIC_MODELS:
        return name
    matches = [m for m in KNOWN_ANTHROPIC_MODELS if name in m]
    return matches[0] if len(matches) == 1 else name


def _normalize_anthropic_name(name: str) -> str:
    return _resolve_anthropic_alias(_fix_anthropic_dots(name))


def normalize_model(model: str) -> str:
    """Add provider prefix if litellm can't resolve, and fix well-known ID typos."""
    alias = _MODEL_ALIASES.get(model.lower())
    if alias:
        return alias

    # Split off an explicit provider prefix
    if "/" in model:
        prefix, _, name = model.partition("/")
        if prefix == "anthropic":
            name = _normalize_anthropic_name(name)
        resolved = f"{prefix}/{name}"
        return _MODEL_ALIASES.get(resolved.lower(), resolved)

    # Prefer known local aliases before probing LiteLLM. Some LiteLLM failures
    # print provider-help text as a side effect, which corrupts CLI output.
    m = model.lower()
    for fragment, provider in _PREFIX_MAP.items():
        if fragment in m:
            if provider == "anthropic":
                return f"anthropic/{_normalize_anthropic_name(model)}"
            return f"{provider}/{model}"

    # Try litellm's native detection first
    try:
        _, detected, _, _ = get_llm_provider(model)
        if detected == "anthropic":
            return _normalize_anthropic_name(model)
        return model
    except Exception:
        pass

    # Best-effort: bare names like `sonnet-4-5` or `opus-4-7` that uniquely
    # match a known Anthropic ID get expanded and prefixed.
    fixed = _fix_anthropic_dots(model)
    resolved = _resolve_anthropic_alias(fixed)
    if resolved != fixed and resolved in KNOWN_ANTHROPIC_MODELS:
        return f"anthropic/{resolved}"

    # Fallback: infer provider from a name fragment
    for fragment, provider in _PREFIX_MAP.items():
        if fragment in m:
            if provider == "anthropic":
                return f"anthropic/{_normalize_anthropic_name(model)}"
            return f"{provider}/{model}"
    return model


def detect_strategy(model: str, override: str | None = None) -> ContinuationStrategy:
    if override:
        if override not in REGISTRY:
            valid = ", ".join(REGISTRY)
            raise ValueError(f"Unknown strategy {override!r}. Valid: {valid}")
        return REGISTRY[override]()

    m = model.lower()
    stem = m.split("/")[-1]

    if "claude" in m:
        if stem in NO_PREFILL_MODELS:
            return SystemPromptStrategy()
        return PrefillStrategy()

    if model in _COMPLETION_MODELS or any(s in m for s in _COMPLETION_SUBSTRINGS):
        return CompletionStrategy()

    if any(s in m for s in _FIM_SUBSTRINGS):
        return FIMStrategy()

    return SystemPromptStrategy()

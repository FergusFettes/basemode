from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

from .strategies import (
    REGISTRY,
    CompletionStrategy,
    ContinuationStrategy,
    FIMStrategy,
    PrefillStrategy,
    SystemPromptStrategy,
)
from .strategies.compat import NO_PREFILL_MODELS

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

# Provider prefix to add when litellm can't auto-detect from model name alone
_PREFIX_MAP = {
    "claude": "anthropic",
    "gemini": "gemini",
    "command": "cohere",
}


def normalize_model(model: str) -> str:
    """Add provider prefix if litellm can't resolve the model name."""
    if "/" in model:
        return model
    try:
        get_llm_provider(model)
        return model
    except Exception:
        m = model.lower()
        for fragment, provider in _PREFIX_MAP.items():
            if fragment in m:
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

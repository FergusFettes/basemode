from .strategies import (
    REGISTRY,
    CompletionStrategy,
    ContinuationStrategy,
    FIMStrategy,
    PrefillStrategy,
    SystemPromptStrategy,
)

# Models that use the native completions API
_COMPLETION_MODELS = {
    "gpt-3.5-turbo-instruct",
    "davinci-002",
    "babbage-002",
}
_COMPLETION_SUBSTRINGS = ["text-davinci", "text-curie", "text-babbage", "text-ada"]

# Models where FIM is the right move
_FIM_SUBSTRINGS = ["deepseek-coder", "starcoder", "codellama", "fim"]


def detect_strategy(model: str, override: str | None = None) -> ContinuationStrategy:
    if override:
        if override not in REGISTRY:
            valid = ", ".join(REGISTRY)
            raise ValueError(f"Unknown strategy {override!r}. Valid: {valid}")
        return REGISTRY[override]()

    m = model.lower()

    if "claude" in m:
        return PrefillStrategy()

    if model in _COMPLETION_MODELS or any(s in m for s in _COMPLETION_SUBSTRINGS):
        return CompletionStrategy()

    if any(s in m for s in _FIM_SUBSTRINGS):
        return FIMStrategy()

    return SystemPromptStrategy()

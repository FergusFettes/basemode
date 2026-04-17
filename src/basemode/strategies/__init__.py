from .base import ContinuationStrategy
from .completion import CompletionStrategy
from .few_shot import FewShotStrategy
from .fim import FIMStrategy
from .prefill import PrefillStrategy
from .system import SystemPromptStrategy

REGISTRY: dict[str, type[ContinuationStrategy]] = {
    s.name: s
    for s in [
        CompletionStrategy,
        PrefillStrategy,
        SystemPromptStrategy,
        FewShotStrategy,
        FIMStrategy,
    ]
}

__all__ = [
    "ContinuationStrategy",
    "CompletionStrategy",
    "FewShotStrategy",
    "FIMStrategy",
    "PrefillStrategy",
    "SystemPromptStrategy",
    "REGISTRY",
]

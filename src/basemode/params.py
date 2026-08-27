from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationParams:
    model: str
    max_tokens: int = 200
    temperature: float = 0.9
    context: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

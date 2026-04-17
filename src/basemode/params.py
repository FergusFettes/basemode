from dataclasses import dataclass, field


@dataclass
class GenerationParams:
    model: str
    max_tokens: int = 200
    temperature: float = 0.9
    extra: dict = field(default_factory=dict)

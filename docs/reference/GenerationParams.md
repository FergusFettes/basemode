# GenerationParams

`basemode.params.GenerationParams`

```python
@dataclass
class GenerationParams:
    model: str
    max_tokens: int = 200
    temperature: float = 0.9
    context: str = ""
    extra: dict = field(default_factory=dict)
```

Container for model and generation settings passed into strategies.

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str` | Normalized model identifier |
| `max_tokens` | `int` | Requested max output tokens |
| `temperature` | `float` | Sampling temperature |
| `context` | `str` | Optional context injected into strategy prompting |
| `extra` | `dict` | Provider-specific LiteLLM kwargs |

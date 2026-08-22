# continue_text

`basemode.continue_.continue_text`

```python
async def continue_text(
    prefix: str,
    model: str = "gpt-4o-mini",
    *,
    max_tokens: int = 200,
    temperature: float = 0.9,
    context: str = "",
    strategy: str | None = None,
    rewind: bool = False,
    strict_max_tokens: bool = False,
    **extra,
) -> AsyncGenerator[str, None]
```

Stream a single continuation token-by-token.

## Notes

- Model names are normalized before strategy selection.
- `strategy` overrides auto-detection.
- `rewind=True` rewinds short trailing word fragments for `system`/`few_shot` strategies.
- `strict_max_tokens=True` stops the visible stream at `max_tokens` using client-side token counting.
- `extra` is forwarded to LiteLLM request kwargs.
- Raises `EmptyCompletionError` (with `model`, `strategy`, and `finish_reason`) if the
  provider stream ends without yielding any content.

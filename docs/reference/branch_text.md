# branch_text

`basemode.continue_.branch_text`

```python
async def branch_text(
    prefix: str,
    model: str = "gpt-4o-mini",
    *,
    n: int = 4,
    max_tokens: int = 200,
    temperature: float = 0.9,
    strategy: str | None = None,
    rewind: bool = False,
    strict_max_tokens: bool = False,
    **extra,
) -> AsyncGenerator[tuple[int, str], None]
```

Stream `n` parallel continuations.

Yields `(branch_idx, token)` tuples until all branches finish.

## Notes

- Branches run concurrently with `asyncio` tasks.
- Stream order is interleaved across branches.
- `branch_idx` is zero-based.
- `n` must be at least one; a provider error from any branch is propagated and cancels the rest.
  This includes `EmptyCompletionError` (with `model`, `strategy`, and `finish_reason`) when
  a branch's stream ends without yielding any content.
- `strict_max_tokens=True` stops each visible branch stream at `max_tokens` using client-side token counting.
  Some models are sent a wider budget than requested, because their hidden reasoning
  would otherwise consume it (see `basemode.strategies.compat`); pass this flag when the
  visible length must match `max_tokens` exactly.

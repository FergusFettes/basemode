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
    **extra,
) -> AsyncGenerator[tuple[int, str], None]
```

Stream `n` parallel continuations.

Yields `(branch_idx, token)` tuples until all branches finish.

## Notes

- Branches run concurrently with `asyncio` tasks.
- Stream order is interleaved across branches.
- `branch_idx` is zero-based.

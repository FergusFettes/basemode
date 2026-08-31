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
    record_health: bool = True,
    observation: ObservationContext | None = None,
    retry_empty_completion: bool = True,
    on_usage: BranchUsageCallback | None = None,
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
- `observation` attaches the same allow-listed caller provenance to each independent branch
  operation. It never carries prompt or response content.
- `record_health` is retained for source compatibility but does not bypass the unified ledger.
- `retry_empty_completion=False` disables the default retry after an empty response.
- `on_usage` is called once per branch with its index and provider usage payloads.

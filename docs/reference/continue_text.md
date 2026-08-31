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
    record_health: bool = True,
    observation: ObservationContext | None = None,
    retry_empty_completion: bool = True,
    on_raw_head: RawHeadCallback | None = None,
    raw_head_chars: int = 32,
    on_usage: UsageCallback | None = None,
    **extra,
) -> AsyncGenerator[str, None]
```

Stream a single continuation token-by-token.

## Notes

- Model names are normalized before strategy selection.
- `strategy` overrides auto-detection.
- `rewind=True` holds the prefix's last short token back from `system`/`few_shot`
  generation so a word split across the boundary is rejoined exactly rather than
  heuristically; the request is reissued in full if the model does not re-emit it.
- `strict_max_tokens=True` stops the visible stream at `max_tokens` using client-side token counting.
  Some models are sent a wider budget than requested, because their hidden reasoning
  would otherwise consume it (see `basemode.strategies.compat`); pass this flag when the
  visible length must match `max_tokens` exactly.
- `extra` is forwarded to LiteLLM request kwargs.
- `observation` attaches allow-listed caller provenance such as `source="loom"` and a
  package version. It never carries prompt or response content.
- `record_health` is retained for source compatibility but no longer bypasses the unified
  observation ledger. Set the documented global recording opt-out to disable local recording.
- `retry_empty_completion=False` disables the default retry after an empty first response.
- `on_raw_head` receives up to `raw_head_chars` unprocessed opening characters; `on_usage`
  receives every provider usage payload after the stream ends. See [[Python API]].
- Raises `EmptyCompletionError` (with `model`, `strategy`, and `finish_reason`) if both the
  initial request and enabled retry end without content.

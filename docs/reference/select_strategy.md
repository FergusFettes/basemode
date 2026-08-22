# select_strategy

`basemode.detect.select_strategy`

```python
def select_strategy(
    model: str,
    override: str | None = None,
    *,
    allow_user_override: bool = True,
) -> StrategyChoice
```

Resolve the continuation strategy for a model and report what decided it.
`detect_strategy` wraps this and returns the strategy instance instead.

```python
from basemode import select_strategy

choice = select_strategy("moonshot/kimi-k3")
print(choice.name, choice.source)  # prefill registry
```

## Returns

`StrategyChoice(name, source)`, where `source` is one of:

| Source | Meaning |
|---|---|
| `explicit` | Came from the `override` argument |
| `user` | Pinned on this machine by `basemode bench --save` |
| `registry` | The `prompt_method` verified for this model |
| `heuristic` | Derived from the model name |

## Resolution order

1. `override`, if set — raises `ValueError` if it isn't a known strategy.
2. A locally pinned strategy for this model, unless `allow_user_override=False`.
3. The `prompt_method` recorded for the model in the verified-models registry.
4. Model-name heuristics: Claude → `prefill` (or `system` when `no_prefill`
   applies), known completion models → `completion`, FIM families → `fim`,
   everything else → `system`.

Steps 2 and 3 are skipped when the strategy they name is `prefill` and the
model carries the `no_prefill` quirk, so stale data degrades to the heuristic
rather than to a failed request.

Pass `allow_user_override=False` in scripts that generate committed data, so
one developer's local pin can't leak into a shared file.

See [[Strategies]] for the surrounding workflow, and [[Model Normalization]]
for provider-prefix and alias handling.

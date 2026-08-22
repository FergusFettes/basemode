# detect_strategy

`basemode.detect.detect_strategy`

```python
def detect_strategy(model: str, override: str | None = None) -> ContinuationStrategy
```

Select the continuation strategy for a model, and return an instance of it.

## Resolution order

1. `override`, if set (or `ValueError` if it isn't a known strategy).
2. A strategy pinned for this model by `basemode bench --save`.
3. The `prompt_method` verified for this model in the registry.
4. Model-name heuristics: Claude models use `prefill`, except no-prefill
   models which use `system`; known completion models use `completion`; FIM
   model families use `fim`; the fallback is `system`.

Use [[select_strategy]] instead when you need to know *which* of those
applied — it returns the name alongside its source. See [[Strategies]] for
the full picture and [[Model Normalization]] for provider-prefix and alias
handling.

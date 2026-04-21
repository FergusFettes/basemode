# detect_strategy

`basemode.detect.detect_strategy`

```python
def detect_strategy(model: str, override: str | None = None) -> ContinuationStrategy
```

Select the continuation strategy for a model.

## Resolution order

1. If `override` is set, return that strategy (or raise `ValueError` if invalid).
2. Claude models use `prefill`, except known no-prefill models which use `system`.
3. Known completion models use `completion`.
4. FIM model families use `fim`.
5. Fallback is `system`.

For provider-prefix and alias handling, see [[Model Normalization]].

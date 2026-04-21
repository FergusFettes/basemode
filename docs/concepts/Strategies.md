# Strategies

`basemode` supports five continuation strategies.

| Strategy | When used | Notes |
|----------|-----------|-------|
| `completion` | Base completion models (`davinci-002`, etc.) | Uses text completion API directly |
| `prefill` | Anthropic Claude models that still allow assistant prefill | Seeds assistant turn with suffix of prefix |
| `system` | Default fallback for most chat models | Strict continuation-only system prompt |
| `few_shot` | Manual override for stubborn models | Uses varied continuation examples |
| `fim` | FIM-capable code models | Uses model-family-specific FIM tokens |

## Auto-selection

`detect_strategy(model)` picks a strategy from normalized model ID.

- Claude models default to `prefill`
- Newer Claude models in a no-prefill allowlist are forced to `system`
- Known completion models go to `completion`
- FIM-family names go to `fim`
- Everything else uses `system`

## Manual override

```bash
basemode "Text to continue" --strategy few_shot
```

If you pass an unknown strategy, `basemode` raises a clear validation error with valid names.

# Model Normalization

`normalize_model()` resolves shorthand names into provider-qualified IDs.

Examples:

- `claude-sonnet-4-6` -> `anthropic/claude-sonnet-4-6`
- `gpt-4o-mini` -> `openai/gpt-4o-mini`
- `kimi-k2` -> `moonshot/kimi-k2-0905-preview`

## What normalization handles

- Provider prefix inference from model-name fragments
- Anthropic version formatting fixes (`4.6` -> `4-6`)
- Alias expansion for known shorthand names
- Best-effort disambiguation for Claude family inputs

Normalization runs before strategy detection and before pricing/usage lookup.

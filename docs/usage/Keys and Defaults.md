# Keys and Defaults

`basemode` supports both environment variables and a persistent local key store.

## Persistent key storage

Keys are saved to:

- `~/.config/basemode/auth.json`

File mode is restricted to user-only (`0600`).

Use:

```bash
basemode keys set openai
basemode keys list
basemode keys get openai
```

## Load order

At startup, settings are loaded in this order (later wins):

1. `~/.config/basemode/auth.json`
2. `.env` (project-local if present)
3. Existing process environment variables

Environment variables are never overwritten by stored keys.

## Default model

Set a default model once:

```bash
basemode default claude-sonnet-4-6
```

Show current default:

```bash
basemode default
```

Clear it:

```bash
basemode default --unset
```

When unset, CLI generation defaults to `gpt-4o-mini`.

## Pinned strategies

`basemode bench --save` records the winning strategy for a model in the same
file, under `strategy_overrides`:

```json
{
  "keys": {"anthropic": "sk-..."},
  "default_model": "claude-sonnet-4-6",
  "strategy_overrides": {"moonshot/kimi-k3": "few_shot"}
}
```

A pin outranks the strategy shipped in the verified-models registry but is
still overridden by an explicit `--strategy`, and is ignored if the model has
since gained a quirk that rules it out. Pins are keyed by the normalized model
ID, so `kimi-k3` and `moonshot/kimi-k3` are the same pin.

```bash
basemode strategies              # lists pins under the strategy table
basemode strategies --unpin kimi-k3
```

See [[Strategies]] for the full precedence order.

## Model ratings

`basemode rate` records a personal thumbs up or thumbs down for a model in the
same file, under `model_ratings`:

```json
{
  "keys": {"anthropic": "sk-..."},
  "model_ratings": {"anthropic/claude-opus-5": 1, "openai/gpt-4o": -1}
}
```

```bash
basemode rate claude-opus-5 up
basemode rate gpt-4o down
basemode rate gpt-4o clear
basemode rate                      # list every rated model
```

A rating only affects ordering: models you rated up sort to the top of
`basemode models` (and of any picker built on `list_model_picker_entries`),
models you rated down sort to the bottom, and nothing is ever hidden. Ratings
are keyed like strategy pins — by the normalized model ID.

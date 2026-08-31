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

## Observed model health

A rating is what you think of a model. Health is what the model has actually
done here: every continuation records its outcome against the model it ran on,
so a failure rate builds up from real usage rather than from the weekly
registry probe.

New call observations live in `~/.local/share/basemode/observations.sqlite`
rather than in `auth.json`,
because branches generate in parallel and read-modify-write on a JSON file
loses counts under exactly that load.

```bash
basemode health                    # every model seen, worst failure rate first
basemode health claude-opus-5      # one model
basemode health --days 7           # narrow the failure breakdown
basemode health --json             # raw records
basemode health --clear            # forget everything
basemode info gpt-4o               # health and rating alongside pricing
```

Failures are recorded by category — `authentication`, `quota`, `rate_limit`,
`timeout`, `network`, `provider_unavailable`, `invalid_request`,
`empty_response`, `content_filter`, `provider_error`, `cancelled`, or `unknown`
— which is the difference between "fix your key", "wait",
and "pick another model". All-time totals are kept forever; the per-category
breakdown comes from an event log pruned after 30 days.

For invalid requests, the event record also keeps the provider's safe machine
details when supplied: an error code and rejected parameter name. It does not
retain provider error messages, which can include request content.

A cancelled stream counts as a success if any tokens had already arrived: a
consumer walking away is not a verdict on the model.

Set `BASEMODE_NO_HEALTH=1` to turn recording off entirely.

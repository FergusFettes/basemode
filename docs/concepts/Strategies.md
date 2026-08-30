# Strategies

A *strategy* is how `basemode` asks a chat model to behave like a base model.
Five are available.

| Strategy | When used | Notes |
|----------|-----------|-------|
| `completion` | Base completion models (`davinci-002`, etc.) | Uses text completion API directly |
| `prefill` | Models that allow seeding the assistant turn | Seeds assistant turn with suffix of prefix |
| `system` | Default fallback for most chat models | Strict continuation-only system prompt |
| `few_shot` | Models that ignore a plain instruction | Continuation examples across four registers |
| `fim` | FIM-capable code models | Uses model-family-specific FIM tokens |

## How a strategy gets chosen

`select_strategy(model)` resolves the strategy and reports where the choice
came from. `detect_strategy(model)` is the same resolution, returning the
strategy instance. Precedence, highest first:

| Source | Set by | Scope |
|---|---|---|
| `explicit` | `--strategy` / `strategy=` | One call |
| `user` | `basemode bench --save` | This machine, this model |
| `registry` | The verified-models registry's `prompt_method` | Shipped with the package |
| `heuristic` | Model-name rules in `detect.py` | Anything unregistered |

The registry's `prompt_method` records a strategy that passed model discovery.
The [[Verified Models]] table and runtime selection read the same field.

A registry or user choice of `prefill` is skipped for any model carrying the
`no_prefill` quirk. Providers can withdraw prefill support between releases;
an incompatible entry therefore falls back to the heuristic.

The name heuristics are the last resort, for models nobody has probed:

- Claude models default to `prefill`, or `system` when `no_prefill` applies
- Known completion models go to `completion`
- FIM-family names go to `fim`
- Everything else uses `system`

Check what a given model will do, and why:

```bash
basemode info claude-opus-5
```

## Comparing strategies

`basemode bench` runs the candidate strategies against the same model and
ranks them by how cleanly they continue text.

```bash
basemode bench claude-opus-5 --samples
```

```
┏━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Strategy ┃ Score ┃ Flags / error                ┃ Mean s ┃
┡━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ few_shot │ 0.93  │ quoted                       │ 2.07   │
│ system   │ 0.75  │ empty                        │ 2.24   │
│ prefill  │ 0.00  │ BadRequestError: ...         │ 0.30   │
└──────────┴───────┴──────────────────────────────┴────────┘
```

Each strategy runs against narrative prose, technical instructions, poetry,
and dialogue. The different registers expose prompt shapes that only work for
plain prose.

`--save` pins the winner in `~/.config/basemode/auth.json`; see [[Keys and
Defaults]]. Latency breaks ranking ties, but `bench` recommends changing the
current strategy only when another strategy has a higher score.

## Continuation scoring

`basemode.scoring.score_continuation(prefix, text)` returns a number between
0.0 and 1.0 plus flags for detected problems. It is what `bench` ranks with and what
`scripts/discover_new_models.py` uses to decide whether a newly-listed model is
worth registering at all.

| Flag | What it caught | Penalty |
|---|---|---:|
| `empty` | Nothing but whitespace came back | 1.00 |
| `refusal` | "I'm sorry, I can't…" | 0.90 |
| `preamble` | "Sure! Here's…", "Certainly…" | 0.60 |
| `echoed_prefix` | The model restated the prefix before continuing | 0.50 |
| `chat_turn` | A `User:` / `Assistant:` transcript appeared | 0.40 |
| `meta_commentary` | It talked *about* the text instead of being it | 0.40 |
| `quoted` | The continuation came back wrapped in quotes | 0.30 |
| `code_fence` | A prose prefix answered with a code fence | 0.30 |
| `list_formatting` | A prose prefix answered with bullets or a heading | 0.25 |
| `bad_boundary` | A word welded onto the prefix's last word | 0.15 |

Penalties are subtracted from 1.0 and the result is clamped at zero, so one
heavy flag disqualifies a strategy while a couple of light ones only reorder
the ranking. Anything at 0.75 or above counts as clean. `bad_boundary` is only
reported when nothing else fired — on top of a preamble it would double-count a
failure already accounted for.

The scorer detects assistant behavior, not prose quality. Fluent and dull
continuations can both score 1.0.

## Manual override

```bash
basemode "Text to continue" --strategy few_shot
```

An unknown strategy name raises a validation error listing the valid ones.
Pins are per model and easy to inspect or drop:

```bash
basemode strategies                        # lists strategies, then any pins
basemode strategies --unpin claude-opus-5
```

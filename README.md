# basemode

[![PyPI](https://img.shields.io/pypi/v/basemode.svg)](https://pypi.org/project/basemode/)
[![Python](https://img.shields.io/pypi/pyversions/basemode.svg)](https://pypi.org/project/basemode/)
[![CI](https://github.com/FergusFettes/basemode/actions/workflows/ci.yml/badge.svg)](https://github.com/FergusFettes/basemode/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://fergusfettes.github.io/basemode/)
[![License](https://img.shields.io/pypi/l/basemode.svg)](https://github.com/FergusFettes/basemode/blob/main/LICENSE)

Make chat-tuned LLMs continue text instead of answering it.

`basemode` provides raw text continuation through the terminal, an async Python
API, and an OpenAI-compatible completions server. It selects a prompting method
for each model and repairs common artifacts at the join between the prompt and
the generated text.

## The problem

Ask a chat model to continue a passage and it may answer with an introduction,
quote the continuation, or offer to revise it:

```
> The ship rounded the headland and

I'd be happy to help continue this passage! Here's one possibility:

"The ship rounded the headland and the harbour opened before them..."

Would you like me to continue in a different style?
```

With `basemode`, the same prefix produces continuation text directly:

```console
$ basemode "The ship rounded the headland and"
The ship rounded the headland and the wind dropped all at once, as though
the cliffs had swallowed it. Sails that had been drum-tight for three days
went slack, and in the sudden quiet the crew could hear water moving along
the hull.
```


## Install

```bash
pip install basemode
```

Requires Python 3.11 or later. Set the relevant provider key in the environment
or a `.env` file, or save it with `basemode keys set PROVIDER`.

## Quickstart

```bash
# Continue a prefix
basemode "The ship rounded the headland and"

# Parallel continuations
basemode "The ship rounded the headland and" -n 3

# Inspect selected strategy and pricing metadata
basemode info claude-sonnet-4-6

# Compare prompting methods and save the best result
basemode bench claude-sonnet-4-6 --save

# Show only key-configured models
basemode models --available
```

Set a default model with `basemode default MODEL`. Without one, generation uses
`gpt-4o-mini`. See the [installation guide](https://fergusfettes.github.io/basemode/getting-started/Installation/)
for provider-key names and source installation.

## Commands

```bash
basemode --help
basemode run --help
basemode models --help
basemode info --help
basemode strategies --help
```

- `basemode run`: stream one or more continuations; this is the default command
- `basemode models`: list known models and filter by provider or availability
- `basemode info`: inspect a model's normalized ID, strategy, quirks, price, and local health
- `basemode bench`: compare strategies using live requests and optionally pin the winner
- `basemode verify`: collect durable compatibility evidence with bounded live probes
- `basemode serve`: expose an OpenAI-compatible `/v1/completions` endpoint
- `basemode keys`, `default`, and `rate`: manage local preferences

## How strategy selection works

Different models respond to different prompt shapes: native completion,
assistant prefill, a system instruction, few-shot examples, or FIM tokens.
`basemode` chooses in this order:

1. A strategy supplied for the current call.
2. A local pin saved by `basemode bench --save`.
3. The strategy recorded in the verified-model registry.
4. A model-name heuristic.

`basemode info MODEL` reports the selected strategy and its source. `bench`
compares strategies on prose, technical writing, poetry, and dialogue, then
scores assistant preambles, refusals, echoed prefixes, chat turns, and stray
formatting.

```bash
basemode bench claude-opus-5 --samples   # compare
basemode bench claude-opus-5 --save      # pin the winner for this model
```

```
┏━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Strategy ┃ Score ┃ Flags / error        ┃ Mean s ┃
┡━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ few_shot │ 0.93  │ quoted               │ 2.07   │
│ system   │ 0.75  │ empty                │ 2.24   │
│ prefill  │ 0.00  │ BadRequestError: ... │ 0.30   │
└──────────┴───────┴──────────────────────┴────────┘
```

See [Strategies](https://fergusfettes.github.io/basemode/concepts/Strategies/)
for the prompt shapes, scoring rules, and override behavior.

## Server

`basemode serve` exposes `POST /v1/completions` for clients that support the
classic OpenAI completions API, including
[Tapestry Loom](https://github.com/transkatgirl/Tapestry-Loom).

```bash
pip install 'basemode[server]'
basemode serve --port 8080
```

Set the client endpoint to `http://127.0.0.1:8080/v1/completions`. Responses
are synchronous JSON; streaming and logprobs are not supported. The
[CLI reference](https://fergusfettes.github.io/basemode/usage/CLI-Reference/#serve)
lists accepted fields.

## Python API

```python
from basemode import continue_text, branch_text

async for token in continue_text(
    "The ship rounded the headland and",
    model="gpt-4o-mini",
    max_tokens=120,
):
    print(token, end="", flush=True)

async for idx, token in branch_text(
    "The ship rounded the headland and",
    model="gpt-4o-mini",
    n=3,
    max_tokens=80,
):
    print(idx, token, end="", flush=True)
```

## Documentation and development

The [documentation](https://fergusfettes.github.io/basemode/) covers
configuration, strategy behavior, model evidence, and the complete CLI and
Python APIs. To build it locally:

```bash
make docs-serve
```

For repository setup and test commands, see the
[agent quickstart](https://fergusfettes.github.io/basemode/Agent-Quickstart/).

<!-- verified-models:start -->

## Verified models

The registry currently contains 285 model endpoints with a tested prompt strategy. The [full model table](https://fergusfettes.github.io/basemode/usage/Verified-Models/) lists prices, release dates, strategies, and compatibility quirks. It is generated from the same registry used at runtime.

<!-- verified-models:end -->

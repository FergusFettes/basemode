# Python API

Public API exports:

```python
from basemode import (
    ContinuationScore,
    EmptyCompletionError,
    GenerationParams,
    StrategyChoice,
    bench_model,
    branch_text,
    build_model_picker_state,
    continue_text,
    detect_strategy,
    list_model_picker_entries,
    score_continuation,
    select_strategy,
)
```

## `continue_text`

Stream a single continuation.

```python
from basemode import continue_text

async for token in continue_text(
    "She opened the letter with trembling hands.",
    model="gpt-4o-mini",
    max_tokens=200,
    temperature=0.9,
):
    print(token, end="", flush=True)
```

If a provider stream ends without yielding any content — a content filter, a
stop sequence hit immediately, a truncated response — the strategy raises
`EmptyCompletionError` instead of silently completing with nothing. It carries
`model`, `strategy`, and `finish_reason` (when the provider reported one):

```python
from basemode import EmptyCompletionError, continue_text

try:
    async for token in continue_text(prefix, model="moonshot/kimi-k3"):
        print(token, end="")
except EmptyCompletionError as exc:
    print(f"no tokens from {exc.model} ({exc.strategy}): {exc.finish_reason}")
```

## `branch_text`

Stream `n` parallel continuations as `(branch_idx, token)` tuples.

```python
from basemode import branch_text

async for idx, token in branch_text(
    "She opened the letter",
    model="anthropic/claude-sonnet-4-6",
    n=4,
    max_tokens=200,
):
    print(f"[{idx}] {token}", end="")
```

## `detect_strategy`

Get the strategy object that will be used for a model.

```python
from basemode import detect_strategy

strategy = detect_strategy("anthropic/claude-sonnet-4-6")
print(strategy.name)  # system
```

## `select_strategy`

Same resolution, but reporting where the choice came from — `explicit`,
`user`, `registry`, or `heuristic`. See [[select_strategy]].

```python
from basemode import select_strategy

choice = select_strategy("moonshot/kimi-k3")
print(choice.name, choice.source)  # prefill registry
```

## `score_continuation`

Score a continuation from 0.0 to 1.0 and get the flags explaining it. See
[[score_continuation]].

```python
from basemode import score_continuation

result = score_continuation("The ship rounded the headland and", "Sure! Here you go:")
print(result.score, result.flags)  # 0.4 ('preamble',)
```

## `bench_model`

Rank strategies for a model by running them. Makes real API calls.

```python
import asyncio

from basemode import bench_model

results = asyncio.run(bench_model("anthropic/claude-opus-5"))
for result in results:
    print(result.strategy, result.score, result.flags)
```

Returns `StrategyResult` objects, best first. See [[Strategies]] for how the
probes and scoring work.

## `GenerationParams`

Dataclass passed into strategy implementations:

```python
from basemode import GenerationParams

params = GenerationParams(
    model="gpt-4o-mini",
    max_tokens=200,
    temperature=0.9,
    context="",
    extra={},
)
```

## Model picker helpers

Use structured model metadata for frontend pickers:

```python
from basemode import build_model_picker_state, list_model_picker_entries

entries = list_model_picker_entries(verified_only=True)
state = build_model_picker_state(
    selected=["openai/gpt-4o-mini", "anthropic/claude-sonnet-4-6"],
    max_models=3,
    verified_only=True,
)
```

Each entry carries a `rating` field — this user's thumb for the model (`1`,
`-1`, or `None`) — and rated models sort ahead of or behind the reliability
ordering, so an explicit opinion outranks the shipped data. Read and write
thumbs with `basemode.keys`:

```python
from basemode.keys import RATING_UP, list_model_ratings, set_model_rating

set_model_rating("anthropic/claude-opus-5", RATING_UP)
list_model_ratings()  # {"anthropic/claude-opus-5": 1}
```

Pass `ratings={...}` to `list_model_picker_entries` to rank with thumbs from
somewhere other than the local store.

# Python API

Public API exports:

```python
from basemode import (
    GenerationParams,
    branch_text,
    build_model_picker_state,
    continue_text,
    detect_strategy,
    list_model_picker_entries,
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

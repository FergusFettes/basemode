# score_continuation

`basemode.scoring.score_continuation`

```python
def score_continuation(prefix: str, text: str) -> ContinuationScore
```

Score how well `text` behaves as a continuation of `prefix`: `1.0` is clean,
`0.0` is unusable.

```python
from basemode import score_continuation

score_continuation("The ship rounded the headland and", " the harbour opened out.")
# ContinuationScore(score=1.0, flags=())

score_continuation("The ship rounded the headland and", "Sure! Here's a continuation:")
# ContinuationScore(score=0.4, flags=('preamble',))
```

## Returns

`ContinuationScore(score, flags)` with two convenience members:

- `clean` — `True` when `score >= 0.75`
- `detail` — the flags as a string, or `"clean"`

Flags and their penalties are listed in [[Strategies]]. Penalties subtract
from 1.0 and clamp at zero.

## `looks_clean`

```python
def looks_clean(prefix: str, text: str) -> tuple[bool, str | None]
```

Pass/fail form for probes, returning `(ok, reason)` — `reason` is `None` on
success and otherwise names the flags with a snippet of the offending output.

## Caveats

The scorer recognizes the *shape* of assistant behavior, not the quality of
the writing. A fluent continuation and a dull one both score `1.0`; a good
continuation that happens to open with a quotation mark is flagged `quoted`.
It is built to compare strategies for the same model, not to judge prose.

# Model Normalization

`normalize_model()` resolves whatever you typed into a fully-qualified LiteLLM
model ID. It runs before strategy detection and before pricing lookup, so every
downstream step sees one canonical name.

```python
from basemode.detect import normalize_model

normalize_model("claude-sonnet-4-6")  # "anthropic/claude-sonnet-4-6"
normalize_model("gpt-4o-mini")        # "openai/gpt-4o-mini"
normalize_model("kimi-k2")            # "moonshot/kimi-k2-0905-preview"
```

The point is that you should be able to type the name you actually remember.

## Resolution order

1. **Exact alias.** A lookup table of shorthand and of models newer than
   LiteLLM's baked-in list. `kimi-k2` resolves to a specific dated snapshot
   (`moonshot/kimi-k2-0905-preview`) because the bare name is not routable.
2. **Explicit provider prefix.** Anything containing `/` is taken as
   provider-qualified and passed through, with Anthropic name fixes applied to
   the part after the slash.
3. **Provider inference from name fragments.** `claude`, `opus`, `sonnet` and
   `haiku` imply `anthropic`; `gemini` and `gemma` imply `gemini`; `gpt`, `o1`,
   `o3`, `o4` imply `openai`; also `glm` (zai), `grok` (xai), `command`
   (cohere), `kimi` (moonshot), `deepseek`.
4. **LiteLLM's own resolution**, as a last resort.

Local aliases are checked *before* LiteLLM is asked. This is deliberate and
worth preserving: some LiteLLM resolution failures print provider-help text to
stdout as a side effect, which corrupts CLI output that other tools parse.

## Anthropic-specific handling

Anthropic IDs use dashes between version digits, but everyone types dots:

```python
normalize_model("claude-opus-4.6")  # "anthropic/claude-opus-4-6"
```

Anthropic names also expand by unique substring match, so you can skip the
`claude-` prefix and the date suffix:

```python
normalize_model("sonnet-4-5")  # "anthropic/claude-sonnet-4-5-20250929"
normalize_model("opus-4-7")    # "anthropic/claude-opus-4-7"
```

Expansion only fires when the fragment matches **exactly one** known model. An
ambiguous or unknown fragment passes through untouched and the provider returns
a 404 — normalization guesses, but it never guesses between two candidates.

## Checking what you got

`basemode info` shows the resolved ID alongside the strategy it selected:

```bash
basemode info sonnet-4-5
```

If a model 404s unexpectedly, that is the first thing to look at: normalization
may have passed an ambiguous fragment through unchanged.

## Adding an alias

Aliases live in `_MODEL_ALIASES` and provider fragments in `_PREFIX_MAP`, both
in `src/basemode/detect.py`. Add a regression in `tests/` alongside. See
[[Agent Quickstart]].

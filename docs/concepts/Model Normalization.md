# Model Normalization

`normalize_model()` resolves whatever you typed into a fully-qualified LiteLLM
wire ID. It runs before generation, strategy detection and pricing lookup.
User-facing model names use lowercase `provider/creator/model`, with creator
aliases folded to one spelling. The wire ID preserves the provider's exact
spelling and capitalization for requests.

```python
from basemode.detect import normalize_model

normalize_model("claude-sonnet-4-6")  # "anthropic/claude-sonnet-4-6"
normalize_model("gpt-4o-mini")        # "openai/gpt-4o-mini"
normalize_model("kimi-k2")            # "moonshot/kimi-k2-0905-preview"
normalize_model("deepinfra/zai/glm-5.3-flash")  # "deepinfra/zai-org/GLM-5.3-Flash"
```

This lets the CLI accept familiar short names while the rest of the pipeline
uses provider-qualified IDs.

## Resolution order

1. **Exact alias.** A lookup table of shorthand and of models newer than
   LiteLLM's baked-in list. `kimi-k2` resolves to a specific dated snapshot
   (`moonshot/kimi-k2-0905-preview`) because the bare name is not routable.
2. **Explicit provider prefix.** Anything containing `/` is taken as
   provider-qualified, with Anthropic name fixes applied to the part after
   the slash.
3. **Provider inference from name fragments.** `claude`, `opus`, `sonnet` and
   `haiku` imply `anthropic`; `gemini` and `gemma` imply `gemini`; `gpt`, `o1`,
   `o3`, `o4` imply `openai`; also `glm` (zai), `grok` (xai), `command`
   (cohere), `kimi` (moonshot), `deepseek`.
4. **LiteLLM's own resolution**, as a last resort.

The resulting name is matched against the local catalog: exact wire IDs take
precedence, followed by case-insensitive wire IDs and canonical identities.
Known names resolve to the provider's spelling; ambiguous canonical identities
require an exact wire ID. Unknown names pass through for custom endpoints and
models newer than the catalog. `basemode default` additionally rejects unknown
names unless `--force` is given.

This resolution is shared by CLI model arguments and the Python streaming
APIs, including continuation, branching and chat. It makes no provider requests.
The catalog index is cached for the life of the process.

`basemode models`, including `--live`, displays canonical names. Use
`--wire-ids` to inspect exact provider IDs. JSON picker entries expose the
canonical name as `model`, `display` and `canonical_id`, and the exact provider
ID as `wire_id`. Snapshot IDs and picker selections use canonical names too.

Local aliases are checked before LiteLLM. Some LiteLLM resolution failures
print provider guidance to stdout, which would corrupt machine-readable CLI
output.

## Anthropic-specific handling

Anthropic IDs use dashes between version digits; dotted versions are accepted:

```python
normalize_model("claude-opus-4.6")  # "anthropic/claude-opus-4-6"
```

Anthropic names also expand by unique substring match, so you can skip the
`claude-` prefix and the date suffix:

```python
normalize_model("sonnet-4-5")  # "anthropic/claude-sonnet-4-5-20250929"
normalize_model("opus-4-7")    # "anthropic/claude-opus-4-7"
```

Expansion requires exactly one known match. Ambiguous or unknown fragments pass
through unchanged and may be rejected by the provider.

## Checking what you got

`basemode info` shows the canonical name and wire ID alongside the strategy it selected:

```bash
basemode info sonnet-4-5
```

If a model unexpectedly returns 404, check whether normalization left an
ambiguous fragment unchanged.

## Adding an alias

Aliases live in `_MODEL_ALIASES` and provider fragments in `_PREFIX_MAP`, both
in `src/basemode/detect.py`. Add a regression in `tests/` alongside. See
[[Agent Quickstart]].

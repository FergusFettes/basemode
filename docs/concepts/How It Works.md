# How It Works

## Core flow

Each continuation passes through this pipeline:

1. Normalize model name (`normalize_model`)
2. Select a strategy (`select_strategy` / `detect_strategy`)
3. Stream tokens via that strategy
4. Heal token boundaries/newlines so `prefix + tokens` is clean text

Selection uses an explicit override first, followed by a local pin, verified
registry data, and finally model-name heuristics. See [[Strategies]].

## Strategy abstraction

Every strategy implements a shared interface:

- `stream(prefix, params) -> AsyncGenerator[str, None]`

Provider-specific prompting stays behind this interface.

## Transport and model data

Strategies send chat or text-completion requests through a transport interface.
LiteLLM is the default transport and normalizes request and streaming response
shapes across providers. Providers whose protocols cannot be represented by
LiteLLM can supply another transport.

Basemode owns endpoint identity, verified strategies, compatibility quirks,
and model evidence. Live provider catalogs and direct verification take
precedence over LiteLLM's bundled model metadata. A provider-qualified ID can
therefore be called before it appears in LiteLLM's catalog. Pricing and token
metadata from LiteLLM are best-effort and may be unavailable.

## Why continuation needs coercion

Chat models tend to acknowledge, format, or discuss a prompt. Strategies reduce
that behavior using:

- Using native completions APIs when available
- Using Anthropic-style prefill where supported
- Falling back to strict system-prompt coercion
- Using few-shot coercion for stubborn models

## Scoring coercion

`basemode.scoring` assigns each continuation a score based on recognizable
failures: preambles, refusals, echoed prefixes, chat transcripts, and stray
formatting. `basemode bench` uses the score to rank strategies; model discovery
uses it when deciding whether an endpoint is suitable for registration.

## Token healing

Prompt wrappers can remove the space at the prefix boundary or place it inside
a word the model is finishing: `counted, a` + `nd` can render as `a nd`.
Stream output repairs this seam and related artifacts:

- Missing spaces between the prefix and first token.
- Split words at the generation boundary. The two halves are rejoined when the
  evidence says they are one word: the fragment cannot stand alone as a word
  (`nd`, `anks`, `urgery`) and either the merged form is a real word or the
  prefix already ended mid-word. A prefix ending on a complete word followed by
  another complete word is left alone, so `measured a` + `way`, `Section B` +
  `undefined`, and `his brother` + `Ed` all keep their space. This runs at the
  boundary only, never on later chunks of the same stream.
- Split compounds like `any one` where the model intended `anyone`. Phrases
  whose split reading is the real one are protected: `every one of them`,
  `some body of water`, `laid out side by side`, `her self-esteem`, and
  `cars in side streets` all survive intact.
- Newline artifacts that break prose flow.
- A line break the model opened with in the middle of a word. A continuation
  may legitimately start a new paragraph, a heading, or a scene break — but
  not when the prefix stops mid-word, since nothing starts a paragraph inside
  `thermodynamic`. The evidence required is the same as for the space case, so
  a real break after a complete word is never touched.

The package includes a compressed copy of the public-domain BSD `web2` word
list, avoiding platform-dependent behavior from `/usr/share/dict/words`.
Inflected forms are checked against their stems, and words already present in
the document are treated as known. This covers plurals, past tenses, proper
nouns, and repeated coinages without joining every unfamiliar boundary.

### Exact joins with `rewind`

Some joins are inherently ambiguous: `for` + `ever` may be one word or two.
With `rewind=True`, the last short token is held back before generation. If the
model repeats it as part of the continuation, the duplicate prefix is removed,
giving an exact join.

If the model does not repeat the held-back text, basemode reissues the request
with the full prefix before yielding output and falls back to heuristic repair.
This can add a second provider request, so rewind is disabled by default.

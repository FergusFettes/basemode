# How It Works

## Core flow

`basemode` runs a small pipeline:

1. Normalize model name (`normalize_model`)
2. Select a strategy (`select_strategy` / `detect_strategy`)
3. Stream tokens via that strategy
4. Heal token boundaries/newlines so `prefix + tokens` is clean text

Step 2 prefers a verified answer to a guess: the strategy recorded for the
model in the verified-models registry, or one pinned locally by
`basemode bench --save`, before falling back to model-name heuristics. See
[[Strategies]].

## Strategy abstraction

Every strategy implements a shared interface:

- `stream(prefix, params) -> AsyncGenerator[str, None]`

This keeps provider-specific behavior behind a common API.

## Transport and model data

Continuation strategies send chat or text-completion requests through a small
transport interface. LiteLLM is the default transport: it translates request
and streaming response shapes for many providers without requiring basemode to
depend on every provider SDK. A different transport can be installed for a
provider whose protocol cannot be represented safely by LiteLLM.

The transport is not the model catalog. Basemode owns endpoint identity,
verified strategies, compatibility quirks, and model evidence. Live provider
catalogs and direct verification take precedence over LiteLLM's bundled model
metadata. Consequently, a provider-qualified model can be called even when it
has not yet appeared in LiteLLM's model list; the qualified ID is passed to the
transport unchanged. LiteLLM pricing and token metadata remain best-effort
fallbacks and are reported as unavailable when they are missing.

## Why continuation needs coercion

Most chat models default to assistant behavior (acknowledgments, headings, commentary). `basemode` avoids that by:

- Using native completions APIs when available
- Using Anthropic-style prefill where supported
- Falling back to strict system-prompt coercion
- Using few-shot coercion for stubborn models

## Scoring coercion

Whether coercion actually worked is measurable, not a matter of taste. The
recognizable failures — preamble, refusal, echoed prefix, a chat transcript,
stray markdown — are scored by `basemode.scoring` into a single number per
continuation. That score is what `basemode bench` ranks strategies with and
what model discovery uses to accept or reject a newly-listed model, so the
same definition of "clean" governs both.

## Token healing

A chat model cannot be shown where the prefix ends without also being handed a
trailing space, and its first token comes back without one. The space has to be
put back, and sometimes it lands inside a word the model was in the middle of
finishing: `counted, a` + `nd` renders as `a nd` rather than `and`. Stream
output is post-processed to undo that and a few related artifacts:

- Missing space between prefix and first token.
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

The dictionary these rules consult ships with the package: Webster's Second
International, the BSD `web2` list whose 1934 copyright has lapsed, held as a
compressed word list in `basemode/data/`. It used to be read from
`/usr/share/dict/words`, which meant that on a host without that file — most
Linux images, every stock CI runner — no word was known, every prefix looked
like it ended mid-word, and boundaries were joined that should have been left
alone. Packaging the list makes healing behave identically everywhere; the
system dictionary is now only a fallback for an install whose package data
went missing. The list is extended two ways, because on its own it is a poor judge of running prose: it lists `flank`
but not `flanks`, so plurals and past tenses are checked against their stems,
and it knows nothing of invented or proper nouns, so the words already used in
the document count as words too. That second source is what lets a coinage like
`Xenosurgery` be repaired once the document has used it.

### Exact joins with `rewind`

Repair is inference, and inference has a ceiling: `for` + `ever` is ambiguous
from the text alone. `rewind=True` removes the guesswork instead of improving
it. The last short token is held back rather than sent — the model continues
from `counted, ` — and if it comes back with `and`, the `a` already on the page
is stripped from the output and the join is exact, whatever the word was.

When the model writes something else, its text was composed to follow a prefix
the reader never sees, and pasting it on would read as a non-sequitur rather
than a spacing glitch. The request is therefore reissued with the full prefix
before anything has been shown, and the heuristics above take over. The cost is
a second request whenever the model declines the rewind, which is why it is
off by default.

This is why `prefix + ''.join(tokens)` remains readable and stable across providers.

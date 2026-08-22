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

Stream output is post-processed to avoid common boundary artifacts:

- Missing space between prefix and first token
- Split words at the prefix boundary, where the injected space lands inside a
  word the model was finishing — `counted, a` + `nd` becoming `a nd` instead of
  `and`. The join fires when the prefix tail plus the fragment is a dictionary
  word and the fragment alone is not, so a prefix ending in a standalone letter
  (`Section B` + ` undefined`) is left alone. It is applied only at the
  boundary itself, never to later chunks of the same stream.
- Split compounds like `any one` where the model intended `anyone`
- Newline artifacts that break prose flow

This is why `prefix + ''.join(tokens)` remains readable and stable across providers.

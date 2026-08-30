# basemode

**basemode** makes chat-tuned LLMs behave like continuation engines.

Most modern models want to answer prompts. `basemode` does the opposite: it coerces models to continue raw text naturally, with no assistant preamble.

## The problem

Hand a chat-tuned model a fragment of prose and it replies to you instead of
continuing it — "I'd be happy to help! Here's one possibility:", the passage in
quotation marks, an offer to try another style. Base models continue text
natively; instruction tuning trains it out.

`basemode` puts it back, by wrapping your prefix in whatever prompt shape a
given model will actually continue: a prefilled assistant turn, a system
instruction, a few-shot frame, an FIM template. No single trick works
everywhere — Anthropic models took a prefill until they didn't, and some
providers reject the parameter outright — so the choice is per-model data
rather than a guess. See [[How It Works]].

## What it does

- Auto-selects a continuation strategy per model (`completion`, `prefill`, `system`, `few_shot`, `fim`), preferring the method verified for that model over a guess
- Streams text token-by-token from CLI or Python
- Supports parallel branching (`-n/--branches`)
- Normalizes model names across providers (`claude-*`, `gemini-*`, etc.)
- Includes usage and cost estimates using LiteLLM metadata
- Scores how cleanly a model continues text, so strategy choice can be measured (`basemode bench`)

## Interfaces

| Interface | Use case |
|-----------|----------|
| [[CLI Reference]] | Terminal usage, streaming output, branch generation |
| [[Python API]] | Integration into applications and scripts |
| [[Keys and Defaults]] | Manage API keys and preferred model |

## Quick example

```bash
basemode "The ship rounded the headland and"
```

```bash
# Parallel branches
basemode "The ship rounded the headland and" -n 4

# Inspect strategy + pricing metadata
basemode info claude-sonnet-4-6

# Rank coercion strategies for a model, then pin the winner
basemode bench claude-sonnet-4-6 --save
```

See [[Quickstart]] for a 5-minute walkthrough.

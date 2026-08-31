# basemode

[![PyPI version](https://img.shields.io/pypi/v/basemode.svg)](https://pypi.org/project/basemode/)

**basemode** makes chat-tuned LLMs continue text instead of answering it.

It provides streaming continuation through a CLI and async Python API, plus an
OpenAI-compatible completions server.

## The problem

Chat models often introduce, quote, or discuss a requested continuation.
`basemode` instead wraps the prefix in a model-appropriate prompt: native
completion, assistant prefill, a system instruction, few-shot examples, or an
FIM template. Verified models carry a recorded strategy and compatibility
quirks. Unregistered models use conservative name-based defaults. See
[[How It Works]].

## What it does

- Selects a continuation strategy per model (`completion`, `prefill`, `system`, `few_shot`, `fim`)
- Streams text token-by-token from CLI or Python
- Supports parallel branching (`-n/--branches`)
- Normalizes common model aliases and provider prefixes
- Includes usage and cost estimates using LiteLLM metadata
- Measures and stores strategy compatibility with live probes

## Interfaces

| Interface | Use case |
|-----------|----------|
| [[CLI Reference]] | Terminal usage, streaming output, branch generation |
| [[Python API]] | Integration into applications and scripts |
| [[Keys and Defaults]] | Manage API keys and preferred model |
| [[Model Evidence]] | Verification, compatibility evidence, and model status |

## Quick example

```bash
basemode "The ship rounded the headland and"
```

```bash
# Parallel branches
basemode "The ship rounded the headland and" -n 4

# Inspect strategy and pricing metadata
basemode info claude-sonnet-4-6

# Compare strategies and pin the winner
basemode bench claude-sonnet-4-6 --save
```

See [[Quickstart]] for a 5-minute walkthrough.

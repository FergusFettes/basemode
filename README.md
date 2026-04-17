# basemode

**Make any LLM do raw text continuation.**

Most language models today are chat-tuned — they want to respond, acknowledge, and help. This is the opposite of what you need for loom-style tree exploration, creative writing, or any workflow that requires the model to simply *continue* text as if it wrote it. Feed a chat model a sentence mid-thought and you'll get "Sure! Here's a continuation:" instead of the next word.

`basemode` solves this. It wraps any LLM in the right coercion strategy for that provider — native completions API, assistant prefill, or a carefully-tuned system prompt — so you always get a clean continuation back.

```bash
basemode "The defendant, who had been sitting quietly throughout the proceedings, suddenly"
```

```
The defendant, who had been sitting quietly throughout the proceedings, suddenly
rose to his feet, overturning his chair. "You have no idea what actually happened
that night," he said, his voice barely above a whisper.
```

---

## Install

```bash
pip install basemode
```

Requires Python 3.11+. Uses [LiteLLM](https://docs.litellm.ai/) under the hood, so any model LiteLLM supports is available.

Set API keys as environment variables or in a `.env` file:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=...
TOGETHER_API_KEY=...
```

---

## CLI

```bash
# Stream a single continuation (defaults to gpt-4o-mini)
basemode "The ship rounded the headland and"

# Choose a model
basemode "The ship rounded" --model anthropic/claude-3-haiku-20240307

# Generate 4 parallel branches, shown side by side
basemode "The ship rounded" --branches 4

# Pipe text in
cat chapter1.txt | basemode --model groq/llama-3.3-70b-versatile

# Show which coercion strategy is being used
basemode "text" --show-strategy

# Override the auto-detected strategy
basemode "text" --strategy prefill

# Explore available models
basemode models --available          # only models with keys configured
basemode models --provider openai
basemode models --search claude

# Check strategy detection for any model
basemode info mistral-large-latest
```

---

## Python API

```python
from basemode import continue_text, branch_text

# Stream a single continuation
async for token in continue_text(
    "She opened the letter with trembling hands.",
    model="gpt-4o-mini",
    max_tokens=200,
    temperature=0.9,
):
    print(token, end="", flush=True)

# Generate n parallel branches as (branch_idx, token) tuples
async for idx, token in branch_text(
    "She opened the letter",
    model="anthropic/claude-3-haiku-20240307",
    n=4,
    max_tokens=200,
):
    print(f"[{idx}] {token}", end="", flush=True)
```

Token boundaries are handled correctly: `prefix + "".join(tokens)` always produces clean, properly-spaced text regardless of which strategy was used.

---

## How it works

Chat-tuned models need different tricks depending on the provider. `basemode` auto-detects the best strategy from the model name:

| Strategy | When used | How |
|----------|-----------|-----|
| `completion` | OpenAI base models (`davinci-002`, `gpt-3.5-turbo-instruct`) | Native `/completions` endpoint — no coercion needed |
| `prefill` | Anthropic (`claude-*`) | Splits the prefix: last 50 chars become the start of the assistant turn, forcing the model to continue from exactly that point |
| `system` | Everything else | System prompt instructs the model to output only continuation, plus trailing-space normalization to prevent word smashing |
| `few_shot` | Stubborn models | Four varied examples (fiction, technical, poetry, dialogue) in the system prompt |
| `fim` | DeepSeek Coder, StarCoder, CodeLlama | Fill-in-the-middle special tokens |

Override auto-detection with `--strategy` / `strategy=` parameter.

---

## Model compatibility

Tested results from the integration suite. Reliability is assessed on two axes: **does it produce output** and **is the output a clean continuation** (no "Sure!", no preamble, correct word boundaries).

### OpenAI

| Model | Strategy | Reliability | Notes |
|-------|----------|-------------|-------|
| `gpt-4o` | `system` | ⭐⭐⭐⭐⭐ | Excellent continuation quality, zero preamble |
| `gpt-4o-mini` | `system` | ⭐⭐⭐⭐⭐ | Fast, cheap, reliable. Recommended default |
| `gpt-3.5-turbo-instruct` | `completion` | ⭐⭐⭐⭐⭐ | Native completions endpoint, no coercion needed |
| `davinci-002` | `completion` | ⭐⭐⭐⭐⭐ | True base model, best raw continuation behavior |

### Anthropic

| Model | Strategy | Reliability | Notes |
|-------|----------|-------------|-------|
| `anthropic/claude-3-opus-20240229` | `prefill` | ⭐⭐⭐⭐⭐ | Superb quality, stays in voice perfectly |
| `anthropic/claude-3-5-sonnet-20241022` | `prefill` | ⭐⭐⭐⭐⭐ | Best balance of quality and speed |
| `anthropic/claude-3-haiku-20240307` | `prefill` | ⭐⭐⭐⭐⭐ | Fast and reliable, tested extensively here |
| `anthropic/claude-3-5-haiku-20241022` | `prefill` | ⭐⭐⭐⭐ | Use dated model ID; `claude-3-5-haiku-latest` alias not supported by all API tiers |

> **The prefill trick**: Anthropic's API lets you pre-fill the assistant's response. `basemode` puts the last 50 characters of your prefix into the assistant turn, so Claude is literally mid-sentence before it generates a single new token. This produces the cleanest continuation of any strategy tested.

### Groq

| Model | Strategy | Reliability | Notes |
|-------|----------|-------------|-------|
| `groq/llama-3.3-70b-versatile` | `system` | ⭐⭐⭐⭐⭐ | Very fast inference, high-quality output, no preamble |
| `groq/llama-3.1-70b-versatile` | `system` | ⭐⭐⭐⭐ | Slightly older, still reliable |
| `groq/mixtral-8x7b-32768` | `system` | ⭐⭐⭐⭐ | Good for longer context continuations |

Groq's speed (often under 1s for short continuations) makes it excellent for interactive loom exploration.

### Google Gemini

| Model | Strategy | Reliability | Notes |
|-------|----------|-------------|-------|
| `gemini/gemini-2.5-flash` | `system` | ⭐⭐⭐⭐ | **Thinking model** — `basemode` automatically allocates a thinking token budget. Do not use `max_tokens < 1536` |
| `gemini/gemini-2.5-pro` | `system` | ⭐⭐⭐⭐ | Same thinking model caveat |
| `gemini/gemini-flash-latest` | `system` | ⭐⭐⭐ | Non-thinking, faster, but output sometimes truncates at low token counts |

> **Gemini 2.5 thinking models**: These models spend tokens on internal reasoning before producing visible output. `basemode` detects Gemini 2.5 models and automatically sets a `thinking` budget, ensuring the visible output isn't starved. If you're passing `max_tokens` directly, use at least `1536`.

### Together AI

| Model | Strategy | Reliability | Notes |
|-------|----------|-------------|-------|
| `together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo` | `system` | ⭐⭐⭐⭐⭐ | Excellent quality and speed, well-behaved with system prompt |
| `together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo` | `system` | ⭐⭐⭐⭐⭐ | Best open-weight quality available |
| `together_ai/mistralai/Mixtral-8x22B-Instruct-v0.1` | `system` | ⭐⭐⭐⭐ | Strong multilingual continuation |

### OpenRouter

OpenRouter routes to many providers. Any model available there works via `openrouter/<provider>/<model>`:

```bash
basemode "text" --model openrouter/openai/gpt-4o
basemode "text" --model openrouter/anthropic/claude-3-5-sonnet
basemode "text" --model openrouter/mistralai/mistral-large
```

| Reliability | Notes |
|-------------|-------|
| ⭐⭐⭐⭐ | Model availability varies; use `basemode models --provider openrouter` to check |

OpenRouter is particularly useful for accessing base model variants (e.g. `openrouter/mistralai/mistral-7b`) which don't require any coercion.

---

## Reliability ratings

| ⭐⭐⭐⭐⭐ | Zero preamble, correct spacing, stays in voice |
|-----------|---|
| ⭐⭐⭐⭐ | Occasionally needs strategy override or specific model ID |
| ⭐⭐⭐ | Works but has edge cases or quirks |
| ⭐⭐ | Usable with manual `--strategy` override |
| ⭐ | Unreliable, avoid |

---

## Development

```bash
git clone https://github.com/FergusFettes/basemode
cd basemode
uv sync
cp .env.example .env  # add your API keys

# Unit tests (no API calls)
uv run pytest

# Integration tests (hits real APIs, costs ~$0.01)
uv run pytest -m integration
```

Pre-commit hooks (ruff format + lint):

```bash
uv run pre-commit install
```

---

## Strategies reference

```bash
basemode strategies  # list all strategies
basemode info <model>  # show detected strategy for any model
```

To force a specific strategy:

```bash
basemode "text" --strategy prefill   # Anthropic prefill trick
basemode "text" --strategy system    # system prompt coercion
basemode "text" --strategy few_shot  # few-shot examples
basemode "text" --strategy completion # OpenAI /completions endpoint
basemode "text" --strategy fim       # fill-in-the-middle
```

---

## Why this exists

This is the model layer for [loom](https://github.com/FergusFettes/loom) — a multiverse writing interface for human-AI collaboration. Every loom node needs raw continuation. Getting that reliably out of modern chat models is surprisingly hard, and the solution is different for every provider. `basemode` packages that knowledge into a single interface so loom (and anything else that needs it) doesn't have to think about it.

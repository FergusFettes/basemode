# basemode

**Make any LLM do raw text continuation.**

Most language models today are chat-tuned — they want to respond, acknowledge, and help. This is the opposite of what you need when you want a model to simply *continue* text as if it wrote it. Feed a chat model a sentence mid-thought and you'll get "Sure! Here's a continuation:" instead of the next word.

`basemode` solves this. It wraps any LLM in the right coercion strategy for that provider — native completions API, assistant prefill, or a carefully-tuned system prompt — so you always get a clean continuation back.

For persistent branching exploration, use `basemode loom`. The core `basemode` command stays stateless and does not manage sessions.

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

# Persistent branching exploration in loom mode
basemode loom "The ship rounded" --branches 4
basemode loom continue -b 2
basemode loom nodes
basemode loom active
basemode loom show <node-id>
basemode loom children <node-id>
basemode loom select <node-id>
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

## Persistence

`basemode loom` can persist generation trees in SQLite. The schema is node-based:

- root nodes contain the starting text
- child nodes contain only the generated continuation segment
- branches are siblings with the same parent
- continuing from a branch adds new child nodes under that branch
- full text is reconstructed by concatenating a node's ancestors

`loom` is opt-in and does not change `basemode` itself.

```bash
basemode loom "The ship rounded" --branches 4
basemode loom continue -b 2
basemode loom nodes
basemode loom active
basemode loom show <child-node-id>
basemode loom select <node-id>
```

Node ids can be abbreviated to any unique substring. If the substring matches multiple nodes, `basemode loom` will ask for a more specific id instead of guessing.

`loom nodes` marks the active node with `*`, and `loom active` shows the current cursor directly.

By default the database lives at `~/.local/share/basemode/generations.sqlite`. Override it with `--db /path/to/generations.sqlite` or the `BASEMODE_DB` environment variable.

When a loom tree reaches roughly 500 tokens, `basemode loom` tries to name it with a short slug like `this-is-the-topic`. Naming is best-effort and only runs when an OpenAI or Anthropic key is configured; generated text is still saved normally if naming is unavailable.

If you are embedding loom, you can use the storage API directly:

```python
from basemode import GenerationStore

store = GenerationStore()
parent, children = store.save_continuations(
    "The ship rounded",
    [" the headland", " into fog"],
    model="gpt-4o-mini",
    strategy="system",
    max_tokens=200,
    temperature=0.9,
)

text = store.full_text(children[0].id)
```

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
| `gpt-5.4` | `system` | ⭐⭐⭐⭐⭐ | Frontier quality, exceptional prose continuation |
| `gpt-5.4-mini` | `system` | ⭐⭐⭐⭐⭐ | Fast, cheap, excellent. **Recommended default** |
| `gpt-5.4-nano` | `system` | ⭐⭐⭐⭐ | Ultra-cheap, good for high-volume branching |
| `gpt-4o-mini` | `system` | ⭐⭐⭐⭐⭐ | Solid legacy option |
| `gpt-3.5-turbo-instruct` | `completion` | ⭐⭐⭐⭐⭐ | Native completions endpoint, no coercion needed |
| `davinci-002` | `completion` | ⭐⭐⭐⭐⭐ | True base model |

### Anthropic

| Model | Strategy | Reliability | Notes |
|-------|----------|-------------|-------|
| `anthropic/claude-opus-4-7` | `system` | ⭐⭐⭐⭐⭐ | Best available. **Prefill and temperature deprecated** — auto-routed to system strategy |
| `anthropic/claude-sonnet-4-6` | `system` | ⭐⭐⭐⭐⭐ | Prefill deprecated (temperature still supported) — auto-routed to system |
| `anthropic/claude-opus-4-6` | `system` | ⭐⭐⭐⭐⭐ | Prefill deprecated (temperature still supported) — auto-routed to system |
| `anthropic/claude-opus-4-5-20251101` | `prefill` | ⭐⭐⭐⭐⭐ | Last opus that accepts prefill |
| `anthropic/claude-sonnet-4-5-20250929` | `prefill` | ⭐⭐⭐⭐⭐ | |
| `anthropic/claude-haiku-4-5-20251001` | `prefill` | ⭐⭐⭐⭐⭐ | Fast, very clean continuation |
| `anthropic/claude-opus-4-1-20250805` | `prefill` | ⭐⭐⭐⭐⭐ | |
| `anthropic/claude-opus-4-20250514` | `prefill` | ⭐⭐⭐⭐⭐ | Original Opus 4 |
| `anthropic/claude-sonnet-4-20250514` | `prefill` | ⭐⭐⭐⭐⭐ | Original Sonnet 4 |
| `anthropic/claude-3-haiku-20240307` | `prefill` | ⭐⭐⭐⭐⭐ | Legacy, rock solid |

> **The prefill trick**: The full prefix goes in the system prompt for context, then the last 20 characters seed the assistant turn. Claude is mid-sentence before generating a single new token — the cleanest strategy for models that support it. Anthropic has been phasing prefill out on newer models — opus 4.7, sonnet 4.6, and opus 4.6 reject it; `basemode` auto-detects and falls back to the system strategy. The 4.5 and 4.0/4.1 families still accept it.

### Groq

| Model | Strategy | Reliability | Notes |
|-------|----------|-------------|-------|
| `groq/llama-3.3-70b-versatile` | `system` | ⭐⭐⭐⭐⭐ | Sub-second inference, excellent quality |
| `groq/meta-llama/llama-4-scout-17b-16e-instruct` | `system` | ⭐⭐⭐⭐ | Llama 4 Scout, very fast |

### Google Gemini

| Model | Strategy | Reliability | Notes |
|-------|----------|-------------|-------|
| `gemini/gemini-2.5-flash` | `system` | ⭐⭐⭐⭐⭐ | **Thinking model** — `basemode` auto-allocates thinking budget |
| `gemini/gemini-2.5-pro` | `system` | ⭐⭐⭐⭐⭐ | Same |
| `gemini/gemini-3-flash-preview` | `system` | ⭐⭐⭐⭐ | Latest flash, very fast |

> **Gemini thinking models**: Spend tokens on internal reasoning before producing visible output. `basemode` auto-detects Gemini 2.5 models and allocates a thinking budget, so `max_tokens` isn't exhausted before visible output begins.

### Together AI

| Model | Strategy | Reliability | Notes |
|-------|----------|-------------|-------|
| `together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo` | `system` | ⭐⭐⭐⭐⭐ | Reliable and fast |
| `together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo` | `system` | ⭐⭐⭐⭐⭐ | Highest open-weight quality |

### OpenRouter

OpenRouter routes to many providers. Any model available there works via `openrouter/<provider>/<model>`:

```bash
basemode "text" --model openrouter/moonshotai/kimi-k2.5
basemode "text" --model openrouter/deepseek/deepseek-v3.2
basemode "text" --model openrouter/meta-llama/llama-4-maverick
basemode "text" --model openrouter/qwen/qwen3-235b-a22b
```

| Model | Reliability | Notes |
|-------|-------------|-------|
| `openrouter/moonshotai/kimi-k2.5` | ⭐⭐⭐⭐⭐ | **Thinking model** — spectacular prose quality, auto thinking budget |
| `openrouter/moonshotai/kimi-k2` | ⭐⭐⭐⭐⭐ | Non-thinking variant, faster |
| `openrouter/deepseek/deepseek-v3.2` | ⭐⭐⭐⭐⭐ | Excellent continuation, very cost-effective |
| `openrouter/meta-llama/llama-4-maverick` | ⭐⭐⭐⭐⭐ | Llama 4 Maverick, strong quality |
| `openrouter/qwen/qwen3-235b-a22b` | ⭐⭐⭐⭐ | Qwen 3 flagship |
| `openrouter/z-ai/glm-5-turbo` | ⭐⭐ | GLM-5 Turbo — provider errors intermittently |

> OpenRouter is also useful for accessing base model variants (e.g. `openrouter/mistralai/mistral-7b`) which don't require any coercion.

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

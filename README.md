# basemode

Make any LLM do raw text continuation.

`basemode` coerces chat-tuned models into clean next-token continuation mode (instead of assistant-style replies), with strategy selection handled per model/provider.

## Install

```bash
pip install basemode
```

Set provider keys via environment variables or `.env` (for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`).

## Quickstart

```bash
# Single continuation (default model if configured, else fallback)
basemode "The ship rounded the headland and"

# Parallel continuations
basemode "The ship rounded the headland and" -n 3

# Inspect selected strategy and pricing metadata
basemode info claude-sonnet-4-6

# Show only key-configured models
basemode models --available
```

## CLI

```bash
basemode --help
basemode run --help
basemode models --help
basemode info --help
basemode strategies --help
```

Useful commands:

- `basemode run` (default): stream continuation text
- `basemode models`: list models (supports `--verified` and `--json` for picker UIs)
- `basemode providers`: list provider IDs
- `basemode info`: show normalized model + prompt strategy + pricing metadata
- `basemode default`: get/set your default model
- `basemode keys`: manage stored API keys

## Python API

```python
from basemode import continue_text, branch_text

async for token in continue_text(
    "The ship rounded the headland and",
    model="gpt-4o-mini",
    max_tokens=120,
):
    print(token, end="", flush=True)

async for idx, token in branch_text(
    "The ship rounded the headland and",
    model="gpt-4o-mini",
    n=3,
    max_tokens=80,
):
    print(idx, token, end="", flush=True)
```

## Docs

Full docs are in `docs/` and can be served with MkDocs:

```bash
make docs-serve
```

Then open `http://localhost:8001`.

## Integration Health Checks

Run live provider checks (real APIs, key-aware skips):

```bash
uv run pytest -m integration tests/test_integration.py -q
```

This writes a machine-readable report to `dist/integration/provider_health.json` with per-model status, latency, token estimates, and estimated USD cost.

<!-- verified-models:start -->

## Verified Models

Single generated table, refreshed by CI.

| Model | Input cost (/1M) | Output cost (/1M) | Release date | Prompt method | Reliability |
|---|---:|---:|---|---|---|
| `anthropic/claude-haiku-4-5-20251001` | $1.00 | $5.00 | 2025-10-01 | `prefill` | ✓ |
| `anthropic/claude-opus-4-1-20250805` | $15.00 | $75.00 | 2025-08-05 | `prefill` | ✓ |
| `anthropic/claude-opus-4-20250514` | $15.00 | $75.00 | 2025-05-22 | `prefill` | ✓ |
| `anthropic/claude-opus-4-5-20251101` | $5.00 | $25.00 | 2025-11-24 | `prefill` | ✓ |
| `anthropic/claude-opus-4-6` | $5.00 | $25.00 | 2026-02-05 | `system` | ✓ |
| `anthropic/claude-opus-4-7` | $5.00 | $25.00 | 2026-04-16 | `system` | ✓ |
| `anthropic/claude-sonnet-4-20250514` | $3.00 | $15.00 | 2025-05-22 | `prefill` | ✓ |
| `anthropic/claude-sonnet-4-5-20250929` | $3.00 | $15.00 | 2025-09-29 | `prefill` | ✓ |
| `anthropic/claude-sonnet-4-6` | $3.00 | $15.00 | 2026-02-17 | `system` | ✓ |
| `gemini/gemini-2.5-flash` | $0.30 | $2.50 | 2025-06-17 | `system` | ⚠ |
| `gemini/gemini-2.5-pro` | $1.25 | $10.00 | 2025-06-17 | `system` | ⚠ |
| `gemini/gemma-4-26b-a4b-it` | $0.07 | $0.35 | 2026-04-03 | `system` | ⚠ |
| `gemini/gemma-4-31b-it` | $0.13 | $0.38 | 2026-04-02 | `system` | ⚠ |
| `moonshot/kimi-k2-0905-preview` | $0.60 | $2.50 | 2025-07-11 | `system` | ⚠ |
| `moonshot/kimi-k2.5` | $0.60 | $3.00 | 2026-01-27 | `system` | ⚠ |
| `openai/gpt-4o-mini` | $0.15 | $0.60 | 2024-07-18 | `system` | ⚠ |
| `openai/gpt-5.4-mini` | $0.75 | $4.50 | 2026-03-17 | `system` | ✓ |
| `openrouter/moonshotai/kimi-k2.6` | $0.60 | $2.80 | 2026-04-20 | `system` | ⚠ |
| `zai/glm-4.7` | $0.60 | $2.20 | 2025-12-22 | `system` | ✓ |
| `zai/glm-5` | $1.00 | $3.20 | 2026-02-11 | `system` | ✓ |

Legend: `✓` = LiteLLM pricing present and release date available; `⚠` = missing/approximate field or known issue.

<!-- verified-models:end -->

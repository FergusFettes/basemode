# Verified Models

Single generated table, refreshed by CI.

| Model | Input cost (/1M) | Output cost (/1M) | Release date | Prompt method | Reliability | Quirks |
|---|---:|---:|---|---|---|---|
| `anthropic/claude-fable-5` | $10.00 | $50.00 | 2026-06-09 | `system` | ✓ | `no_prefill`, `no_temperature` |
| `anthropic/claude-haiku-4-5-20251001` | $1.00 | $5.00 | 2025-10-01 | `prefill` | ✓ |  |
| `anthropic/claude-opus-4-1-20250805` | $15.00 | $75.00 | 2025-08-05 | `prefill` | ✓ |  |
| `anthropic/claude-opus-4-20250514` | $15.00 | $75.00 | 2025-05-22 | `prefill` | ✓ |  |
| `anthropic/claude-opus-4-5-20251101` | $5.00 | $25.00 | 2025-11-24 | `prefill` | ✓ |  |
| `anthropic/claude-opus-4-6` | $5.00 | $25.00 | 2026-02-05 | `system` | ✓ | `no_prefill` |
| `anthropic/claude-opus-4-7` | $5.00 | $25.00 | 2026-04-16 | `system` | ✓ | `no_prefill`, `no_temperature` |
| `anthropic/claude-opus-4-8` | $5.00 | $25.00 | 2026-05-28 | `system` | ✓ | `no_temperature` |
| `anthropic/claude-opus-5` | $5.00 | $25.00 | 2026-07-23 | `few_shot` | ✓ | `no_prefill`, `no_temperature` |
| `anthropic/claude-sonnet-4-20250514` | $3.00 | $15.00 | 2025-05-22 | `prefill` | ✓ |  |
| `anthropic/claude-sonnet-4-5-20250929` | $3.00 | $15.00 | 2025-09-29 | `prefill` | ✓ |  |
| `anthropic/claude-sonnet-4-6` | $3.00 | $15.00 | 2026-02-17 | `system` | ✓ | `no_prefill` |
| `anthropic/claude-sonnet-5` | $2.00 | $10.00 | 2026-06-30 | `system` | ✓ | `no_prefill`, `no_temperature` |
| `deepseek/deepseek-v4-flash` | $0.44 | $1.32 | unknown | `system` | ⚠ | `reasoning_budget` |
| `deepseek/deepseek-v4-flash-vision-exp` | unknown | unknown | unknown | `system` | ⚠ | `reasoning_budget` |
| `deepseek/deepseek-v4-pro` | $1.32 | $3.96 | unknown | `system` | ⚠ | `reasoning_budget` |
| `gemini/gemini-2.5-flash` | $0.30 | $2.50 | 2025-06-17 | `system` | ⚠ |  |
| `gemini/gemini-2.5-flash-lite` | $0.10 | $0.40 | 2025-07-22 | `system` | ⚠ |  |
| `gemini/gemini-2.5-pro` | $1.25 | $10.00 | 2025-06-17 | `system` | ⚠ |  |
| `gemini/gemini-3-flash-preview` | $0.50 | $3.00 | 2025-12-17 | `system` | ✓ |  |
| `gemini/gemini-3-pro-image` | $2.00 | $12.00 | 2026-05-28 | `prefill` | ✓ |  |
| `gemini/gemini-3-pro-image-preview` | $2.00 | $12.00 | 2026-05-28 | `prefill` | ✓ |  |
| `gemini/gemini-3.1-flash-image` | $0.50 | $3.00 | 2026-05-28 | `system` | ✓ |  |
| `gemini/gemini-3.1-flash-image-preview` | $0.50 | $3.00 | 2026-05-28 | `system` | ✓ |  |
| `gemini/gemini-3.1-flash-lite` | $0.25 | $1.50 | 2026-06-30 | `system` | ✓ |  |
| `gemini/gemini-3.1-flash-lite-image` | $0.25 | $1.50 | 2026-06-30 | `system` | ✓ |  |
| `gemini/gemini-3.1-flash-lite-preview` | $0.25 | $1.50 | 2026-05-07 | `system` | ✓ |  |
| `gemini/gemini-3.1-pro-preview` | $2.00 | $12.00 | 2026-02-19 | `system` | ✓ |  |
| `gemini/gemini-3.1-pro-preview-customtools` | $2.00 | $12.00 | 2026-02-19 | `prefill` | ✓ |  |
| `gemini/gemini-3.5-flash` | $1.50 | $9.00 | 2026-07-21 | `system` | ✓ |  |
| `gemini/gemini-3.5-flash-lite` | $0.30 | $2.50 | 2026-07-21 | `system` | ✓ |  |
| `gemini/gemini-3.6-flash` | $0.75 | $3.75 | 2026-07-21 | `system` | ✓ |  |
| `gemini/gemini-flash-lite-latest` | $0.10 | $0.40 | unknown | `system` | ⚠ |  |
| `gemini/gemini-pro-latest` | $1.25 | $10.00 | 2026-04-27 | `system` | ⚠ |  |
| `gemini/gemini-robotics-er-1.6-preview` | $1.00 | $5.00 | unknown | `prefill` | ⚠ |  |
| `gemini/gemini-robotics-er-2-preview` | $2.00 | $10.00 | unknown | `system` | ⚠ |  |
| `gemini/gemma-4-26b-a4b-it` | $0.07 | $0.34 | 2026-04-03 | `system` | ⚠ | `reasoning_budget` |
| `gemini/gemma-4-31b-it` | $0.10 | $0.34 | 2026-04-02 | `system` | ⚠ | `reasoning_budget` |
| `gemini/nano-banana-pro-preview` | unknown | unknown | unknown | `prefill` | ⚠ |  |
| `groq/allam-2-7b` | unknown | unknown | unknown | `system` | ⚠ |  |
| `groq/groq/compound` | unknown | unknown | unknown | `system` | ⚠ |  |
| `groq/groq/compound-mini` | unknown | unknown | unknown | `system` | ⚠ |  |
| `groq/qwen/qwen3.6-27b` | $0.60 | $3.00 | 2026-04-22 | `system` | ✓ |  |
| `moonshot/kimi-k2-0905-preview` | $0.60 | $2.50 | 2025-07-11 | `system` | ⚠ |  |
| `moonshot/kimi-k2.5` | $0.60 | $3.00 | 2026-01-27 | `system` | ⚠ | `no_temperature` |
| `moonshot/kimi-k2.6` | $0.95 | $4.00 | 2026-04-20 | `system` | ✓ | `no_temperature` |
| `moonshot/kimi-k2.7-code` | unknown | unknown | unknown | `system` | ⚠ | `no_temperature`, `reasoning_budget` |
| `moonshot/kimi-k2.7-code-highspeed` | unknown | unknown | unknown | `system` | ⚠ | `no_temperature`, `reasoning_budget` |
| `moonshot/kimi-k3` | $3.00 | $15.00 | 2026-07-15 | `prefill` | ✓ | `no_temperature` |
| `moonshot/moonshot-v1-128k` | $2.00 | $5.00 | unknown | `system` | ⚠ |  |
| `moonshot/moonshot-v1-128k-vision-preview` | $2.00 | $5.00 | unknown | `system` | ⚠ |  |
| `moonshot/moonshot-v1-32k` | $1.00 | $3.00 | unknown | `system` | ⚠ |  |
| `moonshot/moonshot-v1-32k-vision-preview` | $1.00 | $3.00 | unknown | `system` | ⚠ |  |
| `moonshot/moonshot-v1-8k` | $0.20 | $2.00 | unknown | `system` | ⚠ |  |
| `moonshot/moonshot-v1-8k-vision-preview` | $0.20 | $2.00 | unknown | `system` | ⚠ |  |
| `moonshot/moonshot-v1-auto` | $2.00 | $5.00 | 2023-11-08 | `system` | ⚠ |  |
| `openai/gpt-4-turbo-2024-04-09` | $10.00 | $30.00 | 2024-04-09 | `system` | ⚠ |  |
| `openai/gpt-4.1` | $2.00 | $8.00 | 2025-04-14 | `system` | ✓ |  |
| `openai/gpt-4.1-2025-04-14` | $2.00 | $8.00 | 2025-04-14 | `system` | ✓ |  |
| `openai/gpt-4.1-mini` | $0.40 | $1.60 | 2025-04-14 | `system` | ✓ |  |
| `openai/gpt-4.1-mini-2025-04-14` | $0.40 | $1.60 | 2025-04-14 | `system` | ✓ |  |
| `openai/gpt-4.1-nano` | $0.10 | $0.40 | 2025-04-14 | `system` | ✓ |  |
| `openai/gpt-4.1-nano-2025-04-14` | $0.10 | $0.40 | 2025-04-14 | `system` | ✓ |  |
| `openai/gpt-4o` | $2.50 | $10.00 | 2024-11-20 | `system` | ✓ |  |
| `openai/gpt-4o-2024-05-13` | $5.00 | $15.00 | 2024-05-13 | `system` | ⚠ |  |
| `openai/gpt-4o-2024-08-06` | $2.50 | $10.00 | 2024-08-06 | `system` | ✓ |  |
| `openai/gpt-4o-2024-11-20` | $2.50 | $10.00 | 2024-11-20 | `system` | ✓ |  |
| `openai/gpt-4o-mini` | $0.15 | $0.60 | 2024-07-18 | `system` | ⚠ |  |
| `openai/gpt-4o-mini-2024-07-18` | $0.15 | $0.60 | 2024-07-18 | `system` | ⚠ |  |
| `openai/gpt-5.1` | $1.25 | $10.00 | 2025-12-04 | `system` | ✓ |  |
| `openai/gpt-5.1-2025-11-13` | $1.25 | $10.00 | 2025-11-13 | `system` | ✓ |  |
| `openai/gpt-5.2` | $1.75 | $14.00 | 2026-01-14 | `system` | ✓ |  |
| `openai/gpt-5.2-2025-12-11` | $1.75 | $14.00 | 2025-12-11 | `system` | ✓ |  |
| `openai/gpt-5.2-pro` | $21.00 | $168.00 | 2025-12-11 | `system` | ✓ | `no_temperature` |
| `openai/gpt-5.2-pro-2025-12-11` | $21.00 | $168.00 | 2025-12-11 | `system` | ✓ | `no_temperature` |
| `openai/gpt-5.3-codex` | $1.75 | $14.00 | 2026-02-24 | `system` | ✓ | `no_temperature` |
| `openai/gpt-5.4` | $2.50 | $15.00 | 2026-04-21 | `system` | ✓ |  |
| `openai/gpt-5.4-2026-03-05` | $2.50 | $15.00 | 2026-03-05 | `system` | ✓ |  |
| `openai/gpt-5.4-mini` | $0.75 | $4.50 | 2026-03-17 | `system` | ✓ |  |
| `openai/gpt-5.4-mini-2026-03-17` | $0.75 | $4.50 | 2026-03-17 | `system` | ✓ |  |
| `openai/gpt-5.4-nano` | $0.20 | $1.25 | 2026-03-17 | `system` | ✓ |  |
| `openai/gpt-5.4-nano-2026-03-17` | $0.20 | $1.25 | 2026-03-17 | `system` | ✓ |  |
| `openai/gpt-5.5` | $5.00 | $30.00 | 2026-04-23 | `system` | ✓ | `no_temperature` |
| `openai/gpt-5.5-2026-04-23` | $5.00 | $30.00 | 2026-04-23 | `system` | ✓ | `no_temperature` |
| `openai/gpt-5.6-sol` | $4.00 | $20.00 | 2026-07-09 | `system` | ✓ | `no_temperature` |
| `openai/gpt-5.6-terra` | $2.00 | $12.00 | 2026-07-09 | `system` | ✓ | `no_temperature` |
| `openrouter/moonshotai/kimi-k2.6` | $0.95 | $4.00 | 2026-04-20 | `system` | ⚠ | `no_temperature` |
| `together_ai/deepseek-ai/DeepSeek-V4-Pro-0813` | unknown | unknown | unknown | `system` | ⚠ | `reasoning_budget` |
| `xai/grok-4.20-0309-non-reasoning` | $1.25 | $2.50 | 2026-03-09 | `system` | ✓ |  |
| `xai/grok-4.20-0309-reasoning` | $1.25 | $2.50 | 2026-03-09 | `system` | ✓ |  |
| `xai/grok-4.3` | $1.25 | $2.50 | 2026-04-30 | `system` | ✓ |  |
| `xai/grok-4.5` | $2.00 | $6.00 | 2026-07-08 | `system` | ✓ |  |
| `xai/grok-4.6` | $2.00 | $6.00 | 2026-08-10 | `system` | ✓ |  |
| `xai/grok-build-0.1` | $1.00 | $2.00 | 2026-05-20 | `system` | ✓ |  |
| `zai/glm-4.5` | $0.60 | $2.20 | 2025-08-11 | `system` | ⚠ |  |
| `zai/glm-4.5-air` | $0.20 | $1.10 | 2025-07-25 | `system` | ⚠ |  |
| `zai/glm-4.6` | $0.60 | $2.20 | 2025-12-08 | `system` | ✓ |  |
| `zai/glm-4.7` | $0.60 | $2.20 | 2025-12-22 | `system` | ✓ |  |
| `zai/glm-5` | $1.00 | $3.20 | 2026-02-11 | `system` | ✓ |  |
| `zai/glm-5-turbo` | $1.20 | $4.00 | 2026-03-15 | `system` | ⚠ |  |
| `zai/glm-5.1` | $1.40 | $4.40 | 2026-04-06 | `system` | ✓ |  |
| `zai/glm-5.2` | $0.97 | $3.04 | 2026-06-16 | `system` | ⚠ |  |

Legend: `✓` = LiteLLM pricing present and release date available; `⚠` = missing/approximate field or known issue.

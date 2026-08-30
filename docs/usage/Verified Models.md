# Verified Models

This table lists every endpoint in the verified-model registry. Its `Prompt method` and `Quirks` columns are runtime configuration; the remaining fields help compare endpoints and identify incomplete metadata. The page is regenerated from the registry by CI.

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
| `cerebras/gemma-4-31b` | unknown | unknown | unknown | `system` | ⚠ |  |
| `cerebras/gpt-oss-120b` | $0.35 | $0.75 | unknown | `system` | ⚠ | `reasoning_budget` |
| `deepinfra/anthropic/claude-fable-5` | $10.00 | $50.00 | unknown | `system` | ⚠ |  |
| `deepinfra/anthropic/claude-haiku-4-5` | $1.00 | $5.00 | unknown | `prefill` | ⚠ |  |
| `deepinfra/anthropic/claude-opus-4-7` | $5.00 | $25.00 | unknown | `system` | ⚠ |  |
| `deepinfra/anthropic/claude-opus-4-8` | $5.00 | $25.00 | unknown | `system` | ⚠ |  |
| `deepinfra/anthropic/claude-opus-5` | $5.00 | $25.00 | unknown | `few_shot` | ⚠ |  |
| `deepinfra/anthropic/claude-sonnet-4-6` | $3.00 | $15.00 | unknown | `system` | ⚠ |  |
| `deepinfra/anthropic/claude-sonnet-5` | $2.00 | $10.00 | unknown | `system` | ⚠ |  |
| `deepinfra/bytedance/seed-1.8` | $0.25 | $2.00 | unknown | `system` | ⚠ |  |
| `deepinfra/bytedance/seed-2.0-code` | $0.50 | $3.00 | unknown | `system` | ⚠ |  |
| `deepinfra/bytedance/seed-2.0-mini` | $0.10 | $0.40 | unknown | `system` | ⚠ |  |
| `deepinfra/bytedance/seed-2.0-pro` | $0.50 | $3.00 | unknown | `system` | ⚠ |  |
| `deepinfra/deepseek-ai/deepseek-r1-0528` | $0.50 | $2.15 | unknown | `system` | ⚠ |  |
| `deepinfra/deepseek-ai/deepseek-v3` | $0.32 | $0.89 | unknown | `system` | ⚠ |  |
| `deepinfra/deepseek-ai/deepseek-v3-0324` | $0.24 | $0.90 | unknown | `system` | ⚠ |  |
| `deepinfra/deepseek-ai/deepseek-v3.1` | $0.25 | $0.95 | unknown | `system` | ⚠ |  |
| `deepinfra/deepseek-ai/deepseek-v3.2` | $0.26 | $0.38 | unknown | `system` | ⚠ |  |
| `deepinfra/deepseek-ai/deepseek-v4-flash` | $0.09 | $0.18 | unknown | `system` | ⚠ |  |
| `deepinfra/deepseek-ai/deepseek-v4-flash-0731` | $0.08 | $0.18 | unknown | `system` | ⚠ |  |
| `deepinfra/deepseek-ai/deepseek-v4-pro` | $1.30 | $2.60 | unknown | `system` | ⚠ |  |
| `deepinfra/deepseek-ai/deepseek-v4-pro-0813` | $1.30 | $2.60 | unknown | `system` | ⚠ |  |
| `deepinfra/google/gemini-2.5-flash` | $0.30 | $2.50 | unknown | `system` | ⚠ |  |
| `deepinfra/google/gemini-2.5-pro` | $1.25 | $10.00 | unknown | `system` | ⚠ |  |
| `deepinfra/google/gemini-3.1-flash-lite` | $0.25 | $1.50 | unknown | `system` | ⚠ |  |
| `deepinfra/google/gemini-3.1-pro` | $2.00 | $12.00 | unknown | `system` | ⚠ |  |
| `deepinfra/google/gemma-3-12b-it` | $0.05 | $0.15 | unknown | `system` | ⚠ |  |
| `deepinfra/google/gemma-3-27b-it` | $0.08 | $0.16 | unknown | `system` | ⚠ |  |
| `deepinfra/google/gemma-3-4b-it` | $0.05 | $0.10 | unknown | `system` | ⚠ |  |
| `deepinfra/google/gemma-4-26b-a4b-it` | $0.07 | $0.34 | unknown | `system` | ⚠ |  |
| `deepinfra/google/gemma-4-31b-it` | $0.13 | $0.38 | unknown | `system` | ⚠ |  |
| `deepinfra/google/gemma-4-31b-it-turbo` | $0.09 | $0.34 | unknown | `system` | ⚠ |  |
| `deepinfra/google/gemma-4-31b-it-ultra` | $0.27 | $0.76 | unknown | `system` | ⚠ |  |
| `deepinfra/google/gemma-4-e4b-it` | $0.02 | $0.10 | unknown | `system` | ⚠ |  |
| `deepinfra/gryphe/mythomax-l2-13b` | $0.40 | $0.40 | unknown | `system` | ⚠ |  |
| `deepinfra/inclusionai/ling-3.0-flash` | $0.06 | $0.18 | unknown | `system` | ⚠ |  |
| `deepinfra/meta-llama/llama-3.3-70b-instruct-turbo` | $0.10 | $0.32 | unknown | `system` | ⚠ |  |
| `deepinfra/meta-llama/llama-4-scout-17b-16e-instruct` | $0.10 | $0.30 | unknown | `system` | ⚠ |  |
| `deepinfra/meta-llama/meta-llama-3.1-70b-instruct-turbo` | $0.40 | $0.40 | unknown | `system` | ⚠ |  |
| `deepinfra/meta-llama/meta-llama-3.1-8b-instruct-turbo` | $0.02 | $0.04 | unknown | `system` | ⚠ |  |
| `deepinfra/microsoft/phi-4` | $0.07 | $0.14 | unknown | `system` | ⚠ |  |
| `deepinfra/minimaxai/minimax-m2.7` | $0.25 | $1.00 | unknown | `system` | ⚠ |  |
| `deepinfra/minimaxai/minimax-m3` | $0.28 | $1.10 | unknown | `system` | ⚠ |  |
| `deepinfra/mistralai/mistral-nemo-instruct-2407` | $0.02 | $0.03 | unknown | `system` | ⚠ |  |
| `deepinfra/mistralai/mistral-small-24b-instruct-2501` | $0.05 | $0.08 | unknown | `system` | ⚠ |  |
| `deepinfra/mistralai/mistral-small-3.2-24b-instruct-2506` | $0.07 | $0.20 | unknown | `system` | ⚠ |  |
| `deepinfra/moonshotai/kimi-k2.5` | $0.45 | $2.25 | unknown | `system` | ⚠ |  |
| `deepinfra/moonshotai/kimi-k2.6` | $0.75 | $3.50 | unknown | `system` | ⚠ |  |
| `deepinfra/moonshotai/kimi-k2.7-code` | $0.68 | $3.40 | unknown | `system` | ⚠ |  |
| `deepinfra/moonshotai/kimi-k3` | $2.85 | $14.25 | unknown | `prefill` | ⚠ | `no_temperature` |
| `deepinfra/nousresearch/hermes-3-llama-3.1-405b` | $1.00 | $1.00 | unknown | `system` | ⚠ |  |
| `deepinfra/nousresearch/hermes-3-llama-3.1-70b` | $0.70 | $0.70 | unknown | `system` | ⚠ |  |
| `deepinfra/nvidia/nemotron-3-nano-30b-a3b` | $0.05 | $0.20 | unknown | `system` | ⚠ |  |
| `deepinfra/nvidia/nvidia-nemotron-3-super-120b-a12b` | $0.08 | $0.40 | unknown | `system` | ⚠ |  |
| `deepinfra/nvidia/nvidia-nemotron-3.5-lightning` | $0.08 | $0.20 | unknown | `system` | ⚠ |  |
| `deepinfra/openai/gpt-oss-120b-turbo` | $0.15 | $0.60 | unknown | `system` | ⚠ | `reasoning_budget` |
| `deepinfra/openai/gpt-oss-120b-ultra` | $0.20 | $0.95 | unknown | `system` | ⚠ | `reasoning_budget` |
| `deepinfra/openai/gpt-oss-20b` | $0.03 | $0.14 | unknown | `system` | ⚠ | `reasoning_budget` |
| `deepinfra/qwen/qwen2.5-72b-instruct` | $0.36 | $0.40 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3-14b` | $0.12 | $0.24 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3-235b-a22b-instruct-2507` | $0.09 | $0.55 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3-30b-a3b` | $0.12 | $0.50 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3-32b` | $0.08 | $0.28 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3-coder-480b-a35b-instruct-turbo` | $0.30 | $1.00 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3-max` | $1.20 | $6.00 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3-max-thinking` | $1.20 | $6.00 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3-next-80b-a3b-instruct` | $0.09 | $1.10 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3-vl-235b-a22b-instruct` | $0.20 | $0.88 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3-vl-30b-a3b-instruct` | $0.15 | $0.60 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3.5-122b-a10b` | $0.29 | $2.40 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3.5-27b` | $0.26 | $2.60 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3.5-35b-a3b` | $0.14 | $1.00 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3.5-397b-a17b` | $0.45 | $3.00 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3.5-9b` | $0.10 | $0.15 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3.6-35b-a3b` | $0.10 | $0.95 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3.7-max` | $2.50 | $7.50 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3.8-2.4t-a95b` | $2.00 | $6.00 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3.8-27b` | $0.40 | $3.00 | unknown | `system` | ⚠ |  |
| `deepinfra/qwen/qwen3.8-max` | $1.65 | $4.95 | unknown | `system` | ⚠ |  |
| `deepinfra/sao10k/l3-8b-lunaris-v1-turbo` | $0.04 | $0.05 | unknown | `system` | ⚠ |  |
| `deepinfra/sao10k/l3.1-70b-euryale-v2.2` | $0.85 | $0.85 | unknown | `system` | ⚠ |  |
| `deepinfra/thinkingmachines/inkling` | $0.95 | $4.05 | unknown | `system` | ⚠ |  |
| `deepinfra/thinkingmachines/inkling-small` | $0.45 | $1.20 | unknown | `system` | ⚠ |  |
| `deepinfra/xiaomimimo/mimo-v2.5` | $0.40 | $2.00 | unknown | `system` | ⚠ |  |
| `deepinfra/xiaomimimo/mimo-v2.5-pro` | $1.00 | $3.00 | unknown | `system` | ⚠ |  |
| `deepinfra/zai-org/glm-4.6` | $0.50 | $2.00 | unknown | `system` | ⚠ |  |
| `deepinfra/zai-org/glm-4.7` | $0.40 | $1.75 | unknown | `system` | ⚠ | `reasoning_budget` |
| `deepinfra/zai-org/glm-4.7-flash` | $0.06 | $0.40 | unknown | `system` | ⚠ | `reasoning_budget` |
| `deepinfra/zai-org/glm-5` | $0.60 | $2.08 | unknown | `system` | ⚠ |  |
| `deepinfra/zai-org/glm-5.1` | $1.05 | $3.50 | unknown | `system` | ⚠ |  |
| `deepinfra/zai-org/glm-5.2` | $0.75 | $2.40 | unknown | `system` | ⚠ |  |
| `deepseek/deepseek-v4-flash` | $0.44 | $1.32 | unknown | `system` | ⚠ | `reasoning_budget` |
| `deepseek/deepseek-v4-flash-vision-exp` | $0.44 | $1.32 | unknown | `system` | ⚠ | `reasoning_budget` |
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
| `gemini/gemma-4-31b-it` | $0.09 | $0.34 | 2026-04-02 | `system` | ⚠ | `reasoning_budget` |
| `gemini/nano-banana-pro-preview` | $2.00 | $12.00 | unknown | `prefill` | ⚠ |  |
| `groq/allam-2-7b` | unknown | unknown | unknown | `system` | ⚠ |  |
| `groq/groq/compound` | unknown | unknown | unknown | `system` | ⚠ |  |
| `groq/groq/compound-mini` | unknown | unknown | unknown | `system` | ⚠ |  |
| `groq/openai/gpt-oss-120b` | $0.15 | $0.60 | unknown | `system` | ⚠ | `reasoning_budget` |
| `groq/openai/gpt-oss-20b` | $0.07 | $0.30 | unknown | `system` | ⚠ | `reasoning_budget` |
| `groq/qwen/qwen3.6-27b` | $0.60 | $3.00 | 2026-04-22 | `system` | ✓ |  |
| `moonshot/kimi-k2-0905-preview` | $0.60 | $2.50 | 2025-07-11 | `system` | ⚠ |  |
| `moonshot/kimi-k2.5` | $0.60 | $3.00 | 2026-01-27 | `system` | ⚠ | `no_temperature` |
| `moonshot/kimi-k2.6` | $0.95 | $4.00 | 2026-04-20 | `system` | ✓ | `no_temperature` |
| `moonshot/kimi-k2.7-code` | $0.95 | $4.00 | unknown | `system` | ⚠ | `no_temperature`, `reasoning_budget` |
| `moonshot/kimi-k2.7-code-highspeed` | unknown | unknown | unknown | `system` | ⚠ | `no_temperature`, `reasoning_budget` |
| `moonshot/kimi-k3` | $3.00 | $15.00 | 2026-07-15 | `prefill` | ✓ | `no_temperature` |
| `moonshot/moonshot-v1-128k` | $2.00 | $5.00 | unknown | `system` | ⚠ |  |
| `moonshot/moonshot-v1-128k-vision-preview` | $2.00 | $5.00 | unknown | `system` | ⚠ |  |
| `moonshot/moonshot-v1-32k` | $1.00 | $3.00 | unknown | `system` | ⚠ |  |
| `moonshot/moonshot-v1-32k-vision-preview` | $1.00 | $3.00 | unknown | `system` | ⚠ |  |
| `moonshot/moonshot-v1-8k` | $0.20 | $2.00 | unknown | `system` | ⚠ |  |
| `moonshot/moonshot-v1-8k-vision-preview` | $0.20 | $2.00 | unknown | `system` | ⚠ |  |
| `moonshot/moonshot-v1-auto` | $2.00 | $5.00 | 2023-11-08 | `system` | ⚠ |  |
| `novita/baidu/ernie-4.5-vl-424b-a47b` | $0.42 | $1.25 | unknown | `system` | ⚠ |  |
| `novita/deepseek/deepseek-r1` | $4.00 | $4.00 | unknown | `system` | ⚠ |  |
| `novita/deepseek/deepseek-r1-0528` | $0.70 | $2.50 | unknown | `system` | ⚠ |  |
| `novita/deepseek/deepseek-r1-distill-llama-70b` | $0.80 | $0.80 | unknown | `system` | ⚠ |  |
| `novita/deepseek/deepseek-r1-turbo` | $0.70 | $2.50 | unknown | `system` | ⚠ |  |
| `novita/deepseek/deepseek-v3-0324` | $0.27 | $1.12 | unknown | `system` | ⚠ |  |
| `novita/deepseek/deepseek-v3-turbo` | $0.40 | $1.30 | unknown | `system` | ⚠ |  |
| `novita/deepseek/deepseek-v3.1` | $0.27 | $1.00 | unknown | `system` | ⚠ |  |
| `novita/deepseek/deepseek-v3.1-terminus` | $0.27 | $1.00 | unknown | `system` | ⚠ |  |
| `novita/deepseek/deepseek-v3.2` | $0.27 | $0.40 | unknown | `system` | ⚠ |  |
| `novita/deepseek/deepseek-v3.2-exp` | $0.27 | $0.41 | unknown | `system` | ⚠ |  |
| `novita/deepseek/deepseek-v4-flash` | $0.14 | $0.28 | unknown | `system` | ⚠ |  |
| `novita/deepseek/deepseek-v4-flash-0731` | $0.44 | $1.32 | unknown | `system` | ⚠ |  |
| `novita/deepseek/deepseek-v4-flash-vision-exp` | $0.44 | $1.32 | unknown | `system` | ⚠ |  |
| `novita/deepseek/deepseek-v4-pro` | $1.60 | $3.20 | unknown | `system` | ⚠ |  |
| `novita/deepseek/deepseek-v4-pro-0813` | $1.32 | $3.96 | unknown | `system` | ⚠ |  |
| `novita/deepseek/deepseek_v3` | $0.89 | $0.89 | unknown | `system` | ⚠ |  |
| `novita/google/gemma-3-27b-it` | $0.12 | $0.20 | unknown | `system` | ⚠ |  |
| `novita/google/gemma-4-26b-a4b-it` | $0.13 | $0.40 | unknown | `system` | ⚠ |  |
| `novita/google/gemma-4-31b-it` | $0.14 | $0.40 | unknown | `system` | ⚠ |  |
| `novita/inclusionai/ling-3.0-flash` | $0.06 | $0.18 | unknown | `system` | ⚠ |  |
| `novita/inclusionai/ling-3.0-flash-fast` | $0.06 | $0.18 | unknown | `system` | ⚠ |  |
| `novita/inclusionai/ling-3.0-flash-fin` | unknown | unknown | unknown | `system` | ⚠ |  |
| `novita/kwaipilot/kat-coder-pro` | $0.30 | $1.20 | unknown | `system` | ⚠ |  |
| `novita/meta-llama/llama-3.1-8b-instruct` | $0.02 | $0.05 | unknown | `system` | ⚠ |  |
| `novita/meta-llama/llama-3.3-70b-instruct` | $0.14 | $0.40 | unknown | `system` | ⚠ |  |
| `novita/meta-llama/llama-4-maverick-17b-128e-instruct-fp8` | $0.27 | $0.85 | unknown | `system` | ⚠ |  |
| `novita/meta-llama/llama-4-scout-17b-16e-instruct` | $0.18 | $0.59 | unknown | `system` | ⚠ |  |
| `novita/microsoft/wizardlm-2-8x22b` | $0.62 | $0.62 | unknown | `system` | ⚠ |  |
| `novita/mindai/macaron-v1-venti` | $1.50 | $4.50 | unknown | `system` | ⚠ |  |
| `novita/minimax/m2-her` | unknown | unknown | unknown | `system` | ⚠ |  |
| `novita/minimax/minimax-m2` | $0.30 | $1.20 | unknown | `system` | ⚠ |  |
| `novita/minimax/minimax-m2.1` | $0.30 | $1.20 | unknown | `system` | ⚠ |  |
| `novita/minimax/minimax-m2.5` | $0.30 | $1.20 | unknown | `system` | ⚠ |  |
| `novita/minimax/minimax-m2.5-highspeed` | $0.60 | $2.40 | unknown | `system` | ⚠ |  |
| `novita/minimax/minimax-m2.7` | $0.30 | $1.20 | unknown | `system` | ⚠ |  |
| `novita/minimax/minimax-m2.7-highspeed` | $0.60 | $2.40 | unknown | `system` | ⚠ |  |
| `novita/minimax/minimax-m3` | $0.30 | $1.20 | unknown | `system` | ⚠ |  |
| `novita/minimaxai/minimax-m1-80k` | $0.55 | $2.20 | unknown | `system` | ⚠ |  |
| `novita/mistralai/mistral-nemo` | $0.04 | $0.17 | unknown | `system` | ⚠ |  |
| `novita/moonshotai/kimi-k2-0905` | $0.60 | $2.50 | unknown | `system` | ⚠ |  |
| `novita/moonshotai/kimi-k2-instruct` | $0.57 | $2.30 | unknown | `system` | ⚠ |  |
| `novita/moonshotai/kimi-k2.5` | $0.60 | $3.00 | unknown | `system` | ⚠ |  |
| `novita/moonshotai/kimi-k2.6` | $0.80 | $3.40 | unknown | `system` | ⚠ |  |
| `novita/moonshotai/kimi-k2.7-code` | $0.95 | $4.00 | unknown | `system` | ⚠ |  |
| `novita/moonshotai/kimi-k3` | $3.00 | $15.00 | unknown | `prefill` | ⚠ | `no_temperature` |
| `novita/nvidia/nemotron-3-nano-30b-a3b` | $0.05 | $0.20 | unknown | `system` | ⚠ |  |
| `novita/openai/gpt-oss-120b` | $0.05 | $0.25 | unknown | `system` | ⚠ | `reasoning_budget` |
| `novita/openai/gpt-oss-20b` | $0.04 | $0.15 | unknown | `system` | ⚠ | `reasoning_budget` |
| `novita/qwen/qwen-2.5-72b-instruct` | $0.38 | $0.40 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3-235b-a22b-fp8` | $0.20 | $0.80 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3-235b-a22b-instruct-2507` | $0.09 | $0.58 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3-235b-a22b-thinking-2507` | $0.30 | $3.00 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3-coder-30b-a3b-instruct` | $0.07 | $0.27 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3-coder-480b-a35b-instruct` | $0.38 | $1.55 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3-coder-next` | $0.20 | $1.50 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3-max` | $2.11 | $8.45 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3-next-80b-a3b-instruct` | $0.15 | $1.50 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3-omni-30b-a3b-instruct` | $0.25 | $0.97 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3-omni-30b-a3b-thinking` | $0.25 | $0.97 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3-vl-235b-a22b-instruct` | $0.30 | $1.50 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3-vl-235b-a22b-thinking` | $0.98 | $3.95 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3-vl-30b-a3b-instruct` | $0.20 | $0.70 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3.5-122b-a10b` | $0.40 | $3.20 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3.5-27b` | $0.30 | $2.40 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3.5-35b-a3b` | $0.25 | $2.00 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3.5-397b-a17b` | $0.60 | $3.60 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3.5-plus` | unknown | unknown | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3.6-27b` | $0.60 | $3.60 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3.6-35b-a3b` | $0.25 | $1.49 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3.6-plus` | unknown | unknown | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3.7-max` | $1.25 | $3.75 | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3.8-27b` | unknown | unknown | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3.8-flash` | unknown | unknown | unknown | `system` | ⚠ |  |
| `novita/qwen/qwen3.8-max` | $2.00 | $6.00 | unknown | `system` | ⚠ |  |
| `novita/sao10k/l3-8b-lunaris` | $0.05 | $0.05 | unknown | `system` | ⚠ |  |
| `novita/sao10k/l31-70b-euryale-v2.2` | $1.48 | $1.48 | unknown | `system` | ⚠ |  |
| `novita/thudm/glm-4-32b-0414` | $0.55 | $1.66 | unknown | `system` | ⚠ |  |
| `novita/xiaomimimo/mimo-v2.5` | $0.17 | $0.34 | unknown | `system` | ⚠ |  |
| `novita/xiaomimimo/mimo-v2.5-pro` | $0.52 | $1.04 | unknown | `system` | ⚠ |  |
| `novita/zai-org/autoglm-phone-9b-multilingual` | $0.04 | $0.14 | unknown | `system` | ⚠ |  |
| `novita/zai-org/glm-4.5-air` | $0.13 | $0.85 | unknown | `system` | ⚠ |  |
| `novita/zai-org/glm-4.5v` | $0.60 | $1.80 | unknown | `system` | ⚠ |  |
| `novita/zai-org/glm-4.7` | $0.60 | $2.20 | unknown | `system` | ⚠ |  |
| `novita/zai-org/glm-5` | $1.00 | $3.20 | unknown | `system` | ⚠ |  |
| `novita/zai-org/glm-5-turbo` | $1.20 | $4.00 | unknown | `system` | ⚠ |  |
| `novita/zai-org/glm-5.1` | $1.38 | $4.40 | unknown | `system` | ⚠ |  |
| `novita/zai-org/glm-5.2` | $1.40 | $4.40 | unknown | `system` | ⚠ |  |
| `novita/zai-org/glm-5.3` | $1.40 | $4.40 | unknown | `system` | ⚠ |  |
| `novita/zai-org/glm-5v-turbo` | $1.20 | $4.00 | unknown | `system` | ⚠ |  |
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
| `together_ai/deepseek-ai/DeepSeek-V4-Pro-0813` | $1.32 | $3.96 | unknown | `system` | ⚠ | `reasoning_budget` |
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
| `zai/glm-5.2` | $1.19 | $3.74 | 2026-06-16 | `system` | ⚠ |  |

Legend: `✓` = LiteLLM pricing present and release date available; `⚠` = missing/approximate field or known issue.

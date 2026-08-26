# Provider Health

Results from the weekly live-provider health check. Models without a configured API key are skipped. A quota/credit exhaustion is recorded as an expected failure so monitoring remains green; other errors are real failures.

Latest recorded run: `2026-08-26T07:30:05.933874+00:00`

| Model | Runs | Success | Last status | Median TTFT | Median tok/s |
|---|---:|---:|---|---:|---:|
| `anthropic/claude-fable-5` | 1 | 100% | ok | 3.095s | 35.2 |
| `anthropic/claude-haiku-4-5-20251001` | 4 | 100% | ok | 0.681s | 4427.1 |
| `anthropic/claude-opus-4-1-20250805` | 1 | 0% | xfail_retired_model | — | — |
| `anthropic/claude-opus-4-20250514` | 1 | 0% | xfail_retired_model | — | — |
| `anthropic/claude-opus-4-5-20251101` | 1 | 100% | ok | 1.596s | 2792.3 |
| `anthropic/claude-opus-4-6` | 4 | 100% | ok | 1.742s | 4731.6 |
| `anthropic/claude-opus-4-7` | 8 | 100% | ok | 1.548s | 4570.6 |
| `anthropic/claude-opus-4-8` | 1 | 100% | ok | 1.443s | 1843.4 |
| `anthropic/claude-opus-5` | 1 | 100% | ok | 1.699s | 49.8 |
| `anthropic/claude-sonnet-4-20250514` | 4 | 0% | xfail_retired_model | — | — |
| `anthropic/claude-sonnet-4-5-20250929` | 4 | 100% | ok | 1.912s | 118.9 |
| `anthropic/claude-sonnet-4-6` | 8 | 100% | ok | 1.937s | 2328.3 |
| `anthropic/claude-sonnet-5` | 1 | 100% | ok | 1.811s | 43.5 |
| `openai/gpt-4-turbo-2024-04-09` | 1 | 100% | ok | 4.949s | 2793.8 |
| `openai/gpt-4.1` | 1 | 100% | ok | 1.760s | 3521.9 |
| `openai/gpt-4.1-2025-04-14` | 1 | 100% | ok | 1.004s | 3732.3 |
| `openai/gpt-4.1-mini` | 1 | 100% | ok | 1.062s | 56.5 |
| `openai/gpt-4.1-mini-2025-04-14` | 1 | 100% | ok | 0.610s | 2094.6 |
| `openai/gpt-4.1-nano` | 1 | 100% | ok | 0.931s | 3716.1 |
| `openai/gpt-4.1-nano-2025-04-14` | 1 | 100% | ok | 0.987s | 3945.4 |
| `openai/gpt-4o` | 1 | 100% | ok | 0.983s | 3844.4 |
| `openai/gpt-4o-2024-05-13` | 1 | 100% | ok | 0.835s | 3751.5 |
| `openai/gpt-4o-2024-08-06` | 1 | 100% | ok | 0.736s | 3788.4 |
| `openai/gpt-4o-2024-11-20` | 1 | 100% | ok | 0.591s | 3440.5 |
| `openai/gpt-4o-mini` | 11 | 100% | ok | 0.673s | 2183.2 |
| `openai/gpt-4o-mini-2024-07-18` | 1 | 100% | ok | 0.609s | 135.7 |
| `openai/gpt-5.1` | 1 | 100% | ok | 0.695s | 298.8 |
| `openai/gpt-5.1-2025-11-13` | 1 | 100% | ok | 0.592s | 12.2 |
| `openai/gpt-5.2` | 1 | 100% | ok | 0.962s | 2434.5 |
| `openai/gpt-5.2-2025-12-11` | 1 | 100% | ok | 0.969s | 2536.9 |
| `openai/gpt-5.2-pro` | 1 | 0% | error | — | — |
| `openai/gpt-5.2-pro-2025-12-11` | 1 | 0% | error | — | — |
| `openai/gpt-5.3-codex` | 1 | 0% | error | — | — |
| `openai/gpt-5.4` | 1 | 100% | ok | 0.837s | 2233.8 |
| `openai/gpt-5.4-2026-03-05` | 1 | 100% | ok | 0.874s | 2557.2 |
| `openai/gpt-5.4-mini` | 8 | 100% | ok | 0.541s | 1524.3 |
| `openai/gpt-5.4-mini-2026-03-17` | 1 | 100% | ok | 0.506s | 2497.0 |
| `openai/gpt-5.4-nano` | 1 | 100% | ok | 0.733s | 2428.0 |
| `openai/gpt-5.4-nano-2026-03-17` | 1 | 100% | ok | 0.664s | 2135.9 |
| `openai/gpt-5.5` | 1 | 0% | error | — | — |
| `openai/gpt-5.5-2026-04-23` | 1 | 0% | error | — | — |
| `openai/gpt-5.6-sol` | 1 | 100% | ok | 1.783s | 111.1 |
| `openai/gpt-5.6-terra` | 1 | 100% | ok | 1.198s | 3937.0 |

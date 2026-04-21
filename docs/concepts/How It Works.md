# How It Works

## Core flow

`basemode` runs a small pipeline:

1. Normalize model name (`normalize_model`)
2. Detect strategy (`detect_strategy`)
3. Stream tokens via that strategy
4. Heal token boundaries/newlines so `prefix + tokens` is clean text

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

## Token healing

Stream output is post-processed to avoid common boundary artifacts:

- Missing space between prefix and first token
- Split compounds like `any one` where the model intended `anyone`
- Newline artifacts that break prose flow

This is why `prefix + ''.join(tokens)` remains readable and stable across providers.

# Quickstart

## 1. Generate one continuation

```bash
basemode "Once upon a time there was a" -m gpt-4o-mini -M 200
```

`basemode` prints your prefix dimmed, then streams continuation tokens.

## 2. Generate parallel branches

```bash
basemode "Once upon a time there was a" -n 3
```

This opens a live panel and streams all branches concurrently.

## 3. Inspect strategy behavior

```bash
basemode "Once upon a time" --show-strategy
basemode strategies
basemode info claude-sonnet-4-6
```

## 4. Save your default model

```bash
basemode default claude-sonnet-4-6
basemode default
```

You can now omit `--model`.

## 5. Pipe text from stdin

```bash
cat chapter1.txt | basemode --model groq/llama-3.3-70b-versatile
```

See [[CLI Reference]] for all commands and options.

# CLI Reference

All commands are under `basemode`. Running `basemode` with no explicit subcommand defaults to `run`.

## Generation

### `run`

Generate continuation text (single stream or parallel branches).

```bash
basemode [PREFIX] [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--model` | stored default or `gpt-4o-mini` | Model to use |
| `-n`, `--branches` | `1` | Number of parallel continuations |
| `-M`, `--max-tokens` | `200` | Max output tokens |
| `-t`, `--temperature` | `0.9` | Sampling temperature (when model allows) |
| `-s`, `--strategy` | auto | Force strategy selection |
| `--rewind` | `false` | Rewind short trailing fragments before generation |
| `--show-strategy` | `false` | Print selected strategy |
| `--show-usage` | `false` | Print token estimate after generation |
| `--show-cost` | `false` | Print estimated cost after generation |

`PREFIX` can come from stdin when omitted.

## Discovery

### `models`

List LiteLLM-known models.

```bash
basemode models [--provider openai] [--search claude] [--available] [--verified] [--json]
```

- `--available` limits to providers with configured keys.
- `--verified` limits to models tracked in the verified-models registry.
- `--json` emits structured picker metadata (provider, availability, reliability, pricing fields when known).

### `providers`

List all known providers.

```bash
basemode providers
```

### `strategies`

List supported continuation strategies.

```bash
basemode strategies
```

### `info`

Show normalized model ID, selected strategy, token limits, and pricing metadata.

```bash
basemode info claude-sonnet-4-6
```

## Server

### `serve`

Run an OpenAI-completions-compatible server (`POST /v1/completions`), backed by `continue_text`/`branch_text`. Requires the `server` extra: `pip install 'basemode[server]'`.

```bash
basemode serve [--host 127.0.0.1] [--port 8080]
```

Point tools that expect a llama.cpp-style `/v1/completions` endpoint (e.g. [Tapestry Loom](https://github.com/transkatgirl/Tapestry-Loom)) at `http://<host>:<port>/v1/completions`. Also exposes `GET /v1/models`. `stream` requests are always answered synchronously (no SSE); no logprobs are returned, since this coerces chat models rather than serving a real base model.

## Configuration

### `keys`

Manage persisted API keys (`~/.config/basemode/auth.json`).

```bash
basemode keys set openai
basemode keys list
basemode keys get anthropic
```

### `default`

Show/set/unset default model.

```bash
basemode default
basemode default gpt-4o-mini
basemode default --unset
```

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
| `--rewind` | `false` | Hold back the last short token so word joins are exact (may reissue the request) |
| `--strict-max-tokens` | `false` | Enforce the visible output limit client-side |
| `--show-strategy` | `false` | Print selected strategy |
| `--show-usage` | `false` | Print token estimate after generation |
| `--show-cost` | `false` | Print estimated cost after generation |

`PREFIX` can come from stdin when omitted.

## Discovery

### `models`

List known model endpoints. The list combines basemode's verified data, cached
live provider catalogs, and LiteLLM's bundled catalog; fresh provider evidence
takes precedence over stale bundled entries.

```bash
basemode models [--provider openai] [--search claude] [--available] [--verified] [--json]
```

- `--available` limits to providers with configured keys.
- `--verified` limits to models tracked in the verified-models registry.
- `--full` retains dated snapshots that are collapsed in the default view.
- `--all-modes` includes image, audio, embedding, and other non-text models.
- `--since 6m` limits results by release age; ISO dates are also accepted.
- `--live` queries a provider's current catalog and requires `--provider`.
- `--json` emits structured picker metadata (provider, availability, reliability, pricing fields when known).

The `Rating` column shows your own thumbs (see `rate`); rated models sort to
the top or the bottom of the list.

### `rate`

Rate a model up or down. Ratings are personal, stored in
`~/.config/basemode/auth.json`, and reorder every list built from the model
picker — `basemode models`, and any frontend on top of it.

```bash
basemode rate claude-opus-5 up
basemode rate gpt-4o down
basemode rate gpt-4o clear
basemode rate                      # list every rated model
```

### `health`

Show what models actually did on this machine: attempts, failures, failure
rate, and the categories they failed with. Recorded automatically by every
continuation; see [[Keys and Defaults]].

```bash
basemode health [MODEL] [--days 7] [--json] [--clear] [--verification]
```

`--verification` shows durable verification-probe health instead of local
generation history.

### `providers`

List all known providers.

```bash
basemode providers
```

### `strategies`

List supported continuation strategies, followed by any per-model strategy
pins stored on this machine.

```bash
basemode strategies
basemode strategies --unpin claude-opus-5
```

### `info`

Show normalized model ID, selected strategy and where that choice came from
(`verified models registry`, `model-name heuristic`, or a local pin), quirks,
token limits, pricing metadata, your rating, and observed health.

```bash
basemode info claude-sonnet-4-6
```

## Tuning

### `verify`

Run durable quick, thorough, or transient-failure verification. This makes
real provider requests and stores each attempt in the shared model-evidence
database. See [[Verification]] for suite selection, planning, limits, recovery,
and status effects.

```bash
basemode verify [MODEL...] [--suite quick|thorough|transient-recheck] [--attempts N] [--max-tokens N]
  [--provider NAME] [--status STATUS] [--from-catalog]
  [--released-since YYYY-MM-DD | --max-release-age-days N]
  [--stale-after-days N] [--dry-run] [--json]
```

Sweeps run with bounded global and per-provider concurrency (`--concurrency` and
`--per-provider-concurrency`). Bound work with `--max-probes`, `--max-requests`,
`--max-elapsed` (seconds), and `--max-cost-usd`. Every underlying self-healing
request is retained and counts toward the request limit. A bounded run finishes
with status `limited`; continue it with `basemode verify --resume RUN_ID`.

Models are required for quick and thorough suites. With no models, the
transient-recheck suite selects endpoints whose latest operational failure is
suspected to be temporary and whose stored backoff timestamp is due. Explicit
model arguments override queue selection for a deliberate manual recheck.

Selectors can plan deterministic catalog sweeps by provider, release date, and
current status (`never-tested`, `reachable`, `broken`, `transient`, `verified`,
or `stale`). Repeat provider and status options to form unions. `--dry-run`
makes no provider requests and reports the ordered stages, provider counts,
logical probes, maximum self-healing requests, and a best-effort price ceiling.

### `evidence`

Inspect the shared database without making provider requests. Reports exclude
endpoints explicitly identified as image, video, audio, embedding, reranking,
or moderation models.

```bash
basemode evidence
basemode evidence providers
basemode evidence statuses --json
basemode evidence failures
basemode evidence transient
basemode evidence rechecks --json
basemode evidence runs
basemode evidence corpus
basemode evidence endpoint openai/gpt-5.4
basemode evidence export > evidence.jsonl
basemode evidence export --json
```

The export omits prompts, generated text, request/configuration JSON, output
fingerprints, and account identifiers. It contains safe structured outcomes,
measurements, text endpoint metadata, and aggregated corpus quality.

### `bench`

Run each candidate strategy against a model and rank them by continuation
quality. Makes real API calls — four short probes per strategy, a fraction of
a cent for a full run. See [[Strategies]] for how scoring works.

```bash
basemode bench claude-opus-5 [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-s`, `--strategies` | `system,prefill,few_shot` | Comma-separated strategies to compare |
| `-M`, `--max-tokens` | `60` | Max output tokens per probe |
| `-t`, `--temperature` | `1.0` | Sampling temperature (the one value every provider accepts) |
| `--samples` | `false` | Print a sample continuation (or the provider error) per strategy |
| `--save` | `false` | Pin the winning strategy for this model |
| `--json` | `false` | Emit the ranking as JSON |

Exits `1` when no strategy produced a usable continuation — usually a missing
key or a model ID the provider doesn't recognize. Run with `--samples` to see
the underlying errors.

```bash
# Compare, then pin the winner
basemode bench kimi-k3 --samples
basemode bench kimi-k3 --save

# Drop the pin again
basemode strategies --unpin kimi-k3
```

## Server

### `serve`

Run an OpenAI-completions-compatible server (`POST /v1/completions`), backed by `continue_text`/`branch_text`. Requires the `server` extra: `pip install 'basemode[server]'`.

```bash
basemode serve [--host 127.0.0.1] [--port 8080]
```

Set a compatible client's endpoint to
`http://<host>:<port>/v1/completions`. Requests accept `model`, `prompt`,
`max_tokens`, `temperature`, `n`, `echo`, and `strategy`. The server also
exposes `GET /v1/models`.

Responses are synchronous JSON, including when a request supplies `stream`;
SSE and logprobs are not supported. The endpoint works with clients such as
[Tapestry Loom](https://github.com/transkatgirl/Tapestry-Loom) that accept the
classic completions format.

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

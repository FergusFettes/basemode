# Agent Quickstart

Use this page when changing the `basemode` repository. It is intentionally about
working on the project; [[Quickstart]] is the guide for using the installed CLI.

## First five minutes

```bash
uv sync --all-groups
make check
```

`make check` is the standard local gate: Ruff linting and formatting (including
scripts), the non-integration test suite, a strict documentation build, and
wheel/sdist builds. Run a focused test while iterating, then run this full gate
before handing off a change.

Integration tests make paid requests to live providers and are intentionally
opt-in:

```bash
make test-integration
```

They skip providers without keys and write a health report under `dist/`. Never
print or commit provider credentials; see [[Keys and Defaults]].

## Repository map

| Area | Start here | Use it when |
|---|---|---|
| Public streaming API | `src/basemode/continue_.py` | Changing single or parallel continuation behaviour |
| Prompt coercion | `src/basemode/strategies/` | Adding or adjusting a continuation strategy; see [[Strategies]] |
| Provider quirks | `src/basemode/strategies/compat.py` | A model rejects a parameter, prefill, or needs thinking-budget handling |
| Text repair | `src/basemode/healing.py` | Fixing spaces, fragmented words, or streamed-newline behaviour |
| Model IDs and selection | `src/basemode/detect.py` | Adding aliases or changing strategy selection; see [[Model Normalization]] |
| Continuation scoring | `src/basemode/scoring.py` | Tuning what counts as assistant leakage rather than a clean continuation |
| Strategy bake-off | `src/basemode/bench.py` | Changing how `basemode bench` probes or ranks strategies |
| Model listing metadata | `src/basemode/models.py`, `src/basemode/live_models.py` | Updating picker data or live-provider discovery |
| User model opinion and history | `src/basemode/keys.py`, `src/basemode/health.py` | Changing stored ratings, or how generation outcomes are recorded |
| Provider health | `tests/test_integration.py`, `scripts/model_reliability.py` | Maintaining live-provider checks and the [[Provider Health]] dashboard |
| Interfaces | `src/basemode/cli.py`, `src/basemode/server.py` | Changing CLI or OpenAI-completions-compatible server behaviour |
| Tests | `tests/` | Add a regression beside the feature area; real-API coverage is in `test_integration.py` |

For the public contract and examples, see [[Python API]], [[CLI Reference]], and
[[How It Works]]. Keep these pages in sync with behaviour changes.

## Model-data workflow

The editable source of verified-model intent is
`data/verified_models_registry.json`. Its `prompt_method` field is read at
runtime by `detect.select_strategy`, so an edit there changes which strategy
ships for that model — the published table and the runtime behaviour are the
same field. Regenerate the packaged copy after editing it. Generated outputs are:

- `README.md` verified-model table
- `docs/usage/Verified Models.md`
- `src/basemode/data/verified_models_details.json`

Regenerate them with `make models-table`; do not hand-edit generated tables.
`scripts/refresh_live_models.py` refreshes the packaged live-model cache.
Scheduled workflows in `.github/workflows/` open pull requests for these
updates. Read [[Verified Models]] for the user-facing meaning of that data.

The weekly integration workflow records reliability, TTFT, and output throughput
for every verified model with a configured key. Regenerate its docs view locally
with `make health-report`.

## Change checklist

1. Make the smallest coherent change and add/adjust a regression test.
2. Update the relevant user-facing docs and reference signature.
3. Run focused tests, then `make check`.
4. Run `uv run pre-commit run --all-files` if hooks or formatting tooling changed.
5. Commit small logical units. CI uses `uv sync --locked`; update `uv.lock` with
   `uv lock` whenever dependency constraints change.

## Further reading

- [[Installation]] — environment and provider-key setup
- [[Strategies]] — coercion trade-offs and overrides
- [[Model Normalization]] — aliases and provider-qualified IDs
- [[Keys and Defaults]] — key precedence and persisted configuration
- [[Verified Models]] — reliability, pricing, and generated model records
- [[CLI Reference]] and [[Python API]] — public interfaces

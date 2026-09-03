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

## How a request flows

Almost every change lands somewhere on this path. Read it once before touching
the strategy or detection code.

```
continue_text(prefix, model)          src/basemode/continue_.py
  |
  |  normalize_model()                src/basemode/detect.py
  |    "claude-opus-5" -> "anthropic/claude-opus-5"
  |    aliases, provider-prefix inference, Anthropic "4.6" -> "4-6"
  v
  |  select_strategy()                src/basemode/detect.py
  |    explicit arg > user pin > verified registry > name heuristic
  |    returns a StrategyChoice carrying *which* of those applied
  v
  |  Strategy.stream(prefix, params)  src/basemode/strategies/*.py
  |    the model-specific prompt shape: prefill / system / few_shot / fim
  v
  |  build_kwargs()                   src/basemode/strategies/compat.py
  |    drops or rewrites params the model rejects (no_temperature,
  |    no_prefill, reasoning_budget floors)
  v
  |  litellm.acompletion(...)         the actual provider call
  v
  |  healing                          src/basemode/healing.py
  |    seam repair at the prefix/continuation boundary, newline normalization
  v
  |  Operation/Attempt recorder       src/basemode/observations.py
  |    unified local health and verification provenance
  v
  streamed tokens
```

Two things about this path are easy to get wrong:

- **`select_strategy` reports its own provenance.** It returns a
  `StrategyChoice` with a source string, not a bare name, because `basemode
  info` has to explain *why* a model got the strategy it did. Preserve that
  when adding a precedence level.
- **`on_raw_head` fires before healing.** It is the only hook that sees text as
  the strategy produced it. That is what distinguishes "the model never emitted
  the space" from "healing ate the space" — do not move it downstream of
  `healing.py`.

## Adding a strategy

1. Subclass `ContinuationStrategy` in `src/basemode/strategies/`, set a `name`
   class attribute, and implement `async def stream(self, prefix, params)`.
   Build provider kwargs with `compat.build_kwargs(params)` rather than passing
   `params` straight to litellm — that is where quirk handling lives.
2. Register the class in `REGISTRY` in `strategies/__init__.py`. The dict is
   keyed off each class's `name`, so the attribute is the public identifier
   used by `--strategy`, the registry's `prompt_method`, and pins.
3. Add it to the heuristic fallback in `detect._heuristic_strategy` only if a
   model-name pattern should reach for it *without* verification.
4. Add a regression in `tests/` covering the prompt shape it builds.
5. Document it in [[Strategies]] — that page is the user-facing contract for
   what each strategy costs and when it fails.

`basemode bench` picks the new strategy up automatically; it enumerates
`REGISTRY`, so no separate registration is needed there.

## Adding a model or provider quirk

A model that misbehaves needs data, not code. Edit
`data/verified_models_registry.json`:

- `prompt_method` — the strategy that verifiably works. Read at runtime by
  `select_strategy`, so this field *is* the shipped behaviour.
- `quirks` — parameter-level workarounds consumed by `compat.build_kwargs`.
  Existing ones: `no_prefill`, `no_temperature`, `reasoning_budget`.

Then run `make models-table` to regenerate the README table, the docs page, and
the packaged JSON. Never hand-edit those three.

A genuinely new *kind* of misbehaviour — a parameter no existing quirk covers —
needs a new quirk name handled in `compat.py` plus a test, then the registry
entry. Prefer an existing quirk where one fits; the vocabulary is deliberately
small.

## Gotchas

- **Never let litellm's exceptions reach stdout.** Some failures print
  provider-help text as a side effect, which corrupts CLI output that callers
  parse. `normalize_model` deliberately checks local aliases before probing
  litellm for this reason.
- **Integration tests cost real money.** They are excluded from `make check` by
  the `not integration` marker. Keep it that way.
- **`uv.lock` is checked in and CI runs `uv sync --locked`.** A dependency
  change without `uv lock` fails CI rather than silently resolving.
- **Version bumps trigger publishing.** `publish.yml` compares the
  `pyproject.toml` version against PyPI on every push to `main` and uploads
  when they differ. Bump the version only when you intend a release.

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
| Provider checks | `tests/test_integration.py` | Running opt-in live-provider integration checks |
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

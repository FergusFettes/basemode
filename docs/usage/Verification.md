# Verification

`basemode verify` checks whether a provider endpoint can produce clean text
continuations and records every attempt in the model-evidence database. It is
intended for controlled compatibility testing, not ordinary generation.

Verification makes live provider requests and may incur cost. Start with a dry
run when selecting more than a few endpoints.

## Choose a suite

| Suite | Probes | Use it for | Effect on status |
|---|---:|---|---|
| `quick` | One prefix | Reachability and request-shape checks | Can mark an endpoint reachable or broken |
| `thorough` | Three prefixes | Continuation compatibility across prose styles | A completed passing run can mark an endpoint verified |
| `transient-recheck` | One prefix | Retrying due operational failures | Updates the recheck schedule and recovery state |

`--attempts N` repeats every logical probe. A thorough run passes only when
each probe has at least one successful attempt. Failed recovery requests remain
in the evidence database but do not fail a probe that later succeeds.

## Verify named endpoints

Use provider-qualified model IDs so the evidence identifies the endpoint that
was actually called:

```bash
# Inexpensive reachability check
basemode verify openai/gpt-4o-mini

# Compatibility check with repeated probes
basemode verify anthropic/claude-sonnet-4-6 \
  --suite thorough --attempts 3
```

Quick and transient-recheck suites default to 64 output tokens per probe;
thorough defaults to 160. Override this with `--max-tokens`.

Only text-generating endpoints are eligible. Multimodal models that produce
text remain eligible; image-, video-, audio-, embedding-, reranking-,
transcription-, and moderation-only endpoints are excluded.

## Plan a catalog sweep

Selectors build a deterministic target list from stored catalog and
verification evidence:

```bash
basemode verify \
  --from-catalog \
  --provider openai \
  --status never-tested \
  --released-since 2026-08-01 \
  --dry-run
```

Available selectors are:

- `--available`, selecting models from providers with configured keys
- `--provider NAME`, repeatable
- `--status STATUS`, repeatable
- `--from-catalog`, requiring the latest catalog observation to be available
- `--priced`, selecting only models with known pricing
- `--unpriced`, selecting only models without known pricing
- `--released-since YYYY-MM-DD`
- `--max-release-age-days N`
- `--stale-after-days N`, which defaults to 30

Statuses are `never-tested`, `reachable`, `broken`, `transient`, `verified`,
and `stale`. Multiple providers or statuses form a union within that selector.
Different selector types are combined, so an endpoint must satisfy all of them.
`--priced` and `--unpriced` are mutually exclusive. Pricing is evaluated while
the target list is built, before any run limits are applied.

For a first local sweep, use `--available` rather than `--from-catalog`:

```bash
basemode verify --available --status never-tested --dry-run
```

To inspect the pricing gaps without making requests:

```bash
basemode verify --available --unpriced --dry-run
```

Targets are ordered by prior state: transient, broken, never-tested, stale,
reachable, then verified. Provider and model ID break ties.

### What a dry run reports

`--dry-run` does not contact providers or create a verification run. It reports
the selected endpoints, logical probes, maximum underlying requests, provider
counts, and a known-price cost ceiling. Add `--json` for a machine-readable
plan.

The cost ceiling allows up to three requests per logical probe: the initial
request and both self-healing paths. Pricing comes from best-effort LiteLLM
metadata. The output separates priced and unpriced targets; unknown prices are
not treated as free.

## Bound a live run

Use limits for any sweep:

```bash
basemode verify \
  --from-catalog --provider openai --status never-tested \
  --concurrency 2 --per-provider-concurrency 1 \
  --max-probes 5 --max-requests 10 \
  --max-elapsed 300 --max-cost-usd 0.25 \
  -v
```

| Option | Limits |
|---|---|
| `--concurrency` | All requests in flight; default 4 |
| `--per-provider-concurrency` | Requests in flight to one provider; default 2 |
| `--max-probes` | Logical probes started |
| `--max-requests` | Provider requests, including recovery attempts |
| `--max-elapsed` | Wall-clock seconds |
| `--max-cost-usd` | Known-price request reservations and recorded cost |

An unpriced endpoint cannot be guaranteed by the cost limit; use request and
probe limits as well. Provider-specific queues are interleaved so a large
provider does not monopolize the run.

With `-v`, Basemode reports the model, strategy, attempt kind, safe failure
class, status eligibility, latency, token counts, cost, and final logical
outcome while the run is active. It never prints or records prompt or response
content in these events. The same view is available for an ordinary generation:

```bash
basemode run -v "The next paragraph begins"
```

## Recovery and retained attempts

If a request fails because of its shape or returns no text, verification may
retry with reasoning disabled and then with a larger output budget. Each
provider request is stored as a separate attempt and counts toward limits.

Recorded data includes structured safe error details, latency, time to first
token, usage, and cost. Prompt text, generated text, request parameters,
fingerprints, provider error bodies, keys, and account identifiers are not stored.

## Resume an interrupted or limited run

Attempts commit individually. Reaching a configured limit marks the run
`limited`; cancellation or an unexpected exception marks it `aborted`. Both
can be resumed without repeating completed probes:

```bash
basemode verify --resume RUN_ID --max-requests 200
```

The resumed run keeps its original suite, targets, attempts, and token budget.
New runtime limits apply to the resumed invocation.

## Operational rechecks

Rate limits, timeouts, network failures, and provider outages are treated as
operational failures rather than model incompatibility. Their default backoff
is 15 minutes, 2 hours, then 24 hours. Three failures observed in separate runs
become `persistent_operational` and move to weekly checks.

Run all due checks:

```bash
basemode verify --suite transient-recheck
```

Authentication and quota failures become `account_limited` and are not queued
automatically. A later successful check becomes `recovered`. Repeated model
unavailability on one provider may become `provider_route_unavailable` when
another route to the same model family succeeds.

## Read the results

```bash
basemode health --verification
basemode health openai/gpt-4o-mini
```

See [[Model Evidence]] for the unified ledger, derived-status rules, and
opt-in aggregate contributions. See [[Strategies]] for the separate `basemode bench`
workflow, which compares prompt strategies for one model.

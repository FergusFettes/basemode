# Unified call observation and contribution specification

## Objective

Make basemode the single owner of endpoint-call observations. Every logical
continuation and every physical provider request must pass through one recorder,
regardless of whether it originated from the CLI, Python API, HTTP server,
Loom, a controlled verification suite, or a scheduled recheck.

Verification is provenance attached to ordinary call records, not a separate
kind of call. Organic usage and verification both inform operational endpoint
health. Controlled suites retain a separate `verified` claim because their
inputs and configuration are reproducible.

Move the public evidence ledger, contribution validation, cross-contributor
compilation, public reports, and published dataset artifacts to the sibling
`basemode-evidence` repository.

The existing local evidence can be discarded. Do not spend implementation time
preserving questionable historical observations. Preserve only curated model
configuration that is still required for runtime strategy and quirk selection.

## Non-goals

- Do not store prompt or response content in the observation system.
- Do not make flags, edits, branch selection, or prose quality core call-health
  fields.
- Do not implement a hosted telemetry collector.
- Do not solve malicious evidence poisoning in this iteration.
- Do not silently upload anything.
- Do not make ordinary generation depend on successful observation recording.
- Do not move provider calling, normalization, healing, or strategy selection
  out of basemode.

## Core invariants

1. Every user-visible continuation branch creates exactly one logical operation.
2. Every actual provider request creates exactly one attempt linked to that
   operation, including retries and recovery requests.
3. An operation records what the caller experienced; attempts record what each
   provider request did.
4. Recording failures are logged and swallowed. They never break generation.
5. No content, content hash, arbitrary provider error body, key, account ID, or
   caller document identifier enters the ledger.
6. Endpoint identity is provider-route-specific.
7. All call sources use the same failure taxonomy and recorder.
8. Public contribution is explicit, inspectable, aggregate-only, and opt-in.

## Fresh local schema

Replace the current `health.sqlite` and `model_evidence.sqlite` split with one
fresh database, provisionally:

```text
~/.local/share/basemode/observations.sqlite
```

Use SQLite WAL mode, foreign keys, a busy timeout, schema versioning, and short
transactions. A different final filename is acceptable, but there must be only
one authoritative local observation store.

### `model_endpoints`

Retain a normalized endpoint table with provider route, provider model ID,
upstream family where known, modality/text eligibility, release metadata, and
first/last-seen timestamps.

### `call_operations`

One row per logical continuation branch:

```text
id                         local integer primary key
event_id                   globally unique ID for export/deduplication
endpoint_id                FK model_endpoints
started_at, finished_at    UTC timestamps
source                     cli/python/server/loom/verification/recheck/other allow-list
source_version             optional package version
basemode_version           package version
strategy                   selected strategy
strategy_source            explicit/user/registry/heuristic where known
logical_outcome            success/failure/cancelled/inconclusive
returned_content           boolean
finish_reason              safe allow-listed/length-limited provider finish reason
attempt_count              derived or maintained count
total_latency_ms
total_prompt_tokens
total_completion_tokens
total_reasoning_tokens
total_cost_usd
cost_source
contribution_eligible      boolean set by explicit local preference/context
verification_probe_id      nullable FK
```

Do not store prefix length, context length, document identity, or arbitrary
request JSON in the core ledger unless a later privacy review explicitly adds
an aggregate-safe field.

### `call_attempts`

One row per physical provider request:

```text
id
operation_id
attempt_index
started_at, finished_at
attempt_kind                initial/rewind_retry/empty_retry/reasoning_off/larger_budget
outcome                     success/failure/cancelled/inconclusive
returned_content
failure_class
failure_transience
failure_attribution         provider/endpoint/account/basemode/client/unknown
http_status
safe_error_code
safe_error_parameter
finish_reason
latency_ms
ttft_ms
generation_ms
prompt_tokens
completion_tokens
reasoning_tokens
output_characters
output_tokens_per_second
cost_usd
cost_source
status_eligible
status_exclusion_reason
```

No output fingerprint is needed for organic or contributed calls. If controlled
verification genuinely requires content deduplication, keep any fingerprint in
a verification-only private table and exclude it from exports.

### Verification tables

Keep small tables for experimental structure only:

- `verification_runs`: suite, suite version, software versions, controlled
  configuration, target policy, lifecycle status.
- `verification_probes`: run, endpoint, probe identifier, repetition/logical
  probe number, required status, linked operation.
- `probe_metrics`: optional controlled scoring results linked to a probe or
  operation.

Do not duplicate endpoint request outcomes in a `verification_attempts` table.
The linked operations and attempts are the source of truth.

### Operational tables

- `recheck_schedules`: derived scheduling state per endpoint.
- `operational_assessments`: optional materialized history of state changes and
  the observation/rule that caused each change.
- `daily_call_aggregates`: rollups used after raw organic retention expires.
- `contribution_batches`: exported bundle IDs, windows, paths/status, and which
  local aggregates they cover so a window is not submitted twice.

### Optional quality tables

Flags, edits, corrections, branch acceptance, and other Loom/caller quality
signals must remain optional and separate from endpoint-call health. A future
`quality_observations` or `quality_aggregates` table may reference operation IDs
or aggregate windows. It must not be required to record or export calls.

## Observation API

Introduce a typed public context, for example:

```python
ObservationContext(
    source="loom",
    source_version="0.8.0",
    contribution_eligible=True,
)
```

Accept it on `continue_text` and `branch_text`. Defaults identify ordinary
Python use without enabling contribution.

The recorder itself should be internal and structured around an operation
scope plus attempt lifecycle calls. Exact names are flexible, but it must be
impossible for a normal provider request path to bypass attempt recording by
accident.

Suggested shape:

```python
with observe_operation(model, strategy, context) as operation:
    with operation.attempt("initial") as attempt:
        ... provider request ...
```

Async equivalents or explicit begin/finish methods are fine. Ensure partially
failed streams, cancellation, retries, and exceptions finalize records exactly
once.

`branch_text` creates one logical operation per branch. A shared invocation ID
may be retained locally for diagnostics only if it is excluded from contribution
payloads.

## Outcome semantics

### Logical operation

- `success`: content reached the caller after healing/normalization.
- `failure`: the operation ended without usable content because all allowed
  attempts failed.
- `cancelled`: the client deliberately stopped after content arrived; count as
  operational success for endpoint-health summaries while retaining cancellation.
- `inconclusive`: cancellation or shutdown before the endpoint produced a
  classifiable result.

### Physical attempt

Classify each provider request independently. A failed initial request followed
by a successful retry yields a successful operation, a failed attempt, a
successful attempt, and a recovery dependency.

### Failure taxonomy

Use a single public allow-list across ordinary use and verification:

```text
authentication
quota
rate_limit
timeout
network
provider_unavailable
invalid_request
empty_response
content_filter
provider_error
cancelled
unknown
```

Add `failure_attribution` so account failures and basemode request-shape bugs do
not condemn an endpoint globally. Existing safe error-code and parameter
extraction should be reused; raw error messages must never be stored.

## Derived health

Expose controlled and operational state independently.

### Controlled status

```text
never_tested
reachable
verified
failed
stale
account_limited
retired
```

`failed` means the endpoint answered and the answer was unusable. A run that
produced no success only because the account cannot reach the endpoint is
`account_limited`; one whose every failure was a 404 is `retired`. Neither is
a verdict on the model, and neither is swept again by default.

Only a completed controlled suite can establish `verified`. The latest thorough
run passes when every required logical probe has at least one successful linked
operation.

### Operational status

```text
unknown
healthy
degraded
failing
suspected_transient
persistent_operational
account_limited
provider_route_unavailable
recovered
```

Operational status uses all status-eligible calls, regardless of source.
Organic Loom/Python/server calls are not second-class observations.

At minimum calculate and expose:

- logical operation success rate;
- initial-attempt success rate;
- recovery rate;
- attempt failures by category and attribution;
- sample size and observation window;
- source counts;
- last successful and last failed observation.

Keep policy thresholds in a small versioned rules module or data structure.
Initial conservative rules may reuse the existing recheck timings. Do not bury
status transitions inside write functions without tests.

Organic eligible failures should create/advance recheck schedules. Organic
successes should resolve appropriate transient/unavailable schedules. Account-
attributed failures remain visible locally but do not affect public endpoint
health.

## Verification runner

Retain `basemode verify`, deterministic planning, suite definitions, work/cost/
time limits, fair provider concurrency, self-healing attempts, resume support,
and transient rechecks.

Refactor it to:

1. create a verification run and probe record;
2. invoke the ordinary continuation path with verification provenance;
3. let the common recorder capture operations and attempts;
4. attach controlled probe metrics;
5. derive run and endpoint status from the linked records.

Remove its bespoke attempt storage and any duplicate call classification.

## Contribution subsystem

Add a `basemode contribute` command group:

```bash
basemode contribute status
basemode contribute enable
basemode contribute disable
basemode contribute preview [--since ...] [--until ...]
basemode contribute export [--output PATH]
basemode contribute pr [--repo OWNER/basemode-evidence]
basemode contribute release <bundle-id>
basemode contribute clear-pending
```

### Local versus public recording

All calls may be recorded locally unless the existing global opt-out is set.
Public contribution is a separate, explicit opt-in. Enabling it affects future
operations only unless the user explicitly chooses a historical window.

`preview` and `export` must use the same serializer and validation path. The
preview must show the exact JSON that would be written.

### Aggregation

Export content-free aggregates grouped by:

- provider-qualified endpoint;
- strategy;
- source application;
- source version where known;
- a coarse UTC time window.

Export operation/attempt counts, safe failure counts, and aggregate performance/
usage/cost metrics. Never export individual rows.

The canonical v1 JSON shape is specified in `../basemode-evidence/SPEC.md`.
Implement it as typed dataclasses/models plus a checked-in JSON Schema. Reject
unknown fields during local validation as well as in the evidence repository.

Generate a random bundle ID. Record exported windows/bundle IDs locally, and
mark each counted operation submitted, so repeated commands do not accidentally
resubmit the same observations. Windows are free-form and overlap constantly, so
the marker belongs on the operation rather than the window; aggregate rows carry
neither timestamps nor contributor identity, so the evidence repository can only
refuse a repeated bundle ID and cannot deduplicate observations itself. An
export that is never submitted must be releasable.

### Pull request command

`basemode contribute pr` should shell out to an authenticated GitHub CLI (`gh`)
rather than read or store GitHub tokens. It should:

1. create the exact bundle via the normal exporter;
2. run local schema and semantic validation;
3. check `gh auth status`;
4. fork or reuse a fork of the configured evidence repository;
5. create a branch;
6. add the bundle at `contributions/v1/YYYY/MM/<bundle-id>.json`;
7. commit only that file;
8. push and open a pull request;
9. record the PR URL and bundle status locally.

If `gh` is absent or unauthenticated, fail with the exported path and clear
manual instructions. Do not implement a hosted fallback.

## Importing public evidence

Basemode may provide an explicit downloader/importer for compiled release
artifacts from `basemode-evidence`:

```bash
basemode evidence update
basemode evidence public [MODEL]
```

Requirements:

- never download during generation;
- verify the release manifest and SHA-256 checksum;
- store imported public summaries separately from local raw calls;
- retain source release/commit/schema provenance;
- allow deletion and refresh;
- do not allow public aggregates to masquerade as local observations.

This feature can follow contribution export and is not required for the first
recorder milestone.

## Retention

- Controlled verification operations and attempts: retain indefinitely.
- Recent organic attempts/failures: retain long enough for diagnosis.
- Organic successes: eligible for daily aggregation and pruning after a
  configurable period, initially 30 days.
- Daily aggregates and contribution-batch metadata: retain.
- Never prune unexported contributable observations without a clear policy.

Implement retention only after query and contribution tests prove that rollups
preserve required totals.

## Remove or move from basemode

The refactor should actively slim basemode down.

### Remove obsolete local machinery

- `src/basemode/health.py` and its separate `~/.config/basemode/health.sqlite`
  store, after `basemode health` has been reimplemented over the common ledger.
- The current evidence schema/migrations and historical import complexity in
  `src/basemode/evidence/`; replace them with the fresh schema rather than
  migrating dubious data.
- `corpus_observations` as a core evidence concept. Optional Loom quality data
  belongs in a distinct extension, not endpoint-health status.
- Legacy importers for old sweep JSONL, legacy health SQLite, scheduled health
  JSONL, rejected registries, and other obsolete formats unless a specific
  bootstrap input remains demonstrably useful.
- Duplicate ratings/annotation persistence inside evidence if `keys.py` remains
  the authoritative local preference store.
- Output fingerprints from the general evidence path.

### Move to `basemode-evidence`

- cross-contributor bundle storage;
- contribution PR validation;
- public dataset compilation;
- public aggregate reports/dashboard generation;
- public provenance and bundle revocation;
- release artifact publication.

### Review generated repository data

Review and likely remove or replace:

- `docs/usage/Provider Health.md` as a committed public report;
- scripts/workflows whose primary job is producing a repository-wide public
  health dataset;
- packaged evidence snapshots that are not required for runtime strategy/
  quirk configuration;
- scheduled public recheck artifact handling that belongs with the public
  evidence repository.

Keep the editable verified-model registry only as bootstrap runtime intent until
the new evidence system can safely replace its roles. Do not conflate curated
intent with measured evidence.

## CLI and documentation

Retain clear user-facing commands:

- `basemode health`: local operational calls from the unified ledger.
- `basemode verify`: controlled probes.
- `basemode evidence`: inspection of local operations, verification, rechecks,
  and optionally imported public summaries.
- `basemode contribute`: opt-in aggregate export/PR workflow.

Update `basemode info` and model-picker metadata to report controlled and
operational status separately, with sample counts and windows.

Rewrite the Verification and Model Evidence wiki pages once behavior lands.
Document exactly what is stored locally and what can be contributed publicly.

## Testing requirements

At minimum cover:

- one operation and one attempt for a normal successful stream;
- branch operations recorded independently;
- each retry creates a new attempt under one operation;
- recovered operation semantics;
- empty stream, provider exception, cancellation before content, and
  cancellation after content;
- callback/recorder errors never breaking generation;
- consistent classification across CLI, Python, server, Loom provenance, and
  verification;
- account/basemode-attributed failures excluded from global endpoint health;
- organic failures scheduling rechecks and successes resolving them;
- controlled verification derived through linked ordinary calls;
- exact contribution aggregation and arithmetic;
- absence of content-bearing fields in schema and payloads;
- preview and export byte-for-byte agreement;
- idempotent bundle/window tracking;
- `gh` missing/unauthenticated failure behavior without token handling;
- public snapshot checksum/provenance validation if import is implemented.

## Suggested implementation sequence

1. Delete/reset the current observation schema and define the fresh tables.
2. Implement the internal operation/attempt recorder and failure taxonomy.
3. Route `continue_text`, `branch_text`, CLI, and server calls through it.
4. Rebuild local health queries and `basemode info` on the ledger.
5. Refactor verification to attach provenance to the common call path.
6. Drive rechecks from all eligible calls.
7. Remove old health/evidence/import/reporting code and obsolete docs/workflows.
8. Add typed observation context for Loom and other clients.
9. Implement contribution aggregation, preview, export, and JSON Schema.
10. Implement optional `gh` pull-request orchestration.
11. Add public snapshot import only after `basemode-evidence` publishes a stable
    release format.

## Acceptance criteria

- Ordinary and controlled calls share the same operation/attempt records.
- No provider request can be made through a supported generation path without
  exactly one attempt record, barring a safely handled recorder failure.
- Loom no longer disables basemode recording or classifies the same provider
  outcome independently.
- Local health and verification status are projections over one database.
- A user can preview and export a contribution containing no content or stable
  user/install identifier.
- The export validates in `basemode-evidence` without importing provider code.
- Public dataset storage and compilation have been removed from basemode.
- Existing dubious observational databases are not silently treated as new
  schema evidence.

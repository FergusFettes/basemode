# Model Evidence

Basemode keeps durable model observations in
`~/.local/share/basemode/model_evidence.sqlite`. This is separate from the
short-lived generation-health log: controlled verification results are never
deleted merely because they are old.

The database distinguishes provider endpoints, catalog observations,
verification runs and attempts, structured probe results, user annotations,
and privacy-preserving corpus aggregates published by applications such as
Loom. Provider routes for the same upstream model remain separate because
their availability, request compatibility, speed, and price can differ.

## Verification suites

The quick suite performs an inexpensive continuation health check. The
thorough suite uses several prefixes and a larger output allowance; only a
completed thorough suite can add evidence-backed verified status. The
transient-recheck suite selects endpoints whose latest attempt failed with a
rate limit, timeout, network error, or provider outage.

```bash
basemode verify --suite quick openai/gpt-4o-mini
basemode verify --suite thorough --attempts 3 openai/gpt-4o-mini
basemode verify --suite transient-recheck
```

Use `basemode evidence` for read-only summaries and sanitized exports. Views
cover the overall dataset, providers, current statuses, failure classes, the
transient queue, runs, Loom corpus quality (including depth buckets), and a
single endpoint's history. `basemode evidence export` writes JSONL; add
`--json` for one JSON array. Reports are text-model-only.

These commands make provider requests and may incur cost. Every request and
self-healing step is retained, including parameters, safe structured failure
details, latency, TTFT, usage, estimated or provider-derived cost, and output
fingerprints. Prompt and response text and provider error bodies are not
stored.

On request-shape failure or empty output, verification can retry with
reasoning disabled and with a larger output budget. Operational errors such as
rate limits are recorded for later rechecking rather than being mistaken for
model incompatibility.

Verification and derived status are text-generation-only. Provider modality is
preferred where it exists; conservative model-family classification excludes
clear image, video, audio, embedding, reranking, transcription, and moderation
endpoints when catalogs omit modality. Unknown models remain eligible because
many provider chat catalogs do not publish this field.

`enforce_text_only_and_supersede_obsolete_failures()` is an idempotent evidence
maintenance operation. It retains every raw observation while excluding
non-text endpoints from model lists, status, and transient rechecks. It also
marks an old `invalid_request` attempt as status-ineligible when a later request
to the same endpoint succeeded. Explicit provider rejections of unsupported
parameters or values are also retained as compatibility evidence but excluded
from endpoint health. Other invalid requests without demonstrated recovery, and
genuine provider availability failures, remain status-bearing evidence.

## Importing existing evidence

`basemode.evidence` provides idempotent import functions for generic sweep
JSONL, the legacy health SQLite database, scheduled provider-health JSONL, the
rejected-model registry, live catalog snapshots, curated verified-registry
intent, and legacy ratings. Curated intent remains an annotation and does not
pretend to be a measured success. Imports append evidence and never remove the
original source.

Loom keeps private node text and tree structure in its own database. It calls
`publish_corpus_observations` with aggregate counts and timing summaries. A
repeat publication for the same source and time window replaces that window,
so scheduled exports do not double-count observations.

The packaged verified registry remains a compatibility fallback. Fresh,
durable evidence can contradict it; missing or expired evidence is not treated
as a successful check.

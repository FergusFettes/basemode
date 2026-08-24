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

These commands make provider requests and may incur cost. Every request and
self-healing step is retained, including parameters, safe structured failure
details, latency, TTFT, usage, estimated or provider-derived cost, and output
fingerprints. Prompt and response text and provider error bodies are not
stored.

On request-shape failure or empty output, verification can retry with
reasoning disabled and with a larger output budget. Operational errors such as
rate limits are recorded for later rechecking rather than being mistaken for
model incompatibility.

## Importing existing evidence

`basemode.evidence` provides idempotent import functions for generic sweep
JSONL, the legacy health SQLite database, scheduled provider-health JSONL, the
rejected-model registry, and legacy ratings. Imports append evidence and never
remove the original source.

Loom keeps private node text and tree structure in its own database. It calls
`publish_corpus_observations` with aggregate counts and timing summaries. A
repeat publication for the same source and time window replaces that window,
so scheduled exports do not double-count observations.

The packaged verified registry remains a compatibility fallback. Fresh,
durable evidence can contradict it; missing or expired evidence is not treated
as a successful check.

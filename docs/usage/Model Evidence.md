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

## Inspecting evidence

`basemode evidence` is read-only. Its views cover the overall dataset,
providers, current statuses, failure classes, operational rechecks, runs, Loom
corpus quality, and individual endpoint history:

```bash
basemode evidence
basemode evidence providers
basemode evidence statuses
basemode evidence failures
basemode evidence runs
basemode evidence endpoint openai/gpt-4o-mini
basemode evidence rechecks --json
```

See [[Verification]] for suites, target planning, limits, recovery requests,
resuming runs, and the operational recheck lifecycle.

## Derived status

Status is calculated from completed runs and status-eligible attempts. The
latest eligible result determines current reachability and whether a failure
looks durable or transient. A model is verified only when every logical probe
in its latest thorough run has a successful attempt. Missing or expired
evidence never counts as success.

An unsupported parameter or value is compatibility evidence about basemode's
request shape, not evidence that the endpoint is unhealthy. Such attempts are
retained but excluded from current status. An older invalid request is also
excluded when a later request to the same endpoint succeeds. Other invalid
requests and genuine provider failures continue to affect status.

## Text endpoint scope

Verification and derived status are text-generation-only. Provider modality is
preferred where it exists; conservative model-family classification excludes
clear image, video, audio, embedding, reranking, transcription, and moderation
endpoints when catalogs omit modality. Unknown models remain eligible because
many provider chat catalogs do not publish this field.

Live catalog snapshots retain available capability metadata, including
OpenRouter modalities and supported parameters, Gemini generation methods, and
generic provider model types. Output capability decides eligibility: a model
that accepts images and generates text is included; an image-only model is not.

## Sanitized exports

```bash
basemode evidence export > evidence.jsonl
basemode evidence export --json
```

Exports contain structured outcomes, timings, text-endpoint metadata, and
aggregated corpus quality. They omit keys, account identifiers, prompts,
responses, provider error bodies, request configuration, and output
fingerprints.

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

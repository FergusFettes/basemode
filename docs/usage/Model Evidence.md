# Model observations and contributions

Basemode records every logical continuation and physical provider attempt in one
content-free ledger at `~/.local/share/basemode/observations.sqlite`. CLI,
Python, server, Loom, verification, and recheck calls all use the same recorder.
Provider routes remain distinct because their availability and performance can
differ even when they expose the same upstream model.

The ledger never stores prompts, responses, content hashes, arbitrary provider
error bodies, keys, account identifiers, or caller document identifiers.
Controlled verification adds run and probe provenance to ordinary operations;
it does not maintain a second set of call records.

## Reading local status

```bash
basemode health
basemode health openai/gpt-4o-mini
basemode health --verification
```

Operational health uses status-eligible observations from ordinary use and
verification. Account-, client-, and basemode-attributed failures remain useful
for local diagnosis but do not make a public endpoint look unhealthy. A
`verified` claim is stricter: every required probe in the latest controlled run
must pass, and stale runs cease to count as verified.

Eligible organic transient failures create a durable recheck schedule. Repeated
failures back off from 15 minutes to two hours, one day, and then seven days. A
later organic success resolves the schedule.

## Opt-in public contribution

Local recording and public contribution are separate. Contribution is disabled
by default, affects future calls only, and exports aggregate rows rather than
individual operations.

```bash
basemode contribute status
basemode contribute enable
basemode contribute preview --since 2026-08-25T00:00:00Z
basemode contribute export --output contribution.json
basemode contribute disable
```

Preview and export share the same serializer and validation path. Rows are
grouped by provider-qualified endpoint, strategy, source application, and source
version. They contain counts, safe failure categories, aggregate percentiles,
token totals, and cost totals where available. The output is validated against
the public contribution v1 contract used by the sibling `basemode-evidence`
repository.

Basemode does not import that repository as a runtime dependency. The public
repository owns cross-contributor validation, compilation, reports, revocation,
and published dataset artifacts. A future explicit downloader may import its
compiled releases as clearly labelled public aggregates; they must never
masquerade as local observations.

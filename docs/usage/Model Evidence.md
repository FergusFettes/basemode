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

The ordinary health table separates logical calls from underlying provider
requests. A call can recover after one or more failed requests, so its attempt
failure counts may be nonzero while its logical failure rate remains zero.

Operational health uses status-eligible observations from ordinary use and
verification. Account-, client-, and basemode-attributed failures remain useful
for local diagnosis but do not make a public endpoint look unhealthy. A
`verified` claim is stricter: every required probe in the latest controlled run
must pass, and stale runs cease to count as verified.

Eligible organic transient failures create a durable recheck schedule. Repeated
failures back off from 15 minutes to two hours, one day, and then seven days. A
later organic success resolves the schedule.

## Opt-in public contribution

Local recording and public contribution are separate. Existing content-free
observations can be previewed and exported at any time; running the explicit
export or PR command is the consent boundary. Exports contain aggregate rows
rather than individual operations.

```bash
basemode contribute preview --since 2026-08-25T00:00:00Z
basemode contribute export --output contribution.json
basemode contribute release <bundle-id>
```

Preview and export share the same serializer and validation path.

An export marks every operation it counted as submitted, and later bundles skip
those. That is what makes repeating a command safe: `--since`/`--until` are
free-form, so two runs overlap constantly, and a window is far too coarse a
thing to deduplicate on. It is also the only place the question can be answered
— aggregate rows carry no timestamps and no contributor identity, so the public
repository can tell one bundle from another but never one contributor's
Wednesday from another's. Basemode-evidence refuses a repeated `bundle_id`;
avoiding a double count of the same observations is this machine's job.

`preview` marks nothing. An export you decide not to submit — or one whose
submission fails — would otherwise hold its observations back forever, so
`basemode contribute release <bundle-id>` frees exactly the operations that
bundle counted. It refuses once the bundle has been submitted upstream, where
releasing it really would contribute twice.

`basemode contribute pr` forks the evidence repository, unless the
authenticated account already owns it: GitHub will not let one account own both
a parent and a fork, so the owner pushes a branch to the repository itself. Rows are
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

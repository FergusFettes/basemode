"""Aggregate-only contribution bundles for the public evidence repository."""

from __future__ import annotations

import json
import math
import shutil
import sqlite3
import subprocess
import tempfile
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from . import observations
from .identity import canonical_id

SCHEMA_VERSION = 1
PUBLIC_SOURCES = {"cli", "python", "server", "loom", "verification"}
FAILURES = {
    "authentication",
    "quota",
    "rate_limit",
    "timeout",
    "network",
    "provider_unavailable",
    "invalid_request",
    "empty_response",
    "content_filter",
    "provider_error",
    "cancelled",
    "unknown",
}


def _timestamp(value: datetime | str) -> str:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _metric(values: list[float]) -> dict[str, int | float] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
    }


@dataclass(frozen=True)
class Contribution:
    """One aggregate bundle and the exact operations it counted.

    The operation IDs travel with the bundle so exporting can mark precisely
    what was counted. Deriving them again from the window would be a
    different question asked a moment later, and an operation that finished
    in between would be marked submitted without ever being counted.
    """

    bundle: dict[str, Any]
    operation_ids: tuple[int, ...]


def build_bundle(
    *,
    since: datetime,
    until: datetime,
    bundle_id: str | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build and validate one aggregate bundle without marking it exported."""
    return build_contribution(
        since=since, until=until, bundle_id=bundle_id, generated_at=generated_at
    ).bundle


def build_contribution(
    *,
    since: datetime,
    until: datetime,
    bundle_id: str | None = None,
    generated_at: datetime | None = None,
) -> Contribution:
    """Aggregate every not-yet-submitted operation in the window.

    A window is whatever `--since`/`--until` say, so two runs overlap freely.
    Submission is therefore tracked per operation rather than per window: an
    operation already counted into an exported bundle is skipped here, which
    makes a repeated export a no-op instead of a double count.
    """
    start, end = _timestamp(since), _timestamp(until)
    if not observations._DB_FILE.exists():
        raise ValueError("no local observations")
    conn = sqlite3.connect(observations._DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        operations = conn.execute(
            """SELECT o.*,e.provider_route,e.provider_model_id,e.canonical_model_id
               FROM call_operations o JOIN model_endpoints e ON e.id=o.endpoint_id
               WHERE o.finished_at IS NOT NULL AND o.is_submitted=0
                 AND o.started_at>=? AND o.started_at<?
               ORDER BY o.id""",
            (start.replace("Z", "+00:00"), end.replace("Z", "+00:00")),
        ).fetchall()
        grouped: dict[tuple[str, str, str, str | None], list[sqlite3.Row]] = (
            defaultdict(list)
        )
        for operation in operations:
            if operation["source"] not in PUBLIC_SOURCES:
                continue
            # Contributions are compared across contributors and providers,
            # so they carry the canonical provider/creator/model identity
            # rather than whatever each provider happens to call the model.
            endpoint = operation["canonical_model_id"] or canonical_id(
                operation["provider_model_id"]
                if operation["provider_route"] == "unknown"
                else f"{operation['provider_route']}/{operation['provider_model_id']}"
            )
            grouped[
                (
                    endpoint,
                    operation["strategy"],
                    operation["source"],
                    operation["source_version"],
                )
            ].append(operation)
        rows = [
            _aggregate(conn, dimensions, items) for dimensions, items in grouped.items()
        ]
        counted = tuple(
            int(operation["id"]) for items in grouped.values() for operation in items
        )
    finally:
        conn.close()
    if not rows:
        raise ValueError("no unsubmitted observations in window")
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": bundle_id or str(uuid.uuid4()),
        "generated_at": _timestamp(generated_at or datetime.now(UTC)),
        "basemode_version": observations._package_version(),
        "window_start": start,
        "window_end": end,
        "observations": sorted(
            rows,
            key=lambda row: (
                row["endpoint"],
                row["strategy"],
                row["source"],
                row.get("source_version", ""),
            ),
        ),
    }
    validate_bundle(bundle)
    return Contribution(bundle, counted)


def _aggregate(
    conn: sqlite3.Connection,
    dimensions: tuple[str, str, str, str | None],
    operations: list[sqlite3.Row],
) -> dict[str, Any]:
    endpoint, strategy, source, source_version = dimensions
    operation_ids = [int(row["id"]) for row in operations]
    placeholders = ",".join("?" for _ in operation_ids)
    attempts = conn.execute(
        f"SELECT * FROM call_attempts WHERE operation_id IN ({placeholders})",
        operation_ids,
    ).fetchall()
    initial = [row for row in attempts if row["attempt_index"] == 0]
    failures = Counter(
        failure if failure in FAILURES else "unknown"
        for item in attempts
        if item["outcome"] == "failure"
        for failure in [item["failure_class"] or "unknown"]
    )
    row: dict[str, Any] = {
        "endpoint": endpoint,
        "strategy": strategy,
        "source": source,
        "operations": len(operations),
        "successful_operations": sum(
            item["logical_outcome"] == "success" for item in operations
        ),
        "initial_attempts": len(initial),
        "successful_initial_attempts": sum(
            item["outcome"] == "success" for item in initial
        ),
        "recovered_operations": sum(
            item["logical_outcome"] == "success" and item["attempt_count"] > 1
            for item in operations
        ),
        "attempts": len(attempts),
        "failures": {key: failures[key] for key in sorted(failures)},
    }
    if source_version:
        row["source_version"] = source_version
    successful = [item for item in operations if item["logical_outcome"] == "success"]
    for key, values in (
        (
            "latency_ms",
            [
                float(item["total_latency_ms"])
                for item in successful
                if item["total_latency_ms"] is not None
            ],
        ),
        (
            "ttft_ms",
            [
                float(item["ttft_ms"])
                for item in attempts
                if item["outcome"] == "success" and item["ttft_ms"] is not None
            ],
        ),
    ):
        metric = _metric(values)
        if metric:
            row[key] = metric
    for key, column in (
        ("input_tokens", "total_prompt_tokens"),
        ("output_tokens", "total_completion_tokens"),
    ):
        values = [item[column] for item in operations]
        if all(value is not None for value in values):
            row[key] = sum(values)
    costs = [item["total_cost_usd"] for item in operations]
    if all(cost is not None for cost in costs):
        row["cost_usd"] = sum(costs)
    return row


def validate_bundle(bundle: dict[str, Any]) -> None:
    """Enforce the sibling repository's contribution-v1 semantic invariants."""
    if bundle.get("schema_version") != 1 or set(bundle) != {
        "schema_version",
        "bundle_id",
        "generated_at",
        "basemode_version",
        "window_start",
        "window_end",
        "observations",
    }:
        raise ValueError("invalid contribution-v1 envelope")
    start = datetime.fromisoformat(bundle["window_start"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(bundle["window_end"].replace("Z", "+00:00"))
    generated = datetime.fromisoformat(bundle["generated_at"].replace("Z", "+00:00"))
    if start >= end or end - start > timedelta(days=31):
        raise ValueError("contribution window must be positive and at most 31 days")
    if generated < end:
        raise ValueError("generated_at must not precede window_end")
    if not 1 <= len(bundle["observations"]) <= 1000:
        raise ValueError("bundle must contain 1 to 1000 aggregate rows")
    dimensions: set[tuple[str, str, str, str | None]] = set()
    for row in bundle["observations"]:
        allowed = {
            "endpoint",
            "strategy",
            "source",
            "source_version",
            "operations",
            "successful_operations",
            "initial_attempts",
            "successful_initial_attempts",
            "recovered_operations",
            "attempts",
            "failures",
            "latency_ms",
            "ttft_ms",
            "input_tokens",
            "output_tokens",
            "cost_usd",
        }
        required = {
            "endpoint",
            "strategy",
            "source",
            "operations",
            "successful_operations",
            "initial_attempts",
            "successful_initial_attempts",
            "recovered_operations",
            "attempts",
            "failures",
        }
        if not required <= set(row) or set(row) - allowed:
            raise ValueError("invalid contribution-v1 observation fields")
        if row["source"] not in PUBLIC_SOURCES:
            raise ValueError("invalid public contribution source")
        count_fields = {
            "operations",
            "successful_operations",
            "initial_attempts",
            "successful_initial_attempts",
            "recovered_operations",
            "attempts",
            "input_tokens",
            "output_tokens",
        }
        if any(
            key in row
            and (
                not isinstance(row[key], int)
                or isinstance(row[key], bool)
                or row[key] < 0
            )
            for key in count_fields
        ):
            raise ValueError("contribution counts must be nonnegative integers")
        comparisons = (
            ("successful_operations", "operations"),
            ("recovered_operations", "successful_operations"),
            ("successful_initial_attempts", "initial_attempts"),
            ("initial_attempts", "attempts"),
        )
        if any(row[smaller] > row[larger] for smaller, larger in comparisons):
            raise ValueError("contribution count invariant violated")
        if set(row["failures"]) - FAILURES or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in row["failures"].values()
        ):
            raise ValueError("invalid failure counts")
        if sum(row["failures"].values()) > row["attempts"]:
            raise ValueError("failures exceed attempts")
        for metric, population in (
            # latency is measured once per logical operation; time to first
            # token is measured on each provider request that produced one,
            # and an operation can have more than one of those — a resumed
            # verification probe re-runs a configuration that already
            # succeeded. Bounding TTFT by operations rejected honest data.
            ("latency_ms", "successful_operations"),
            ("ttft_ms", "attempts"),
        ):
            if metric not in row:
                continue
            value = row[metric]
            if (
                value.get("count", -1) < 0
                or value["count"] > row[population]
                or value["p50"] > value["p95"]
                or not all(math.isfinite(value[key]) for key in ("p50", "p95"))
            ):
                raise ValueError(f"invalid {metric} summary")
        if "cost_usd" in row and (
            not math.isfinite(row["cost_usd"]) or row["cost_usd"] < 0
        ):
            raise ValueError("invalid cost_usd")
        dimension = (
            row["endpoint"],
            row["strategy"],
            row["source"],
            row.get("source_version"),
        )
        if dimension in dimensions:
            raise ValueError("duplicate observation dimensions")
        dimensions.add(dimension)


def export_bundle(contribution: Contribution | dict[str, Any], path: Path) -> Path:
    """Write an exact validated bundle and mark what it counted as submitted.

    Marking happens at export rather than at PR creation because the exported
    file is already a submittable artifact: whether it reaches GitHub through
    `contribute pr` or by hand, those operations have left the machine. A
    bundle that is abandoned rather than submitted can be released again with
    `contribute unmark`.
    """
    if isinstance(contribution, dict):
        contribution = Contribution(contribution, ())
    bundle = contribution.bundle
    validate_bundle(bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    with observations._db() as conn:
        conn.execute(
            """INSERT INTO contribution_batches(
                   bundle_id,window_start,window_end,path,status,created_at
               ) VALUES(?,?,?,?,?,?)""",
            (
                bundle["bundle_id"],
                bundle["window_start"],
                bundle["window_end"],
                str(path),
                "exported",
                observations._now(),
            ),
        )
        _mark_submitted(conn, contribution.operation_ids, bundle["bundle_id"])
    return path


def _mark_submitted(
    conn: sqlite3.Connection, operation_ids: tuple[int, ...], bundle_id: str
) -> None:
    if not operation_ids:
        return
    placeholders = ",".join("?" for _ in operation_ids)
    conn.execute(
        f"""UPDATE call_operations SET is_submitted=1,submitted_bundle_id=?
            WHERE id IN ({placeholders})""",
        (bundle_id, *operation_ids),
    )


def release_bundle(bundle_id: str) -> int:
    """Undo an export: forget the batch and let its operations be counted again.

    An export that was never submitted would otherwise hold its observations
    out of every future bundle.
    """
    with observations._db() as conn:
        batch = conn.execute(
            "SELECT status FROM contribution_batches WHERE bundle_id=?", (bundle_id,)
        ).fetchone()
        if batch is None:
            raise ValueError(f"no local record of bundle {bundle_id}")
        if batch["status"] == "submitted":
            raise ValueError(
                f"bundle {bundle_id} was submitted upstream; releasing it would "
                "contribute the same observations twice"
            )
        released = conn.execute(
            """UPDATE call_operations SET is_submitted=0,submitted_bundle_id=NULL
               WHERE submitted_bundle_id=?""",
            (bundle_id,),
        ).rowcount
        conn.execute("DELETE FROM contribution_batches WHERE bundle_id=?", (bundle_id,))
    return released


def _owns_repository(command: Any, repo: str) -> bool:
    """Whether the authenticated account owns the evidence repository."""
    login = command(["gh", "api", "user", "--jq", ".login"]).stdout.strip()
    return bool(login) and login.lower() == repo.split("/", 1)[0].lower()


def open_contribution_pr(
    contribution: Contribution | dict[str, Any],
    *,
    repo: str,
    exported_path: Path,
    run: Any = subprocess.run,
) -> str:
    """Submit one already-approved bundle through authenticated GitHub CLI."""
    if isinstance(contribution, dict):
        contribution = Contribution(contribution, ())
    bundle = contribution.bundle
    export_bundle(contribution, exported_path)

    def command(
        args: list[str], *, cwd: Path | None = None
    ) -> subprocess.CompletedProcess:
        try:
            return run(
                args,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            detail = getattr(error, "stderr", "") or str(error)
            raise RuntimeError(
                f"GitHub submission stopped; bundle remains at {exported_path}. "
                f"Resolve gh authentication/access and submit it manually, or run "
                f"`basemode contribute release {bundle['bundle_id']}` to contribute "
                f"those observations another time. {detail.strip()}"
            ) from error

    command(["gh", "auth", "status"])
    with tempfile.TemporaryDirectory(prefix="basemode-contribution-") as temporary:
        work = Path(temporary)
        if _owns_repository(command, repo):
            # GitHub refuses to let one account own both a parent and a fork,
            # so the repository's own owner contributes from a branch on it.
            command(["gh", "repo", "clone", repo, "--"], cwd=work)
        else:
            command(["gh", "repo", "fork", repo, "--clone", "--remote"], cwd=work)
        checkout = work / repo.rsplit("/", 1)[-1]
        branch = f"basemode-contribution-{bundle['bundle_id']}"
        command(["git", "switch", "-c", branch], cwd=checkout)
        end = datetime.fromisoformat(bundle["window_end"].replace("Z", "+00:00"))
        relative = Path(
            f"contributions/v1/{end:%Y}/{end:%m}/{bundle['bundle_id']}.json"
        )
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(exported_path, target)
        command(["git", "add", "--", str(relative)], cwd=checkout)
        command(
            ["git", "commit", "-m", f"Add contribution {bundle['bundle_id']}"],
            cwd=checkout,
        )
        command(["git", "push", "-u", "origin", branch], cwd=checkout)
        result = command(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                repo,
                "--title",
                f"Evidence contribution {bundle['bundle_id']}",
                "--body",
                "Aggregate-only Basemode contribution generated locally.",
                "--head",
                branch,
            ],
            cwd=checkout,
        )
    pr_url = result.stdout.strip().splitlines()[-1]
    with observations._db() as conn:
        conn.execute(
            """UPDATE contribution_batches SET status='submitted',pr_url=?
               WHERE bundle_id=?""",
            (pr_url, bundle["bundle_id"]),
        )
    return pr_url

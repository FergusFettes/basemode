"""Aggregate-only contribution bundles for the public evidence repository."""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from . import observations

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


def build_bundle(
    *,
    since: datetime,
    until: datetime,
    bundle_id: str | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build and validate one aggregate bundle without marking it exported."""
    start, end = _timestamp(since), _timestamp(until)
    if not observations._DB_FILE.exists():
        raise ValueError("no local observations")
    conn = sqlite3.connect(observations._DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        operations = conn.execute(
            """SELECT o.*,e.provider_route,e.provider_model_id
               FROM call_operations o JOIN model_endpoints e ON e.id=o.endpoint_id
               WHERE o.contribution_eligible=1 AND o.finished_at IS NOT NULL
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
            endpoint = (
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
    finally:
        conn.close()
    if not rows:
        raise ValueError("no contribution-eligible observations in window")
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
    return bundle


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
            ("latency_ms", "successful_operations"),
            ("ttft_ms", "successful_operations"),
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


def export_bundle(bundle: dict[str, Any], path: Path) -> Path:
    """Write an exact validated bundle and remember its exported window."""
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
    return path

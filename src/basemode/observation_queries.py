"""Read-only projections over the unified local observation ledger."""

from __future__ import annotations

import sqlite3
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

from . import observations
from .health_rules import RULES_VERSION, operational_status


def endpoint_health(model: str, *, days: int | None = None) -> dict[str, Any] | None:
    """Return operational endpoint health without exposing individual content."""
    return list_endpoint_health(days=days).get(model.lower())


def list_endpoint_health(*, days: int | None = None) -> dict[str, dict[str, Any]]:
    if not observations._DB_FILE.exists():
        return {}
    conn = sqlite3.connect(observations._DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        where = "WHERE o.finished_at IS NOT NULL"
        params: list[Any] = []
        if days is not None:
            cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            where += " AND o.started_at >= ?"
            params.append(cutoff)
        rows = conn.execute(
            f"""SELECT
                    o.*, e.provider_route, e.provider_model_id
                FROM call_operations o
                JOIN model_endpoints e ON e.id = o.endpoint_id
                {where}
                ORDER BY o.started_at""",
            params,
        ).fetchall()
        result: dict[str, dict[str, Any]] = {}
        for endpoint_id in sorted({int(row["endpoint_id"]) for row in rows}):
            endpoint_rows = [row for row in rows if row["endpoint_id"] == endpoint_id]
            model = _model_name(endpoint_rows[0])
            result[model] = _summarize(conn, endpoint_id, endpoint_rows)
        return result
    finally:
        conn.close()


def _summarize(
    conn: sqlite3.Connection,
    endpoint_id: int,
    rows: list[sqlite3.Row],
) -> dict[str, Any]:
    operation_ids = [int(row["id"]) for row in rows]
    placeholders = ",".join("?" for _ in operation_ids)
    attempts = conn.execute(
        f"SELECT * FROM call_attempts WHERE operation_id IN ({placeholders})",
        operation_ids,
    ).fetchall()
    attempts_by_operation: dict[int, list[sqlite3.Row]] = {}
    for attempt in attempts:
        attempts_by_operation.setdefault(int(attempt["operation_id"]), []).append(
            attempt
        )

    eligible_rows = [
        row
        for row in rows
        if any(
            bool(attempt["status_eligible"])
            for attempt in attempts_by_operation.get(int(row["id"]), [])
        )
    ]
    successes = [
        row
        for row in eligible_rows
        if row["logical_outcome"] in {"success", "cancelled"}
        and bool(row["returned_content"])
    ]
    eligible_attempts = [attempt for attempt in attempts if attempt["status_eligible"]]
    initial_attempts = [
        attempt for attempt in eligible_attempts if attempt["attempt_kind"] == "initial"
    ]
    recovered = [
        row
        for row in successes
        if len(attempts_by_operation.get(int(row["id"]), [])) > 1
    ]
    failures = Counter(
        str(attempt["failure_class"])
        for attempt in eligible_attempts
        if attempt["failure_class"]
    )
    transient_failures = sum(
        1
        for attempt in eligible_attempts
        if attempt["failure_transience"] == "transient"
    )
    persistent_failures = sum(
        1
        for attempt in eligible_attempts
        if attempt["failure_transience"] == "persistent"
    )
    account_failures = sum(
        1 for attempt in attempts if attempt["failure_attribution"] == "account"
    )
    last_outcome = eligible_rows[-1]["logical_outcome"] if eligible_rows else None
    operations = len(eligible_rows)
    successful_operations = len(successes)
    return {
        "operational_status": operational_status(
            operations=operations,
            successful_operations=successful_operations,
            transient_failures=transient_failures,
            persistent_failures=persistent_failures,
            account_failures=account_failures,
            last_outcome=last_outcome,
        ),
        "rules_version": RULES_VERSION,
        "operations": operations,
        "successful_operations": successful_operations,
        "logical_success_rate": (
            successful_operations / operations if operations else None
        ),
        "initial_attempts": len(initial_attempts),
        "successful_initial_attempts": sum(
            1 for attempt in initial_attempts if attempt["outcome"] == "success"
        ),
        "recovered_operations": len(recovered),
        "attempts": len(eligible_attempts),
        "failures": dict(sorted(failures.items())),
        "account_failures": account_failures,
        "source_counts": dict(Counter(str(row["source"]) for row in eligible_rows)),
        "window_start": eligible_rows[0]["started_at"] if eligible_rows else None,
        "window_end": eligible_rows[-1]["finished_at"] if eligible_rows else None,
        "last_successful_at": successes[-1]["finished_at"] if successes else None,
        "last_failed_at": next(
            (
                row["finished_at"]
                for row in reversed(eligible_rows)
                if row["logical_outcome"] == "failure"
            ),
            None,
        ),
    }


def _model_name(row: sqlite3.Row) -> str:
    route = str(row["provider_route"])
    model = str(row["provider_model_id"])
    return model if route == "unknown" else f"{route}/{model}"

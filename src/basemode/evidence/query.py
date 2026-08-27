"""Read path: derived status, recheck scheduling, and rating lookups."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ._util import _catalog_text_modality, _now, classify_text_endpoint
from .schema import connect


def current_status(
    *, conn: sqlite3.Connection | None = None
) -> dict[str, dict[str, Any]]:
    """Derive orthogonal current states; absence/staleness never becomes success."""
    own = conn is None
    db = conn or connect()
    rows = db.execute("""SELECT e.normalized_model_id, r.suite, a.outcome, a.failure_class,
      a.failure_transience, a.finished_at, a.run_id, a.attempt_number FROM verification_attempts a
      JOIN model_endpoints e ON e.id=a.endpoint_id JOIN verification_runs r ON r.id=a.run_id
      WHERE r.status='completed' AND e.text_eligible=1 AND a.status_eligible=1
      ORDER BY a.finished_at DESC, a.id DESC""").fetchall()
    catalogs = db.execute("""SELECT e.normalized_model_id,c.available,c.observed_at FROM catalog_observations c
      JOIN model_endpoints e ON e.id=c.endpoint_id WHERE e.text_eligible=1
      ORDER BY c.observed_at DESC,c.id DESC""").fetchall()
    result: dict[str, dict[str, Any]] = {}
    for row in catalogs:
        entry = result.setdefault(row["normalized_model_id"], {})
        entry.setdefault("available", bool(row["available"]))
        entry.setdefault("catalog_observed_at", row["observed_at"])
    for row in rows:
        entry = result.setdefault(row["normalized_model_id"], {})
        if "reachable" not in entry:
            entry.update(
                reachable=row["outcome"] == "success",
                last_checked_at=row["finished_at"],
                currently_broken=row["outcome"] != "success"
                and row["failure_transience"] == "durable",
                transient_failure=row["outcome"] != "success"
                and row["failure_transience"] == "suspected",
            )
    # A thorough run passes only when every logical probe has at least one
    # successful attempt. Failed self-healing steps remain evidence but do
    # not condemn a probe that subsequently recovered.
    thorough: dict[str, dict[str, dict[int, bool]]] = {}
    for row in rows:
        if row["suite"] != "thorough":
            continue
        model_runs = thorough.setdefault(row["normalized_model_id"], {})
        groups = model_runs.setdefault(row["run_id"], {})
        group = int(row["attempt_number"]) // 10
        groups[group] = groups.get(group, False) or row["outcome"] == "success"
    for model, runs in thorough.items():
        # Rows are newest-first, so insertion order gives the latest run.
        groups = next(iter(runs.values()))
        result.setdefault(model, {})["verified"] = bool(groups) and all(groups.values())
    for model, schedule in recheck_statuses(conn=db).items():
        result.setdefault(model, {}).update(schedule)
    if own:
        db.close()
    return result


def recheck_statuses(
    *, conn: sqlite3.Connection | None = None
) -> dict[str, dict[str, Any]]:
    """Return durable operational assessments and their next scheduled check."""
    own = conn is None
    db = conn or connect()
    rows = db.execute(
        """SELECT s.*,e.normalized_model_id,e.model_family_id FROM recheck_schedules s
        JOIN model_endpoints e ON e.id=s.endpoint_id WHERE e.text_eligible=1"""
    ).fetchall()
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        status = row["operational_status"]
        if (
            status == "persistent_operational"
            and row["failure_class"] == "provider_unavailable"
            and row["model_family_id"]
        ):
            alternative = db.execute(
                """SELECT 1 FROM verification_attempts a
                JOIN verification_runs r ON r.id=a.run_id
                JOIN model_endpoints e ON e.id=a.endpoint_id
                WHERE e.model_family_id=? AND e.id!=? AND a.outcome='success'
                  AND r.status='completed' AND a.status_eligible=1 LIMIT 1""",
                (row["model_family_id"], row["endpoint_id"]),
            ).fetchone()
            if alternative:
                status = "provider_route_unavailable"
        output[row["normalized_model_id"]] = {
            "operational_status": status,
            "recheck_failure_class": row["failure_class"],
            "consecutive_operational_failures": row["consecutive_failures"],
            "first_operational_failure_at": row["first_failure_at"],
            "last_operational_failure_at": row["last_failure_at"],
            "next_check_at": row["next_check_at"],
        }
    if own:
        db.close()
    return output


def transient_recheck_models(
    *, conn: sqlite3.Connection | None = None, now: str | None = None
) -> list[str]:
    """Return only endpoints whose durable backoff says they are due."""
    own = conn is None
    db = conn or connect()
    due = now or _now()
    rows = db.execute(
        """SELECT e.normalized_model_id FROM recheck_schedules s
        JOIN model_endpoints e ON e.id=s.endpoint_id
        WHERE s.next_check_at IS NOT NULL AND s.next_check_at<=?
          AND s.operational_status IN ('suspected_transient','persistent_operational')
          AND e.text_eligible=1 ORDER BY e.normalized_model_id""",
        (due,),
    ).fetchall()
    if own:
        db.close()
    return [str(row[0]) for row in rows]


def excluded_non_text_models(*, conn: sqlite3.Connection | None = None) -> set[str]:
    """Return endpoints durably excluded from text-generation consumers."""
    own = conn is None
    db = conn or connect()
    rows = db.execute(
        "SELECT normalized_model_id FROM model_endpoints WHERE text_eligible=0"
    ).fetchall()
    if own:
        db.close()
    return {row[0] for row in rows}


def enforce_text_only_and_supersede_obsolete_failures(
    *, conn: sqlite3.Connection | None = None
) -> dict[str, int]:
    """Idempotently clean derived status while retaining every raw observation.

    Non-text endpoints are excluded using provider modality first and conservative
    product-name evidence second. Historical ``invalid_request`` attempts are
    excluded from *current status* only where a later success demonstrates that
    the old request shape, rather than the endpoint, was at fault.
    """
    own = conn is None
    db = conn or connect()
    endpoints_changed = 0
    for row in db.execute(
        "SELECT id,normalized_model_id,modality,text_eligible,exclusion_reason FROM model_endpoints"
    ).fetchall():
        latest = db.execute(
            "SELECT metadata_json FROM catalog_observations WHERE endpoint_id=? "
            "ORDER BY observed_at DESC,id DESC LIMIT 1",
            (row["id"],),
        ).fetchone()
        projected_modality = (
            _catalog_text_modality(json.loads(latest[0])) if latest else None
        )
        modality = projected_modality or row["modality"]
        eligible, reason = classify_text_endpoint(row["normalized_model_id"], modality)
        authoritative_change = projected_modality is not None and (
            row["modality"] != projected_modality
            or bool(row["text_eligible"]) != eligible
            or row["exclusion_reason"] != reason
        )
        fallback_exclusion = not eligible and (
            row["text_eligible"] or row["exclusion_reason"] != reason
        )
        if authoritative_change or fallback_exclusion:
            db.execute(
                "UPDATE model_endpoints SET modality=?,text_eligible=?,exclusion_reason=? WHERE id=?",
                (modality, int(eligible), reason, row["id"]),
            )
            endpoints_changed += 1

    # A structured unsupported parameter/value rejection is evidence about
    # basemode's request shaping, not about endpoint health.
    compatibility_cursor = db.execute(
        """UPDATE verification_attempts
        SET status_eligible=0,
            status_exclusion_reason='basemode request-shaping incompatibility'
        WHERE status_eligible=1 AND failure_class='invalid_request'
          AND safe_error_code IN ('unsupported_parameter','unsupported_value')"""
    )
    compatibility_failures_changed = compatibility_cursor.rowcount

    # A later successful direct request is concrete evidence that an older
    # invalid request was about our then-current request shaping. Other failure
    # classes and invalid requests without recovery remain status-bearing.
    cursor = db.execute(
        """UPDATE verification_attempts AS failed
        SET status_eligible=0,
            status_exclusion_reason='superseded by later successful request'
        WHERE failed.status_eligible=1 AND failed.failure_class='invalid_request'
          AND EXISTS (
            SELECT 1 FROM verification_attempts AS passed
            WHERE passed.endpoint_id=failed.endpoint_id
              AND passed.outcome='success'
              AND (passed.finished_at>failed.finished_at
                   OR (passed.finished_at=failed.finished_at AND passed.id>failed.id))
          )"""
    )
    attempts_changed = cursor.rowcount
    if own:
        db.commit()
        db.close()
    return {
        "non_text_endpoints_excluded": endpoints_changed,
        "request_shaping_failures_superseded": compatibility_failures_changed,
        "obsolete_invalid_requests_superseded": attempts_changed,
    }


def get_model_rating(model: str) -> int | None:
    return list_model_ratings().get(model.lower())


def list_model_ratings() -> dict[str, int]:
    with connect() as db:
        rows = db.execute(
            """SELECT e.normalized_model_id,a.value_json FROM model_annotations a
            JOIN model_endpoints e ON e.id=a.endpoint_id WHERE a.kind='rating'
            ORDER BY a.created_at DESC,a.id DESC"""
        ).fetchall()
    result: dict[str, int] = {}
    seen: set[str] = set()
    for row in rows:
        model = row["normalized_model_id"]
        if model in seen:
            continue
        seen.add(model)
        value = json.loads(row["value_json"])
        if value in {-1, 1} and not isinstance(value, bool):
            result[model] = value
    return result

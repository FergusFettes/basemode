"""Write path: recording endpoints, runs, attempts, and annotations."""

from __future__ import annotations

import os
import sqlite3
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from ._util import _as_utc, _catalog_text_modality, _json, _now, classify_text_endpoint
from .schema import (
    ACCOUNT_FAILURES,
    PERSISTENT_RECHECK_DELAY,
    RECHECK_DELAYS,
    TRANSIENT_FAILURES,
    connect,
)


def ensure_endpoint(
    model: str, *, conn: sqlite3.Connection | None = None, **metadata: Any
) -> int:
    own = conn is None
    db = conn or connect()
    normalized = model.strip().lower()
    provider, _, provider_model = normalized.partition("/")
    if not provider_model:
        provider, provider_model = "unknown", provider
    now = _now()
    text_eligible, exclusion_reason = classify_text_endpoint(
        normalized, metadata.get("modality")
    )
    db.execute(
        """INSERT INTO model_endpoints(normalized_model_id,provider,provider_model_id,
        model_family_id,upstream_provider,display_name,modality,release_date,first_seen_at,last_seen_at,
        text_eligible,exclusion_reason)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(normalized_model_id) DO UPDATE SET
        last_seen_at=excluded.last_seen_at,
        modality=COALESCE(excluded.modality,model_endpoints.modality),
        release_date=COALESCE(excluded.release_date,model_endpoints.release_date),
        display_name=COALESCE(excluded.display_name,model_endpoints.display_name),
        text_eligible=CASE WHEN excluded.modality IS NOT NULL THEN excluded.text_eligible
          ELSE model_endpoints.text_eligible END,
        exclusion_reason=CASE WHEN excluded.modality IS NOT NULL THEN excluded.exclusion_reason
          ELSE model_endpoints.exclusion_reason END""",
        (
            normalized,
            provider,
            provider_model,
            metadata.get("model_family_id"),
            metadata.get("upstream_provider"),
            metadata.get("display_name"),
            metadata.get("modality"),
            metadata.get("release_date"),
            now,
            now,
            int(text_eligible),
            exclusion_reason,
        ),
    )
    endpoint_id = db.execute(
        "SELECT id FROM model_endpoints WHERE normalized_model_id=?", (normalized,)
    ).fetchone()[0]
    if own:
        db.commit()
        db.close()
    return endpoint_id


def record_catalog_observation(
    model: str,
    *,
    source: str,
    available: bool,
    observed_at: str | None = None,
    catalog_snapshot_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Record what one catalog source claimed without treating it as truth."""
    own = conn is None
    db = conn or connect()
    catalog_metadata = metadata or {}
    modality = _catalog_text_modality(catalog_metadata)
    endpoint = ensure_endpoint(
        model,
        conn=db,
        modality=modality,
        release_date=catalog_metadata.get("release_date"),
        display_name=catalog_metadata.get("display_name"),
    )
    cur = db.execute(
        """INSERT INTO catalog_observations(endpoint_id,observed_at,source,available,
        catalog_snapshot_id,metadata_json) VALUES(?,?,?,?,?,?)""",
        (
            endpoint,
            observed_at or _now(),
            source,
            int(available),
            catalog_snapshot_id,
            _json(catalog_metadata),
        ),
    )
    if own:
        db.commit()
        db.close()
    return int(cur.lastrowid)


def start_run(
    suite: str,
    *,
    configuration: Mapping[str, Any] | None = None,
    target_policy: Mapping[str, Any] | None = None,
    conn: sqlite3.Connection | None = None,
    **metadata: Any,
) -> str:
    own = conn is None
    db = conn or connect()
    run_id = str(uuid.uuid4())
    account = metadata.get("account_fingerprint") or os.environ.get(
        "BASEMODE_ACCOUNT_FINGERPRINT"
    )
    db.execute(
        """INSERT INTO verification_runs(id,suite,suite_version,started_at,basemode_version,
        git_commit,litellm_version,target_policy_json,configuration_json,catalog_snapshot_id,
        account_fingerprint,status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            run_id,
            suite,
            int(metadata.get("suite_version", 1)),
            _now(),
            metadata.get("basemode_version"),
            metadata.get("git_commit"),
            metadata.get("litellm_version"),
            _json(target_policy or {}),
            _json(configuration or {}),
            metadata.get("catalog_snapshot_id"),
            account,
            "running",
        ),
    )
    if own:
        db.commit()
        db.close()
    return run_id


def finish_run(
    run_id: str, status: str = "completed", *, conn: sqlite3.Connection | None = None
) -> None:
    own = conn is None
    db = conn or connect()
    db.execute(
        "UPDATE verification_runs SET completed_at=?, status=? WHERE id=?",
        (_now(), status, run_id),
    )
    if status == "completed":
        for row in db.execute(
            """SELECT endpoint_id,outcome,failure_class,failure_transience,
            safe_error_code,coalesce(finished_at,started_at) observed_at
            FROM verification_attempts WHERE run_id=? ORDER BY observed_at,id""",
            (run_id,),
        ):
            _update_recheck_schedule(
                db,
                endpoint_id=row["endpoint_id"],
                outcome=row["outcome"],
                failure_class=row["failure_class"],
                failure_transience=row["failure_transience"],
                safe_error_code=row["safe_error_code"],
                observed_at=row["observed_at"],
                run_id=run_id,
            )
    if own:
        db.commit()
        db.close()


def resume_run(run_id: str, *, conn: sqlite3.Connection | None = None) -> sqlite3.Row:
    """Mark an interrupted or limited run active again and return its metadata."""
    own = conn is None
    db = conn or connect()
    row = db.execute("SELECT * FROM verification_runs WHERE id=?", (run_id,)).fetchone()
    if row is None:
        if own:
            db.close()
        raise ValueError(f"unknown verification run: {run_id}")
    if row["status"] == "completed":
        if own:
            db.close()
        raise ValueError(f"verification run is already completed: {run_id}")
    db.execute(
        "UPDATE verification_runs SET completed_at=NULL,status='running' WHERE id=?",
        (run_id,),
    )
    if own:
        db.commit()
        db.close()
    return row


def record_attempt(
    run_id: str,
    model: str,
    *,
    probe_kind: str,
    attempt_number: int,
    outcome: str,
    conn: sqlite3.Connection | None = None,
    **values: Any,
) -> int:
    own = conn is None
    db = conn or connect()
    endpoint_id = ensure_endpoint(model, conn=db)
    failure = values.get("failure_class")
    transience = values.get("failure_transience")
    if failure and transience is None:
        transience = "suspected" if failure in TRANSIENT_FAILURES else "durable"
    columns = (
        "run_id,endpoint_id,probe_kind,attempt_number,started_at,finished_at,prompt_method,"
        "request_params_json,compatibility_actions_json,outcome,failure_class,failure_transience,"
        "http_status,safe_error_code,safe_error_parameter,latency_ms,ttft_ms,generation_ms,"
        "prompt_tokens,completion_tokens,reasoning_tokens,output_characters,output_tokens_per_second,"
        "cost_usd,cost_source,output_fingerprint,status_eligible,status_exclusion_reason"
    )
    params = (
        run_id,
        endpoint_id,
        probe_kind,
        attempt_number,
        values.get("started_at", _now()),
        values.get("finished_at", _now()),
        values.get("prompt_method"),
        _json(values.get("request_params", {})),
        _json(values.get("compatibility_actions", [])),
        outcome,
        failure,
        transience,
        values.get("http_status"),
        values.get("safe_error_code"),
        values.get("safe_error_parameter"),
        values.get("latency_ms"),
        values.get("ttft_ms"),
        values.get("generation_ms"),
        values.get("prompt_tokens"),
        values.get("completion_tokens"),
        values.get("reasoning_tokens"),
        values.get("output_characters"),
        values.get("output_tokens_per_second"),
        values.get("cost_usd"),
        values.get("cost_source"),
        values.get("output_fingerprint"),
        int(values.get("status_eligible", True)),
        values.get("status_exclusion_reason"),
    )
    cur = db.execute(
        f"INSERT INTO verification_attempts({columns}) VALUES({','.join('?' for _ in params)})",
        params,
    )
    run_status = db.execute(
        "SELECT status FROM verification_runs WHERE id=?", (run_id,)
    ).fetchone()[0]
    if run_status == "completed":
        _update_recheck_schedule(
            db,
            endpoint_id=endpoint_id,
            outcome=outcome,
            failure_class=failure,
            failure_transience=transience,
            safe_error_code=values.get("safe_error_code"),
            observed_at=values.get("finished_at", _now()),
            run_id=run_id,
        )
    if own:
        db.commit()
        db.close()
    return int(cur.lastrowid)


def _update_recheck_schedule(
    db: sqlite3.Connection,
    *,
    endpoint_id: int,
    outcome: str,
    failure_class: str | None,
    failure_transience: str | None,
    safe_error_code: str | None,
    observed_at: str,
    run_id: str,
) -> None:
    previous = db.execute(
        "SELECT * FROM recheck_schedules WHERE endpoint_id=?", (endpoint_id,)
    ).fetchone()
    if outcome == "success":
        if previous is not None:
            db.execute(
                """UPDATE recheck_schedules SET next_check_at=NULL,
                operational_status='recovered',updated_at=? WHERE endpoint_id=?""",
                (observed_at, endpoint_id),
            )
        return
    if not failure_class:
        return
    is_transient = (
        failure_transience == "suspected" or failure_class in TRANSIENT_FAILURES
    )
    normalized_code = (safe_error_code or "").lower()
    is_account = failure_class in ACCOUNT_FAILURES or any(
        marker in normalized_code
        for marker in ("quota", "credit", "billing", "insufficient_funds")
    )
    if not (is_transient or is_account):
        return
    was_recovered = (
        previous is not None and previous["operational_status"] == "recovered"
    )
    is_separate_run = previous is None or previous["last_run_id"] != run_id
    consecutive = (
        1
        if was_recovered
        else int(previous["consecutive_failures"]) + int(is_separate_run)
        if previous
        else 1
    )
    first = str(previous["first_failure_at"]) if previous else observed_at
    if is_account:
        status = "account_limited"
        next_check = None
    elif consecutive >= 3:
        status = "persistent_operational"
        next_check = _as_utc(observed_at) + PERSISTENT_RECHECK_DELAY
    else:
        status = "suspected_transient"
        next_check = _as_utc(observed_at) + RECHECK_DELAYS[consecutive - 1]
    db.execute(
        """INSERT INTO recheck_schedules(endpoint_id,failure_class,first_failure_at,
        last_failure_at,consecutive_failures,next_check_at,operational_status,updated_at,last_run_id)
        VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(endpoint_id) DO UPDATE SET
        failure_class=excluded.failure_class,last_failure_at=excluded.last_failure_at,
        consecutive_failures=excluded.consecutive_failures,next_check_at=excluded.next_check_at,
        operational_status=excluded.operational_status,updated_at=excluded.updated_at,
        last_run_id=excluded.last_run_id""",
        (
            endpoint_id,
            failure_class,
            first,
            observed_at,
            consecutive,
            next_check.isoformat() if next_check else None,
            status,
            observed_at,
            run_id,
        ),
    )


def record_probe_result(
    attempt_id: int,
    metric: str,
    value: object,
    passed: bool | None = None,
    *,
    conn: sqlite3.Connection | None = None,
) -> None:
    own = conn is None
    db = conn or connect()
    db.execute(
        "INSERT OR REPLACE INTO probe_results(attempt_id,metric,value_json,passed) VALUES(?,?,?,?)",
        (attempt_id, metric, _json(value), None if passed is None else int(passed)),
    )
    if own:
        db.commit()
        db.close()


def add_annotation(
    model: str,
    kind: str,
    value: object,
    source: str,
    *,
    conn: sqlite3.Connection | None = None,
) -> int:
    own = conn is None
    db = conn or connect()
    endpoint = ensure_endpoint(model, conn=db)
    cur = db.execute(
        "INSERT INTO model_annotations(endpoint_id,kind,value_json,source,created_at) VALUES(?,?,?,?,?)",
        (endpoint, kind, _json(value), source, _now()),
    )
    if own:
        db.commit()
        db.close()
    return int(cur.lastrowid)


def publish_corpus_observations(
    observations: Iterable[Mapping[str, Any]],
    *,
    source_instance: str,
    window_start: str | None = None,
    window_end: str | None = None,
    loom_version: str | None = None,
    basemode_version: str | None = None,
) -> int:
    """Publish privacy-preserving corpus aggregates (never prompts or text)."""
    count = 0
    with connect() as db:
        db.execute(
            """DELETE FROM corpus_observations WHERE source_instance=?
            AND window_start IS ? AND window_end IS ?""",
            (source_instance, window_start, window_end),
        )
        for observation in observations:
            model = observation.get("model") or observation.get("endpoint_id")
            if not isinstance(model, str) or not model:
                raise ValueError(
                    "each corpus observation requires a model or endpoint_id"
                )
            endpoint = ensure_endpoint(model, conn=db)
            db.execute(
                """INSERT INTO corpus_observations(endpoint_id,source_instance,window_start,window_end,
                basemode_version,loom_version,prompt_method,generated_count,successful_count,
                flagged_count,corrected_count,open_issue_count,depth_bucket,issue_kind,
                timing_summary_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    endpoint,
                    source_instance,
                    window_start,
                    window_end,
                    basemode_version,
                    loom_version,
                    observation.get("prompt_method"),
                    int(observation.get("generated_count", 0)),
                    observation.get("successful_count"),
                    observation.get("flagged_count"),
                    observation.get("corrected_count"),
                    observation.get("open_issue_count"),
                    observation.get("depth_bucket"),
                    observation.get("issue_kind"),
                    _json(observation.get("timing_summary", {})),
                    _now(),
                ),
            )
            count += 1
    return count


def set_model_rating(model: str, rating: int | None) -> None:
    """Set the current evidence-backed thumb (1/-1), or clear it."""
    if rating not in {None, -1, 1} or isinstance(rating, bool):
        raise ValueError("rating must be 1, -1, or None")
    with connect() as db:
        endpoint = ensure_endpoint(model, conn=db)
        prior = db.execute(
            "SELECT id FROM model_annotations WHERE endpoint_id=? AND kind='rating' ORDER BY created_at DESC,id DESC LIMIT 1",
            (endpoint,),
        ).fetchone()
        db.execute(
            "INSERT INTO model_annotations(endpoint_id,kind,value_json,source,created_at,supersedes_id) VALUES(?,?,?,?,?,?)",
            (
                endpoint,
                "rating",
                _json(rating),
                "user",
                _now(),
                prior[0] if prior else None,
            ),
        )

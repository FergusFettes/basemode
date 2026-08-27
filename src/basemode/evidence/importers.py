"""Idempotent importers that fold external and legacy evidence sources in."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from ._util import _json, _now
from .schema import connect
from .store import ensure_endpoint, record_attempt, record_catalog_observation


def import_sweep_jsonl(
    path: Path, *, suite: str = "imported", conn: sqlite3.Connection | None = None
) -> str:
    """Import generic JSONL idempotently via a content-derived run ID."""
    own = conn is None
    db = conn or connect()
    content = path.read_bytes()
    source_observed_at = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
    run_id = "import-" + hashlib.sha256(content).hexdigest()[:24]
    if db.execute("SELECT 1 FROM verification_runs WHERE id=?", (run_id,)).fetchone():
        if own:
            db.close()
        return run_id
    db.execute(
        "INSERT INTO verification_runs(id,suite,suite_version,started_at,completed_at,target_policy_json,configuration_json,status) VALUES(?,?,?,?,?,?,?,?)",
        (
            run_id,
            suite,
            1,
            _now(),
            _now(),
            "{}",
            _json({"source": str(path)}),
            "completed",
        ),
    )
    counts: dict[tuple[str, str], int] = {}
    for line in content.decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        model = row.get("model") or row.get("model_id")
        if not model:
            continue
        probe = row.get("probe_kind", "imported")
        key = (model, probe)
        counts[key] = counts.get(key, 0) + 1
        ok = bool(
            row.get(
                "ok",
                row.get(
                    "success",
                    row.get("outcome") == "success" or row.get("status") == "ok",
                ),
            )
        )
        raw_status = row.get("http_status", row.get("status"))
        http_status = (
            raw_status
            if isinstance(raw_status, int) and not isinstance(raw_status, bool)
            else None
        )
        failure_class = row.get("category") or row.get("failure_class")
        if not ok and failure_class is None and isinstance(raw_status, str):
            failure_class = raw_status
        observed_at = (
            row.get("finished_at")
            or row.get("at")
            or row.get("run_at")
            or source_observed_at
        )
        record_attempt(
            run_id,
            model,
            probe_kind=probe,
            attempt_number=row.get("attempt_number", counts[key]),
            outcome="success" if ok else "failure",
            conn=db,
            started_at=row.get("started_at") or observed_at,
            finished_at=observed_at,
            failure_class=failure_class,
            http_status=http_status,
            safe_error_code=row.get("code") or row.get("safe_error_code"),
            safe_error_parameter=row.get("param") or row.get("safe_error_parameter"),
            latency_ms=row.get("latency_ms")
            or (row.get("elapsed_s") and row["elapsed_s"] * 1000)
            or (row.get("elapsed_seconds") and row["elapsed_seconds"] * 1000)
            or (row.get("latency_s") and row["latency_s"] * 1000),
            cost_usd=row.get("cost_usd") or row.get("estimated_cost_usd"),
            request_params=row.get("request_params", {})
            or {"max_tokens": row.get("requested_max_tokens")},
            compatibility_actions=row.get("compatibility_actions", []),
        )
    db.commit()
    if own:
        db.close()
    return run_id


def import_provider_health_jsonl(
    path: Path, *, conn: sqlite3.Connection | None = None
) -> str:
    """Import the repository's scheduled provider-health history."""
    return import_sweep_jsonl(path, suite="provider-health", conn=conn)


def import_rejected_registry(
    path: Path, *, conn: sqlite3.Connection | None = None
) -> str:
    """Import rejected models as historical, safely structured evidence."""
    payload = json.loads(path.read_text())
    models = payload.get("models", []) if isinstance(payload, dict) else []
    run_id = "rejected-" + hashlib.sha256(path.read_bytes()).hexdigest()[:24]
    own = conn is None
    db = conn or connect()
    if db.execute("SELECT 1 FROM verification_runs WHERE id=?", (run_id,)).fetchone():
        if own:
            db.close()
        return run_id
    started = min((row.get("checked_at_utc", _now()) for row in models), default=_now())
    db.execute(
        """INSERT INTO verification_runs(id,suite,suite_version,started_at,completed_at,
        target_policy_json,configuration_json,status) VALUES(?,?,?,?,?,?,?,?)""",
        (
            run_id,
            "legacy-rejected",
            1,
            started,
            _now(),
            "{}",
            _json({"source": str(path)}),
            "completed",
        ),
    )
    for number, row in enumerate(models, 1):
        strategies = sorted((row.get("attempted_strategies") or {}).keys())
        record_attempt(
            run_id,
            row["model"],
            probe_kind="legacy-rejected",
            attempt_number=number,
            outcome="failure",
            failure_class="invalid_request",
            failure_transience="unknown",
            compatibility_actions=[
                {"action": "try_strategy", "strategy": strategy}
                for strategy in strategies
            ],
            started_at=row.get("checked_at_utc"),
            finished_at=row.get("checked_at_utc"),
            conn=db,
        )
    db.commit()
    if own:
        db.close()
    return run_id


def import_live_catalog_cache(
    path: Path, *, conn: sqlite3.Connection | None = None
) -> int:
    """Import a packaged live-provider catalog snapshot idempotently."""
    payload = json.loads(path.read_text())
    providers = payload.get("providers", {}) if isinstance(payload, dict) else {}
    snapshot_id = "live-" + hashlib.sha256(path.read_bytes()).hexdigest()[:24]
    own = conn is None
    db = conn or connect()
    if db.execute(
        "SELECT 1 FROM catalog_observations WHERE catalog_snapshot_id=? LIMIT 1",
        (snapshot_id,),
    ).fetchone():
        if own:
            db.close()
        return 0
    count = 0
    observed_at = payload.get("generated_at_utc") or payload.get("generated_at")
    for provider, provider_data in providers.items():
        models = provider_data.get("models", {})
        if isinstance(models, list):
            models = {model: None for model in models}
        for provider_model_id, model_data in models.items():
            metadata = (
                dict(model_data)
                if isinstance(model_data, dict)
                else {"release_date": model_data}
            )
            metadata["reliable_dates"] = provider_data.get("reliable_dates")
            model = f"{provider}/{provider_model_id}"
            record_catalog_observation(
                model,
                source="provider_api_cache",
                available=True,
                observed_at=observed_at,
                catalog_snapshot_id=snapshot_id,
                metadata=metadata,
                conn=db,
            )
            count += 1
    db.commit()
    if own:
        db.close()
    return count


def import_verified_registry(
    path: Path, *, conn: sqlite3.Connection | None = None
) -> int:
    """Import curated registry intent as annotations, not measured success."""
    payload = json.loads(path.read_text())
    models = payload.get("models", []) if isinstance(payload, dict) else []
    source = "verified_registry:" + hashlib.sha256(path.read_bytes()).hexdigest()[:24]
    own = conn is None
    db = conn or connect()
    count = 0
    for row in models:
        model = row.get("model")
        if not isinstance(model, str) or not model:
            continue
        endpoint = ensure_endpoint(model, conn=db)
        value = {
            key: row[key]
            for key in (
                "prompt_method",
                "quirks",
                "known_issues",
                "openrouter_id",
                "pricing_url",
            )
            if key in row
        }
        encoded = _json(value)
        if db.execute(
            """SELECT 1 FROM model_annotations WHERE endpoint_id=?
            AND kind='registry_intent' AND value_json=? AND source=?""",
            (endpoint, encoded, source),
        ).fetchone():
            continue
        db.execute(
            """INSERT INTO model_annotations(endpoint_id,kind,value_json,source,created_at)
            VALUES(?,?,?,?,?)""",
            (endpoint, "registry_intent", encoded, source, _now()),
        )
        count += 1
    db.commit()
    if own:
        db.close()
    return count


def import_annotations(
    ratings: Mapping[str, int], *, conn: sqlite3.Connection | None = None
) -> int:
    own = conn is None
    db = conn or connect()
    count = 0
    for model, rating in ratings.items():
        endpoint = ensure_endpoint(model, conn=db)
        exists = db.execute(
            "SELECT 1 FROM model_annotations WHERE endpoint_id=? AND kind='rating' AND value_json=? AND source='auth.json'",
            (endpoint, _json(rating)),
        ).fetchone()
        if not exists:
            db.execute(
                "INSERT INTO model_annotations(endpoint_id,kind,value_json,source,created_at) VALUES(?,?,?,?,?)",
                (endpoint, "rating", _json(rating), "auth.json", _now()),
            )
            count += 1
    db.commit()
    if own:
        db.close()
    return count

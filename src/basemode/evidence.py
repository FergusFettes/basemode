"""Durable, shared evidence about model endpoints.

Unlike :mod:`basemode.health`, this database is an append-only experimental
record.  Verification evidence is never aged out.  Applications such as Loom
may publish aggregate corpus observations here without exposing their text or
tree structure.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 4
_DB_FILE = Path.home() / ".local" / "share" / "basemode" / "model_evidence.sqlite"
TRANSIENT_FAILURES = frozenset(
    {"rate_limit", "timeout", "provider_unavailable", "network"}
)
ACCOUNT_FAILURES = frozenset({"authentication", "quota"})
RECHECK_DELAYS = (timedelta(minutes=15), timedelta(hours=2), timedelta(days=1))
PERSISTENT_RECHECK_DELAY = timedelta(days=7)

# These are deliberately conservative. Unknown endpoints remain eligible: many
# provider chat catalogs have no modality field at all. We only exclude a model
# when provider metadata or its product name makes the non-text purpose clear.
TEXT_MODALITIES = frozenset({"text", "chat", "completion", "responses", "language"})
NON_TEXT_MODALITIES = frozenset(
    {
        "audio",
        "audio_generation",
        "embedding",
        "embeddings",
        "image",
        "image_generation",
        "moderation",
        "rerank",
        "speech",
        "stt",
        "tts",
        "transcription",
        "video",
        "video_generation",
    }
)
_NON_TEXT_NAME_RE = re.compile(
    r"(?:^|[/_.:-])(?:audio|bge|bria|chatterbox|clip-vit|content-safety|csm|dall-e|"
    r"diariz\w*|embedding(?:gemma)?|embed|e5|flux|gte|higgsaudio|ideogram|image|"
    r"imagen|i2v|kokoro|llama-guard|llama-prompt-guard|lyria|moderation|"
    r"nano-banana|ocr|orpheus|pixverse|r2v|realtime|rerank|safeguard|seedance|"
    r"seedream|sentence-transformers|sora|speech|sdxl|stable-diffusion|t2v|text2vec|"
    r"transcrib\w*|tts|veo|video|vidu|wan|whisper)"
    r"(?:$|[/_.:-])",
    re.IGNORECASE,
)
_NON_GENERATION_NAME_RE = re.compile(
    r"(?:^|[/_.:-])(?:content-safety|llama-guard|llama-prompt-guard|moderation|"
    r"rerank|safeguard)(?:$|[/_.:-])",
    re.IGNORECASE,
)


def classify_text_endpoint(
    model: str, modality: str | None = None
) -> tuple[bool, str | None]:
    """Return text eligibility and a durable, human-readable exclusion reason."""
    specialized = _NON_GENERATION_NAME_RE.search(model)
    if specialized:
        family = specialized.group(0).strip("/_.:-").lower()
        return False, f"non-generation model family: {family}"
    normalized_modality = (modality or "").strip().lower()
    if normalized_modality in NON_TEXT_MODALITIES:
        return False, f"provider modality: {normalized_modality}"
    if normalized_modality in TEXT_MODALITIES:
        return True, None
    match = _NON_TEXT_NAME_RE.search(model)
    if match:
        return False, f"non-text model family: {match.group(0).strip('/_.:-').lower()}"
    return True, None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def connect(path: Path | None = None) -> sqlite3.Connection:
    target = path or _DB_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"evidence database schema {version} is newer than supported {SCHEMA_VERSION}"
        )
    if version == 0:
        conn.executescript(
            """
            CREATE TABLE model_endpoints (
              id INTEGER PRIMARY KEY, normalized_model_id TEXT NOT NULL UNIQUE,
              provider TEXT NOT NULL, provider_model_id TEXT NOT NULL,
              model_family_id TEXT, upstream_provider TEXT, display_name TEXT,
              modality TEXT, release_date TEXT, first_seen_at TEXT NOT NULL,
              last_seen_at TEXT NOT NULL
            );
            CREATE TABLE model_aliases (
              alias TEXT PRIMARY KEY, endpoint_id INTEGER NOT NULL REFERENCES model_endpoints(id)
            );
            CREATE TABLE catalog_observations (
              id INTEGER PRIMARY KEY, endpoint_id INTEGER NOT NULL REFERENCES model_endpoints(id),
              observed_at TEXT NOT NULL, source TEXT NOT NULL, available INTEGER NOT NULL,
              catalog_snapshot_id TEXT, metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE verification_runs (
              id TEXT PRIMARY KEY, suite TEXT NOT NULL, suite_version INTEGER NOT NULL,
              started_at TEXT NOT NULL, completed_at TEXT, basemode_version TEXT,
              git_commit TEXT, litellm_version TEXT, target_policy_json TEXT NOT NULL,
              configuration_json TEXT NOT NULL, catalog_snapshot_id TEXT,
              account_fingerprint TEXT, status TEXT NOT NULL
            );
            CREATE TABLE verification_attempts (
              id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES verification_runs(id),
              endpoint_id INTEGER NOT NULL REFERENCES model_endpoints(id), probe_kind TEXT NOT NULL,
              attempt_number INTEGER NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
              prompt_method TEXT, request_params_json TEXT NOT NULL DEFAULT '{}',
              compatibility_actions_json TEXT NOT NULL DEFAULT '[]', outcome TEXT NOT NULL,
              failure_class TEXT, failure_transience TEXT, http_status INTEGER,
              safe_error_code TEXT, safe_error_parameter TEXT, latency_ms REAL, ttft_ms REAL,
              generation_ms REAL, prompt_tokens INTEGER, completion_tokens INTEGER,
              reasoning_tokens INTEGER, output_characters INTEGER, output_tokens_per_second REAL,
              cost_usd REAL, cost_source TEXT, output_fingerprint TEXT,
              UNIQUE(run_id, endpoint_id, probe_kind, attempt_number)
            );
            CREATE TABLE probe_results (
              id INTEGER PRIMARY KEY, attempt_id INTEGER NOT NULL REFERENCES verification_attempts(id),
              metric TEXT NOT NULL, value_json TEXT NOT NULL, passed INTEGER,
              UNIQUE(attempt_id, metric)
            );
            CREATE TABLE model_annotations (
              id INTEGER PRIMARY KEY, endpoint_id INTEGER NOT NULL REFERENCES model_endpoints(id),
              kind TEXT NOT NULL, value_json TEXT NOT NULL, source TEXT NOT NULL,
              created_at TEXT NOT NULL, supersedes_id INTEGER REFERENCES model_annotations(id)
            );
            CREATE TABLE corpus_observations (
              id INTEGER PRIMARY KEY, endpoint_id INTEGER NOT NULL REFERENCES model_endpoints(id),
              source_instance TEXT NOT NULL, window_start TEXT, window_end TEXT,
              basemode_version TEXT, loom_version TEXT, prompt_method TEXT,
              generated_count INTEGER NOT NULL, successful_count INTEGER,
              flagged_count INTEGER, corrected_count INTEGER, open_issue_count INTEGER,
              depth_bucket TEXT, issue_kind TEXT, timing_summary_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL
            );
            CREATE INDEX verification_attempt_endpoint ON verification_attempts(endpoint_id, finished_at);
            CREATE INDEX verification_attempt_failure ON verification_attempts(failure_class, finished_at);
            CREATE INDEX catalog_endpoint_time ON catalog_observations(endpoint_id, observed_at);
            PRAGMA user_version=1;
            """
        )
        conn.commit()
        version = 1
    if version == 1:
        # A publication replaces the same Loom source/window atomically (see
        # publish_corpus_observations), so repeated scheduled exports cannot
        # inflate aggregates.
        conn.execute("PRAGMA user_version=2")
        conn.commit()
        version = 2
    if version == 2:
        conn.executescript(
            """
            ALTER TABLE model_endpoints ADD COLUMN text_eligible INTEGER NOT NULL DEFAULT 1;
            ALTER TABLE model_endpoints ADD COLUMN exclusion_reason TEXT;
            ALTER TABLE verification_attempts ADD COLUMN status_eligible INTEGER NOT NULL DEFAULT 1;
            ALTER TABLE verification_attempts ADD COLUMN status_exclusion_reason TEXT;
            PRAGMA user_version=3;
            """
        )
        conn.commit()
        version = 3
    if version == 3:
        conn.executescript(
            """
            CREATE TABLE recheck_schedules (
              endpoint_id INTEGER PRIMARY KEY REFERENCES model_endpoints(id),
              failure_class TEXT NOT NULL, first_failure_at TEXT NOT NULL,
              last_failure_at TEXT NOT NULL, consecutive_failures INTEGER NOT NULL,
              next_check_at TEXT, operational_status TEXT NOT NULL,
              updated_at TEXT NOT NULL, last_run_id TEXT
            );
            CREATE INDEX recheck_schedule_due ON recheck_schedules(next_check_at);
            """
        )
        # Preserve the existing transient queue when upgrading a v3 database.
        # Replay completed observations chronologically; the scheduler counts
        # distinct run IDs, so self-healing attempts in one run remain one
        # operational observation.
        for row in conn.execute(
            """SELECT a.endpoint_id,a.outcome,a.failure_class,a.failure_transience,
            a.safe_error_code,coalesce(a.finished_at,a.started_at) observed_at,a.run_id
            FROM verification_attempts a JOIN verification_runs r ON r.id=a.run_id
            WHERE r.status='completed' ORDER BY observed_at,a.id"""
        ):
            _update_recheck_schedule(
                conn,
                endpoint_id=row["endpoint_id"],
                outcome=row["outcome"],
                failure_class=row["failure_class"],
                failure_transience=row["failure_transience"],
                safe_error_code=row["safe_error_code"],
                observed_at=row["observed_at"],
                run_id=row["run_id"],
            )
        conn.execute("PRAGMA user_version=4")
        conn.commit()


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


def _catalog_text_modality(metadata: Mapping[str, Any]) -> str | None:
    """Project rich catalog capabilities onto text eligibility.

    Output capability is intentionally decisive. Image input with text output
    is a valid text-generation endpoint; image-only output is not.
    """
    outputs = {
        str(value).strip().lower()
        for value in metadata.get("output_modalities", [])
        if isinstance(value, str)
    }
    if outputs:
        known_non_text = outputs & NON_TEXT_MODALITIES
        if known_non_text:
            return sorted(known_non_text)[0]
        if outputs & TEXT_MODALITIES:
            return "text"
    methods = {
        str(value).strip().lower()
        for value in metadata.get("supported_methods", [])
        if isinstance(value, str)
    }
    if "generatecontent" in methods:
        return "text"
    provider_type = metadata.get("provider_type")
    if isinstance(provider_type, str):
        normalized_type = provider_type.strip().lower()
        if normalized_type in TEXT_MODALITIES | NON_TEXT_MODALITIES:
            return normalized_type
    value = metadata.get("modality") or metadata.get("mode")
    return str(value) if isinstance(value, str) else None


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


def _as_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


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


def import_health_sqlite(
    path: Path, *, conn: sqlite3.Connection | None = None
) -> str | None:
    if not path.exists():
        return None
    source = sqlite3.connect(path)
    source.row_factory = sqlite3.Row
    columns = {r[1] for r in source.execute("PRAGMA table_info(model_events)")}
    if not columns:
        source.close()
        return None
    rows = source.execute(
        "SELECT * FROM model_events WHERE source='verification'"
    ).fetchall()
    source.close()
    if not rows:
        return None
    own = conn is None
    db = conn or connect()
    signature = hashlib.sha256(
        (str(path.resolve()) + str(path.stat().st_mtime_ns)).encode()
    ).hexdigest()[:24]
    run_id = "health-" + signature
    if db.execute("SELECT 1 FROM verification_runs WHERE id=?", (run_id,)).fetchone():
        return run_id
    db.execute(
        "INSERT INTO verification_runs(id,suite,suite_version,started_at,completed_at,target_policy_json,configuration_json,status) VALUES(?,?,?,?,?,?,?,?)",
        (
            run_id,
            "legacy-health",
            1,
            _now(),
            _now(),
            "{}",
            _json({"source": str(path)}),
            "completed",
        ),
    )
    for number, row in enumerate(rows, 1):
        d = dict(row)
        observed_at = d.get("at") or d.get("timestamp") or _now()
        record_attempt(
            run_id,
            d["model"],
            probe_kind="legacy",
            attempt_number=number,
            outcome="success" if d["ok"] else "failure",
            conn=db,
            started_at=observed_at,
            finished_at=observed_at,
            failure_class=d.get("category"),
            http_status=d.get("status"),
            safe_error_code=d.get("error_code"),
            safe_error_parameter=d.get("error_param"),
            cost_usd=d.get("cost_usd"),
        )
    db.commit()
    if own:
        db.close()
    return run_id


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

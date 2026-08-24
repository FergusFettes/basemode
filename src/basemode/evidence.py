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
import sqlite3
import uuid
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
_DB_FILE = Path.home() / ".local" / "share" / "basemode" / "model_evidence.sqlite"
TRANSIENT_FAILURES = frozenset(
    {"rate_limit", "timeout", "provider_unavailable", "network"}
)


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
    db.execute(
        """INSERT INTO model_endpoints(normalized_model_id,provider,provider_model_id,
        model_family_id,upstream_provider,display_name,modality,release_date,first_seen_at,last_seen_at)
        VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(normalized_model_id) DO UPDATE SET
        last_seen_at=excluded.last_seen_at""",
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
    endpoint = ensure_endpoint(model, conn=db)
    cur = db.execute(
        """INSERT INTO catalog_observations(endpoint_id,observed_at,source,available,
        catalog_snapshot_id,metadata_json) VALUES(?,?,?,?,?,?)""",
        (
            endpoint,
            observed_at or _now(),
            source,
            int(available),
            catalog_snapshot_id,
            _json(metadata or {}),
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
    if own:
        db.commit()
        db.close()


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
        "cost_usd,cost_source,output_fingerprint"
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
    )
    cur = db.execute(
        f"INSERT INTO verification_attempts({columns}) VALUES({','.join('?' for _ in params)})",
        params,
    )
    if own:
        db.commit()
        db.close()
    return int(cur.lastrowid)


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
      WHERE r.status='completed' ORDER BY a.finished_at DESC, a.id DESC""").fetchall()
    catalogs = db.execute("""SELECT e.normalized_model_id,c.available,c.observed_at FROM catalog_observations c
      JOIN model_endpoints e ON e.id=c.endpoint_id ORDER BY c.observed_at DESC,c.id DESC""").fetchall()
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
    if own:
        db.close()
    return result


def transient_recheck_models(*, conn: sqlite3.Connection | None = None) -> list[str]:
    return sorted(
        m
        for m, status in current_status(conn=conn).items()
        if status.get("transient_failure")
    )


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
        record_attempt(
            run_id,
            model,
            probe_kind=probe,
            attempt_number=row.get("attempt_number", counts[key]),
            outcome="success" if ok else "failure",
            conn=db,
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

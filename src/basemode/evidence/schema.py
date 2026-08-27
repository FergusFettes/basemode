"""Connection setup and the durable SQLite schema for evidence storage."""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from pathlib import Path

SCHEMA_VERSION = 4
_DB_FILE = Path.home() / ".local" / "share" / "basemode" / "model_evidence.sqlite"
TRANSIENT_FAILURES = frozenset(
    {"rate_limit", "timeout", "provider_unavailable", "network"}
)
ACCOUNT_FAILURES = frozenset({"authentication", "quota"})
RECHECK_DELAYS = (timedelta(minutes=15), timedelta(hours=2), timedelta(days=1))
PERSISTENT_RECHECK_DELAY = timedelta(days=7)


def connect(path: Path | None = None) -> sqlite3.Connection:
    # Fetched from the package namespace (rather than the module-level
    # default above) so that tests can monkeypatch
    # ``basemode.evidence._DB_FILE`` and have it take effect here.
    from . import _DB_FILE as _current_db_file

    target = path or _current_db_file
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    # Imported lazily to avoid a schema<->store import cycle: replaying the
    # v3->v4 upgrade below reuses the same recheck-schedule update logic that
    # store.py's write path uses for live attempts.
    from .store import _update_recheck_schedule

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
        # publish_corpus_observations in store.py), so repeated scheduled
        # exports cannot inflate aggregates.
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

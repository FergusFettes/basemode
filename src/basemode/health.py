"""Observed per-model generation health, recorded from real usage.

Every attempt a caller makes is one row: whether it produced usable text, and
if not, how it failed. That is the record the shipped
`verified_models_registry.json` cannot give — the registry is a weekly lab
measurement of one call per model, while this is what the models actually did
for *this* user, with *this* key, at the volume they actually run.

Stored in ``~/.config/basemode/health.sqlite`` beside the key file: health is
a fact about a model, a provider, and a key, so it belongs to the user rather
than to any one project or corpus.

SQLite rather than JSON because branches are generated in parallel — several
processes and coroutines record at once, and read-modify-write on a JSON file
loses counts under exactly that load.

Recording must never break generation, so :func:`record_outcome` swallows its
own errors. Set ``BASEMODE_NO_HEALTH=1`` to turn recording off entirely.
"""

from __future__ import annotations

import os
import sqlite3
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_CONFIG_DIR = Path.home() / ".config" / "basemode"
_DB_FILE = _CONFIG_DIR / "health.sqlite"

#: Events older than this are pruned; the totals they contributed to survive.
EVENT_RETENTION_DAYS = 30

#: Outcome for a call that raised nothing but returned nothing usable.
EMPTY_RESPONSE = "empty_response"


def classify_error(error: BaseException) -> tuple[str, int | None]:
    """Sort a provider exception into a category and an HTTP status.

    Categories are deliberately coarse: what a user needs from a failure
    record is whether to fix a key, wait, retry, or pick another model.
    """
    from .exceptions import EmptyCompletionError

    if isinstance(error, EmptyCompletionError):
        return EMPTY_RESPONSE, None
    status = error_status(error)
    name = type(error).__name__.lower()
    if status in {401, 403} or "auth" in name or "permission" in name:
        return "authentication", status
    if status == 429 or "ratelimit" in name or "rate_limit" in name:
        return "rate_limit", status
    if isinstance(error, TimeoutError) or "timeout" in name:
        return "timeout", status
    if status is not None and status >= 500:
        return "provider_unavailable", status
    if status in {400, 404, 409, 422}:
        return "invalid_request", status
    if "connection" in name or "network" in name:
        return "network", status
    return "provider_error", status


def error_status(error: BaseException) -> int | None:
    """The HTTP status a provider exception carries, if any."""
    candidates = (getattr(error, "status_code", None), getattr(error, "status", None))
    response = getattr(error, "response", None)
    if response is not None:
        candidates += (getattr(response, "status_code", None),)
    for value in candidates:
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 100 <= value <= 599
        ):
            return value
    return None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _disabled() -> bool:
    return os.environ.get("BASEMODE_NO_HEALTH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _connect() -> sqlite3.Connection:
    _DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_FILE, timeout=5.0)
    conn.row_factory = sqlite3.Row
    # Concurrent writers are the normal case here, not the exception.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_totals (
            model TEXT PRIMARY KEY,
            attempts INTEGER NOT NULL DEFAULT 0,
            successes INTEGER NOT NULL DEFAULT 0,
            failures INTEGER NOT NULL DEFAULT 0,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            last_success_at TEXT,
            last_failure_at TEXT,
            last_category TEXT,
            last_status INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            at TEXT NOT NULL,
            ok INTEGER NOT NULL,
            category TEXT,
            status INTEGER
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_model_at ON model_events(model, at)"
    )
    return conn


def record_outcome(
    model: str,
    *,
    ok: bool,
    category: str | None = None,
    status: int | None = None,
) -> None:
    """Record one generation attempt. Never raises.

    `category` describes a failure the way the caller classified it (
    `rate_limit`, `timeout`, `empty_response`, ...); it is ignored when `ok`.
    """
    if _disabled() or not model.strip():
        return
    model = model.strip().lower()
    category = None if ok else (category or "provider_error")
    now = _now()
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO model_events (model, at, ok, category, status)"
                " VALUES (?, ?, ?, ?, ?)",
                (model, now, int(ok), category, status),
            )
            conn.execute(
                """
                INSERT INTO model_totals (
                    model, attempts, successes, failures, first_seen, last_seen,
                    last_success_at, last_failure_at, last_category, last_status
                ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model) DO UPDATE SET
                    attempts = attempts + 1,
                    successes = successes + excluded.successes,
                    failures = failures + excluded.failures,
                    last_seen = excluded.last_seen,
                    last_success_at = COALESCE(
                        excluded.last_success_at, last_success_at
                    ),
                    last_failure_at = COALESCE(
                        excluded.last_failure_at, last_failure_at
                    ),
                    last_category = CASE
                        WHEN excluded.last_failure_at IS NOT NULL
                        THEN excluded.last_category ELSE last_category END,
                    last_status = CASE
                        WHEN excluded.last_failure_at IS NOT NULL
                        THEN excluded.last_status ELSE last_status END
                """,
                (
                    model,
                    int(ok),
                    int(not ok),
                    now,
                    now,
                    now if ok else None,
                    None if ok else now,
                    category,
                    status,
                ),
            )
            _prune(conn)
    except Exception:
        # A health record is never worth failing a generation over.
        return


def _prune(conn: sqlite3.Connection) -> None:
    cutoff = (datetime.now(UTC) - timedelta(days=EVENT_RETENTION_DAYS)).isoformat()
    conn.execute("DELETE FROM model_events WHERE at < ?", (cutoff,))


def _window_rows(
    conn: sqlite3.Connection, model: str | None, days: int | None
) -> dict[str, list[sqlite3.Row]]:
    sql = "SELECT model, ok, category FROM model_events"
    clauses, params = [], []
    if model:
        clauses.append("model = ?")
        params.append(model)
    if days is not None:
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        clauses.append("at >= ?")
        params.append(cutoff)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    rows: dict[str, list[sqlite3.Row]] = {}
    for row in conn.execute(sql, params):
        rows.setdefault(row["model"], []).append(row)
    return rows


def _summary(row: sqlite3.Row, events: list[sqlite3.Row], days: int | None) -> dict:
    attempts = int(row["attempts"])
    failures = int(row["failures"])
    categories = Counter(
        event["category"] for event in events if not event["ok"] and event["category"]
    )
    recent_attempts = len(events)
    recent_failures = sum(1 for event in events if not event["ok"])
    return {
        "model": row["model"],
        "attempts": attempts,
        "successes": int(row["successes"]),
        "failures": failures,
        "failure_rate": round(failures / attempts, 4) if attempts else None,
        "first_seen": row["first_seen"],
        "last_seen": row["last_seen"],
        "last_success_at": row["last_success_at"],
        "last_failure_at": row["last_failure_at"],
        "last_category": row["last_category"],
        "last_status": row["last_status"],
        # Windowed figures come from the event log, which only reaches back
        # EVENT_RETENTION_DAYS; the totals above are for all time.
        "window_days": days,
        "recent_attempts": recent_attempts,
        "recent_failures": recent_failures,
        "recent_failure_rate": (
            round(recent_failures / recent_attempts, 4) if recent_attempts else None
        ),
        "categories": dict(sorted(categories.items())),
    }


def list_model_health(*, days: int | None = None) -> dict[str, dict[str, Any]]:
    """Health for every model this user has generated with, keyed by model ID.

    `days` limits the windowed figures (`recent_*`, `categories`) to that many
    days back; the all-time totals are unaffected.
    """
    try:
        with _connect() as conn:
            events = _window_rows(conn, None, days)
            return {
                row["model"]: _summary(row, events.get(row["model"], []), days)
                for row in conn.execute("SELECT * FROM model_totals ORDER BY model")
            }
    except Exception:
        return {}


def model_health(model: str, *, days: int | None = None) -> dict[str, Any] | None:
    """Health for one model, or None if it has never been generated with."""
    key = model.strip().lower()
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT * FROM model_totals WHERE model = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            events = _window_rows(conn, key, days)
            return _summary(row, events.get(key, []), days)
    except Exception:
        return None


def clear_model_health(model: str | None = None) -> None:
    """Forget one model's history, or all of it when `model` is None."""
    try:
        with _connect() as conn:
            if model is None:
                conn.execute("DELETE FROM model_totals")
                conn.execute("DELETE FROM model_events")
            else:
                key = model.strip().lower()
                conn.execute("DELETE FROM model_totals WHERE model = ?", (key,))
                conn.execute("DELETE FROM model_events WHERE model = ?", (key,))
    except Exception:
        return

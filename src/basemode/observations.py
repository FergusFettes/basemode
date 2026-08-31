"""Content-free observation ledger for logical continuations and provider calls."""

from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import monotonic
from typing import Final

from .health import classify_error, error_details

log = logging.getLogger(__name__)

_DB_FILE = Path.home() / ".local" / "share" / "basemode" / "observations.sqlite"
_SCHEMA_VERSION: Final = 1
_SOURCES: Final = {
    "cli",
    "python",
    "server",
    "loom",
    "verification",
    "recheck",
    "other",
}
_ATTEMPT_KINDS: Final = {
    "initial",
    "rewind_retry",
    "empty_retry",
    "reasoning_off",
    "larger_budget",
}


def _package_version() -> str:
    try:
        return version("basemode")
    except PackageNotFoundError:  # pragma: no cover - editable source edge case
        return "unknown"


@dataclass(frozen=True, slots=True)
class ObservationContext:
    """Safe caller provenance attached to a continuation operation."""

    source: str = "python"
    source_version: str | None = None
    contribution_eligible: bool = False

    def __post_init__(self) -> None:
        if self.source not in _SOURCES:
            allowed = ", ".join(sorted(_SOURCES))
            raise ValueError(
                f"Unknown observation source {self.source!r}. Valid: {allowed}"
            )
        if self.source_version is not None and len(self.source_version) > 120:
            raise ValueError("source_version must be at most 120 characters")


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
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_endpoints (
            id INTEGER PRIMARY KEY,
            provider_route TEXT NOT NULL,
            provider_model_id TEXT NOT NULL,
            upstream_family TEXT,
            text_eligible INTEGER NOT NULL DEFAULT 1,
            release_date TEXT,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            UNIQUE(provider_route, provider_model_id)
        );
        CREATE TABLE IF NOT EXISTS call_operations (
            id INTEGER PRIMARY KEY,
            event_id TEXT NOT NULL UNIQUE,
            endpoint_id INTEGER NOT NULL REFERENCES model_endpoints(id),
            started_at TEXT NOT NULL,
            finished_at TEXT,
            source TEXT NOT NULL,
            source_version TEXT,
            basemode_version TEXT NOT NULL,
            strategy TEXT NOT NULL,
            strategy_source TEXT NOT NULL,
            logical_outcome TEXT,
            returned_content INTEGER NOT NULL DEFAULT 0,
            finish_reason TEXT,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            total_latency_ms REAL,
            total_prompt_tokens INTEGER,
            total_completion_tokens INTEGER,
            total_reasoning_tokens INTEGER,
            total_cost_usd REAL,
            cost_source TEXT,
            contribution_eligible INTEGER NOT NULL DEFAULT 0,
            verification_probe_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS call_attempts (
            id INTEGER PRIMARY KEY,
            operation_id INTEGER NOT NULL REFERENCES call_operations(id),
            attempt_index INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            attempt_kind TEXT NOT NULL,
            outcome TEXT,
            returned_content INTEGER NOT NULL DEFAULT 0,
            failure_class TEXT,
            failure_transience TEXT,
            failure_attribution TEXT,
            http_status INTEGER,
            safe_error_code TEXT,
            safe_error_parameter TEXT,
            finish_reason TEXT,
            latency_ms REAL,
            ttft_ms REAL,
            generation_ms REAL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            reasoning_tokens INTEGER,
            output_characters INTEGER,
            output_tokens_per_second REAL,
            cost_usd REAL,
            cost_source TEXT,
            status_eligible INTEGER NOT NULL DEFAULT 1,
            status_exclusion_reason TEXT,
            UNIQUE(operation_id, attempt_index)
        );
        CREATE INDEX IF NOT EXISTS idx_operations_endpoint_started
            ON call_operations(endpoint_id, started_at);
        CREATE INDEX IF NOT EXISTS idx_attempts_operation
            ON call_attempts(operation_id, attempt_index);
        """
    )
    existing = conn.execute(
        "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
    ).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO schema_metadata(key, value) VALUES ('schema_version', ?)",
            (str(_SCHEMA_VERSION),),
        )
    elif int(existing["value"]) != _SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported observation schema version {existing['value']}"
        )
    return conn


@contextmanager
def _db():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _endpoint_parts(model: str) -> tuple[str, str]:
    if "/" in model:
        return tuple(model.split("/", 1))  # type: ignore[return-value]
    return "unknown", model


class Operation:
    """Best-effort lifecycle handle for one logical continuation branch."""

    def __init__(
        self,
        model: str,
        strategy: str,
        strategy_source: str,
        context: ObservationContext,
    ) -> None:
        self.id: int | None = None
        self._started = monotonic()
        self._attempts = 0
        self._finished = False
        if _disabled():
            return
        try:
            route, provider_model_id = _endpoint_parts(model)
            now = _now()
            with _db() as conn:
                conn.execute(
                    """INSERT INTO model_endpoints(
                           provider_route, provider_model_id, first_seen, last_seen
                       ) VALUES (?, ?, ?, ?)
                       ON CONFLICT(provider_route, provider_model_id)
                       DO UPDATE SET last_seen = excluded.last_seen""",
                    (route, provider_model_id, now, now),
                )
                endpoint_id = conn.execute(
                    "SELECT id FROM model_endpoints WHERE provider_route=? AND provider_model_id=?",
                    (route, provider_model_id),
                ).fetchone()["id"]
                cursor = conn.execute(
                    """INSERT INTO call_operations(
                           event_id, endpoint_id, started_at, source, source_version,
                           basemode_version, strategy, strategy_source,
                           contribution_eligible
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        endpoint_id,
                        now,
                        context.source,
                        context.source_version,
                        _package_version(),
                        strategy,
                        strategy_source,
                        int(context.contribution_eligible),
                    ),
                )
                self.id = int(cursor.lastrowid)
        except Exception:
            log.warning(
                "could not begin observation operation; ignoring", exc_info=True
            )

    def begin_attempt(self, kind: str) -> Attempt:
        if kind not in _ATTEMPT_KINDS:
            raise ValueError(f"Unknown attempt kind {kind!r}")
        index = self._attempts
        self._attempts += 1
        return Attempt(self, index, kind)

    def finish(self, outcome: str, *, returned_content: bool) -> None:
        if self._finished:
            return
        self._finished = True
        if self.id is None:
            return
        try:
            with _db() as conn:
                conn.execute(
                    """UPDATE call_operations SET
                           finished_at=?, logical_outcome=?, returned_content=?,
                           attempt_count=?, total_latency_ms=?
                       WHERE id=?""",
                    (
                        _now(),
                        outcome,
                        int(returned_content),
                        self._attempts,
                        (monotonic() - self._started) * 1000,
                        self.id,
                    ),
                )
        except Exception:
            log.warning(
                "could not finish observation operation; ignoring", exc_info=True
            )


class Attempt:
    """Best-effort lifecycle handle for one strategy/provider request."""

    def __init__(self, operation: Operation, index: int, kind: str) -> None:
        self.operation = operation
        self.index = index
        self.kind = kind
        self.id: int | None = None
        self._started = monotonic()
        self._finished = False
        self.returned_content = False
        self.output_characters = 0
        if operation.id is None:
            return
        try:
            with _db() as conn:
                cursor = conn.execute(
                    """INSERT INTO call_attempts(
                           operation_id, attempt_index, started_at, attempt_kind
                       ) VALUES (?, ?, ?, ?)""",
                    (operation.id, index, _now(), kind),
                )
                self.id = int(cursor.lastrowid)
        except Exception:
            log.warning("could not begin observation attempt; ignoring", exc_info=True)

    def saw_content(self, text: str) -> None:
        if text:
            self.returned_content = True
            self.output_characters += len(text)

    def finish(self, outcome: str, error: BaseException | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        if self.id is None:
            return
        failure_class = None
        status = None
        error_code = None
        error_param = None
        if error is not None:
            failure_class, status = classify_error(error)
            error_code, error_param = error_details(error)
        elif outcome == "failure" and not self.returned_content:
            failure_class = "empty_response"
        try:
            with _db() as conn:
                conn.execute(
                    """UPDATE call_attempts SET
                           finished_at=?, outcome=?, returned_content=?,
                           failure_class=?, failure_attribution=?, http_status=?,
                           safe_error_code=?, safe_error_parameter=?, latency_ms=?,
                           output_characters=?
                       WHERE id=?""",
                    (
                        _now(),
                        outcome,
                        int(self.returned_content),
                        failure_class,
                        "unknown" if error is not None else None,
                        status,
                        error_code,
                        error_param,
                        (monotonic() - self._started) * 1000,
                        self.output_characters,
                        self.id,
                    ),
                )
        except Exception:
            log.warning("could not finish observation attempt; ignoring", exc_info=True)


def observe_operation(
    model: str,
    strategy: str,
    strategy_source: str,
    context: ObservationContext | None,
) -> Operation:
    """Begin one operation without allowing recorder failure to escape."""
    return Operation(model, strategy, strategy_source, context or ObservationContext())

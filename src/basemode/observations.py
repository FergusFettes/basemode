"""Content-free observation ledger for logical continuations and provider calls."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import monotonic
from typing import Final

from .failure_taxonomy import classify_error, error_details
from .health_rules import recheck_due_at
from .usage import usage_from_events

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
_TRANSIENT_FAILURES: Final = {
    "rate_limit",
    "timeout",
    "network",
    "provider_unavailable",
    "empty_response",
    "provider_error",
}
_SAFE_FINISH_REASONS: Final = {
    "stop",
    "length",
    "content_filter",
    "tool_calls",
    "end_turn",
    "max_tokens",
    "error",
    "cancelled",
    "unknown",
}
_VERSION_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+\-]{0,119}$")


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
    contribution_eligible: bool | None = None
    verification_probe_id: int | None = None

    def __post_init__(self) -> None:
        if self.source not in _SOURCES:
            allowed = ", ".join(sorted(_SOURCES))
            raise ValueError(
                f"Unknown observation source {self.source!r}. Valid: {allowed}"
            )
        if self.contribution_eligible is None:
            from .keys import contribution_enabled

            object.__setattr__(self, "contribution_eligible", contribution_enabled())
        if self.source_version is not None and not _VERSION_RE.fullmatch(
            self.source_version
        ):
            raise ValueError("source_version must be a safe package-version identifier")
        if self.verification_probe_id is not None and self.source != "verification":
            raise ValueError("verification_probe_id requires source='verification'")


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
        CREATE TABLE IF NOT EXISTS verification_runs (
            id INTEGER PRIMARY KEY,
            event_id TEXT NOT NULL UNIQUE,
            suite TEXT NOT NULL,
            suite_version TEXT NOT NULL,
            basemode_version TEXT NOT NULL,
            litellm_version TEXT,
            target_policy TEXT,
            lifecycle_status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT
        );
        CREATE TABLE IF NOT EXISTS verification_probes (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES verification_runs(id),
            endpoint_id INTEGER NOT NULL REFERENCES model_endpoints(id),
            probe_identifier TEXT NOT NULL,
            repetition INTEGER NOT NULL DEFAULT 0,
            required_status INTEGER NOT NULL DEFAULT 1,
            operation_id INTEGER UNIQUE REFERENCES call_operations(id),
            UNIQUE(run_id, endpoint_id, probe_identifier, repetition)
        );
        CREATE TABLE IF NOT EXISTS probe_metrics (
            id INTEGER PRIMARY KEY,
            probe_id INTEGER NOT NULL REFERENCES verification_probes(id),
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            UNIQUE(probe_id, metric_name)
        );
        CREATE TABLE IF NOT EXISTS recheck_schedules (
            endpoint_id INTEGER PRIMARY KEY REFERENCES model_endpoints(id),
            due_at TEXT NOT NULL,
            reason TEXT NOT NULL,
            failure_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS operational_assessments (
            id INTEGER PRIMARY KEY,
            endpoint_id INTEGER NOT NULL REFERENCES model_endpoints(id),
            assessed_at TEXT NOT NULL,
            status TEXT NOT NULL,
            rules_version INTEGER NOT NULL,
            triggering_operation_id INTEGER REFERENCES call_operations(id)
        );
        CREATE TABLE IF NOT EXISTS daily_call_aggregates (
            id INTEGER PRIMARY KEY,
            endpoint_id INTEGER NOT NULL REFERENCES model_endpoints(id),
            utc_day TEXT NOT NULL,
            strategy TEXT NOT NULL,
            source TEXT NOT NULL,
            source_version TEXT,
            operations INTEGER NOT NULL,
            successful_operations INTEGER NOT NULL,
            attempts INTEGER NOT NULL,
            failed_attempts INTEGER NOT NULL,
            UNIQUE(endpoint_id, utc_day, strategy, source, source_version)
        );
        CREATE TABLE IF NOT EXISTS contribution_batches (
            id INTEGER PRIMARY KEY,
            bundle_id TEXT NOT NULL UNIQUE,
            window_start TEXT NOT NULL,
            window_end TEXT NOT NULL,
            path TEXT,
            status TEXT NOT NULL,
            pr_url TEXT,
            created_at TEXT NOT NULL
        );
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
        self.model = model
        self._started = monotonic()
        self._attempts = 0
        self._finished = False
        if _disabled():
            return
        try:
            route, provider_model_id = _endpoint_parts(model)
            now = _now()
            with _db() as conn:
                endpoint_id = _ensure_endpoint(conn, route, provider_model_id, now)
                cursor = conn.execute(
                    """INSERT INTO call_operations(
                           event_id, endpoint_id, started_at, source, source_version,
                           basemode_version, strategy, strategy_source,
                           contribution_eligible, verification_probe_id
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                        context.verification_probe_id,
                    ),
                )
                self.id = int(cursor.lastrowid)
                if context.verification_probe_id is not None:
                    conn.execute(
                        "UPDATE verification_probes SET operation_id=? WHERE id=?",
                        (self.id, context.verification_probe_id),
                    )
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

    @classmethod
    def resume(cls, operation_id: int) -> Operation:
        """Reopen an unfinished operation so a resumed run can append attempts."""
        with _db() as conn:
            row = conn.execute(
                """SELECT o.id,o.finished_at,e.provider_route,e.provider_model_id,
                          COALESCE(MAX(a.attempt_index), -1) AS last_attempt
                   FROM call_operations o
                   JOIN model_endpoints e ON e.id=o.endpoint_id
                   LEFT JOIN call_attempts a ON a.operation_id=o.id
                   WHERE o.id=? GROUP BY o.id""",
                (operation_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown observation operation: {operation_id}")
        if row["finished_at"] is not None:
            raise ValueError(
                f"observation operation is already finished: {operation_id}"
            )
        operation = cls.__new__(cls)
        operation.id = int(row["id"])
        operation.model = (
            str(row["provider_model_id"])
            if row["provider_route"] == "unknown"
            else f"{row['provider_route']}/{row['provider_model_id']}"
        )
        operation._started = monotonic()
        operation._attempts = int(row["last_attempt"]) + 1
        operation._finished = False
        return operation

    def finish(self, outcome: str, *, returned_content: bool) -> None:
        if self._finished:
            return
        self._finished = True
        if self.id is None:
            return
        try:
            with _db() as conn:
                totals = conn.execute(
                    """SELECT
                           SUM(prompt_tokens), SUM(completion_tokens),
                           SUM(reasoning_tokens), SUM(cost_usd)
                       FROM call_attempts WHERE operation_id=?""",
                    (self.id,),
                ).fetchone()
                conn.execute(
                    """UPDATE call_operations SET
                           finished_at=?, logical_outcome=?, returned_content=?,
                           attempt_count=?, total_latency_ms=?,
                           total_prompt_tokens=?, total_completion_tokens=?,
                           total_reasoning_tokens=?, total_cost_usd=?, cost_source=?
                       WHERE id=?""",
                    (
                        _now(),
                        outcome,
                        int(returned_content),
                        self._attempts,
                        (monotonic() - self._started) * 1000,
                        totals[0],
                        totals[1],
                        totals[2],
                        totals[3],
                        "provider" if totals[3] is not None else None,
                        self.id,
                    ),
                )
                _update_recheck_schedule(conn, self.id, outcome, _now())
        except Exception:
            log.warning(
                "could not finish observation operation; ignoring", exc_info=True
            )


def _update_recheck_schedule(
    conn: sqlite3.Connection, operation_id: int, outcome: str, observed_at: str
) -> None:
    """Advance derived organic recheck state after a completed operation."""
    operation = conn.execute(
        "SELECT endpoint_id,source FROM call_operations WHERE id=?", (operation_id,)
    ).fetchone()
    if operation is None or operation["source"] not in {
        "cli",
        "python",
        "server",
        "loom",
        "other",
    }:
        return
    if outcome == "success":
        conn.execute(
            "DELETE FROM recheck_schedules WHERE endpoint_id=?",
            (operation["endpoint_id"],),
        )
        return
    attempt = conn.execute(
        """SELECT failure_class,status_eligible FROM call_attempts
           WHERE operation_id=? ORDER BY attempt_index DESC LIMIT 1""",
        (operation_id,),
    ).fetchone()
    if (
        attempt is None
        or not attempt["status_eligible"]
        or attempt["failure_class"] not in _TRANSIENT_FAILURES
    ):
        return
    prior = conn.execute(
        "SELECT failure_count FROM recheck_schedules WHERE endpoint_id=?",
        (operation["endpoint_id"],),
    ).fetchone()
    failure_count = (int(prior["failure_count"]) if prior else 0) + 1
    conn.execute(
        """INSERT INTO recheck_schedules(
               endpoint_id,due_at,reason,failure_count,updated_at
           ) VALUES(?,?,?,?,?) ON CONFLICT(endpoint_id) DO UPDATE SET
               due_at=excluded.due_at,reason=excluded.reason,
               failure_count=excluded.failure_count,updated_at=excluded.updated_at""",
        (
            operation["endpoint_id"],
            recheck_due_at(observed_at, failure_count),
            attempt["failure_class"],
            failure_count,
            observed_at,
        ),
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
        self._first_content_at: float | None = None
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
            if self._first_content_at is None:
                self._first_content_at = monotonic()
            self.returned_content = True
            self.output_characters += len(text)

    def finish(
        self,
        outcome: str,
        error: BaseException | None = None,
        *,
        usage_events: list[dict] | None = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        if self.id is None:
            return
        failure_class = None
        status = None
        error_code = None
        error_param = None
        finish_reason = None
        if error is not None:
            failure_class, status = classify_error(error)
            error_code, error_param = error_details(error)
            raw_finish_reason = getattr(error, "finish_reason", None)
            if isinstance(raw_finish_reason, str):
                finish_reason = (
                    raw_finish_reason
                    if raw_finish_reason in _SAFE_FINISH_REASONS
                    else "unknown"
                )
        elif outcome == "failure" and not self.returned_content:
            failure_class = "empty_response"
        attribution = _failure_attribution(failure_class)
        transience = _failure_transience(failure_class)
        status_eligible = attribution not in {"account", "basemode", "client"}
        exclusion_reason = None if status_eligible else f"{attribution}_attributed"
        try:
            usage = usage_from_events(self.operation.model, usage_events or [])
            reasoning_tokens = _reasoning_tokens(usage_events or [])
        except Exception:
            log.warning("could not parse attempt usage; ignoring", exc_info=True)
            usage = None
            reasoning_tokens = 0
        now = monotonic()
        latency_ms = (now - self._started) * 1000
        ttft_ms = (
            (self._first_content_at - self._started) * 1000
            if self._first_content_at is not None
            else None
        )
        generation_ms = (
            (now - self._first_content_at) * 1000
            if self._first_content_at is not None
            else None
        )
        output_rate = None
        if usage is not None and generation_ms and generation_ms > 0:
            output_rate = usage.completion_tokens / (generation_ms / 1000)
        try:
            with _db() as conn:
                conn.execute(
                    """UPDATE call_attempts SET
                           finished_at=?, outcome=?, returned_content=?,
                           failure_class=?, failure_transience=?, failure_attribution=?,
                           http_status=?,
                           safe_error_code=?, safe_error_parameter=?, latency_ms=?,
                           output_characters=?, finish_reason=?, ttft_ms=?, generation_ms=?,
                           prompt_tokens=?, completion_tokens=?, reasoning_tokens=?,
                           output_tokens_per_second=?, cost_usd=?, cost_source=?,
                           status_eligible=?, status_exclusion_reason=?
                       WHERE id=?""",
                    (
                        _now(),
                        outcome,
                        int(self.returned_content),
                        failure_class,
                        transience,
                        attribution,
                        status,
                        error_code,
                        error_param,
                        latency_ms,
                        self.output_characters,
                        finish_reason,
                        ttft_ms,
                        generation_ms,
                        usage.prompt_tokens if usage is not None else None,
                        usage.completion_tokens if usage is not None else None,
                        reasoning_tokens if usage is not None else None,
                        output_rate,
                        usage.cost_usd if usage is not None else None,
                        "provider"
                        if usage is not None and not usage.is_estimate
                        else None,
                        int(status_eligible),
                        exclusion_reason,
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


def begin_verification_run(
    suite: str,
    suite_version: str,
    *,
    litellm_version: str | None = None,
    target_policy: str | None = None,
) -> int:
    """Create controlled-run structure without duplicating call outcomes."""
    now = _now()
    with _db() as conn:
        cursor = conn.execute(
            """INSERT INTO verification_runs(
                   event_id, suite, suite_version, basemode_version,
                   litellm_version, target_policy, lifecycle_status, started_at
               ) VALUES (?, ?, ?, ?, ?, ?, 'running', ?)""",
            (
                str(uuid.uuid4()),
                suite,
                suite_version,
                _package_version(),
                litellm_version,
                target_policy,
                now,
            ),
        )
        return int(cursor.lastrowid)


def finish_verification_run(run_id: int, status: str) -> None:
    with _db() as conn:
        conn.execute(
            "UPDATE verification_runs SET lifecycle_status=?, finished_at=? WHERE id=?",
            (status, _now(), run_id),
        )


def resume_verification_run(run_id: int | str) -> sqlite3.Row:
    local_id = int(run_id)
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM verification_runs WHERE id=?", (local_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown verification run: {run_id}")
        if row["lifecycle_status"] == "completed":
            raise ValueError(f"verification run is already completed: {run_id}")
        conn.execute(
            """UPDATE verification_runs SET lifecycle_status='running',finished_at=NULL
               WHERE id=?""",
            (local_id,),
        )
        return row


def begin_verification_probe(
    run_id: int,
    model: str,
    probe_identifier: str,
    *,
    repetition: int = 0,
    required: bool = True,
) -> int:
    route, provider_model_id = _endpoint_parts(model.lower())
    with _db() as conn:
        endpoint_id = _ensure_endpoint(conn, route, provider_model_id, _now())
        cursor = conn.execute(
            """INSERT INTO verification_probes(
                   run_id, endpoint_id, probe_identifier, repetition, required_status
               ) VALUES (?, ?, ?, ?, ?)""",
            (run_id, endpoint_id, probe_identifier, repetition, int(required)),
        )
        return int(cursor.lastrowid)


def verification_probe(
    run_id: int,
    model: str,
    probe_identifier: str,
    *,
    repetition: int = 0,
) -> sqlite3.Row | None:
    route, provider_model_id = _endpoint_parts(model.lower())
    with _db() as conn:
        return conn.execute(
            """SELECT p.* FROM verification_probes p
               JOIN model_endpoints e ON e.id=p.endpoint_id
               WHERE p.run_id=? AND e.provider_route=? AND e.provider_model_id=?
                 AND p.probe_identifier=? AND p.repetition=?""",
            (run_id, route, provider_model_id, probe_identifier, repetition),
        ).fetchone()


def completed_verification_probes(run_id: int) -> set[tuple[str, int]]:
    with _db() as conn:
        rows = conn.execute(
            """SELECT p.probe_identifier,p.repetition
               FROM verification_probes p
               JOIN call_operations o ON o.id=p.operation_id
               WHERE p.run_id=? AND o.finished_at IS NOT NULL""",
            (run_id,),
        ).fetchall()
    return {(str(row["probe_identifier"]), int(row["repetition"])) for row in rows}


def operation_attempt_kinds(operation_id: int) -> set[str]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT attempt_kind FROM call_attempts WHERE operation_id=?",
            (operation_id,),
        ).fetchall()
    return {str(row["attempt_kind"]) for row in rows}


def _ensure_endpoint(
    conn: sqlite3.Connection, route: str, provider_model_id: str, now: str
) -> int:
    conn.execute(
        """INSERT INTO model_endpoints(
               provider_route, provider_model_id, first_seen, last_seen
           ) VALUES (?, ?, ?, ?)
           ON CONFLICT(provider_route, provider_model_id)
           DO UPDATE SET last_seen = excluded.last_seen""",
        (route, provider_model_id, now, now),
    )
    return int(
        conn.execute(
            "SELECT id FROM model_endpoints WHERE provider_route=? AND provider_model_id=?",
            (route, provider_model_id),
        ).fetchone()["id"]
    )


def _reasoning_tokens(events: list[dict]) -> int:
    total = 0
    for event in events:
        details = event.get("completion_tokens_details") or {}
        if isinstance(details, dict):
            total += int(details.get("reasoning_tokens") or 0)
    return total


def _failure_attribution(failure_class: str | None) -> str | None:
    if failure_class is None:
        return None
    if failure_class in {"authentication", "quota"}:
        return "account"
    if failure_class == "invalid_request":
        return "basemode"
    if failure_class == "cancelled":
        return "client"
    if failure_class in {
        "provider_unavailable",
        "provider_error",
        "content_filter",
        "rate_limit",
    }:
        return "provider"
    if failure_class == "empty_response":
        return "endpoint"
    return "unknown"


def _failure_transience(failure_class: str | None) -> str | None:
    if failure_class is None:
        return None
    if failure_class in _TRANSIENT_FAILURES:
        return "transient"
    if failure_class in {"authentication", "quota", "invalid_request"}:
        return "persistent"
    return "unknown"

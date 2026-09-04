import sqlite3
from collections.abc import AsyncGenerator

import pytest

from basemode import (
    ObservationContext,
    branch_text,
    continue_text,
    observations,
    usage_capture,
)
from basemode.exceptions import EmptyCompletionError
from basemode.observation_queries import due_recheck_models

pytestmark = pytest.mark.asyncio


class _Strategy:
    name = "system"

    def __init__(self, scripts: list[list[str] | BaseException]) -> None:
        self.scripts = scripts

    async def stream(self, prefix, params) -> AsyncGenerator[str, None]:
        script = self.scripts.pop(0)
        if isinstance(script, BaseException):
            raise script
        for token in script:
            yield token


def _install(monkeypatch, scripts: list[list[str] | BaseException]) -> _Strategy:
    strategy = _Strategy(scripts)
    monkeypatch.setattr("basemode.continue_.detect_strategy", lambda *args: strategy)
    return strategy


async def _drain(stream):
    return [item async for item in stream]


def _rows(table: str) -> list[sqlite3.Row]:
    conn = sqlite3.connect(observations._DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute(f"SELECT * FROM {table} ORDER BY id"))
    finally:
        conn.close()


async def test_schema_has_operational_tables_and_no_content_columns() -> None:
    operation = observations.observe_operation(
        "openai/gpt-4o-mini", "system", "heuristic", None
    )
    operation.finish("inconclusive", returned_content=False)
    expected = {
        "model_endpoints",
        "call_operations",
        "call_attempts",
        "verification_runs",
        "verification_probes",
        "probe_metrics",
        "recheck_schedules",
        "operational_assessments",
        "daily_call_aggregates",
        "contribution_batches",
    }
    conn = sqlite3.connect(observations._DB_FILE)
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert expected <= tables
        for table in expected:
            columns = {
                row[1].lower() for row in conn.execute(f"PRAGMA table_info({table})")
            }
            assert not columns & {
                "prompt",
                "prefix",
                "response",
                "content",
                "content_hash",
                "document_id",
                "account_id",
            }
    finally:
        conn.close()


async def test_fresh_database_starts_at_current_schema() -> None:
    with observations._db() as conn:
        version = conn.execute(
            "SELECT value FROM schema_metadata WHERE key='schema_version'"
        ).fetchone()[0]
        columns = {row[1] for row in conn.execute("PRAGMA table_info(model_endpoints)")}
    assert version == "2"
    assert {"modality", "catalog_available"} <= columns


async def test_schema_v1_endpoint_metadata_migrates_in_place() -> None:
    conn = sqlite3.connect(observations._DB_FILE)
    conn.executescript(
        """CREATE TABLE schema_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        INSERT INTO schema_metadata VALUES('schema_version','1');
        CREATE TABLE model_endpoints(
          id INTEGER PRIMARY KEY,provider_route TEXT NOT NULL,
          provider_model_id TEXT NOT NULL,upstream_family TEXT,
          text_eligible INTEGER NOT NULL DEFAULT 1,release_date TEXT,
          first_seen TEXT NOT NULL,last_seen TEXT NOT NULL,
          UNIQUE(provider_route,provider_model_id));"""
    )
    conn.close()

    migrated = observations._connect()
    try:
        version = migrated.execute(
            "SELECT value FROM schema_metadata WHERE key='schema_version'"
        ).fetchone()[0]
        columns = {
            row[1] for row in migrated.execute("PRAGMA table_info(model_endpoints)")
        }
    finally:
        migrated.close()
    assert version == "2"
    assert {"modality", "catalog_available"} <= columns


async def test_success_records_one_operation_and_attempt(monkeypatch) -> None:
    _install(monkeypatch, [[" hello"]])

    assert await _drain(
        continue_text(
            "Seed",
            observation=ObservationContext(
                source="loom", source_version="0.8.0", contribution_eligible=True
            ),
        )
    ) == [" hello"]

    operations = _rows("call_operations")
    attempts = _rows("call_attempts")
    assert len(operations) == len(attempts) == 1
    assert operations[0]["source"] == "loom"
    assert operations[0]["source_version"] == "0.8.0"
    assert operations[0]["contribution_eligible"] == 1
    assert operations[0]["logical_outcome"] == "success"
    assert operations[0]["attempt_count"] == 1
    assert attempts[0]["attempt_kind"] == "initial"
    assert attempts[0]["outcome"] == "success"
    assert attempts[0]["output_characters"] == 6


async def test_observation_lifecycle_emits_content_free_verbose_events(
    monkeypatch, caplog
) -> None:
    _install(monkeypatch, [[" private output"]])
    caplog.set_level("INFO", logger="basemode.observations")

    await _drain(continue_text("private prompt", model="openai/example"))

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "operation 1 started: model=openai/example" in messages
    assert "attempt 1.0 started: model=openai/example kind=initial" in messages
    assert "attempt 1.0 recorded: model=openai/example" in messages
    assert "operation 1 recorded: model=openai/example outcome=success" in messages
    assert "private prompt" not in messages
    assert "private output" not in messages


async def test_branches_record_independent_operations(monkeypatch) -> None:
    _install(monkeypatch, [[" a"], [" b"], [" c"]])

    await _drain(branch_text("Seed", n=3))

    assert len(_rows("call_operations")) == 3
    assert len(_rows("call_attempts")) == 3


async def test_empty_recovery_is_two_attempts_under_one_operation(monkeypatch) -> None:
    _install(
        monkeypatch,
        [
            EmptyCompletionError(model="openrouter/example", strategy="system"),
            [" recovered"],
        ],
    )

    assert await _drain(continue_text("Seed", model="openrouter/example")) == [
        " recovered"
    ]

    operations = _rows("call_operations")
    attempts = _rows("call_attempts")
    assert len(operations) == 1
    assert operations[0]["logical_outcome"] == "success"
    assert operations[0]["attempt_count"] == 2
    assert [row["attempt_kind"] for row in attempts] == ["initial", "reasoning_off"]
    assert [row["outcome"] for row in attempts] == ["failure", "success"]
    assert attempts[0]["failure_class"] == "empty_response"


async def test_provider_exception_finalizes_operation_and_attempt(monkeypatch) -> None:
    class RateLimitError(RuntimeError):
        status_code = 429

    _install(monkeypatch, [RateLimitError("do not store this body")])

    with pytest.raises(RateLimitError):
        await _drain(continue_text("Seed"))

    operation = _rows("call_operations")[0]
    attempt = _rows("call_attempts")[0]
    assert operation["logical_outcome"] == "failure"
    assert attempt["failure_class"] == "rate_limit"
    assert attempt["failure_transience"] == "transient"
    assert attempt["failure_attribution"] == "provider"
    assert attempt["status_eligible"] == 1
    assert attempt["http_status"] == 429
    assert "do not store" not in str(dict(attempt))


async def test_organic_transient_failures_schedule_rechecks(monkeypatch) -> None:
    class RateLimitError(RuntimeError):
        status_code = 429

    _install(monkeypatch, [RateLimitError("limited"), RateLimitError("limited")])
    for _ in range(2):
        with pytest.raises(RateLimitError):
            await _drain(continue_text("Seed", model="openai/example"))

    conn = sqlite3.connect(observations._DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        schedule = conn.execute("SELECT * FROM recheck_schedules").fetchone()
    finally:
        conn.close()
    assert schedule["reason"] == "rate_limit"
    assert schedule["failure_count"] == 2
    assert due_recheck_models(now="9999-12-31T00:00:00+00:00") == ["openai/example"]


async def test_organic_success_resolves_recheck_schedule(monkeypatch) -> None:
    class RateLimitError(RuntimeError):
        status_code = 429

    _install(monkeypatch, [RateLimitError("limited"), [" recovered"]])
    with pytest.raises(RateLimitError):
        await _drain(continue_text("Seed", model="openai/example"))
    await _drain(continue_text("Seed", model="openai/example"))

    conn = sqlite3.connect(observations._DB_FILE)
    try:
        assert conn.execute("SELECT COUNT(*) FROM recheck_schedules").fetchone()[0] == 0
    finally:
        conn.close()


async def test_account_and_basemode_failures_are_not_status_eligible(
    monkeypatch,
) -> None:
    class AuthenticationError(RuntimeError):
        status_code = 401

    _install(monkeypatch, [AuthenticationError("secret provider body")])
    with pytest.raises(AuthenticationError):
        await _drain(continue_text("Seed"))

    attempt = _rows("call_attempts")[0]
    assert attempt["failure_attribution"] == "account"
    assert attempt["status_eligible"] == 0
    assert attempt["status_exclusion_reason"] == "account_attributed"


async def test_consumer_close_after_content_is_cancelled_success(monkeypatch) -> None:
    _install(monkeypatch, [[" first", " second"]])
    stream = continue_text("Seed")

    assert await anext(stream) == " first second"
    await stream.aclose()

    operation = _rows("call_operations")[0]
    attempt = _rows("call_attempts")[0]
    assert operation["logical_outcome"] == "cancelled"
    assert operation["returned_content"] == 1
    # The provider stream had completed before healing emitted its buffered head.
    assert attempt["outcome"] == "success"


async def test_recorder_failure_never_breaks_generation(monkeypatch) -> None:
    _install(monkeypatch, [[" hello"]])
    monkeypatch.setattr(
        observations, "_connect", lambda: (_ for _ in ()).throw(OSError("disk down"))
    )

    assert await _drain(continue_text("Seed")) == [" hello"]


async def test_provider_usage_rolls_up_from_attempt_to_operation(monkeypatch) -> None:
    class UsageStrategy:
        name = "system"

        async def stream(self, prefix, params):
            usage_capture.record(
                {
                    "prompt_tokens": 12,
                    "completion_tokens": 7,
                    "completion_tokens_details": {"reasoning_tokens": 2},
                }
            )
            yield " measured"

    monkeypatch.setattr(
        "basemode.continue_.detect_strategy", lambda *args: UsageStrategy()
    )

    await _drain(continue_text("Seed", model="openai/gpt-4o-mini"))

    operation = _rows("call_operations")[0]
    attempt = _rows("call_attempts")[0]
    assert attempt["prompt_tokens"] == operation["total_prompt_tokens"] == 12
    assert attempt["completion_tokens"] == operation["total_completion_tokens"] == 7
    assert attempt["reasoning_tokens"] == operation["total_reasoning_tokens"] == 2
    assert attempt["ttft_ms"] is not None
    assert attempt["generation_ms"] is not None
    assert attempt["cost_source"] == "provider"
    assert operation["cost_source"] == "provider"


async def test_verification_probe_links_to_ordinary_operation(monkeypatch) -> None:
    _install(monkeypatch, [[" verified"]])
    run_id = observations.begin_verification_run("quick", "1")
    probe_id = observations.begin_verification_probe(
        run_id, "openai/gpt-4o-mini", "continuation-1"
    )

    await _drain(
        continue_text(
            "Seed",
            observation=ObservationContext(
                source="verification", verification_probe_id=probe_id
            ),
        )
    )
    observations.finish_verification_run(run_id, "completed")

    operation = _rows("call_operations")[0]
    probe = _rows("verification_probes")[0]
    run = _rows("verification_runs")[0]
    assert operation["verification_probe_id"] == probe_id
    assert probe["operation_id"] == operation["id"]
    assert run["lifecycle_status"] == "completed"


async def test_internal_recovery_calls_share_one_logical_operation(monkeypatch) -> None:
    class RequestShapeError(RuntimeError):
        status_code = 400

    strategy = _install(monkeypatch, [RequestShapeError("bad shape"), [" recovered"]])
    operation = observations.observe_operation(
        "openai/gpt-4o-mini",
        strategy.name,
        "registry",
        ObservationContext(source="verification"),
    )

    with pytest.raises(RequestShapeError):
        await _drain(
            continue_text(
                "Seed",
                _observation_operation=operation,
                _finalize_observation=False,
            )
        )
    await _drain(
        continue_text(
            "Seed",
            _observation_operation=operation,
            _observation_attempt_kind="reasoning_off",
            _finalize_observation=False,
        )
    )
    operation.finish("success", returned_content=True)

    operations = _rows("call_operations")
    attempts = _rows("call_attempts")
    assert len(operations) == 1
    assert operations[0]["attempt_count"] == 2
    assert [row["attempt_kind"] for row in attempts] == ["initial", "reasoning_off"]

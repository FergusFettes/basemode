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

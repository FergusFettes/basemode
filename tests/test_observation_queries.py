import pytest

from basemode import ObservationContext, continue_text, observations
from basemode.observation_queries import (
    controlled_status,
    endpoint_health,
    list_controlled_status,
    list_endpoint_health,
)

pytestmark = pytest.mark.asyncio


class _Strategy:
    name = "system"

    def __init__(self, scripts):
        self.scripts = scripts

    async def stream(self, prefix, params):
        item = self.scripts.pop(0)
        if isinstance(item, BaseException):
            raise item
        yield item


async def _drain(stream):
    return [token async for token in stream]


async def test_health_reports_success_recovery_and_sources(monkeypatch) -> None:
    from basemode.exceptions import EmptyCompletionError

    strategy = _Strategy(
        [
            EmptyCompletionError(model="openrouter/example", strategy="system"),
            " recovered",
        ]
    )
    monkeypatch.setattr("basemode.continue_.detect_strategy", lambda *args: strategy)

    await _drain(continue_text("Seed", model="openrouter/example"))

    health = endpoint_health("openrouter/example")
    assert health is not None
    assert health["operations"] == health["successful_operations"] == 1
    assert health["attempts"] == 2
    assert health["recovered_operations"] == 1
    assert health["failures"] == {"empty_response": 1}
    assert health["source_counts"] == {"python": 1}
    assert health["operational_status"] == "healthy"


async def test_account_failure_is_visible_but_excluded_from_endpoint_rate(
    monkeypatch,
) -> None:
    class AuthenticationError(RuntimeError):
        status_code = 401

    strategy = _Strategy([AuthenticationError("private body")])
    monkeypatch.setattr("basemode.continue_.detect_strategy", lambda *args: strategy)

    with pytest.raises(AuthenticationError):
        await _drain(continue_text("Seed", model="openai/example"))

    health = endpoint_health("openai/example")
    assert health is not None
    assert health["operations"] == 0
    assert health["attempts"] == 0
    assert health["account_failures"] == 1
    assert health["operational_status"] == "account_limited"


async def test_thorough_controlled_status_is_derived_from_linked_operation(
    monkeypatch,
) -> None:
    strategy = _Strategy([" verified"])
    monkeypatch.setattr("basemode.continue_.detect_strategy", lambda *args: strategy)
    run_id = observations.begin_verification_run("thorough", "1")
    probe_id = observations.begin_verification_probe(
        run_id, "openai/example", "openai/example", repetition=1
    )

    await _drain(
        continue_text(
            "Seed",
            model="openai/example",
            observation=ObservationContext(
                source="verification", verification_probe_id=probe_id
            ),
        )
    )
    observations.finish_verification_run(run_id, "completed")

    status = controlled_status("openai/example")
    assert status["controlled_status"] == "verified"
    assert status["successful_probes"] == status["required_probes"] == 1
    assert status["attempts"] == 1


async def _run_controlled_probe(monkeypatch, model, error) -> None:
    strategy = _Strategy([error])
    monkeypatch.setattr("basemode.continue_.detect_strategy", lambda *args: strategy)
    run_id = observations.begin_verification_run("quick", "1")
    probe_id = observations.begin_verification_probe(
        run_id, model, model, repetition=1
    )
    with pytest.raises(type(error)):
        await _drain(
            continue_text(
                "Seed",
                model=model,
                observation=ObservationContext(
                    source="verification", verification_probe_id=probe_id
                ),
            )
        )
    observations.finish_verification_run(run_id, "completed")


async def test_account_failure_leaves_controlled_status_inconclusive(
    monkeypatch,
) -> None:
    class PermissionDenied(RuntimeError):
        status_code = 403

    await _run_controlled_probe(
        monkeypatch, "zai/glm-locked", PermissionDenied("no access")
    )

    status = controlled_status("zai/glm-locked")
    assert status["controlled_status"] == "account_limited"


async def test_missing_model_controlled_status_is_retired(monkeypatch) -> None:
    class NotFound(RuntimeError):
        status_code = 404

    await _run_controlled_probe(monkeypatch, "gemini/gone-1.0", NotFound("no model"))

    status = controlled_status("gemini/gone-1.0")
    assert status["controlled_status"] == "retired"


async def test_answering_endpoint_that_fails_is_still_failed(monkeypatch) -> None:
    class Overloaded(RuntimeError):
        status_code = 503

    await _run_controlled_probe(monkeypatch, "openai/wobbly", Overloaded("busy"))

    assert controlled_status("openai/wobbly")["controlled_status"] == "failed"


async def test_unseen_endpoint_has_never_tested_status() -> None:
    assert controlled_status("openai/unseen")["controlled_status"] == "never_tested"


async def test_queries_initialize_an_existing_empty_ledger() -> None:
    observations._DB_FILE.touch()

    assert list_endpoint_health() == {}
    assert list_controlled_status() == {}

    with observations._db() as conn:
        version = conn.execute(
            "SELECT value FROM schema_metadata WHERE key='schema_version'"
        ).fetchone()[0]
    assert version == "2"

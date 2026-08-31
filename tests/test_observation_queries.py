import pytest

from basemode import continue_text
from basemode.observation_queries import endpoint_health

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

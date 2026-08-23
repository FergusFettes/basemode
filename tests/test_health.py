"""Observed per-model health recorded from real usage."""

from datetime import UTC, datetime, timedelta

from basemode import health


def test_an_unseen_model_has_no_health() -> None:
    assert health.model_health("openai/gpt-4o") is None
    assert health.list_model_health() == {}


def test_outcomes_accumulate_into_totals() -> None:
    health.record_outcome("openai/gpt-4o", ok=True)
    health.record_outcome("openai/gpt-4o", ok=True)
    health.record_outcome("openai/gpt-4o", ok=False, category="rate_limit", status=429)

    summary = health.model_health("openai/gpt-4o")

    assert summary["attempts"] == 3
    assert summary["successes"] == 2
    assert summary["failures"] == 1
    assert summary["failure_rate"] == round(1 / 3, 4)
    assert summary["categories"] == {"rate_limit": 1}
    assert summary["last_category"] == "rate_limit"
    assert summary["last_status"] == 429


def test_a_later_success_does_not_erase_the_last_failure() -> None:
    health.record_outcome("openai/gpt-4o", ok=False, category="timeout", status=None)
    health.record_outcome("openai/gpt-4o", ok=True)

    summary = health.model_health("openai/gpt-4o")

    assert summary["last_category"] == "timeout"
    assert summary["last_failure_at"] is not None
    assert summary["last_success_at"] > summary["last_failure_at"]


def test_a_failure_without_a_category_is_still_classified() -> None:
    health.record_outcome("openai/gpt-4o", ok=False)

    assert health.model_health("openai/gpt-4o")["categories"] == {"provider_error": 1}


def test_model_ids_are_matched_case_insensitively() -> None:
    health.record_outcome("OpenAI/GPT-4o", ok=True)

    assert health.model_health("openai/gpt-4o")["attempts"] == 1
    assert list(health.list_model_health()) == ["openai/gpt-4o"]


def test_the_window_narrows_recent_figures_but_not_totals() -> None:
    health.record_outcome("openai/gpt-4o", ok=False, category="rate_limit", status=429)
    old = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    with health._connect() as conn:
        conn.execute("UPDATE model_events SET at = ?", (old,))

    health.record_outcome("openai/gpt-4o", ok=True)
    summary = health.model_health("openai/gpt-4o", days=7)

    assert summary["attempts"] == 2
    assert summary["failures"] == 1
    assert summary["window_days"] == 7
    assert summary["recent_attempts"] == 1
    assert summary["recent_failures"] == 0
    assert summary["categories"] == {}


def test_events_past_retention_are_pruned_but_totals_survive() -> None:
    health.record_outcome("openai/gpt-4o", ok=False, category="timeout")
    ancient = (
        datetime.now(UTC) - timedelta(days=health.EVENT_RETENTION_DAYS + 1)
    ).isoformat()
    with health._connect() as conn:
        conn.execute("UPDATE model_events SET at = ?", (ancient,))

    health.record_outcome("openai/gpt-4o", ok=True)

    with health._connect() as conn:
        remaining = conn.execute("SELECT COUNT(*) FROM model_events").fetchone()[0]
    summary = health.model_health("openai/gpt-4o")

    assert remaining == 1
    assert summary["attempts"] == 2
    assert summary["failures"] == 1


def test_listing_covers_every_model_seen() -> None:
    health.record_outcome("openai/gpt-4o", ok=True)
    health.record_outcome("anthropic/claude-opus-5", ok=False, category="timeout")

    listed = health.list_model_health()

    assert set(listed) == {"openai/gpt-4o", "anthropic/claude-opus-5"}
    assert listed["anthropic/claude-opus-5"]["failure_rate"] == 1.0


def test_clearing_forgets_one_model_or_all_of_them() -> None:
    health.record_outcome("openai/gpt-4o", ok=True)
    health.record_outcome("anthropic/claude-opus-5", ok=True)

    health.clear_model_health("openai/gpt-4o")
    assert set(health.list_model_health()) == {"anthropic/claude-opus-5"}

    health.clear_model_health()
    assert health.list_model_health() == {}


def test_recording_can_be_turned_off(monkeypatch) -> None:
    monkeypatch.setenv("BASEMODE_NO_HEALTH", "1")

    health.record_outcome("openai/gpt-4o", ok=True)

    assert health.list_model_health() == {}


def test_recording_never_raises_at_the_call_site(monkeypatch) -> None:
    import sqlite3

    def boom(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(health, "_connect", boom)

    health.record_outcome("openai/gpt-4o", ok=True)
    assert health.model_health("openai/gpt-4o") is None


class _FakeStrategy:
    """Stands in for a real strategy; only its name is read once _stream_tokens
    is patched out."""

    name = "system"


def _patch_stream(monkeypatch, tokens=(), error=None):
    from basemode import continue_ as continue_module

    async def fake_stream_tokens(strat, prefix, generation_prefix, fragment, params):
        for token in tokens:
            yield token
        if error is not None:
            raise error

    monkeypatch.setattr(continue_module, "_stream_tokens", fake_stream_tokens)
    monkeypatch.setattr(
        continue_module, "detect_strategy", lambda model, strategy: _FakeStrategy()
    )


async def _drain(agen):
    return [token async for token in agen]


async def test_a_successful_continuation_is_recorded(monkeypatch) -> None:
    from basemode import continue_text

    _patch_stream(monkeypatch, tokens=[" alpha"])

    await _drain(continue_text("Seed", "gpt-4o-mini"))

    summary = health.model_health("openai/gpt-4o-mini")
    assert summary["attempts"] == 1
    assert summary["successes"] == 1


async def test_a_failed_continuation_is_recorded_with_its_category(monkeypatch) -> None:
    import pytest

    from basemode import continue_text

    class RateLimitError(RuntimeError):
        status_code = 429

    _patch_stream(monkeypatch, error=RateLimitError("slow down"))

    with pytest.raises(RateLimitError):
        await _drain(continue_text("Seed", "gpt-4o-mini"))

    summary = health.model_health("openai/gpt-4o-mini")
    assert summary["failures"] == 1
    assert summary["categories"] == {"rate_limit": 1}
    assert summary["last_status"] == 429


async def test_a_stream_that_yields_nothing_counts_as_an_empty_response(
    monkeypatch,
) -> None:
    from basemode import continue_text

    _patch_stream(monkeypatch, tokens=[])

    await _drain(continue_text("Seed", "gpt-4o-mini"))

    summary = health.model_health("openai/gpt-4o-mini")
    assert summary["failures"] == 1
    assert summary["categories"] == {"empty_response": 1}


async def test_a_caller_that_classifies_for_itself_can_opt_out(monkeypatch) -> None:
    from basemode import continue_text

    _patch_stream(monkeypatch, tokens=[" alpha"])

    await _drain(continue_text("Seed", "gpt-4o-mini", record_health=False))

    assert health.list_model_health() == {}


def test_classify_error_reads_a_nested_response_status() -> None:
    class ProviderError(RuntimeError):
        class response:
            status_code = 503

    assert health.classify_error(ProviderError()) == ("provider_unavailable", 503)


def test_classify_error_recognises_an_empty_completion() -> None:
    from basemode.exceptions import EmptyCompletionError

    error = EmptyCompletionError(model="gpt-4o", strategy="system")

    assert health.classify_error(error) == ("empty_response", None)

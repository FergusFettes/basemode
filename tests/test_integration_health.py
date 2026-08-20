import pytest
import test_integration as integration


def test_detects_provider_model_not_found_as_retired() -> None:
    error = Exception(
        "AnthropicException - {'error': {'type': 'not_found_error', "
        "'message': 'model: claude-sonnet-4-20250514'}}"
    )

    assert integration._is_retired_model_error(error)


def test_does_not_treat_unrelated_errors_as_retired() -> None:
    assert not integration._is_retired_model_error(Exception("rate limit exceeded"))


@pytest.mark.asyncio
async def test_records_retired_model_as_an_expected_health_failure(monkeypatch) -> None:
    async def unavailable(*_args, **_kwargs):
        raise Exception(
            "AnthropicException - {'error': {'type': 'not_found_error', "
            "'message': 'model: claude-sonnet-4-20250514'}}"
        )
        yield ""  # pragma: no cover

    integration._HEALTH_ROWS.clear()
    monkeypatch.setattr(integration, "continue_text", unavailable)

    with pytest.raises(pytest.xfail.Exception):
        await integration._run_probe(
            prefix="A short prefix",
            model="anthropic/claude-sonnet-4-20250514",
            strategy=None,
            max_tokens=10,
            test_kind="provider_depth",
        )

    assert integration._HEALTH_ROWS[-1]["status"] == "xfail_retired_model"

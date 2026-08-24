from types import SimpleNamespace

import pytest

from basemode import evidence, verify


@pytest.mark.asyncio
async def test_known_catalog_eligibility_overrides_name_fallback(monkeypatch) -> None:
    async def continuation(*args, **kwargs):
        yield " text output"

    monkeypatch.setattr(verify, "continue_text", continuation)
    monkeypatch.setattr(
        verify,
        "estimate_usage",
        lambda *args: SimpleNamespace(
            prompt_tokens=1, completion_tokens=1, cost_usd=None, is_estimate=True
        ),
    )
    evidence.record_catalog_observation(
        "openrouter/acme/image-reasoner",
        source="test-catalog",
        available=True,
        metadata={"input_modalities": ["image"], "output_modalities": ["text"]},
    )

    summary = await verify.verify_models(["openrouter/acme/image-reasoner"])

    assert summary.successes == 1


@pytest.mark.asyncio
async def test_quick_verification_records_success(monkeypatch) -> None:
    async def continuation(*args, **kwargs):
        callback = kwargs["on_usage"]
        callback([])
        yield " onward"

    monkeypatch.setattr(verify, "continue_text", continuation)
    monkeypatch.setattr(
        verify,
        "estimate_usage",
        lambda *args: SimpleNamespace(
            prompt_tokens=8,
            completion_tokens=2,
            cost_usd=0.001,
            is_estimate=True,
        ),
    )
    summary = await verify.verify_models(["openai/test"], suite="quick")
    assert summary.successes == 1
    with evidence.connect() as db:
        run = db.execute("SELECT suite,status FROM verification_runs").fetchone()
        attempt = db.execute(
            "SELECT outcome,output_characters,cost_source FROM verification_attempts"
        ).fetchone()
    assert tuple(run) == ("quick", "completed")
    assert tuple(attempt) == ("success", 7, "estimated")


@pytest.mark.asyncio
async def test_verification_records_self_healing_steps(monkeypatch) -> None:
    calls = 0

    async def continuation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("unsupported parameter: `thinking`")
        yield " recovered"

    monkeypatch.setattr(verify, "continue_text", continuation)
    monkeypatch.setattr(
        verify,
        "estimate_usage",
        lambda *args: SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            cost_usd=None,
            is_estimate=True,
        ),
    )
    summary = await verify.verify_models(["openai/test"])
    assert summary.successes == 1
    with evidence.connect() as db:
        rows = db.execute(
            "SELECT outcome,compatibility_actions_json FROM verification_attempts ORDER BY id"
        ).fetchall()
    assert [row["outcome"] for row in rows] == ["failure", "success"]
    assert "disable_reasoning" in rows[1]["compatibility_actions_json"]


@pytest.mark.asyncio
async def test_request_limit_is_hard_and_run_resumes(monkeypatch) -> None:
    calls = 0

    async def continuation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("unsupported parameter")
        yield " recovered"

    monkeypatch.setattr(verify, "continue_text", continuation)
    monkeypatch.setattr(
        verify,
        "get_price_info",
        lambda model: SimpleNamespace(pricing_available=False),
    )
    monkeypatch.setattr(
        verify,
        "estimate_usage",
        lambda *args: SimpleNamespace(
            prompt_tokens=1, completion_tokens=1, cost_usd=None, is_estimate=True
        ),
    )
    first = await verify.verify_models(["openai/test"], max_requests=1)
    assert (first.status, first.requests, first.attempts) == ("limited", 1, 0)

    resumed = await verify.verify_models(None, run_id=first.run_id, max_requests=1)
    assert (resumed.status, resumed.requests, resumed.successes) == ("completed", 1, 1)
    with evidence.connect() as db:
        rows = db.execute(
            "SELECT attempt_number,outcome FROM verification_attempts ORDER BY attempt_number"
        ).fetchall()
    assert [tuple(row) for row in rows] == [(10, "failure"), (11, "success")]


@pytest.mark.asyncio
async def test_provider_and_global_concurrency_are_bounded(monkeypatch) -> None:
    active = global_peak = 0
    provider_active: dict[str, int] = {}
    provider_peak: dict[str, int] = {}

    async def continuation(prefix, model, **kwargs):
        nonlocal active, global_peak
        provider = model.split("/", 1)[0]
        active += 1
        provider_active[provider] = provider_active.get(provider, 0) + 1
        global_peak = max(global_peak, active)
        provider_peak[provider] = max(
            provider_peak.get(provider, 0), provider_active[provider]
        )
        await __import__("asyncio").sleep(0.01)
        active -= 1
        provider_active[provider] -= 1
        yield " ok"

    monkeypatch.setattr(verify, "continue_text", continuation)
    monkeypatch.setattr(
        verify,
        "get_price_info",
        lambda model: SimpleNamespace(pricing_available=False),
    )
    monkeypatch.setattr(
        verify,
        "estimate_usage",
        lambda *args: SimpleNamespace(
            prompt_tokens=1, completion_tokens=1, cost_usd=None, is_estimate=True
        ),
    )
    summary = await verify.verify_models(
        ["openai/a", "openai/b", "anthropic/a", "anthropic/b"],
        concurrency=3,
        per_provider_concurrency=1,
    )
    assert summary.successes == 4
    assert global_peak <= 3
    assert all(peak <= 1 for peak in provider_peak.values())


@pytest.mark.asyncio
async def test_known_cost_ceiling_prevents_request(monkeypatch) -> None:
    async def continuation(*args, **kwargs):
        raise AssertionError("request must not start above known cost ceiling")
        yield  # pragma: no cover

    monkeypatch.setattr(verify, "continue_text", continuation)
    monkeypatch.setattr(
        verify,
        "get_price_info",
        lambda model: SimpleNamespace(
            pricing_available=True, output_cost_per_token=1.0
        ),
    )
    monkeypatch.setattr(
        verify,
        "estimate_usage",
        lambda *args: SimpleNamespace(cost_usd=1.0),
    )
    summary = await verify.verify_models(["openai/test"], max_cost_usd=1.0)
    assert (summary.status, summary.requests) == ("limited", 0)

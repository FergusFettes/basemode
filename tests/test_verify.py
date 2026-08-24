from types import SimpleNamespace

import pytest

from basemode import evidence, verify


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

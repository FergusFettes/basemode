from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from basemode import evidence, verification_plan
from basemode.cli import app


def _catalog(
    model: str, *, release: str = "2026-08-01", modality: str = "text"
) -> None:
    evidence.record_catalog_observation(
        model,
        source="test",
        available=True,
        metadata={"release_date": release, "modality": modality},
    )


def test_plan_is_text_only_staged_and_stably_ordered(monkeypatch) -> None:
    _catalog("zeta/new")
    _catalog("alpha/old")
    _catalog("alpha/image", modality="image")
    run = evidence.start_run("quick")
    evidence.record_attempt(
        run,
        "alpha/old",
        probe_kind="continuation",
        attempt_number=1,
        outcome="failure",
        failure_class="rate_limit",
    )
    evidence.finish_run(run)
    monkeypatch.setattr(
        verification_plan,
        "get_price_info",
        lambda model: SimpleNamespace(
            pricing_available=False,
            input_cost_per_token=None,
            output_cost_per_token=None,
        ),
    )

    plan = verification_plan.plan_verification(catalog_available=True)

    assert [target.model for target in plan.targets] == ["alpha/old", "zeta/new"]
    assert [target.stage for target in plan.targets] == ["transient", "never-tested"]
    assert plan.maximum_requests == 6
    assert plan.provider_counts == {"alpha": 1, "zeta": 1}


def test_plan_filters_provider_status_and_release(monkeypatch) -> None:
    _catalog("openai/new", release="2026-08-20")
    _catalog("openai/old", release="2025-01-01")
    _catalog("other/new", release="2026-08-20")
    monkeypatch.setattr(
        verification_plan,
        "get_price_info",
        lambda model: SimpleNamespace(
            pricing_available=True,
            input_cost_per_token=0.000001,
            output_cost_per_token=0.000002,
        ),
    )

    plan = verification_plan.plan_verification(
        providers=["openai"],
        statuses=["never-tested"],
        released_since="2026-08-01",
        suite="thorough",
        attempts=2,
    )

    assert [target.model for target in plan.targets] == ["openai/new"]
    assert plan.logical_probes == 6
    assert plan.maximum_requests == 18
    assert plan.priced_targets == 1
    assert plan.estimated_known_max_cost_usd > 0


def test_dry_run_cli_never_calls_verifier(monkeypatch) -> None:
    _catalog("openai/test")

    async def forbidden(*args, **kwargs):
        raise AssertionError("dry-run made a live call")

    monkeypatch.setattr("basemode.verify.verify_models", forbidden)
    result = CliRunner().invoke(
        app,
        ["verify", "--from-catalog", "--dry-run", "--json"],
    )
    assert result.exit_code == 0, result.output
    assert '"maximum_requests": 3' in result.output
    assert '"model": "openai/test"' in result.output


def test_unknown_status_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown status"):
        verification_plan.plan_verification(statuses=["magic"])


def test_explicit_unknown_text_model_is_plannable_but_media_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        verification_plan,
        "get_price_info",
        lambda model: SimpleNamespace(
            pricing_available=False,
            input_cost_per_token=None,
            output_cost_per_token=None,
        ),
    )
    plan = verification_plan.plan_verification(["newco/chat-v1"])
    assert [target.model for target in plan.targets] == ["newco/chat-v1"]
    with pytest.raises(ValueError, match="not eligible"):
        verification_plan.plan_verification(["newco/video-v1"])

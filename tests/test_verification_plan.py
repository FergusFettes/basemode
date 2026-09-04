from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from basemode import observations, verification_plan
from basemode.cli import app


@pytest.fixture(autouse=True)
def isolated_catalog(monkeypatch) -> None:
    monkeypatch.setattr(verification_plan, "list_catalog_endpoint_metadata", lambda: [])
    monkeypatch.setattr(
        verification_plan, "list_available_endpoint_metadata", lambda: []
    )


def _catalog(
    model: str, *, release: str = "2026-08-01", modality: str = "text"
) -> None:
    observations.record_endpoint_metadata(
        model,
        text_eligible=modality == "text",
        modality=modality,
        release_date=release,
        catalog_available=True,
    )


def test_plan_is_text_only_staged_and_stably_ordered(monkeypatch) -> None:
    _catalog("zeta/new")
    _catalog("alpha/old")
    _catalog("alpha/image", modality="image")
    operation = observations.observe_operation("alpha/old", "system", "heuristic", None)
    attempt = operation.begin_attempt("initial")
    error = RuntimeError("limited")
    error.status_code = 429
    attempt.finish("failure", error)
    operation.finish("failure", returned_content=False)
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


def test_from_catalog_uses_packaged_models_on_a_fresh_ledger(monkeypatch) -> None:
    monkeypatch.setattr(
        verification_plan,
        "list_catalog_endpoint_metadata",
        lambda: [
            {
                "model": "openai/catalog-model",
                "provider": "openai",
                "release_date": "2026-08-01",
                "text_eligible": True,
                "catalog_available": True,
            }
        ],
    )

    plan = verification_plan.plan_verification(
        catalog_available=True, statuses=["never-tested"]
    )

    assert [target.model for target in plan.targets] == ["openai/catalog-model"]


def test_available_discovers_configured_provider_models(monkeypatch) -> None:
    monkeypatch.setattr(
        verification_plan,
        "list_available_endpoint_metadata",
        lambda: [
            {
                "model": "groq/available-model",
                "provider": "groq",
                "release_date": None,
                "text_eligible": True,
                "catalog_available": None,
            }
        ],
    )

    plan = verification_plan.plan_verification(
        available_only=True, statuses=["never-tested"]
    )

    assert [target.model for target in plan.targets] == ["groq/available-model"]


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

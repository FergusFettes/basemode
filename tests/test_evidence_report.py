import json

from typer.testing import CliRunner

from basemode import evidence, evidence_report
from basemode.cli import app

runner = CliRunner()


def _seed() -> None:
    with evidence.connect() as db:
        evidence.ensure_endpoint("openai/text-model", modality="text", conn=db)
        evidence.ensure_endpoint("together/image-model", modality="image", conn=db)
        evidence.record_catalog_observation(
            "openai/text-model", source="test", available=True, conn=db
        )
        run = evidence.start_run("quick", conn=db)
        evidence.record_attempt(
            run,
            "openai/text-model",
            probe_kind="continuation",
            attempt_number=0,
            outcome="failure",
            failure_class="rate_limit",
            conn=db,
        )
        evidence.record_attempt(
            run,
            "together/image-model",
            probe_kind="continuation",
            attempt_number=0,
            outcome="failure",
            failure_class="rate_limit",
            conn=db,
        )
        evidence.finish_run(run, conn=db)


def test_provider_and_transient_reports_exclude_non_text() -> None:
    _seed()
    with evidence.connect() as db:
        assert [row["provider"] for row in evidence_report.providers(db)] == ["openai"]
        assert [row["model"] for row in evidence_report.transient(db)] == [
            "openai/text-model"
        ]


def test_cli_evidence_json_and_tables() -> None:
    _seed()
    result = runner.invoke(app, ["evidence", "providers", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["provider"] == "openai"

    result = runner.invoke(app, ["evidence", "failures"])
    assert result.exit_code == 0
    assert "rate_limit" in result.output

    result = runner.invoke(app, ["evidence", "rechecks", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)[0]["operational_status"] == "suspected_transient"


def test_export_is_jsonl_sanitized_and_text_only() -> None:
    _seed()
    result = runner.invoke(app, ["evidence", "export"])
    assert result.exit_code == 0
    records = [json.loads(line) for line in result.output.splitlines()]
    assert any(row["type"] == "attempt" for row in records)
    assert all(row.get("model") != "together/image-model" for row in records)
    assert all("request_params_json" not in row for row in records)
    assert all("configuration_json" not in row for row in records)
    assert any(row["type"] == "recheck" for row in records)


def test_endpoint_detail_includes_attempt_history() -> None:
    _seed()
    result = runner.invoke(app, ["evidence", "endpoint", "openai/text-model"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["endpoint"]["transient_failure"] is True
    assert payload["attempts"][0]["failure_class"] == "rate_limit"

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import model_reliability as mr  # noqa: E402
import record_provider_health as rph  # noqa: E402


def test_record_appends_rows_tagged_with_run_at(tmp_path, monkeypatch) -> None:
    snapshot_path = tmp_path / "provider_health.json"
    history_path = tmp_path / "history.jsonl"
    monkeypatch.setattr(rph, "SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr(rph, "HISTORY_PATH", history_path)

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-08-11T00:00:00+00:00",
                "rows": [
                    {"model": "openai/gpt-4o-mini", "status": "ok"},
                    {
                        "model": "anthropic/claude-opus-5",
                        "status": "error",
                        "error": "boom",
                    },
                ],
            }
        )
    )

    result = rph.main()
    assert result == 0

    lines = history_path.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["run_at"] == "2026-08-11T00:00:00+00:00"
    assert first["model"] == "openai/gpt-4o-mini"
    assert first["status"] == "ok"


def test_record_is_a_noop_when_snapshot_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rph, "SNAPSHOT_PATH", tmp_path / "missing.json")
    history_path = tmp_path / "history.jsonl"
    monkeypatch.setattr(rph, "HISTORY_PATH", history_path)

    assert rph.main() == 0
    assert not history_path.exists()


def test_record_appends_across_multiple_runs(tmp_path, monkeypatch) -> None:
    snapshot_path = tmp_path / "provider_health.json"
    history_path = tmp_path / "history.jsonl"
    monkeypatch.setattr(rph, "SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr(rph, "HISTORY_PATH", history_path)

    for run_at in ["2026-08-04T00:00:00+00:00", "2026-08-11T00:00:00+00:00"]:
        snapshot_path.write_text(
            json.dumps(
                {
                    "generated_at_utc": run_at,
                    "rows": [{"model": "openai/gpt-4o-mini", "status": "ok"}],
                }
            )
        )
        rph.main()

    lines = history_path.read_text().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["run_at"] for line in lines] == [
        "2026-08-04T00:00:00+00:00",
        "2026-08-11T00:00:00+00:00",
    ]


def test_summarize_computes_success_rate_and_last_status() -> None:
    rows = [
        {
            "model": "openai/gpt-4o-mini",
            "status": "ok",
            "run_at": "2026-08-04T00:00:00Z",
        },
        {
            "model": "openai/gpt-4o-mini",
            "status": "error",
            "run_at": "2026-08-11T00:00:00Z",
            "error": "timeout",
        },
        {
            "model": "anthropic/claude-opus-5",
            "status": "ok",
            "run_at": "2026-08-11T00:00:00Z",
        },
    ]

    summary = mr.summarize(rows)

    by_model = {s["model"]: s for s in summary}
    gpt = by_model["openai/gpt-4o-mini"]
    assert gpt["runs"] == 2
    assert gpt["ok"] == 1
    assert gpt["success_rate"] == 0.5
    assert gpt["last_status"] == "error"
    assert gpt["last_error"] == "timeout"

    claude = by_model["anthropic/claude-opus-5"]
    assert claude["runs"] == 1
    assert claude["success_rate"] == 1.0


def test_summarize_sorts_by_run_at_to_find_last_status() -> None:
    # Out-of-order input shouldn't matter — summarize must sort by run_at itself.
    rows = [
        {"model": "m", "status": "ok", "run_at": "2026-08-11T00:00:00Z"},
        {"model": "m", "status": "error", "run_at": "2026-08-04T00:00:00Z"},
    ]
    summary = mr.summarize(rows)
    assert summary[0]["last_status"] == "ok"


def test_load_rows_returns_empty_when_no_history_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mr, "HISTORY_PATH", tmp_path / "missing.jsonl")
    assert mr.load_rows() == []

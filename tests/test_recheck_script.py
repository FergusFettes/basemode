from __future__ import annotations

import runpy
from pathlib import Path


def test_scheduled_recheck_script_is_opt_in(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.delenv("BASEMODE_SCHEDULED_RECHECKS", raising=False)
    monkeypatch.setenv("BASEMODE_RECHECK_ARTIFACT", str(tmp_path / "result.json"))
    module = runpy.run_path(str(Path("scripts/run_transient_rechecks.py")))

    assert module["main"]() == 0
    assert "disabled" in capsys.readouterr().out
    assert not (tmp_path / "result.json").exists()


def test_enabled_script_with_no_due_configured_models_writes_empty_artifact(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BASEMODE_SCHEDULED_RECHECKS", "1")
    output = tmp_path / "result.json"
    monkeypatch.setenv("BASEMODE_RECHECK_ARTIFACT", str(output))
    script = Path(__file__).parents[1] / "scripts" / "run_transient_rechecks.py"
    module = runpy.run_path(str(script))

    assert module["main"]() == 0
    assert '"selected_models": []' in output.read_text()

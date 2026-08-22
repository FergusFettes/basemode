import json
from pathlib import Path

import pytest

from basemode import keys


@pytest.fixture(autouse=True)
def isolated_key_store(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep unit tests off the developer's real ~/.config/basemode/auth.json.

    Without this, a locally-stored default model or pinned strategy would
    leak into assertions, and a test that writes would clobber real keys.
    Integration tests are exempt: they need the stored provider keys.
    """
    if request.node.get_closest_marker("integration"):
        return
    monkeypatch.setattr(keys, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(keys, "_AUTH_FILE", tmp_path / "auth.json")


@pytest.fixture
def prefix() -> str:
    return "The ship rounded the headland and"


def pytest_terminal_summary(terminalreporter, exitstatus: int, config) -> None:
    path = Path("dist/integration/provider_health.json")
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return
    cost = summary.get("estimated_total_cost_usd")
    rows_total = summary.get("rows_total")
    rows_with_errors = summary.get("rows_with_errors")
    if not isinstance(cost, int | float):
        return
    terminalreporter.write_sep("-", "integration suite estimated cost")
    terminalreporter.write_line(f"estimated_total_cost_usd={cost:.8f}")
    if rows_total is not None:
        terminalreporter.write_line(f"rows_total={rows_total}")
    if rows_with_errors is not None:
        terminalreporter.write_line(f"rows_with_errors={rows_with_errors}")

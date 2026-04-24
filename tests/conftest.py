import json
from pathlib import Path

import pytest


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
    if not isinstance(cost, (int, float)):
        return
    terminalreporter.write_sep("-", "integration suite estimated cost")
    terminalreporter.write_line(f"estimated_total_cost_usd={cost:.8f}")
    if rows_total is not None:
        terminalreporter.write_line(f"rows_total={rows_total}")
    if rows_with_errors is not None:
        terminalreporter.write_line(f"rows_with_errors={rows_with_errors}")

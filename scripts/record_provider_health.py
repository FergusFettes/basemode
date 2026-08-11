#!/usr/bin/env python3
"""Append the latest integration-test provider health report to a running JSONL log.

`tests/test_integration.py` writes a snapshot to dist/integration/provider_health.json
on every `pytest -m integration` run (pass or fail). This script appends each row from
that snapshot — tagged with the run timestamp — onto data/provider_health_history.jsonl,
so reliability can be tracked over time instead of only seeing the latest run.

Safe to run when the snapshot is missing (e.g. the test step crashed before writing
one) — it just does nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "dist" / "integration" / "provider_health.json"
HISTORY_PATH = ROOT / "data" / "provider_health_history.jsonl"


def main() -> int:
    if not SNAPSHOT_PATH.exists():
        print(f"No snapshot at {SNAPSHOT_PATH}; nothing to record.")
        return 0

    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    run_at = snapshot.get("generated_at_utc")
    rows = snapshot.get("rows", [])
    if not rows:
        print("Snapshot has no rows; nothing to record.")
        return 0

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a") as f:
        for row in rows:
            f.write(json.dumps({"run_at": run_at, **row}) + "\n")

    print(f"Appended {len(rows)} row(s) from run {run_at} to {HISTORY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

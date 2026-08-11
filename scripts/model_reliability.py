#!/usr/bin/env python3
"""Summarize per-model reliability from data/provider_health_history.jsonl.

Usage:
    uv run python scripts/model_reliability.py [--model SUBSTRING] [--json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT / "data" / "provider_health_history.jsonl"


def load_rows() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    rows = []
    for line in HISTORY_PATH.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    summary = []
    for model, model_rows in sorted(by_model.items()):
        model_rows.sort(key=lambda r: r.get("run_at") or "")
        total = len(model_rows)
        ok = sum(1 for r in model_rows if r.get("status") == "ok")
        last = model_rows[-1]
        summary.append(
            {
                "model": model,
                "runs": total,
                "ok": ok,
                "success_rate": round(ok / total, 4) if total else None,
                "last_run_at": last.get("run_at"),
                "last_status": last.get("status"),
                "last_error": last.get("error"),
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Only show models whose name contains this")
    parser.add_argument(
        "--json", action="store_true", help="Print JSON instead of a table"
    )
    args = parser.parse_args()

    rows = load_rows()
    summary = summarize(rows)
    if args.model:
        summary = [s for s in summary if args.model.lower() in s["model"].lower()]

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    if not summary:
        print("No history recorded yet.")
        return 0

    print(f"{'Model':<45} {'Runs':>5} {'OK':>4} {'Rate':>7}  Last status")
    for s in summary:
        rate = (
            f"{s['success_rate'] * 100:.0f}%"
            if s["success_rate"] is not None
            else "n/a"
        )
        print(
            f"{s['model']:<45} {s['runs']:>5} {s['ok']:>4} {rate:>7}  {s['last_status']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

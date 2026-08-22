#!/usr/bin/env python3
"""Summarize per-model reliability from data/provider_health_history.jsonl.

Usage:
    uv run python scripts/model_reliability.py [--model SUBSTRING] [--json]
"""

from __future__ import annotations

import argparse
import json
import statistics
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
        ttft_values = [
            row["time_to_first_token_s"]
            for row in model_rows
            if isinstance(row.get("time_to_first_token_s"), int | float)
        ]
        throughput_values = [
            row["output_tokens_per_s"]
            for row in model_rows
            if isinstance(row.get("output_tokens_per_s"), int | float)
        ]
        summary.append(
            {
                "model": model,
                "runs": total,
                "ok": ok,
                "success_rate": round(ok / total, 4) if total else None,
                "last_run_at": last.get("run_at"),
                "last_status": last.get("status"),
                "last_error": last.get("error"),
                "last_time_to_first_token_s": last.get("time_to_first_token_s"),
                "last_output_tokens_per_s": last.get("output_tokens_per_s"),
                "median_time_to_first_token_s": (
                    round(statistics.median(ttft_values), 3) if ttft_values else None
                ),
                "median_output_tokens_per_s": (
                    round(statistics.median(throughput_values), 2)
                    if throughput_values
                    else None
                ),
            }
        )
    return summary


def render_markdown(summary: list[dict]) -> str:
    """Render the health history as a committed MkDocs page."""
    lines = [
        "# Provider Health",
        "",
        "Results from the weekly live-provider health check. Models without a "
        "configured API key are skipped. A quota/credit exhaustion is recorded "
        "as an expected failure so monitoring remains green; other errors are "
        "real failures.",
        "",
    ]
    if not summary:
        return "\n".join([*lines, "No provider-health history recorded yet.", ""])

    latest = max((row.get("last_run_at") or "" for row in summary), default="")
    lines.extend(
        [
            f"Latest recorded run: `{latest}`",
            "",
            "| Model | Runs | Success | Last status | Median TTFT | Median tok/s |",
            "|---|---:|---:|---|---:|---:|",
        ]
    )
    for row in summary:
        ttft = (
            f"{row['median_time_to_first_token_s']:.3f}s"
            if row["median_time_to_first_token_s"] is not None
            else "—"
        )
        throughput = (
            f"{row['median_output_tokens_per_s']:.1f}"
            if row["median_output_tokens_per_s"] is not None
            else "—"
        )
        lines.append(
            f"| `{row['model']}` | {row['runs']} | "
            f"{row['success_rate'] * 100:.0f}% | {row['last_status']} | "
            f"{ttft} | {throughput} |"
        )
    return "\n".join([*lines, ""])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Only show models whose name contains this")
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--json", action="store_true", help="Print JSON instead of a table"
    )
    output.add_argument(
        "--markdown", action="store_true", help="Print a MkDocs health page"
    )
    parser.add_argument(
        "--output", type=Path, help="Write the selected output format to this path"
    )
    args = parser.parse_args()

    rows = load_rows()
    summary = summarize(rows)
    if args.model:
        summary = [s for s in summary if args.model.lower() in s["model"].lower()]

    if args.json:
        rendered = json.dumps(summary, indent=2) + "\n"
        if args.output:
            args.output.write_text(rendered)
        else:
            print(rendered, end="")
        return 0
    if args.markdown:
        rendered = render_markdown(summary)
        if args.output:
            args.output.write_text(rendered)
        else:
            print(rendered, end="")
        return 0

    if not summary:
        print("No history recorded yet.")
        return 0

    print(
        f"{'Model':<45} {'Runs':>5} {'OK':>4} {'Rate':>7} "
        f"{'TTFT':>8} {'tok/s':>8}  Last status"
    )
    for s in summary:
        rate = (
            f"{s['success_rate'] * 100:.0f}%"
            if s["success_rate"] is not None
            else "n/a"
        )
        ttft = (
            f"{s['median_time_to_first_token_s']:.3f}"
            if s["median_time_to_first_token_s"] is not None
            else "n/a"
        )
        throughput = (
            f"{s['median_output_tokens_per_s']:.1f}"
            if s["median_output_tokens_per_s"] is not None
            else "n/a"
        )
        print(
            f"{s['model']:<45} {s['runs']:>5} {s['ok']:>4} {rate:>7} "
            f"{ttft:>8} {throughput:>8}  {s['last_status']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize per-model reliability from data/provider_health_history.jsonl.

Usage:
    uv run python scripts/model_reliability.py [--model SUBSTRING] [--json]
"""

from __future__ import annotations

import argparse
import html
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


def render_html(summary: list[dict]) -> str:
    """Render a standalone, locally viewable provider-health dashboard."""
    total_runs = sum(row["runs"] for row in summary)
    total_ok = sum(row["ok"] for row in summary)
    success_rate = f"{total_ok / total_runs * 100:.1f}%" if total_runs else "—"
    latest = max((row.get("last_run_at") or "" for row in summary), default="—")
    rows = (
        "\n".join(
            "<tr>"
            f"<td>{html.escape(row['model'])}</td>"
            f"<td>{row['runs']}</td>"
            f"<td>{row['success_rate'] * 100:.0f}%</td>"
            f'<td class="{html.escape(row["last_status"])}">'
            f"{html.escape(row['last_status'])}</td>"
            f"<td>{_format_seconds(row['median_time_to_first_token_s'])}</td>"
            f"<td>{_format_throughput(row['median_output_tokens_per_s'])}</td>"
            "</tr>"
            for row in summary
        )
        or '<tr><td colspan="6">No provider-health history recorded yet.</td></tr>'
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>basemode Provider Health</title>
  <style>
    :root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
    body {{ max-width: 1100px; margin: 2rem auto; padding: 0 1rem; background: #10131a; color: #e8edf5; }}
    h1 {{ margin-bottom: .25rem; }}
    .muted {{ color: #9ba8ba; }}
    .cards {{ display: flex; flex-wrap: wrap; gap: 1rem; margin: 1.5rem 0; }}
    .card {{ min-width: 10rem; padding: 1rem; border-radius: .6rem; background: #1a2130; }}
    .label {{ color: #9ba8ba; font-size: .85rem; }}
    .value {{ font-size: 1.35rem; font-weight: 700; margin-top: .2rem; }}
    input {{ width: min(28rem, 100%); box-sizing: border-box; padding: .65rem; border: 1px solid #39465c; border-radius: .4rem; background: #151c28; color: inherit; }}
    table {{ width: 100%; margin-top: 1rem; border-collapse: collapse; background: #151c28; }}
    th, td {{ padding: .65rem .8rem; text-align: left; border-bottom: 1px solid #2a3547; }}
    th {{ color: #9ba8ba; font-size: .8rem; text-transform: uppercase; }}
    td:nth-child(n+2) {{ font-variant-numeric: tabular-nums; }}
    .ok {{ color: #72d6a2; }} .error {{ color: #ff8585; }}
    [class^="xfail_"] {{ color: #f3c969; }}
  </style>
</head>
<body>
  <h1>Provider Health</h1>
  <p class="muted">Latest recorded run: {html.escape(latest)}</p>
  <div class="cards">
    <div class="card"><div class="label">Models observed</div><div class="value">{len(summary)}</div></div>
    <div class="card"><div class="label">Recorded probes</div><div class="value">{total_runs}</div></div>
    <div class="card"><div class="label">Success rate</div><div class="value">{success_rate}</div></div>
  </div>
  <input id="filter" type="search" placeholder="Filter models or statuses" autofocus>
  <table>
    <thead><tr><th>Model</th><th>Runs</th><th>Success</th><th>Last status</th><th>Median TTFT</th><th>Median tok/s</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <script>
    const input = document.querySelector('#filter');
    input.addEventListener('input', () => document.querySelectorAll('tbody tr').forEach(
      row => row.hidden = !row.textContent.toLowerCase().includes(input.value.toLowerCase())
    ));
  </script>
</body>
</html>
"""


def _format_seconds(value: float | None) -> str:
    return f"{value:.3f}s" if value is not None else "—"


def _format_throughput(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "—"


def _write_output(path: Path | None, rendered: str) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
    else:
        print(rendered, end="")


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
    output.add_argument(
        "--html", action="store_true", help="Print a standalone HTML dashboard"
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
        _write_output(args.output, rendered)
        return 0
    if args.markdown:
        rendered = render_markdown(summary)
        _write_output(args.output, rendered)
        return 0
    if args.html:
        _write_output(args.output, render_html(summary))
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

#!/usr/bin/env python3
"""Generate a unified verified-models table for README.

Data sources:
- LiteLLM model metadata/pricing (primary)
- OpenRouter models API (secondary fallback for missing pricing + release-date signal)
- Provider pricing docs URLs (tracked in registry for manual verification)
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from basemode.detect import detect_strategy, normalize_model
from basemode.usage import get_price_info

REGISTRY_PATH = ROOT / "data" / "verified_models_registry.json"
README_PATH = ROOT / "README.md"
DETAILS_PATH = ROOT / "data" / "verified_models_details.json"
DOCS_PATH = ROOT / "docs" / "usage" / "Verified Models.md"
MARKER_START = "<!-- verified-models:start -->"
MARKER_END = "<!-- verified-models:end -->"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


@dataclass(frozen=True)
class Row:
    model: str
    input_cost: float | None
    output_cost: float | None
    release_date: str | None
    prompt_method: str
    reliability: str
    issues: list[str]
    sources: list[str]


def _load_registry() -> list[dict]:
    raw = json.loads(REGISTRY_PATH.read_text())
    return list(raw.get("models", []))


def _fetch_openrouter_models() -> dict[str, dict]:
    try:
        req = urllib.request.Request(
            OPENROUTER_MODELS_URL,
            headers={"User-Agent": "basemode-verified-models/1"},
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return {m["id"]: m for m in payload.get("data", []) if "id" in m}
    except (TimeoutError, urllib.error.URLError, json.JSONDecodeError):
        return {}


def _parse_release_date(meta: dict | None) -> tuple[str | None, str | None]:
    if not meta:
        return None, None

    slug = str(meta.get("canonical_slug") or "")
    m = re.search(r"(20\d{2})[-]?(\d{2})[-]?(\d{2})$", slug)
    if m:
        yyyy, mm, dd = m.groups()
        return f"{yyyy}-{mm}-{dd}", "openrouter_slug"

    created = meta.get("created")
    if isinstance(created, (int, float)) and created > 0:
        dt = datetime.fromtimestamp(created, tz=UTC)
        return dt.date().isoformat(), "openrouter_created"

    return None, None


def _pick_price(
    litellm_input: float | None,
    litellm_output: float | None,
    openrouter_meta: dict | None,
) -> tuple[float | None, float | None, str | None]:
    if litellm_input is not None and litellm_output is not None:
        return litellm_input, litellm_output, "litellm"

    if openrouter_meta:
        pricing = openrouter_meta.get("pricing") or {}
        try:
            prompt = float(pricing.get("prompt"))
            completion = float(pricing.get("completion"))
            return prompt, completion, "openrouter_fallback"
        except (TypeError, ValueError):
            pass

    return litellm_input, litellm_output, None


def _format_per_million(cost: float | None) -> str:
    if cost is None:
        return "unknown"
    return f"${cost * 1_000_000:.2f}"


def _reliability_and_issues(
    *,
    price_source: str | None,
    release_source: str | None,
    known_issues: list[str],
) -> tuple[str, list[str]]:
    issues = list(known_issues)

    if price_source is None:
        issues.append("missing_price")
    elif price_source != "litellm":
        issues.append("price_not_from_litellm")

    if release_source is None:
        issues.append("missing_release_date")
    elif release_source == "openrouter_created":
        issues.append("release_date_is_openrouter_created")

    if issues:
        return "⚠", sorted(set(issues))
    return "✓", []


def _build_rows() -> list[Row]:
    openrouter = _fetch_openrouter_models()
    rows: list[Row] = []

    for entry in _load_registry():
        model = normalize_model(str(entry["model"]))
        price = get_price_info(model)
        prompt_method = detect_strategy(model).name

        or_meta = openrouter.get(entry.get("openrouter_id", ""))
        release_date, release_source = _parse_release_date(or_meta)
        input_cost, output_cost, price_source = _pick_price(
            price.input_cost_per_token,
            price.output_cost_per_token,
            or_meta,
        )

        reliability, issues = _reliability_and_issues(
            price_source=price_source,
            release_source=release_source,
            known_issues=list(entry.get("known_issues", [])),
        )

        sources = [
            "https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json",
            "https://openrouter.ai/docs/api-reference/models/get-models",
        ]
        if entry.get("pricing_url"):
            sources.append(str(entry["pricing_url"]))

        rows.append(
            Row(
                model=model,
                input_cost=input_cost,
                output_cost=output_cost,
                release_date=release_date,
                prompt_method=prompt_method,
                reliability=reliability,
                issues=issues,
                sources=sources,
            )
        )

    return sorted(rows, key=lambda r: r.model)


def _render_table(rows: list[Row]) -> str:
    lines = [
        "| Model | Input cost (/1M) | Output cost (/1M) | Release date | Prompt method | Reliability |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.model}`",
                    _format_per_million(row.input_cost),
                    _format_per_million(row.output_cost),
                    row.release_date or "unknown",
                    f"`{row.prompt_method}`",
                    row.reliability,
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def _inject_readme(table: str) -> None:
    readme = README_PATH.read_text()
    block = (
        f"{MARKER_START}\n"
        "\n"
        "## Verified Models\n"
        "\n"
        "Single generated table, refreshed by CI.\n"
        "\n"
        f"{table}\n"
        "\n"
        "Legend: `✓` = LiteLLM pricing present and release date available; `⚠` = missing/approximate field or known issue.\n"
        "\n"
        f"{MARKER_END}"
    )

    if MARKER_START in readme and MARKER_END in readme:
        pattern = re.compile(
            re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END),
            flags=re.S,
        )
        updated = pattern.sub(block, readme)
    else:
        if not readme.endswith("\n"):
            readme += "\n"
        updated = readme + "\n" + block + "\n"

    README_PATH.write_text(updated)


def _write_details(rows: list[Row]) -> None:
    payload = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "rows": [
            {
                "model": r.model,
                "input_cost_per_token": r.input_cost,
                "output_cost_per_token": r.output_cost,
                "release_date": r.release_date,
                "prompt_method": r.prompt_method,
                "reliability": r.reliability,
                "issues": r.issues,
                "sources": r.sources,
            }
            for r in rows
        ],
    }
    DETAILS_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def _write_docs_page(table: str) -> None:
    content = (
        "# Verified Models\n\n"
        "Single generated table, refreshed by CI.\n\n"
        f"{table}\n\n"
        "Legend: `✓` = LiteLLM pricing present and release date available; "
        "`⚠` = missing/approximate field or known issue.\n"
    )
    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_PATH.write_text(content)


def main() -> int:
    rows = _build_rows()
    table = _render_table(rows)
    _inject_readme(table)
    _write_details(rows)
    _write_docs_page(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

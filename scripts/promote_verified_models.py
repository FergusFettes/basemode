#!/usr/bin/env python3
"""Promote models with clean evidence into the verified-models registry.

`basemode verify --suite thorough` (see `basemode.verify` / `basemode.evidence`)
already probes models and durably records every attempt in the shared
evidence database (`~/.local/share/basemode/model_evidence.sqlite`). A model
counts as `verified` there once its latest thorough run has at least one
success in every logical probe group (`evidence.current_status`) -- that is
a stronger bar than a single "does it respond" quick sweep, which only
proves reachability, not clean output across attempts.

Nothing previously turned that evidence into an entry in
`data/verified_models_registry.json`, so a model could pass every thorough
probe and still never show up in the README/docs table or get picked up by
`detect.select_strategy`. This script closes that gap: for every model the
evidence store calls `verified` that isn't already in the registry, add a
minimal entry (just `"model"`, plus `"pricing_url"` when a mapping is known).
No `prompt_method` is set -- `basemode verify` probes through basemode's own
auto-detected strategy (see `detect.detect_strategy`), so a bare entry lets
that auto-detection keep doing the same job it already proved works.

Run `scripts/generate_verified_models_table.py` afterwards (or `make
models-table`) to propagate any addition into the packaged data file that
`basemode.strategies.compat` reads at runtime.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from basemode.evidence import current_status  # noqa: E402

REGISTRY_PATH = ROOT / "data" / "verified_models_registry.json"

_PRICING_URLS = {
    "openai": "https://openai.com/api/pricing/",
    "anthropic": "https://docs.anthropic.com/en/docs/about-claude/pricing",
    "deepinfra": "https://deepinfra.com/pricing",
    "groq": "https://groq.com/pricing/",
    "cerebras": "https://www.cerebras.ai/pricing",
    "novita": "https://novita.ai/pricing",
}

_INLINE_ARRAY_RE = re.compile(
    r"\[\n\s+(" r'"(?:[^"\\]|\\.)*"(?:,\n\s+"(?:[^"\\]|\\.)*")*' r")\n\s*\]"
)


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _compact_scalar_arrays(rendered: str) -> str:
    """Collapse arrays of strings back onto one line, matching the registry's
    existing hand-formatted style, so a single new entry doesn't reformat
    every other array in the file."""

    def _collapse(match: re.Match[str]) -> str:
        items = re.findall(r'"(?:[^"\\]|\\.)*"', match.group(1))
        return "[" + ", ".join(items) + "]"

    return _INLINE_ARRAY_RE.sub(_collapse, rendered)


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = _compact_scalar_arrays(json.dumps(data, indent=2))
    path.write_text(rendered + "\n")


def main() -> int:
    registry = _load_json(REGISTRY_PATH, {"models": []})
    known = {entry["model"] for entry in registry.get("models", [])}

    status = current_status()
    promoted: list[str] = []
    for model, state in sorted(status.items()):
        if not state.get("verified"):
            continue
        if model in known:
            continue
        provider = model.partition("/")[0]
        entry: dict[str, str] = {"model": model}
        pricing_url = _PRICING_URLS.get(provider)
        if pricing_url:
            entry["pricing_url"] = pricing_url
        registry.setdefault("models", []).append(entry)
        known.add(model)
        promoted.append(model)

    if promoted:
        registry["models"] = sorted(registry["models"], key=lambda e: e["model"])
        _save_json(REGISTRY_PATH, registry)
        print(f"Promoted {len(promoted)} model(s):")
        for model in promoted:
            print(f"  + {model}")
        print("\nRun `make models-table` to propagate into the packaged data file.")
    else:
        print("No new models to promote.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

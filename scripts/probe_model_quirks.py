#!/usr/bin/env python3
"""Sweep every verified model for known API-quirk classes and keep the
registry's `"quirks"` field current.

For each verified model with a configured provider key, this script first
confirms the model works at all through the normal, quirk-respecting path
(a baseline probe), then sends a couple of small probes that deliberately
bypass basemode's own compat layer — constructing `GenerationParams` and
calling the strategy directly — so the request reaches the provider exactly
as if the quirk didn't exist:

- temperature probe: force an explicit, non-default `temperature` through
  `GenerationParams.extra`, which `compat.build_kwargs` applies *last* —
  after any `no_temperature` suppression — so it always reaches the request.
  If the provider rejects it with a temperature-shaped error, the model
  needs `no_temperature`. If it *doesn't* reject it but the registry
  currently claims `no_temperature`, the quirk has healed and is removed.
- prefill probe (Claude models only): pick `strategy="prefill"` directly,
  bypassing `detect.py`'s own `no_prefill` check. Same add/heal logic.
- reasoning-budget probe: runs only when the baseline itself comes back
  completely empty (not an error) and the model isn't already tagged. Retries
  once with a much larger `max_tokens`; if that produces clean output, the
  model was starving on hidden reasoning tokens and gets the generic
  `reasoning_budget` quirk (see `compat.thinking_kwargs`) instead of being
  left silently broken. No heal check for this one — once a model needs
  headroom for reasoning it isn't expected to stop needing it.

Cost is tiny — each probe is capped at ~60 tokens (the reasoning-budget
retry, when it fires, uses a larger `REASONING_PROBE_MAX_TOKENS`), and only
verified models with a configured key are probed (see `make probe-quirks`).
A full weekly sweep across every verified model costs a fraction of a cent.

Run `scripts/generate_verified_models_table.py` afterwards (or via `make
models-table`) to propagate any change into the packaged data file that
`basemode.strategies.compat` actually reads at runtime.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from basemode.continue_ import continue_text  # noqa: E402
from basemode.detect import detect_strategy, normalize_model  # noqa: E402
from basemode.exceptions import EmptyCompletionError  # noqa: E402
from basemode.keys import load_into_environ  # noqa: E402
from basemode.params import GenerationParams  # noqa: E402
from basemode.settings import settings  # noqa: E402

REGISTRY_PATH = ROOT / "data" / "verified_models_registry.json"
SUMMARY_PATH = ROOT / "dist" / "quirk-probe" / "summary.md"

PROBE_PREFIX = "The quick brown fox jumps over the lazy"
PROBE_MAX_TOKENS = 60  # generous enough that reasoning-token overhead on some
# models (e.g. opus-5) doesn't starve visible output and produce a false
# "doesn't work at all" baseline failure.
PROBE_TIMEOUT = 45  # seconds

# A value every provider's default sampler accepts. Used as the *baseline*
# call (still going through the normal compat-aware path) to confirm a model
# works at all before spending a second probe on it.
BASELINE_TEMPERATURE = 1.0
# Distinctly non-default, used to force an explicit temperature through.
PROBE_TEMPERATURE = 0.3

_TEMPERATURE_ERROR_RE = re.compile(r"temperature", re.IGNORECASE)
_PREFILL_ERROR_RE = re.compile(r"prefill|assistant message", re.IGNORECASE)

# Retry budget for the reasoning-budget probe below — generous enough that a
# model burning most of PROBE_MAX_TOKENS on hidden chain-of-thought still has
# room left to write something visible.
REASONING_PROBE_MAX_TOKENS = 5120


@dataclass(frozen=True)
class QuirkChange:
    model: str
    quirk: str
    action: str  # "added" or "removed"
    detail: str


def _load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


_INLINE_ARRAY_RE = re.compile(
    r"\[\n\s+(" r'"(?:[^"\\]|\\.)*"(?:,\n\s+"(?:[^"\\]|\\.)*")*' r")\n\s*\]"
)


def _compact_scalar_arrays(rendered: str) -> str:
    def _collapse(match: re.Match[str]) -> str:
        items = re.findall(r'"(?:[^"\\]|\\.)*"', match.group(1))
        return "[" + ", ".join(items) + "]"

    return _INLINE_ARRAY_RE.sub(_collapse, rendered)


def _save_registry(registry: dict) -> None:
    rendered = _compact_scalar_arrays(json.dumps(registry, indent=2))
    REGISTRY_PATH.write_text(rendered + "\n")


def _provider(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else "unknown"


async def _collect_normal(model: str) -> str:
    """Stream a probe continuation through the real, quirk-respecting path."""
    chunks = [
        token
        async for token in continue_text(
            PROBE_PREFIX,
            model=model,
            max_tokens=PROBE_MAX_TOKENS,
            temperature=BASELINE_TEMPERATURE,
        )
    ]
    return "".join(chunks)


async def _collect_forced(
    model: str,
    *,
    strategy_override: str | None,
    forced_temperature: float | None,
    forced_max_tokens: int = PROBE_MAX_TOKENS,
) -> str:
    """Stream a probe continuation that bypasses compat.py's own suppression.

    `strategy_override` picks the strategy directly (skipping detect.py's
    `no_prefill` check); `forced_temperature`, if set, is injected via
    `GenerationParams.extra`, which `compat.build_kwargs` applies *last* —
    after any `no_temperature` suppression — so it always reaches the
    outgoing request. Going through `continue_text(..., temperature=X)`
    instead would silently bind to its own named `temperature` parameter and
    get suppressed exactly like a normal call, defeating the point of a
    probe that must test what the *provider* accepts regardless of what the
    registry currently believes. `forced_max_tokens` widens the raw budget
    directly — used by the reasoning-budget probe below.
    """
    normalized = normalize_model(model)
    strat = detect_strategy(normalized, strategy_override)
    extra = (
        {"temperature": forced_temperature} if forced_temperature is not None else {}
    )
    params = GenerationParams(
        model=normalized,
        max_tokens=forced_max_tokens,
        temperature=BASELINE_TEMPERATURE,
        extra=extra,
    )
    chunks = [token async for token in strat.stream(PROBE_PREFIX, params)]
    return "".join(chunks)


async def _try(coro) -> tuple[bool, str]:
    """Returns (worked, detail)."""
    try:
        text = await asyncio.wait_for(coro, timeout=PROBE_TIMEOUT)
    except TimeoutError:
        return False, f"timed out after {PROBE_TIMEOUT}s"
    except EmptyCompletionError:
        # continue_text/strat.stream raise this instead of returning "" —
        # same case the empty-string check below handles, so normalize both
        # to the same detail string callers key off of.
        return False, "empty continuation"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return bool(text.strip()), "empty continuation" if not text.strip() else "ok"


async def _probe_model(entry: dict) -> list[QuirkChange]:
    model = str(entry["model"])
    changes: list[QuirkChange] = []
    current_quirks = set(entry.get("quirks", []))

    # Baseline: does the model work at all right now, through the normal
    # (quirk-aware) path? If not, this is a discovery/health problem, not a
    # quirk to record — skip further probing.
    baseline_ok, baseline_detail = await _try(_collect_normal(model))
    if not baseline_ok:
        # Empty (as opposed to an outright provider error) looks like a
        # reasoning model starved of visible-output room. Retry once with a
        # much larger budget before giving up on this model entirely.
        has_reasoning_quirk = "reasoning_budget" in current_quirks
        if baseline_detail == "empty continuation" and not has_reasoning_quirk:
            wide_ok, wide_detail = await _try(
                _collect_forced(
                    model,
                    strategy_override=None,
                    forced_temperature=None,
                    forced_max_tokens=REASONING_PROBE_MAX_TOKENS,
                )
            )
            if wide_ok:
                changes.append(
                    QuirkChange(
                        model,
                        "reasoning_budget",
                        "added",
                        f"empty at {PROBE_MAX_TOKENS} max_tokens, ok at "
                        f"{REASONING_PROBE_MAX_TOKENS} ({wide_detail})",
                    )
                )
        return changes

    # Temperature probe: force a non-default temperature through, bypassing
    # any existing no_temperature quirk.
    temp_ok, temp_detail = await _try(
        _collect_forced(
            model, strategy_override=None, forced_temperature=PROBE_TEMPERATURE
        )
    )
    has_temp_quirk = "no_temperature" in current_quirks
    if not temp_ok and _TEMPERATURE_ERROR_RE.search(temp_detail) and not has_temp_quirk:
        changes.append(QuirkChange(model, "no_temperature", "added", temp_detail))
    elif temp_ok and has_temp_quirk:
        changes.append(QuirkChange(model, "no_temperature", "removed", "accepted now"))

    # Prefill probe: Claude models only — force strategy="prefill" through,
    # bypassing any existing no_prefill quirk.
    if "claude" in model.lower():
        prefill_ok, prefill_detail = await _try(
            _collect_forced(model, strategy_override="prefill", forced_temperature=None)
        )
        has_prefill_quirk = "no_prefill" in current_quirks
        if (
            not prefill_ok
            and _PREFILL_ERROR_RE.search(prefill_detail)
            and not has_prefill_quirk
        ):
            changes.append(QuirkChange(model, "no_prefill", "added", prefill_detail))
        elif prefill_ok and has_prefill_quirk:
            changes.append(QuirkChange(model, "no_prefill", "removed", "accepted now"))

    return changes


PROBE_CONCURRENCY = 8  # bounded so a full sweep of every verified model stays fast


async def _run(model_filter: str | None) -> int:
    load_into_environ()
    registry = _load_registry()
    entries = [
        e for e in registry.get("models", []) if isinstance(e, dict) and e.get("model")
    ]
    if model_filter:
        entries = [
            e for e in entries if model_filter.lower() in str(e["model"]).lower()
        ]

    keyed_entries = [
        e
        for e in entries
        if _provider(str(e["model"])) in settings.available_providers
        or _provider(str(e["model"])) == "openrouter"
    ]

    semaphore = asyncio.Semaphore(PROBE_CONCURRENCY)

    async def _bounded_probe(entry: dict) -> list[QuirkChange]:
        async with semaphore:
            return await _probe_model(entry)

    results = await asyncio.gather(*(_bounded_probe(e) for e in keyed_entries))

    all_changes: list[QuirkChange] = []
    for entry, changes in zip(keyed_entries, results, strict=True):
        model = str(entry["model"])
        for change in changes:
            quirks = set(entry.get("quirks", []))
            if change.action == "added":
                quirks.add(change.quirk)
                print(f"  + {model}: {change.quirk} ({change.detail})")
            else:
                quirks.discard(change.quirk)
                print(f"  - {model}: {change.quirk} healed ({change.detail})")
            if quirks:
                entry["quirks"] = sorted(quirks)
            else:
                entry.pop("quirks", None)
        all_changes.extend(changes)

    if all_changes:
        _save_registry(registry)
    _write_summary(all_changes)
    return 0


def _write_summary(changes: list[QuirkChange]) -> None:
    lines = ["# Model quirk probe", ""]
    added = [c for c in changes if c.action == "added"]
    removed = [c for c in changes if c.action == "removed"]
    if added:
        lines.append("## New quirks detected")
        lines.extend(f"- `{c.model}`: `{c.quirk}` — {c.detail}" for c in added)
        lines.append("")
    if removed:
        lines.append("## Quirks healed (provider now accepts the param)")
        lines.extend(f"- `{c.model}`: `{c.quirk}`" for c in removed)
        lines.append("")
    if not changes:
        lines.append("No quirk drift detected this run.")
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Only probe models whose id contains this")
    args = parser.parse_args()
    return asyncio.run(_run(args.model))


if __name__ == "__main__":
    raise SystemExit(main())

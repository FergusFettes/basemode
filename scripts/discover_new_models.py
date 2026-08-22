#!/usr/bin/env python3
"""Discover new models across every keyed provider, probe them, and register
the ones that work.

For each provider with an API key configured (see `PROVIDER_ENDPOINTS` in
`basemode.live_models`):
1. List models from the provider's own /v1/models endpoint (the source of truth
   for "latest", ahead of LiteLLM's/OpenRouter's baked-in metadata).
2. Drop anything already in the verified-models registry or previously rejected.
3. Sort remaining candidates newest-first and cap how many get probed (network +
   token spend), via --limit / DISCOVER_MODELS_LIMIT.
4. Actually run a short continuation through basemode against each candidate,
   trying basemode's auto-detected strategy first and falling back through
   STRATEGY_FALLBACKS if that one doesn't work — a wrong default guess in
   detect.py shouldn't get a genuinely-working model rejected. Each attempt
   is judged by `basemode.scoring`, the same continuation-purity scorer
   `basemode bench` ranks strategies with. If any strategy scores clean, the
   model is added to the registry (data/verified_models_registry.json)
   tagged with whichever
   strategy actually worked — which `detect.py` then honours at runtime via
   the registry's `prompt_method`. If every strategy fails, it's recorded in
   data/verified_models_rejected.json (with all attempted errors) so we
   don't keep re-paying to re-discover the same dead end every week.

Run `scripts/generate_verified_models_table.py` afterwards to refresh the
README table from the updated registry.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from basemode.continue_ import continue_text  # noqa: E402
from basemode.detect import detect_strategy, normalize_model  # noqa: E402
from basemode.keys import load_into_environ  # noqa: E402
from basemode.live_models import (  # noqa: E402
    PROVIDER_ENDPOINTS,
    LiveModelsError,
    fetch_live_models,
)
from basemode.scoring import looks_clean  # noqa: E402
from basemode.settings import settings  # noqa: E402

REGISTRY_PATH = ROOT / "data" / "verified_models_registry.json"
REJECTED_PATH = ROOT / "data" / "verified_models_rejected.json"
SUMMARY_PATH = ROOT / "dist" / "model-discovery" / "summary.md"

DEFAULT_LIMIT = 6
PROBE_PREFIX = "The quick brown fox jumps over the lazy"
PROBE_MAX_TOKENS = (
    40  # generous enough that reasoning-token overhead doesn't starve visible output
)
# 1.0 is the one value every provider's default-sampling API accepts,
# including newer models that reject any other explicit temperature.
PROBE_TEMPERATURE = 1.0

# Tried in order, after basemode's own auto-detected strategy, until one
# works. Covers the realistic chat-completion-coercion space; deliberately
# excludes completion/fim, which only apply to literal base/code models that
# provider /v1/models listings for chat models won't surface.
STRATEGY_FALLBACKS = ["system", "prefill", "few_shot"]

_OPENAI_BLACKLIST_SUBSTRINGS = [
    "embedding",
    "whisper",
    "tts",
    "dall-e",
    "moderation",
    "image",
    "audio",
    "realtime",
    "transcribe",
    "search",
    "computer-use",
    "text-similarity",
    "text-search",
    "text-edit",
    "code-search",
]

_PRICING_URLS = {
    "openai": "https://openai.com/api/pricing/",
    "anthropic": "https://docs.anthropic.com/en/docs/about-claude/pricing",
}


@dataclass(frozen=True)
class Candidate:
    provider: str
    raw_id: str
    normalized_id: str
    created: float


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    return json.loads(path.read_text())


_INLINE_ARRAY_RE = re.compile(
    r"\[\n\s+(" r'"(?:[^"\\]|\\.)*"(?:,\n\s+"(?:[^"\\]|\\.)*")*' r")\n\s*\]"
)


def _compact_scalar_arrays(rendered: str) -> str:
    """Collapse arrays of strings back onto one line (matches this repo's
    existing hand-formatted style, e.g. `"known_issues": ["foo"]`), so a
    single new entry doesn't reformat every other array in the file."""

    def _collapse(match: re.Match[str]) -> str:
        items = re.findall(r'"(?:[^"\\]|\\.)*"', match.group(1))
        return "[" + ", ".join(items) + "]"

    return _INLINE_ARRAY_RE.sub(_collapse, rendered)


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = _compact_scalar_arrays(json.dumps(data, indent=2))
    path.write_text(rendered + "\n")


def _known_models(registry: dict) -> set[str]:
    return {entry["model"] for entry in registry.get("models", [])}


def _rejected_models(rejected: dict) -> set[str]:
    return {entry["model"] for entry in rejected.get("models", [])}


def _is_openai_blacklisted(model_id: str) -> bool:
    lowered = model_id.lower()
    return any(substr in lowered for substr in _OPENAI_BLACKLIST_SUBSTRINGS)


def _release_date_ts(release_date: str | None) -> float:
    if not release_date:
        return 0.0
    try:
        return datetime.fromisoformat(release_date).timestamp()
    except ValueError:
        return 0.0


def select_candidates(
    *,
    provider: str,
    raw_models: list,  # list[basemode.live_models.LiveModel]
    known: set[str],
    rejected: set[str],
    limit: int,
) -> list[Candidate]:
    """Pure filtering/dedup/sort/cap logic, kept separate from network calls."""
    candidates: list[Candidate] = []

    for entry in raw_models:
        raw_id = entry.id
        if not raw_id:
            continue
        if provider == "openai" and _is_openai_blacklisted(raw_id):
            continue

        normalized_id = normalize_model(f"{provider}/{raw_id}")
        if not normalized_id.startswith(f"{provider}/"):
            # normalize_model didn't recognize it as belonging to this
            # provider (e.g. a stray non-chat id) — skip rather than guess.
            continue
        if normalized_id in known or normalized_id in rejected:
            continue

        candidates.append(
            Candidate(
                provider=provider,
                raw_id=raw_id,
                normalized_id=normalized_id,
                created=_release_date_ts(entry.release_date),
            )
        )

    candidates.sort(key=lambda c: c.created, reverse=True)
    return candidates[:limit]


PROBE_TIMEOUT = 60  # seconds; a stalled provider stream shouldn't hang the whole run


async def _collect_chunks(candidate: Candidate, strategy: str | None) -> list[str]:
    return [
        token
        async for token in continue_text(
            PROBE_PREFIX,
            model=candidate.normalized_id,
            max_tokens=PROBE_MAX_TOKENS,
            temperature=PROBE_TEMPERATURE,
            strategy=strategy,
        )
    ]


async def _probe_strategy(
    candidate: Candidate, strategy: str | None
) -> tuple[bool, str, str]:
    """Returns (worked, detail, sample_text) for a single strategy attempt."""
    try:
        chunks = await asyncio.wait_for(
            _collect_chunks(candidate, strategy), timeout=PROBE_TIMEOUT
        )
    except TimeoutError:
        return False, f"timed out after {PROBE_TIMEOUT}s", ""
    except Exception as exc:  # provider/strategy error
        return False, f"{type(exc).__name__}: {exc}", ""

    text = "".join(chunks)
    ok, reason = looks_clean(PROBE_PREFIX, text)
    return ok, (reason or "ok"), text


async def _probe(candidate: Candidate) -> tuple[bool, str | None, str, dict[str, str]]:
    """Try the auto-detected strategy, then STRATEGY_FALLBACKS, until one works.

    Returns (worked, strategy_used, detail, attempts) where `attempts` maps
    every strategy name tried to its failure detail (empty on success on the
    first try). `strategy_used` is None when the auto-detected default won.
    """
    auto_name = detect_strategy(candidate.normalized_id).name
    attempts: dict[str, str] = {}

    ok, detail, _sample = await _probe_strategy(candidate, None)
    if ok:
        return True, None, detail, attempts
    attempts[auto_name] = detail

    for strategy in STRATEGY_FALLBACKS:
        if strategy in attempts:
            continue  # already tried as the auto-detected default
        ok, detail, _sample = await _probe_strategy(candidate, strategy)
        if ok:
            return True, strategy, detail, attempts
        attempts[strategy] = detail

    return False, None, attempts[auto_name], attempts


def _dashed(text: str) -> str:
    return text.lower().replace(".", "-").replace("_", "-")


def _guess_openrouter_id(
    normalized_id: str, openrouter_index: dict[str, dict]
) -> str | None:
    stem = _dashed(normalized_id.split("/")[-1])
    for or_id in openrouter_index:
        or_stem = _dashed(or_id.split("/")[-1])
        if stem == or_stem or stem in or_stem or or_stem in stem:
            return or_id
    return None


def _fetch_openrouter_index() -> dict[str, dict]:
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"User-Agent": "basemode-model-discovery/1"},
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return {m["id"]: m for m in payload.get("data", []) if "id" in m}
    except (TimeoutError, urllib.error.URLError, json.JSONDecodeError):
        return {}


async def _run(limit: int, providers: set[str] | None = None) -> int:
    load_into_environ()  # pull persisted keys (~/.config/basemode/auth.json) in, if any
    registry = _load_json(REGISTRY_PATH, {"models": []})
    rejected = _load_json(REJECTED_PATH, {"models": []})
    known = _known_models(registry)
    already_rejected = _rejected_models(rejected)

    all_candidates: list[Candidate] = []
    for provider in sorted(providers if providers is not None else PROVIDER_ENDPOINTS):
        api_key = settings.api_key_for(provider)
        if not api_key and provider != "openrouter":
            print(f"skip {provider}: no API key configured")
            continue
        try:
            raw_models = fetch_live_models(provider, api_key)
        except LiveModelsError as exc:
            print(f"skip {provider}: failed to list models ({exc})")
            continue

        candidates = select_candidates(
            provider=provider,
            raw_models=raw_models,
            known=known,
            rejected=already_rejected,
            limit=limit,
        )
        print(f"{provider}: {len(candidates)} candidate(s) to probe")
        all_candidates.extend(candidates)

    if not all_candidates:
        _write_summary(added=[], rejected_now=[])
        print("No new candidates to probe.")
        return 0

    openrouter_index = _fetch_openrouter_index()

    added: list[tuple[Candidate, str]] = []
    rejected_now: list[tuple[Candidate, str]] = []

    for candidate in all_candidates:
        worked, strategy_used, detail, attempts = await _probe(candidate)
        if worked:
            label = f"strategy={strategy_used}" if strategy_used else "auto strategy"
            print(f"  + {candidate.normalized_id}: works ({label})")
            entry = {"model": candidate.normalized_id}
            or_id = _guess_openrouter_id(candidate.normalized_id, openrouter_index)
            if or_id:
                entry["openrouter_id"] = or_id
            pricing_url = _PRICING_URLS.get(candidate.provider)
            if pricing_url:
                entry["pricing_url"] = pricing_url
            if strategy_used:
                entry["prompt_method"] = strategy_used
            registry.setdefault("models", []).append(entry)
            known.add(candidate.normalized_id)
            added.append((candidate, label))
        else:
            print(f"  - {candidate.normalized_id}: all strategies failed ({detail})")
            rejected.setdefault("models", []).append(
                {
                    "model": candidate.normalized_id,
                    "reason": detail,
                    "attempted_strategies": attempts,
                    "checked_at_utc": datetime.now(tz=UTC).isoformat(),
                }
            )
            already_rejected.add(candidate.normalized_id)
            rejected_now.append((candidate, detail))

    if added:
        registry["models"] = sorted(registry["models"], key=lambda e: e["model"])
        _save_json(REGISTRY_PATH, registry)
    if rejected_now:
        rejected["models"] = sorted(rejected["models"], key=lambda e: e["model"])
        _save_json(REJECTED_PATH, rejected)
    _write_summary(added=added, rejected_now=rejected_now)

    return 0


def _write_summary(
    *,
    added: list[tuple[Candidate, str]],
    rejected_now: list[tuple[Candidate, str]],
) -> None:
    lines = ["# Model discovery run", ""]
    if added:
        lines.append("## Added")
        lines.extend(f"- `{c.normalized_id}`" for c, _ in added)
        lines.append("")
    if rejected_now:
        lines.append("## Rejected")
        lines.extend(f"- `{c.normalized_id}`: {detail}" for c, detail in rejected_now)
        lines.append("")
    if not added and not rejected_now:
        lines.append("No new candidate models found this run.")
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("DISCOVER_MODELS_LIMIT") or DEFAULT_LIMIT),
        help="Max candidates to probe per provider (default: %(default)s)",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default=None,
        help="Comma-separated subset of providers to probe (default: all keyed "
        f"providers, from {', '.join(sorted(PROVIDER_ENDPOINTS))})",
    )
    args = parser.parse_args()
    providers = (
        {p.strip() for p in args.providers.split(",") if p.strip()}
        if args.providers
        else None
    )
    return asyncio.run(_run(args.limit, providers=providers))


if __name__ == "__main__":
    raise SystemExit(main())

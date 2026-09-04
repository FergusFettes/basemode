#!/usr/bin/env python3
"""Refresh data/live_models_cache.json from each provider's own /v1/models.

litellm's bundled model list lags real provider releases by weeks (see
basemode.live_models' docstring). This script hits every provider we have a
configured key for directly and caches the result, so a released basemode
can show new models (and better release dates, where the provider's
`created`/`created_at` field is trustworthy) without needing a live API call
at lookup time.

Trustworthiness check: some providers (confirmed: moonshot) return the exact
same `created` timestamp for every model in the list — that's a list-refresh
time, not a per-model release date, and is worse than no data at all. A
provider is marked `reliable_dates: false` when more than one model shares
identical distinct-looking timestamps across >50% of its catalog; its models
are still cached (for the id/NEW signal) but with release_date stripped.

The cache is merged rather than replaced. Only providers this machine holds a
key for can be fetched, so a scheduled refresh that reaches three providers
must not delete the twelve someone else contributed — that is how every
Together model ended up unpriced. Each provider carries its own
`refreshed_at`, so a kept entry can be seen aging; `--prune-after-days` drops
the ones that have aged out, and `--replace` rebuilds from only what this run
could reach.

Run this periodically (for example through the live-model refresh workflow) and
commit the refreshed cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from basemode.live_models import (  # noqa: E402
    PROVIDER_ENDPOINTS,
    LiveModelsError,
    dates_look_trustworthy,
    fetch_live_models,
)
from basemode.settings import settings  # noqa: E402

CACHE_PATH = ROOT / "src" / "basemode" / "data" / "live_models_cache.json"


def _load_cache() -> dict:
    try:
        payload = json.loads(CACHE_PATH.read_text())
    except (OSError, ValueError):
        return {}
    providers = payload.get("providers")
    return providers if isinstance(providers, dict) else {}


def _age_days(entry: dict, now: datetime) -> float | None:
    stamp = entry.get("refreshed_at")
    if not isinstance(stamp, str):
        return None
    try:
        return (now - datetime.fromisoformat(stamp)).total_seconds() / 86400
    except ValueError:
        return None


def _describe_age(entry: dict, now: datetime) -> str:
    age = _age_days(entry, now)
    return "age unknown" if age is None else f"{age:.0f}d old"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Rebuild from only what this run fetched, dropping every other provider.",
    )
    parser.add_argument(
        "--prune-after-days",
        type=float,
        default=None,
        metavar="N",
        help="Drop kept providers last refreshed more than N days ago.",
    )
    args = parser.parse_args(argv)

    now = datetime.now(tz=UTC)
    providers_out: dict[str, dict] = {} if args.replace else _load_cache()
    fetched: set[str] = set()

    for provider in sorted(PROVIDER_ENDPOINTS):
        api_key = settings.api_key_for(provider)
        if not api_key and provider != "openrouter":
            continue

        try:
            live = fetch_live_models(provider, api_key)
        except LiveModelsError as exc:
            print(f"skip {provider}: {exc}")
            continue

        reliable = dates_look_trustworthy(live)
        providers_out[provider] = {
            "refreshed_at": now.isoformat(),
            "reliable_dates": reliable,
            "models": {
                m.id: {
                    "release_date": m.release_date if reliable else None,
                    "release_date_confidence": (
                        m.release_date_confidence if reliable else "unknown"
                    ),
                    "input_modalities": list(m.input_modalities),
                    "output_modalities": list(m.output_modalities),
                    "supported_methods": list(m.supported_methods),
                    "provider_type": m.provider_type,
                    "supported_parameters": list(m.supported_parameters),
                    "input_price_per_m": m.input_price_per_m,
                    "output_price_per_m": m.output_price_per_m,
                }
                for m in live
            },
        }
        fetched.add(provider)
        flag = "" if reliable else " (dates look bogus, dropped)"
        print(f"{provider}: {len(live)} models{flag}")

    for provider in sorted(set(providers_out) - fetched):
        entry = providers_out[provider]
        age = _age_days(entry, now)
        if args.prune_after_days is not None and (
            age is None or age > args.prune_after_days
        ):
            del providers_out[provider]
            print(f"pruned {provider}: {_describe_age(entry, now)}")
            continue
        models = entry.get("models") or {}
        print(
            f"kept {provider}: {len(models)} models from an earlier refresh "
            f"({_describe_age(entry, now)}; no key here)"
        )

    CACHE_PATH.write_text(
        json.dumps(
            {
                "generated_at_utc": now.isoformat(),
                "providers": providers_out,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Wrote {CACHE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

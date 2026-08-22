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

Run this periodically (e.g. via CI, alongside record_provider_health.py) and
commit the refreshed cache.
"""

from __future__ import annotations

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


def main() -> int:
    providers_out: dict[str, dict] = {}

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
            "reliable_dates": reliable,
            "models": {
                m.id: (m.release_date if reliable else None) for m in live
            },
        }
        flag = "" if reliable else " (dates look bogus, dropped)"
        print(f"{provider}: {len(live)} models{flag}")

    CACHE_PATH.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(tz=UTC).isoformat(),
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

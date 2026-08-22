"""Query providers' own /v1/models endpoints directly, bypassing litellm.

litellm's model list is a community-maintained snapshot that lags real
provider releases by weeks (see e.g. kimi-k3, which ships under
`azure_ai/FW-Kimi-K3` upstream but has no `moonshot/kimi-k3` entry). This
module hits each provider's own model-listing endpoint so `basemode models
--live` can see what a provider actually serves today.

Release-date quality varies a lot by provider:
- Anthropic's `created_at` is documented as the true release date.
- OpenAI/Groq/Together/Moonshot/xAI/zai's `created` is a Unix timestamp of
  when the model was registered in that provider's system — usually close
  to release, but not guaranteed (especially for third-party open models).
- OpenRouter's `created` is similar, but the endpoint needs no API key.
- Gemini's `models.list` has no release-date field at all.
"""

from __future__ import annotations

import datetime
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

_TIMEOUT = 15


class LiveModelsError(Exception):
    """Raised when a provider's models endpoint can't be reached or parsed."""


@dataclass(frozen=True)
class LiveModel:
    id: str
    release_date: str | None
    release_date_confidence: str  # "release" | "registered" | "unknown"


def _bearer_headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _anthropic_headers(key: str) -> dict[str, str]:
    return {"x-api-key": key, "anthropic-version": "2023-06-01"}


def _unix_to_date(value: object) -> str | None:
    if not isinstance(value, int | float) or value <= 0:
        return None
    return datetime.datetime.fromtimestamp(value, tz=datetime.UTC).date().isoformat()


def _iso_to_date(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return (
            datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
            .date()
            .isoformat()
        )
    except ValueError:
        return None


def _parse_openai_style(payload: dict | list) -> list[LiveModel]:
    # Most providers wrap the list as {"data": [...]}; a couple (e.g.
    # together_ai) return the bare list.
    data = payload.get("data", []) if isinstance(payload, dict) else payload
    return [
        LiveModel(
            id=str(m["id"]),
            release_date=_unix_to_date(m.get("created")),
            release_date_confidence="registered",
        )
        for m in data
        if isinstance(m, dict) and "id" in m
    ]


def _parse_anthropic(payload: dict) -> list[LiveModel]:
    return [
        LiveModel(
            id=str(m["id"]),
            release_date=_iso_to_date(m.get("created_at")),
            release_date_confidence="release",
        )
        for m in payload.get("data", [])
        if isinstance(m, dict) and "id" in m
    ]


def _parse_gemini(payload: dict) -> list[LiveModel]:
    return [
        LiveModel(
            id=str(m["name"]).removeprefix("models/"),
            release_date=None,
            release_date_confidence="unknown",
        )
        for m in payload.get("models", [])
        if isinstance(m, dict) and "name" in m
    ]


@dataclass(frozen=True)
class _Endpoint:
    url: str
    headers: Callable[[str], dict[str, str]] | None  # None means no key needed
    parse: Callable[[dict], list[LiveModel]]


PROVIDER_ENDPOINTS: dict[str, _Endpoint] = {
    "openai": _Endpoint(
        "https://api.openai.com/v1/models", _bearer_headers, _parse_openai_style
    ),
    "anthropic": _Endpoint(
        "https://api.anthropic.com/v1/models", _anthropic_headers, _parse_anthropic
    ),
    "openrouter": _Endpoint(
        "https://openrouter.ai/api/v1/models", None, _parse_openai_style
    ),
    "groq": _Endpoint(
        "https://api.groq.com/openai/v1/models", _bearer_headers, _parse_openai_style
    ),
    "together_ai": _Endpoint(
        "https://api.together.xyz/v1/models", _bearer_headers, _parse_openai_style
    ),
    "moonshot": _Endpoint(
        "https://api.moonshot.ai/v1/models", _bearer_headers, _parse_openai_style
    ),
    "xai": _Endpoint(
        "https://api.x.ai/v1/models", _bearer_headers, _parse_openai_style
    ),
    "zai": _Endpoint(
        "https://api.z.ai/api/paas/v4/models", _bearer_headers, _parse_openai_style
    ),
    "deepseek": _Endpoint(
        "https://api.deepseek.com/v1/models", _bearer_headers, _parse_openai_style
    ),
    "gemini": _Endpoint(
        "https://generativelanguage.googleapis.com/v1beta/models", None, _parse_gemini
    ),
}


def dates_look_trustworthy(models: list[LiveModel]) -> bool:
    """False if too many models share an identical release_date.

    A per-model release date should vary across a catalog. Some providers
    (confirmed: moonshot) return the exact same `created` timestamp for
    every model — that's a list-refresh time, not a release date.
    """
    known = [m.release_date for m in models if m.release_date]
    if len(known) < 2:
        return True
    from collections import Counter

    most_common_count = Counter(known).most_common(1)[0][1]
    return most_common_count / len(known) < 0.5


def fetch_live_models(provider: str, api_key: str) -> list[LiveModel]:
    """Query `provider`'s own models endpoint. Raises LiveModelsError on failure."""
    endpoint = PROVIDER_ENDPOINTS.get(provider)
    if endpoint is None:
        raise LiveModelsError(
            f"no live endpoint wired up for provider {provider!r} "
            f"(known: {', '.join(sorted(PROVIDER_ENDPOINTS))})"
        )

    url = endpoint.url
    headers = {"User-Agent": "basemode-live-models/1"}
    if provider == "gemini":
        url = f"{url}?key={api_key}"
    elif endpoint.headers is not None:
        headers.update(endpoint.headers(api_key))

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise LiveModelsError(f"{provider}: HTTP {exc.code} from {url}") from exc
    except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise LiveModelsError(f"{provider}: {exc}") from exc

    return sorted(endpoint.parse(payload), key=lambda m: m.id)

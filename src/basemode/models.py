import datetime
import json
import re
from importlib import resources

import litellm

from .settings import settings

_EXTRA_MODELS_BY_PROVIDER = {
    # Official Gemini API Gemma 4 IDs. LiteLLM 1.83.9 does not list these yet.
    "gemini": [
        "gemini/gemma-4-26b-a4b-it",
        "gemini/gemma-4-31b-it",
    ],
}

# litellm.model_cost["mode"] values that represent text-generation models,
# as opposed to image/audio/embedding/rerank/etc. side models.
TEXT_MODES = {"chat", "responses", "completion"}

# Matches a trailing dated-snapshot suffix, e.g. "-2026-03-05" or "-20251001".
_DATE_SUFFIX_RE = re.compile(r"-(?:(\d{4})-(\d{2})-(\d{2})|(\d{8}))$")


def _strip_date_suffix(model_id: str) -> tuple[str, str | None]:
    """Split a model id into (base_id, iso_date) if it ends in a dated snapshot."""
    match = _DATE_SUFFIX_RE.search(model_id)
    if not match:
        return model_id, None
    year, month, day, compact = match.groups()
    if compact is None:
        date = f"{year}-{month}-{day}"
    else:
        date = f"{compact[0:4]}-{compact[4:6]}-{compact[6:8]}"
    return model_id[: match.start()], date


_SINCE_RE = re.compile(r"^(\d+)([dwmy])$", re.IGNORECASE)


def _shift_months(d: datetime.date, delta_months: int) -> datetime.date:
    month_index = d.month - 1 + delta_months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    next_month_first = datetime.date(year + (month // 12), month % 12 + 1, 1)
    days_in_month = (next_month_first - datetime.timedelta(days=1)).day
    return datetime.date(year, month, min(d.day, days_in_month))


def parse_since(spec: str) -> str:
    """Parse a relative duration like '10d', '4w', '6m', '1y' into an ISO cutoff date."""
    match = _SINCE_RE.match(spec.strip())
    if not match:
        raise ValueError(f"invalid --since value {spec!r}; use e.g. 10d, 4w, 6m, 1y")
    amount, unit = int(match.group(1)), match.group(2).lower()
    today = datetime.date.today()
    if unit == "d":
        cutoff = today - datetime.timedelta(days=amount)
    elif unit == "w":
        cutoff = today - datetime.timedelta(weeks=amount)
    elif unit == "m":
        cutoff = _shift_months(today, -amount)
    else:
        cutoff = _shift_months(today, -amount * 12)
    return cutoff.isoformat()


def _model_mode(provider: str, model: str) -> str | None:
    """Best-effort litellm mode ('chat', 'image_generation', ...) for a model."""
    info = litellm.model_cost.get(model) or litellm.model_cost.get(
        f"{provider}/{model}"
    )
    return info.get("mode") if info else None


def _display_model_id(provider: str, model: str) -> str:
    prefix = f"{provider}/"
    return model[len(prefix) :] if model.startswith(prefix) else model


def _read_package_data(filename: str) -> dict:
    try:
        text = resources.files("basemode").joinpath("data", filename).read_text()
        return json.loads(text)
    except Exception:
        return {}


def _verified_rows_by_model() -> dict[str, dict]:
    payload = _read_package_data("verified_models_details.json")
    rows = payload.get("rows", [])
    return {
        row.get("model"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("model"), str)
    }


def _live_rows_by_provider() -> dict[str, dict]:
    """Cached provider-API model listings (see scripts/refresh_live_models.py).

    Patches over litellm's lag: new model ids litellm doesn't know about yet
    (e.g. a just-released `kimi-k3`) show up here, and providers with a
    trustworthy `created` field (not `moonshot`, whose API returns the same
    timestamp for every model) contribute a release date litellm never has.
    """
    payload = _read_package_data("live_models_cache.json")
    providers = payload.get("providers", {})
    return providers if isinstance(providers, dict) else {}


def _provider_models(provider: str, by_provider: dict[str, list[str]]) -> list[str]:
    return list(by_provider.get(provider, [])) + _EXTRA_MODELS_BY_PROVIDER.get(
        provider, []
    )


def list_models(
    provider: str | None = None,
    search: str | None = None,
    available_only: bool = False,
) -> list[str]:
    by_provider: dict[str, list[str]] = litellm.models_by_provider

    if available_only:
        providers = settings.available_providers
        models = [m for p in providers for m in _provider_models(p, by_provider)]
    elif provider:
        models = _provider_models(provider, by_provider)
    else:
        models = [m for ms in by_provider.values() for m in ms]
        models.extend(m for ms in _EXTRA_MODELS_BY_PROVIDER.values() for m in ms)

    if search:
        models = [m for m in models if search.lower() in m.lower()]

    return sorted(set(models))


def list_providers() -> list[str]:
    return sorted(litellm.models_by_provider.keys())


# Vendor tags a re-hosting provider sometimes prepends to a model id that
# the original provider doesn't use (e.g. azure_ai lists Fireworks-hosted
# models as "FW-Kimi-K3" where moonshot itself just calls it "kimi-k3").
_NOISE_PREFIXES = ("fw-",)


def _canonical_model_key(display: str) -> str:
    """Normalize a display id so the same model looks the same across providers.

    `moonshot/kimi-k3`, `openrouter/moonshotai/kimi-k3`,
    `together_ai/moonshotai/Kimi-K3`, and `azure_ai/FW-Kimi-K3` should all
    collapse to the same key so a release date known for one can stand in
    for the others.
    """
    tail = display.rsplit("/", 1)[-1].lower()
    for prefix in _NOISE_PREFIXES:
        if tail.startswith(prefix):
            tail = tail[len(prefix) :]
            break
    return re.sub(r"[^a-z0-9]", "", tail)


def _known_release_date(
    model_provider: str, model: str, verified: dict, live: dict
) -> str | None:
    display = _display_model_id(model_provider, model)
    v = verified.get(model) or verified.get(f"{model_provider}/{model}", {})
    _, suffix_date = _strip_date_suffix(display)
    live_date = live.get(model_provider, {}).get("models", {}).get(display)
    return v.get("release_date") or suffix_date or live_date


def _cross_provider_release_dates(verified: dict, live: dict) -> dict[str, str]:
    """canonical model key -> earliest known release date across every

    provider/alias we know about, regardless of the current call's provider/
    search filters. Used to fill in a release date for a model that's blank
    under its own provider but dated under another (e.g. `moonshot/kimi-k3`
    has no date yet, but `openrouter/moonshotai/kimi-k3` does).
    """
    pairs = _all_provider_pairs()
    known_display = {(p, _display_model_id(p, m)) for p, m in pairs}
    for m in verified:
        v_provider = m.split("/", 1)[0] if "/" in m else "unknown"
        v_display = _display_model_id(v_provider, m)
        if (v_provider, v_display) not in known_display:
            pairs.append((v_provider, m))
            known_display.add((v_provider, v_display))

    index: dict[str, str] = {}
    for model_provider, model in set(pairs):
        date = _known_release_date(model_provider, model, verified, live)
        if not date:
            continue
        key = _canonical_model_key(_display_model_id(model_provider, model))
        if key not in index or date < index[key]:
            index[key] = date
    return index


def _all_provider_pairs() -> list[tuple[str, str]]:
    """(provider, model) pairs from litellm plus anything the live cache

    has that litellm doesn't — provider always correct.
    """
    by_provider: dict[str, list[str]] = litellm.models_by_provider
    pairs = [(p, m) for p, ms in by_provider.items() for m in ms]
    pairs.extend((p, m) for p, ms in _EXTRA_MODELS_BY_PROVIDER.items() for m in ms)
    known_display = {(p, _display_model_id(p, m)) for p, m in pairs}
    for provider, row in _live_rows_by_provider().items():
        for model_id in row.get("models", {}):
            if (provider, model_id) not in known_display:
                pairs.append((provider, f"{provider}/{model_id}"))
                known_display.add((provider, model_id))
    return pairs


def list_model_picker_entries(
    provider: str | None = None,
    search: str | None = None,
    available_only: bool = False,
    verified_only: bool = False,
    text_only: bool = False,
    compact: bool = False,
    since: str | None = None,
) -> list[dict]:
    """Structured model metadata for frontend pickers.

    Includes:
    - stable model id (`model`)
    - provider and key-availability
    - litellm `mode` (chat, image_generation, embedding, ...)
    - verified pricing/reliability/prompt-method when known

    `text_only` drops non text-generation models (image/audio/embedding/...).
    `compact` collapses dated snapshots (e.g. `gpt-5.4-2026-03-05`) into their
    undated alias (`gpt-5.4`) when one exists, keeping a `snapshots` list of
    the dated ids it absorbed.
    `since` is a relative duration (`10d`, `4w`, `6m`, `1y`) parsed with
    `parse_since`; models with no known release date are dropped.
    """
    verified = _verified_rows_by_model()
    live = _live_rows_by_provider()
    cross_provider_dates = _cross_provider_release_dates(verified, live)

    if verified_only:
        pairs = [(m.split("/", 1)[0] if "/" in m else "unknown", m) for m in verified]
    else:
        pairs = _all_provider_pairs()
        known_display = {(p, _display_model_id(p, m)) for p, m in pairs}
        for m in verified:
            v_provider = m.split("/", 1)[0] if "/" in m else "unknown"
            v_display = _display_model_id(v_provider, m)
            if (v_provider, v_display) not in known_display:
                pairs.append((v_provider, m))
                known_display.add((v_provider, v_display))

    if provider:
        pairs = [(p, m) for p, m in pairs if p == provider]
    if search:
        needle = search.lower()
        pairs = [(p, m) for p, m in pairs if needle in m.lower()]

    available_providers = set(settings.available_providers)
    entries: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for model_provider, model in sorted(set(pairs), key=lambda pm: (pm[0], pm[1])):
        if (model_provider, model) in seen:
            continue
        seen.add((model_provider, model))

        mode = _model_mode(model_provider, model)
        if text_only and mode is not None and mode not in TEXT_MODES:
            continue

        v = verified.get(model) or verified.get(f"{model_provider}/{model}", {})
        available = model_provider in available_providers
        if available_only and not available:
            continue
        display = _display_model_id(model_provider, model)
        own_date = _known_release_date(model_provider, model, verified, live)
        if own_date:
            release_date, release_date_inferred = own_date, False
        else:
            guess = cross_provider_dates.get(_canonical_model_key(display))
            release_date, release_date_inferred = guess, guess is not None
        entries.append(
            {
                "model": model,
                "display": display,
                "provider": model_provider,
                "mode": mode,
                "available": available,
                "verified": bool(v),
                "prompt_method": v.get("prompt_method"),
                "reliability": v.get("reliability"),
                "release_date": release_date,
                "release_date_inferred": release_date_inferred,
                "input_cost_per_token": v.get("input_cost_per_token"),
                "output_cost_per_token": v.get("output_cost_per_token"),
                "issues": list(v.get("issues", [])),
            }
        )

    if compact:
        entries = _compact_entries(entries)

    if since:
        cutoff = parse_since(since)
        entries = [e for e in entries if (e.get("release_date") or "") >= cutoff]

    def sort_key(item: dict) -> tuple[int, int, int, str, str]:
        reliability_rank = 0 if item.get("reliability") == "✓" else 1
        return (
            0 if item["available"] else 1,
            0 if item["verified"] else 1,
            reliability_rank,
            item["provider"],
            item["model"],
        )

    return sorted(entries, key=sort_key)


def _compact_entries(entries: list[dict]) -> list[dict]:
    """Collapse dated snapshots of the same base model into one entry."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for e in entries:
        base, date = _strip_date_suffix(e["display"])
        groups.setdefault((e["provider"], base), []).append({**e, "_date": date})

    compacted: list[dict] = []
    for _key, group in groups.items():
        undated = [g for g in group if g["_date"] is None]
        dated = sorted(
            (g for g in group if g["_date"] is not None),
            key=lambda g: g["_date"],
        )
        representative = undated[0] if undated else dated[-1]
        snapshots = sorted(
            {g["model"] for g in group if g["model"] != representative["model"]}
        )
        release_date = representative.get("release_date") or (
            dated[-1]["_date"] if dated else None
        )
        compacted.append(
            {
                **{k: v for k, v in representative.items() if k != "_date"},
                "release_date": release_date,
                "snapshots": snapshots,
            }
        )
    return compacted


def build_model_picker_state(
    *,
    selected: list[str] | None = None,
    max_models: int = 3,
    provider: str | None = None,
    search: str | None = None,
    available_only: bool = False,
    verified_only: bool = False,
    text_only: bool = False,
    compact: bool = False,
) -> dict:
    """Frontend-friendly state blob for single- or multi-model picker UIs."""
    entries = list_model_picker_entries(
        provider=provider,
        search=search,
        available_only=available_only,
        verified_only=verified_only,
        text_only=text_only,
        compact=compact,
    )
    selected = selected or []
    selected_set = set(selected)
    available_models = {e["model"] for e in entries}
    selected_missing = [m for m in selected if m not in available_models]
    too_many_selected = len(selected) > max_models
    return {
        "max_models": max_models,
        "selected": selected,
        "selected_missing": selected_missing,
        "too_many_selected": too_many_selected,
        "models": [
            {
                **e,
                "selected": e["model"] in selected_set,
                "disabled_for_selection": (
                    e["model"] not in selected_set
                    and len(selected) >= max_models
                    and max_models > 0
                ),
            }
            for e in entries
        ],
    }

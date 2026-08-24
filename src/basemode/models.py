import datetime
import json
import re
from importlib import resources

import litellm

from .health import list_model_health, verification_history
from .keys import list_model_ratings
from .settings import settings
from .strategies.compat import model_quirks

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

# `text_only` deliberately lets an untagged (`mode is None`) model through --
# most of litellm's chat-model catalog has no mode at all, so treating
# "unknown" as "exclude" would drop legitimate text models by the hundreds.
# A handful of non-text models carry no mode tag either (probed live
# 2026-08-24: xai's grok-imagine-image-*/grok-imagine-video-* both 404 on
# text completions with "is an image model and is therefore not available"),
# so those are named explicitly instead.
_NON_TEXT_NAME_FRAGMENTS = ("grok-imagine-image", "grok-imagine-video")

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


def _lowered(ratings: dict[str, int]) -> dict[str, int]:
    return {str(model).lower(): value for model, value in ratings.items()}


def _rating_for(thumbs: dict[str, int], provider: str, model: str) -> int | None:
    """A thumb stored under either the bare or provider-qualified model ID."""
    if not thumbs:
        return None
    return thumbs.get(model.lower()) or thumbs.get(f"{provider}/{model}".lower())


def _health_for(health: dict[str, dict], provider: str, model: str) -> dict | None:
    """Observed health stored under either the bare or qualified model ID."""
    if not health:
        return None
    return health.get(model.lower()) or health.get(f"{provider}/{model}".lower())


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


def _keep_litellm_pair(
    provider: str,
    model: str,
    *,
    live: dict[str, dict],
    verified: dict[str, dict],
    extra_display: set[tuple[str, str]],
) -> bool:
    """False if this litellm-only pair is a stale entry a fresh live listing
    contradicts.

    litellm's bundled catalog lags real deprecations as well as releases:
    probed live 2026-08-24, both groq and cerebras had several litellm-known
    model ids the provider's own `/v1/models` no longer lists at all, and a
    direct call confirmed them dead (404 / "decommissioned" / "archived").
    So a litellm model missing from a provider's live listing is dropped --
    unless a human already confirmed it works (the verified registry, which
    outranks a live snapshot that might just be incomplete for this key's
    access tier) or it was hand-added to `_EXTRA_MODELS_BY_PROVIDER`.
    """
    row = live.get(provider)
    if not row or not row.get("models"):
        return True  # no live signal for this provider -- trust litellm as-is
    display = _display_model_id(provider, model)
    if display in row["models"]:
        return True
    if (provider, display) in extra_display:
        return True
    return model in verified or f"{provider}/{model}" in verified


def _all_provider_pairs() -> list[tuple[str, str]]:
    """(provider, model) pairs from litellm and the live cache, taking the

    best of each: a fresh live listing adds ids litellm doesn't know about
    yet and drops ones litellm still lists but the provider no longer serves
    (see `_keep_litellm_pair`); a provider with no live cache entry is
    unaffected.
    """
    by_provider: dict[str, list[str]] = litellm.models_by_provider
    pairs = [(p, m) for p, ms in by_provider.items() for m in ms]
    pairs.extend((p, m) for p, ms in _EXTRA_MODELS_BY_PROVIDER.items() for m in ms)

    live = _live_rows_by_provider()
    verified = _verified_rows_by_model()
    extra_display = {
        (p, _display_model_id(p, m))
        for p, ms in _EXTRA_MODELS_BY_PROVIDER.items()
        for m in ms
    }
    pairs = [
        (p, m)
        for p, m in pairs
        if _keep_litellm_pair(
            p, m, live=live, verified=verified, extra_display=extra_display
        )
    ]

    known_display = {(p, _display_model_id(p, m)) for p, m in pairs}
    for provider, row in live.items():
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
    ratings: dict[str, int] | None = None,
    health_days: int | None = None,
) -> list[dict]:
    """Structured model metadata for frontend pickers.

    Includes:
    - stable model id (`model`)
    - provider and key-availability
    - litellm `mode` (chat, image_generation, embedding, ...)
    - verified pricing/reliability/prompt-method when known
    - `quirks`: known API-acceptance rules (e.g. `no_temperature`,
      `no_prefill`) a frontend can use to grey out or hide controls the
      model will reject; see `basemode.strategies.compat.model_quirks`
    - `rating`: this user's thumb for the model (`1`, `-1`, or `None`), stored
      in `~/.config/basemode/auth.json`; see `basemode.keys.set_model_rating`
    - `health`: what this model actually did for this user — attempts,
      failures, failure rate, and failure categories — or `None` if they have
      never generated with it; see `basemode.health`. `health_days` narrows
      its windowed figures.

    A rated model sorts ahead of (thumbs up) or behind (thumbs down) the
    reliability ordering, so an explicit opinion outranks the shipped data
    for every consumer of this list. Pass `ratings` to override the stored
    thumbs.

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
    thumbs = list_model_ratings() if ratings is None else _lowered(ratings)
    health = list_model_health(days=health_days)
    verification = verification_history()
    from .evidence import classify_text_endpoint, excluded_non_text_models
    from .evidence import current_status as evidence_current_status

    durable_evidence = evidence_current_status()
    evidence_non_text = excluded_non_text_models()
    # A registry entry means a human confirmed this model works at some
    # point; a verification probe can contradict that with live evidence.
    # Only the versioned thorough suite can independently grant verified
    # status to an endpoint outside the curated registry. Quick and imported
    # probes establish reachability but are not equivalent to verification.
    broken = {m for m, e in verification.items() if e["currently_broken"]}
    # Thorough-suite evidence is durable and reproducible. Quick legacy
    # probes remain a compatibility fallback, but cannot independently grant
    # the stronger evidence-backed verified state.
    broken |= {m for m, e in durable_evidence.items() if e.get("currently_broken")}
    evidence_verified = {m for m, e in durable_evidence.items() if e.get("verified")}
    verified_models = (set(verified) | evidence_verified) - broken

    if verified_only:
        pairs = [
            (m.split("/", 1)[0] if "/" in m else "unknown", m) for m in verified_models
        ]
    else:
        pairs = _all_provider_pairs()
        known_display = {(p, _display_model_id(p, m)) for p, m in pairs}
        for m in verified_models:
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
        qualified = f"{model_provider}/{model}"
        if model.lower() in evidence_non_text or qualified.lower() in evidence_non_text:
            continue
        if not classify_text_endpoint(qualified, mode)[0]:
            continue
        if text_only and mode is not None and mode not in TEXT_MODES:
            continue
        if text_only and any(
            fragment in model.lower() for fragment in _NON_TEXT_NAME_FRAGMENTS
        ):
            continue

        v = verified.get(model) or verified.get(f"{model_provider}/{model}", {})
        is_broken = model in broken or qualified in broken
        is_evidence_verified = (
            model in evidence_verified or qualified in evidence_verified
        )
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
                "verified": (bool(v) or is_evidence_verified) and not is_broken,
                "prompt_method": v.get("prompt_method"),
                "reliability": v.get("reliability"),
                "release_date": release_date,
                "release_date_inferred": release_date_inferred,
                "input_cost_per_token": v.get("input_cost_per_token"),
                "output_cost_per_token": v.get("output_cost_per_token"),
                "issues": list(v.get("issues", [])),
                "quirks": sorted(model_quirks(model)),
                "rating": _rating_for(thumbs, model_provider, model),
                "health": _health_for(health, model_provider, model),
            }
        )

    if compact:
        entries = _compact_entries(entries)

    if since:
        cutoff = parse_since(since)
        entries = [e for e in entries if (e.get("release_date") or "") >= cutoff]

    def sort_key(item: dict) -> tuple[int, int, int, int, str, str]:
        reliability_rank = 0 if item.get("reliability") == "✓" else 1
        # -1 → 1 (last), None → 0, 1 → -1 (first).
        rating_rank = -(item.get("rating") or 0)
        return (
            0 if item["available"] else 1,
            rating_rank,
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

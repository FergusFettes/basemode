import json
from pathlib import Path

import litellm

from .settings import settings

_EXTRA_MODELS_BY_PROVIDER = {
    # Official Gemini API Gemma 4 IDs. LiteLLM 1.83.9 does not list these yet.
    "gemini": [
        "gemini/gemma-4-26b-a4b-it",
        "gemini/gemma-4-31b-it",
    ],
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _verified_rows_by_model() -> dict[str, dict]:
    path = _project_root() / "data" / "verified_models_details.json"
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    rows = payload.get("rows", [])
    return {
        row.get("model"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("model"), str)
    }


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


def list_model_picker_entries(
    provider: str | None = None,
    search: str | None = None,
    available_only: bool = False,
    verified_only: bool = False,
) -> list[dict]:
    """Structured model metadata for frontend pickers.

    Includes:
    - stable model id (`model`)
    - provider and key-availability
    - verified pricing/reliability/prompt-method when known
    """
    verified = _verified_rows_by_model()

    if verified_only:
        models = sorted(verified.keys())
    else:
        models = list_models(provider=provider, search=search, available_only=False)
        for m in verified:
            if m not in models:
                models.append(m)

    if provider:
        models = [m for m in models if m.split("/", 1)[0] == provider]
    if search:
        needle = search.lower()
        models = [m for m in models if needle in m.lower()]

    available_providers = set(settings.available_providers)
    entries: list[dict] = []
    for model in sorted(set(models)):
        model_provider = model.split("/", 1)[0] if "/" in model else "unknown"
        v = verified.get(model, {})
        available = model_provider in available_providers
        if available_only and not available:
            continue
        entries.append(
            {
                "model": model,
                "provider": model_provider,
                "available": available,
                "verified": bool(v),
                "prompt_method": v.get("prompt_method"),
                "reliability": v.get("reliability"),
                "release_date": v.get("release_date"),
                "input_cost_per_token": v.get("input_cost_per_token"),
                "output_cost_per_token": v.get("output_cost_per_token"),
                "issues": list(v.get("issues", [])),
            }
        )

    def sort_key(item: dict) -> tuple[int, int, int, str]:
        reliability_rank = 0 if item.get("reliability") == "✓" else 1
        return (
            0 if item["available"] else 1,
            0 if item["verified"] else 1,
            reliability_rank,
            item["model"],
        )

    return sorted(entries, key=sort_key)


def build_model_picker_state(
    *,
    selected: list[str] | None = None,
    max_models: int = 3,
    provider: str | None = None,
    search: str | None = None,
    available_only: bool = False,
    verified_only: bool = False,
) -> dict:
    """Frontend-friendly state blob for single- or multi-model picker UIs."""
    entries = list_model_picker_entries(
        provider=provider,
        search=search,
        available_only=available_only,
        verified_only=verified_only,
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

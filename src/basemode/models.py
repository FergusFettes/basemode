import litellm

from .settings import settings

_EXTRA_MODELS_BY_PROVIDER = {
    # Official Gemini API Gemma 4 IDs. LiteLLM 1.83.9 does not list these yet.
    "gemini": [
        "gemini/gemma-4-26b-a4b-it",
        "gemini/gemma-4-31b-it",
    ],
}


def _provider_models(provider: str, by_provider: dict[str, list[str]]) -> list[str]:
    return list(by_provider.get(provider, [])) + _EXTRA_MODELS_BY_PROVIDER.get(provider, [])


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

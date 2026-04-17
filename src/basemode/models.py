import litellm

from .settings import settings


def list_models(
    provider: str | None = None,
    search: str | None = None,
    available_only: bool = False,
) -> list[str]:
    by_provider: dict[str, list[str]] = litellm.models_by_provider

    if available_only:
        providers = settings.available_providers
        models = [m for p in providers for m in by_provider.get(p, [])]
    elif provider:
        models = by_provider.get(provider, [])
    else:
        models = [m for ms in by_provider.values() for m in ms]

    if search:
        models = [m for m in models if search.lower() in m.lower()]

    return sorted(set(models))


def list_providers() -> list[str]:
    return sorted(litellm.models_by_provider.keys())

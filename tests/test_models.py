from basemode.models import list_models, list_providers


def test_list_providers_nonempty() -> None:
    assert len(list_providers()) > 0


def test_list_models_all() -> None:
    models = list_models()
    assert len(models) > 0


def test_list_models_by_provider() -> None:
    models = list_models(provider="openai")
    assert len(models) > 0
    assert any("gpt" in m for m in models)


def test_list_models_includes_extra_gemini_models() -> None:
    models = list_models(provider="gemini")
    assert "gemini/gemma-4-26b-a4b-it" in models
    assert "gemini/gemma-4-31b-it" in models


def test_list_models_search() -> None:
    models = list_models(search="claude")
    assert all("claude" in m for m in models)


def test_list_models_search_case_insensitive() -> None:
    lower = list_models(search="claude")
    upper = list_models(search="CLAUDE")
    assert lower == upper


def test_list_models_no_duplicates() -> None:
    models = list_models()
    assert len(models) == len(set(models))

from basemode.models import (
    build_model_picker_state,
    list_model_picker_entries,
    list_models,
    list_providers,
)


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


def test_model_picker_entries_have_expected_shape() -> None:
    entries = list_model_picker_entries(search="gpt-4o-mini")
    assert len(entries) > 0
    first = entries[0]
    assert "model" in first
    assert "provider" in first
    assert "available" in first
    assert "verified" in first
    assert "issues" in first


def test_model_picker_verified_only_filters_results() -> None:
    verified_entries = list_model_picker_entries(verified_only=True)
    assert len(verified_entries) > 0
    assert all(e["verified"] for e in verified_entries)


def test_build_model_picker_state_supports_multi_select() -> None:
    selected = ["openai/gpt-4o-mini", "openai/gpt-5.4-mini", "zai/glm-5"]
    state = build_model_picker_state(
        selected=selected, max_models=3, verified_only=True
    )
    assert state["max_models"] == 3
    assert state["selected"] == selected
    assert state["too_many_selected"] is False


def test_build_model_picker_state_flags_over_selection() -> None:
    state = build_model_picker_state(
        selected=["a", "b", "c", "d"],
        max_models=3,
        verified_only=True,
    )
    assert state["too_many_selected"] is True

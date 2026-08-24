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


def test_live_listing_drops_stale_litellm_models(monkeypatch) -> None:
    """A litellm-known model missing from a provider's fresh live listing is
    dropped from the picker -- probed live 2026-08-24, groq's own /v1/models
    no longer lists several models litellm still has, and a direct call to
    one (llama-3.3-70b-versatile) confirmed a 404, not just a stale live
    snapshot."""
    import basemode.models as models_mod

    monkeypatch.setattr(
        models_mod,
        "_live_rows_by_provider",
        lambda: {"groq": {"models": {"allam-2-7b": "2025-01-23"}, "reliable_dates": True}},
    )

    entries = list_model_picker_entries(provider="groq", text_only=False)
    ids = {e["model"] for e in entries}
    assert "groq/allam-2-7b" in ids
    assert not any("llama-3.3-70b" in m for m in ids)


def test_live_listing_keeps_verified_models_even_if_absent_live(monkeypatch) -> None:
    """A model a human already confirmed works (the verified registry)
    outranks a live snapshot that might just be incomplete for this key's
    access tier -- it should survive the live-listing prune."""
    import basemode.models as models_mod

    monkeypatch.setattr(
        models_mod,
        "_live_rows_by_provider",
        lambda: {"openai": {"models": {}, "reliable_dates": True}},
    )

    entries = list_model_picker_entries(search="gpt-4o-mini", text_only=False)
    assert any(e["model"] == "gpt-4o-mini" for e in entries)


def test_live_listing_with_no_signal_keeps_litellm_as_is(monkeypatch) -> None:
    """A provider with no live cache entry (or an empty one, e.g. a failed
    refresh) is unaffected -- no signal means trust litellm."""
    import basemode.models as models_mod

    monkeypatch.setattr(models_mod, "_live_rows_by_provider", lambda: {})

    entries = list_model_picker_entries(provider="groq", text_only=False)
    assert any("llama-3.3-70b" in e["model"] for e in entries)


def test_text_only_drops_untagged_image_models_by_name() -> None:
    """xai's grok-imagine-image-*/grok-imagine-video-* carry no litellm mode
    tag at all (probed live 2026-08-24: both 404 a text completion with "is
    an image model and is therefore not available"), so the mode-based
    text_only filter can't catch them -- named explicitly instead."""
    entries = list_model_picker_entries(provider="xai", text_only=True)
    assert not any("grok-imagine" in e["model"] for e in entries)

    untagged = list_model_picker_entries(provider="xai", text_only=False)
    assert any("grok-imagine" in e["model"] for e in untagged)


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


def test_model_picker_entries_carry_the_stored_rating() -> None:
    from basemode.keys import set_model_rating

    set_model_rating("openai/gpt-4o-mini", -1)

    entries = list_model_picker_entries(search="gpt-4o-mini")
    rated = [e for e in entries if e["model"] in ("gpt-4o-mini", "openai/gpt-4o-mini")]
    assert rated
    assert all(e["rating"] == -1 for e in rated)
    assert all(e["rating"] is None for e in entries if e not in rated)


def test_model_picker_ratings_argument_overrides_the_store() -> None:
    from basemode.keys import set_model_rating

    set_model_rating("openai/gpt-4o-mini", -1)

    entries = list_model_picker_entries(search="gpt-4o-mini", ratings={})
    assert all(e["rating"] is None for e in entries)


def test_rated_models_sort_ahead_of_and_behind_unrated_ones() -> None:
    from basemode.keys import set_model_rating

    entries = list_model_picker_entries(provider="openai", available_only=False)
    assert len(entries) > 2
    liked, disliked = entries[-1]["model"], entries[0]["model"]
    set_model_rating(liked, 1)
    set_model_rating(disliked, -1)

    resorted = [e["model"] for e in list_model_picker_entries(provider="openai")]
    assert resorted[0] == liked
    assert resorted[-1] == disliked


def test_model_picker_entries_carry_observed_health() -> None:
    from basemode import health

    health.record_outcome("openai/gpt-4o-mini", ok=False, category="rate_limit")

    entries = list_model_picker_entries(search="gpt-4o-mini")
    rated = [e for e in entries if e["model"] in ("gpt-4o-mini", "openai/gpt-4o-mini")]

    assert rated
    assert all(e["health"]["failures"] == 1 for e in rated)
    assert all(e["health"]["categories"] == {"rate_limit": 1} for e in rated)


def test_model_picker_health_is_none_for_a_model_never_used() -> None:
    entries = list_model_picker_entries(search="gpt-4o-mini")

    assert all(e["health"] is None for e in entries)

import json

import pytest

from basemode import keys


def test_ratings_round_trip_case_insensitively() -> None:
    keys.set_model_rating("OpenAI/GPT-4o", keys.RATING_UP)

    assert keys.get_model_rating("openai/gpt-4o") == keys.RATING_UP
    assert keys.list_model_ratings() == {"openai/gpt-4o": 1}


def test_clearing_a_rating_removes_it() -> None:
    keys.set_model_rating("openai/gpt-4o", keys.RATING_DOWN)
    keys.set_model_rating("openai/gpt-4o", None)

    assert keys.get_model_rating("openai/gpt-4o") is None
    assert keys.list_model_ratings() == {}


@pytest.mark.parametrize("rating", [5, 0, True, False])
def test_only_thumbs_are_accepted(rating) -> None:
    with pytest.raises(ValueError):
        keys.set_model_rating("openai/gpt-4o", rating)


def test_ratings_live_beside_keys_without_disturbing_them() -> None:
    keys.set_key("openai", "sk-test-value")
    keys.set_model_rating("openai/gpt-4o", keys.RATING_UP)

    stored = json.loads(keys._AUTH_FILE.read_text())
    assert stored["keys"] == {"openai": "sk-test-value"}
    assert stored["model_ratings"] == {"openai/gpt-4o": 1}
    assert keys.get_key("openai") == "sk-test-value"


def test_hand_edited_junk_ratings_are_dropped_rather_than_raised_on() -> None:
    keys._AUTH_FILE.write_text(
        json.dumps(
            {
                "keys": {"openai": "sk-test-value"},
                "model_ratings": {"a": 3, "b": True, "c": "up", "d": -1},
            }
        )
    )

    assert keys.list_model_ratings() == {"d": -1}
    assert keys.get_key("openai") == "sk-test-value"


def test_legacy_flat_file_gains_ratings_on_write() -> None:
    keys._AUTH_FILE.write_text(json.dumps({"openai": "sk-legacy"}))

    keys.set_model_rating("openai/gpt-4o", keys.RATING_UP)

    stored = json.loads(keys._AUTH_FILE.read_text())
    assert stored == {
        "keys": {"openai": "sk-legacy"},
        "model_ratings": {"openai/gpt-4o": 1},
    }


def test_ratings_and_pins_match_canonical_and_legacy_wire_ids() -> None:
    from basemode.keys import (
        get_model_rating,
        get_strategy_override,
        set_model_rating,
        set_strategy_override,
    )

    wire = "deepinfra/zai-org/GLM-5.3-Flash"
    canonical = "deepinfra/zai/glm-5.3-flash"
    set_model_rating(wire, 1)
    set_strategy_override(wire, "system")
    assert get_model_rating(canonical) == 1
    assert get_strategy_override(canonical) == "system"
    set_model_rating(canonical, None)
    set_strategy_override(canonical, None)
    assert get_model_rating(wire) is None
    assert get_strategy_override(wire) is None

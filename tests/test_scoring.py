from basemode.scoring import looks_clean, score_continuation

PREFIX = "The ship rounded the headland and"


def test_clean_continuation_scores_full_marks() -> None:
    result = score_continuation(PREFIX, " the harbour opened out below them.")

    assert result.score == 1.0
    assert result.flags == ()
    assert result.clean
    assert result.detail == "clean"


def test_empty_output_scores_zero() -> None:
    result = score_continuation(PREFIX, "   \n ")

    assert result.score == 0.0
    assert result.flags == ("empty",)
    assert not result.clean


def test_preamble_is_flagged() -> None:
    result = score_continuation(PREFIX, "Sure! Here's a continuation for you:")

    assert "preamble" in result.flags
    assert not result.clean


def test_refusal_outweighs_preamble_and_is_reported_alone() -> None:
    result = score_continuation(PREFIX, "I'm sorry, I can't help with that.")

    assert result.flags == ("refusal",)
    assert result.score == 0.1


def test_echoed_prefix_is_flagged() -> None:
    result = score_continuation(PREFIX, " rounded the headland and slipped into fog.")

    assert "echoed_prefix" in result.flags


def test_short_coincidental_overlap_is_not_an_echo() -> None:
    result = score_continuation(PREFIX, " and then the fog closed in behind them.")

    assert "echoed_prefix" not in result.flags


def test_chat_turn_markers_are_flagged() -> None:
    result = score_continuation(PREFIX, " the wind rose.\n\nUser: keep going")

    assert "chat_turn" in result.flags


def test_meta_commentary_is_flagged() -> None:
    result = score_continuation(
        PREFIX, " here is how the text continues from the passage you gave."
    )

    assert "meta_commentary" in result.flags


def test_prose_prefix_penalises_list_and_fence_openings() -> None:
    assert "list_formatting" in score_continuation(PREFIX, "\n- first point").flags
    assert "code_fence" in score_continuation(PREFIX, "\n```python\nx = 1").flags


def test_code_prefix_does_not_penalise_a_fence() -> None:
    result = score_continuation("```python\ndef f():", "\n    return 1\n")

    assert "code_fence" not in result.flags


def test_quoted_output_is_flagged() -> None:
    result = score_continuation(PREFIX, '"the harbour opened out below them."')

    assert "quoted" in result.flags


def test_missing_boundary_space_is_flagged_lightly() -> None:
    result = score_continuation(PREFIX, "slipped into fog.")

    assert result.flags == ("bad_boundary",)
    assert result.score == 0.85
    assert result.clean


def test_boundary_flag_does_not_pile_onto_a_bad_answer() -> None:
    result = score_continuation(PREFIX, "Sure! Here you go:")

    assert result.flags == ("preamble",)


def test_punctuation_join_is_not_a_boundary_error() -> None:
    result = score_continuation("The ship rounded the headland", ", then vanished.")

    assert "bad_boundary" not in result.flags


def test_penalties_accumulate_and_clamp_at_zero() -> None:
    result = score_continuation(
        PREFIX,
        'Certainly! Here is the text you provided:\n\n"rounded the headland and"'
        "\n\nUser: more",
    )

    assert result.score == 0.0
    assert len(result.flags) >= 3


def test_looks_clean_returns_reason_on_failure() -> None:
    ok, reason = looks_clean(PREFIX, "Sure! Here you go:")

    assert not ok
    assert "preamble" in reason

    ok, reason = looks_clean(PREFIX, " the harbour opened out.")
    assert ok
    assert reason is None

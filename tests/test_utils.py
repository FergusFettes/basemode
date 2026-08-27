"""Unit tests for prefix normalization edge cases."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from basemode.healing import (
    needs_leading_space,
    normalize_completion_segment,
    normalize_prefix,
    normalize_stream_newlines,
    probe_rewind_overlap,
    rewind_prefix_to_word_boundary,
)


@pytest.mark.parametrize(
    "inp,expected",
    [
        # Normal cases
        ("The ship rounded the headland and", "The ship rounded the headland and "),
        ("Hello world", "Hello world "),
        # Already has trailing space — should have exactly one
        ("text with space ", "text with space "),
        ("text with two spaces  ", "text with two spaces "),
        ("text with tab\t", "text with tab "),
        # Trailing newline — common when piping text
        ("line one\nline two\n", "line one\nline two "),
        ("line one\nline two\n\n", "line one\nline two "),
        # Mid-word — shouldn't add space inside words
        ("The quick bro", "The quick bro "),
        # Punctuation endings
        ("She said:", "She said: "),
        ("The end.", "The end. "),
        ("Wait—", "Wait— "),
        # Empty string
        ("", " "),
        # Only whitespace
        ("   ", " "),
        ("\n\n", " "),
        # Unicode
        ("café", "café "),
        ("雨が降る", "雨が降る "),
        # Poetry — ends mid-line, no trailing space
        (
            "the rain falls like static\nbetween stations, the city\nblurs into signal and",
            "the rain falls like static\nbetween stations, the city\nblurs into signal and ",
        ),
    ],
)
def test_normalize_prefix(inp: str, expected: str) -> None:
    assert normalize_prefix(inp) == expected


def test_normalize_prefix_idempotent() -> None:
    text = "Hello world"
    once = normalize_prefix(text)
    twice = normalize_prefix(once)
    assert once == twice


def test_normalize_prefix_preserves_internal_whitespace() -> None:
    text = "line one\n\nline two"
    result = normalize_prefix(text)
    assert "\n\n" in result
    assert result.endswith(" ")


def test_normalize_prefix_no_double_space() -> None:
    result = normalize_prefix("text ")
    assert not result.endswith("  ")


def test_normalize_prefix_newline_then_space() -> None:
    # \n followed by spaces — normalise to single trailing space
    result = normalize_prefix("text\n  ")
    assert result.endswith(" ")
    assert not result.endswith("  ")


def test_rewind_prefix_to_word_boundary_rewinds_short_fragment() -> None:
    generation_prefix, fragment = rewind_prefix_to_word_boundary(
        "twas brilig and the sli"
    )

    assert generation_prefix == "twas brilig and the "
    assert fragment == "sli"


def test_rewind_prefix_to_word_boundary_rewinds_common_word() -> None:
    # "and" is exactly the case that produces "a nd": the model wants to finish
    # the word, so hold it back and let the model write it whole.
    generation_prefix, fragment = rewind_prefix_to_word_boundary(
        "The ship rounded the headland and"
    )

    assert generation_prefix == "The ship rounded the headland "
    assert fragment == "and"


def test_rewind_prefix_to_word_boundary_keeps_long_tail() -> None:
    generation_prefix, fragment = rewind_prefix_to_word_boundary(
        "The ship rounded the headland"
    )

    assert generation_prefix == "The ship rounded the headland"
    assert fragment == ""


def test_rewind_prefix_to_word_boundary_needs_whitespace_boundary() -> None:
    # "nks" is the tail of "flanks", not a token of its own.
    generation_prefix, fragment = rewind_prefix_to_word_boundary(
        "muscle across their flanks"
    )

    assert generation_prefix == "muscle across their flanks"
    assert fragment == ""


async def test_probe_rewind_overlap_removes_repeated_fragment() -> None:
    async def gen():
        yield "sli"
        yield "vey toves"

    stream = gen()
    matched, head = await probe_rewind_overlap(stream, "sli")
    rest = "".join([token async for token in stream])

    assert matched
    assert head + rest == "vey toves"


async def test_probe_rewind_overlap_removes_spaced_repeated_fragment() -> None:
    async def gen():
        yield " slivey"
        yield " toves"

    stream = gen()
    matched, head = await probe_rewind_overlap(stream, "sli")
    rest = "".join([token async for token in stream])

    assert matched
    assert head + rest == "vey toves"


async def test_probe_rewind_overlap_reports_mismatch() -> None:
    async def gen():
        yield "measured"
        yield " and maximized"

    matched, head = await probe_rewind_overlap(gen(), "a")

    assert not matched
    assert head == "measured"  # discarded by the caller, which retries


# ── needs_leading_space ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "prefix,token,expected",
    [
        # Needs space: prefix ends with word char, token starts with word char
        ("The ship rounded the headland and", "suddenly", True),
        ("Hello world", "foo", True),
        ("end", "start", True),
        # No space needed: prefix ends with space
        ("ends with space ", "word", False),
        ("ends with space ", " word", False),
        # No space needed: token starts with space
        ("no trailing space", " word", False),
        # Punctuation: trailing punctuation + word token still smashes
        ("She said:", "hello", True),  # "She said:hello" is wrong
        ("She said:", " hello", False),  # space in token is fine
        # No space needed: empty inputs
        ("", "word", False),
        ("prefix", "", False),
        ("", "", False),
        # No space needed: prefix ends with newline
        ("line one\n", "line two", False),
        ("line one\n", " line two", False),
    ],
)
def test_needs_leading_space(prefix: str, token: str, expected: bool) -> None:
    assert needs_leading_space(prefix, token) == expected


async def _collect_stream(prefix: str, chunks: list[str]) -> str:
    async def gen():
        for chunk in chunks:
            yield chunk

    return "".join([token async for token in normalize_stream_newlines(prefix, gen())])


async def test_normalize_stream_newlines_collapses_prose_wraps() -> None:
    result = await _collect_stream(
        "This is a prose paragraph about ideas.",
        [" To 'not seek\n", "to understand' is to preserve the boundary."],
    )

    assert result == " To 'not seek to understand' is to preserve the boundary."


async def test_normalize_stream_newlines_preserves_paragraph_breaks() -> None:
    result = await _collect_stream(
        "This is a prose paragraph.",
        [" First paragraph ends.\n\n", "Second paragraph begins."],
    )

    assert result == " First paragraph ends.\n\nSecond paragraph begins."


async def test_normalize_stream_newlines_preserves_markdown_starts() -> None:
    result = await _collect_stream(
        "This is a prose paragraph.",
        [" Items:\n", "- first\n", "- second"],
    )

    assert result == " Items:\n- first\n- second"


async def test_normalize_stream_newlines_preserves_poetry_like_prefixes() -> None:
    result = await _collect_stream(
        "the rain falls like static\nbetween stations, the city\nblurs into",
        [" signal\n", "and the wires hum"],
    )

    assert result == " signal\nand the wires hum"


async def test_normalize_stream_newlines_preserves_first_capitalized_line_break() -> (
    None
):
    result = await _collect_stream(
        "Seeing himself an atom in a shroud—",
        ["\nMan hears himself an engine in a cloud!"],
    )

    assert result == "\nMan hears himself an engine in a cloud!"


async def test_normalize_stream_newlines_collapses_capitalized_sentence_wrap() -> None:
    result = await _collect_stream(
        "This is a prose paragraph that ends.",
        ["\nThe next sentence starts here."],
    )

    assert result == " The next sentence starts here."


async def test_normalize_stream_repairs_split_compound_across_chunks() -> None:
    result = await _collect_stream(
        "It was not",
        [" coward", " ice but something more like engineering."],
    )

    assert result == " cowardice but something more like engineering."


async def test_normalize_stream_repairs_split_compound_at_prefix_boundary() -> None:
    result = await _collect_stream(
        "The recommendation was not born of laziness or coward",
        [" ice but of a kind of cognitive hygiene."],
    )

    assert result == "ice but of a kind of cognitive hygiene."


async def test_normalize_stream_repairs_hyphenated_word_at_prefix_boundary() -> None:
    result = await _collect_stream(
        "some impossible marriage of Jiu-J",
        [" itsu, Capoeira, and Fandango."],
    )

    assert result == "itsu, Capoeira, and Fandango."


async def test_normalize_stream_repairs_short_fragment_at_prefix_boundary() -> None:
    result = await _collect_stream(
        "muscle and motion across their fl",
        [" anks, abdomens, thighs."],
    )

    assert result == "anks, abdomens, thighs."


async def test_normalize_stream_leaves_no_space_hyphenated_boundary_alone() -> None:
    result = await _collect_stream(
        "some impossible marriage of Jiu-J",
        ["itsu, Capoeira, and Fandango."],
    )

    assert result == "itsu, Capoeira, and Fandango."


async def test_normalize_stream_repairs_split_compound_with_space_in_first_chunk() -> (
    None
):
    result = await _collect_stream(
        "The wall made the",
        [" out ", "side feel theoretical."],
    )

    assert result == " outside feel theoretical."


async def test_normalize_stream_does_not_join_unlisted_words() -> None:
    result = await _collect_stream(
        "The argument was",
        [" real", " ice on the page."],
    )

    assert result == " real ice on the page."


async def test_normalize_stream_does_not_repair_unlisted_prefix_boundary() -> None:
    result = await _collect_stream(
        "The argument was real",
        [" ice on the page."],
    )

    assert result == " ice on the page."


def test_normalize_completion_segment_repairs_hyphenated_prefix_boundary() -> None:
    result = normalize_completion_segment(
        "some impossible marriage of Jiu-J",
        " itsu, Capoeira, and Fandango.",
    )

    assert result == "itsu, Capoeira, and Fandango."


def test_normalize_completion_segment_trims_dangling_hyphenated_tail() -> None:
    result = normalize_completion_segment(
        "They learned",
        " a form assembled out of Jiu-J",
    )

    assert result == " a form assembled out of"


def test_normalize_completion_segment_trims_dangling_short_tail() -> None:
    result = normalize_completion_segment(
        "Under the lights",
        " their bodies seemed less illuminated than sm",
    )

    assert result == " their bodies seemed less illuminated than"


def test_normalize_completion_segment_keeps_common_short_final_word() -> None:
    result = normalize_completion_segment(
        "The comparison failed.",
        " The aim was",
    )

    assert result == " The aim was"


def test_normalize_completion_segment_keeps_finished_hyphenated_word() -> None:
    result = normalize_completion_segment(
        "They learned",
        " a form assembled out of Jiu-Jitsu",
    )

    assert result == " a form assembled out of Jiu-Jitsu"


def test_normalize_completion_segment_strips_echoed_word() -> None:
    result = normalize_completion_segment(
        "The cellular automaton Langton ",
        "Langton found he called it lambda.",
    )

    assert result == "found he called it lambda."


def test_normalize_completion_segment_strips_echoed_phrase() -> None:
    result = normalize_completion_segment(
        "he thought about the cellular automaton ",
        "the cellular automaton was stranger than expected.",
    )

    assert result == "was stranger than expected."


def test_normalize_completion_segment_keeps_ordinary_word_recurrence() -> None:
    result = normalize_completion_segment(
        "he said ",
        "he would go tomorrow.",
    )

    assert result == "he would go tomorrow."


async def test_normalize_stream_joins_single_letter_prefix_boundary() -> None:
    result = await _collect_stream(
        "welfare was counted, a",
        ["nd", " recalculated"],
    )

    assert result == "nd recalculated"


async def test_normalize_stream_repairs_boundary_only_once() -> None:
    tail = " and the institution counted the ratings again and again."
    result = await _collect_stream(
        "welfare was counted, a",
        ["nd recalculated" + tail, " nd so the ratings held."],
    )

    assert result == "nd recalculated" + tail + " nd so the ratings held."


# ── boundary repair: what joins and what must not ────────────────────────────


@pytest.mark.parametrize(
    "prefix,completion,expected",
    [
        # Joined: the injected space landed inside one word
        ("welfare was counted, a", " nd recalculated", "nd recalculated"),
        ("the institution kept I", " ts ratings", "ts ratings"),
        ("they were the", " ir own reward", "ir own reward"),
        ("motion across their fl", " anks, abdomens", "anks, abdomens"),
        ("they had recalculat", " ed the sums", "ed the sums"),
        ("it was the wa", " y out", "y out"),
        ("she had be", " en waiting", "en waiting"),
        # Kept apart: the second word stands on its own
        ("welfare was measured a", " way from the ratings", " way from the ratings"),
        ("labelled Section B", " undefined by the board", " undefined by the board"),
        ("a letter from Dr", " Smith arrived", " Smith arrived"),
        ("she reread Mrs", " Dalloway again", " Dalloway again"),
        ("the train to St", " Petersburg left", " Petersburg left"),
        ("a row of TV", " screens flickered", " screens flickered"),
        ("the institution counted the", " ratings again", " ratings again"),
        ("they had", " maximized the welfare", " maximized the welfare"),
        # Invented and proper nouns the dictionary has never heard of
        ("in the city of", " Vorthal they waited", " Vorthal they waited"),
        ("she greeted", " Kaal without warmth", " Kaal without warmth"),
        ("his brother", " Ed said nothing", " Ed said nothing"),
        # A hyphenated compound is not a split word
        ("the thing needed a", " re-evaluation", " re-evaluation"),
    ],
)
def test_normalize_completion_segment_boundary_cases(
    prefix: str, completion: str, expected: str
) -> None:
    assert normalize_completion_segment(prefix, completion) == expected


def test_normalize_completion_segment_uses_document_vocabulary() -> None:
    # "Xenosurgery" is in no dictionary, but the document has already used it.
    result = normalize_completion_segment(
        "Xenosurgery was routine by then. He had trained in Xenos",
        " urgery for eleven years.",
    )

    assert result == "urgery for eleven years."


def test_normalize_completion_segment_needs_document_evidence() -> None:
    result = normalize_completion_segment(
        "He had trained in Xenos",
        " urgery for eleven years.",
    )

    assert result == " urgery for eleven years."


# ── compound joining: ambiguous phrases must survive ─────────────────────────


@pytest.mark.parametrize(
    "prefix,chunk,expected",
    [
        # Joined
        (
            "The wall made",
            [" the out ", "side feel theoretical."],
            " the outside feel theoretical.",
        ),
        ("He kept", [" every thing in its place."], " everything in its place."),
        ("There was", [" no where left to go."], " nowhere left to go."),
        ("He found", [" him self alone."], " himself alone."),
        # Kept apart: the split reading is the real one
        (
            "The institution",
            [" counted every one of them again."],
            " counted every one of them again.",
        ),
        ("It was", [" some body of water, vast."], " some body of water, vast."),
        (
            "They laid the tools",
            [" out side by side."],
            " out side by side.",
        ),
        (
            "The review broke",
            [" her self-esteem."],
            " her self-esteem.",
        ),
        (
            "The lamps lit",
            [" the parked cars in side streets."],
            " the parked cars in side streets.",
        ),
        (
            "The search found",
            [" no body in the room."],
            " no body in the room.",
        ),
    ],
)
async def test_normalize_stream_compound_cases(
    prefix: str, chunk: list[str], expected: str
) -> None:
    assert await _collect_stream(prefix, chunk) == expected


async def test_normalize_stream_joins_compound_at_any_alignment() -> None:
    # The pair used to be missed whenever an earlier pair consumed its left half.
    result = await _collect_stream(
        "The recommendation",
        [" was not coward ice but engineering."],
    )

    assert result == " was not cowardice but engineering."


# ── rewind retry ─────────────────────────────────────────────────────────────


class _ScriptedStrategy:
    """Strategy stub that records the prefixes it is asked to continue."""

    name = "system"

    def __init__(self, *responses: list[str]) -> None:
        self._responses = list(responses)
        self.prefixes: list[str] = []

    async def stream(self, prefix, params):
        self.prefixes.append(prefix)
        for token in self._responses.pop(0):
            yield token


async def test_stream_tokens_uses_rewound_generation_when_word_completed() -> None:
    from basemode.continue_ import _stream_tokens

    strat = _ScriptedStrategy(["and", " recalculated"])
    tokens = _stream_tokens(
        strat, "welfare was counted, a", "welfare was counted, ", "a", None
    )

    assert "".join([token async for token in tokens]) == "nd recalculated"
    assert strat.prefixes == ["welfare was counted, "]


async def test_stream_tokens_retries_when_rewind_not_taken_up() -> None:
    from basemode.continue_ import _stream_tokens

    strat = _ScriptedStrategy(["measured", " again"], [" measured again"])
    tokens = _stream_tokens(
        strat, "welfare was counted, a", "welfare was counted, ", "a", None
    )

    # The rewound text is discarded — it was written to follow "counted, ".
    assert "".join([token async for token in tokens]) == " measured again"
    assert strat.prefixes == ["welfare was counted, ", "welfare was counted, a"]


def test_normalize_completion_segment_keeps_short_coinage_from_document() -> None:
    # A three-letter tail that the document has established is a word, not the
    # stump of one cut off by the token limit.
    result = normalize_completion_segment(
        "The Vor had ruled for centuries.",
        " Their envoys came from Vor",
    )

    assert result == " Their envoys came from Vor"


def test_healing_uses_the_packaged_dictionary_not_the_host_one(monkeypatch) -> None:
    """The repair rules must behave the same on a host with no /usr/share/dict.

    With an empty dictionary every word looks like a fragment, so the prefix
    always reads as mid-word and boundaries get joined that should not be —
    "Seed beta" + " gamma" became "Seed betagamma" on stock CI runners.
    """
    from basemode import healing

    monkeypatch.setattr(healing, "_DICTIONARY", None)
    monkeypatch.setattr(healing, "_DICT_PATH", Path("/nonexistent/dict/words"))

    assert len(healing._dictionary()) > 100_000
    assert healing._is_word("gamma")
    assert normalize_completion_segment("Seed beta", " gamma") == " gamma"


def test_healing_falls_back_to_the_system_dictionary(tmp_path, monkeypatch) -> None:
    from basemode import healing

    fallback = tmp_path / "words"
    fallback.write_text("Alpha\nbeta\ngamma\n")
    monkeypatch.setattr(healing, "_DICTIONARY", None)
    monkeypatch.setattr(healing, "_load_bundled_dictionary", lambda: None)
    monkeypatch.setattr(healing, "_DICT_PATH", fallback)

    assert healing._dictionary() == frozenset({"alpha", "beta", "gamma"})


# --- a line break inside a word at the generation boundary ---


def test_a_newline_inside_a_split_word_is_dropped() -> None:
    """Seen in the wild: opus continued "the thermod" with "\\n\\nynamic sense"."""
    healed = normalize_completion_segment(
        "to work at all, since work in the thermod",
        "\n\nynamic sense is only possible across a gradient",
    )

    assert healed == "ynamic sense is only possible across a gradient"


def test_a_single_newline_inside_a_split_word_is_dropped() -> None:
    healed = normalize_completion_segment(
        "a state of recalculat", "\ned figures for the quarter"
    )

    assert healed == "ed figures for the quarter"


def test_a_paragraph_break_after_a_whole_word_survives() -> None:
    healed = normalize_completion_segment(
        "and that xe would want for nothing.\n\n# Xe",
        "\n\nXe was the most hominiform of xer broodset",
    )

    assert healed == "\n\nXe was the most hominiform of xer broodset"


def test_a_paragraph_break_after_a_sentence_survives() -> None:
    healed = normalize_completion_segment(
        "It was the end of the chapter.", "\n\nThe next morning was clear"
    )

    assert healed == "\n\nThe next morning was clear"


def test_a_fragment_that_does_not_complete_a_word_keeps_its_break() -> None:
    """Without a word to make, the break is the model changing subject."""
    healed = normalize_completion_segment(
        "to work at all, since work in the thermod", "\n\nThe next chapter begins"
    )

    assert healed == "\n\nThe next chapter begins"


def test_a_prefix_ending_in_whitespace_keeps_the_break() -> None:
    healed = normalize_completion_segment(
        "the sentence ended here ", "\n\nAnother thought entirely"
    )

    assert healed == "\n\nAnother thought entirely"


@pytest.mark.asyncio
async def test_the_stream_drops_a_newline_inside_a_split_word() -> None:
    async def tokens():
        for token in ["\n", "\n", "ynamic", " sense is only possible across a grad"]:
            yield token

    prefix = "to work at all, since work in the thermod"
    streamed = "".join([t async for t in normalize_stream_newlines(prefix, tokens())])

    assert streamed.startswith("ynamic sense")


# --- GenerationParams value semantics ---


def test_generation_params_is_frozen() -> None:
    from basemode.params import GenerationParams

    params = GenerationParams(model="gpt-4o-mini")
    with pytest.raises(FrozenInstanceError):
        params.model = "other-model"  # type: ignore[misc]


# --- branch_text's empty-completion retry ---


class _AlwaysEmptyStrategy:
    """Raises EmptyCompletionError on every call; records how it was asked."""

    name = "system"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def stream(self, prefix, params):
        from basemode.exceptions import EmptyCompletionError

        self.calls.append(
            {"prefix": prefix, "reasoning": params.extra.get("reasoning")}
        )
        if False:
            yield ""  # pragma: no cover - makes this an async generator
        raise EmptyCompletionError(
            model=params.model, strategy="system", finish_reason="length"
        )


async def _drain_branch_text(**kwargs):
    from basemode.continue_ import branch_text

    tokens = []
    with pytest.raises(RuntimeError):
        async for item in branch_text(**kwargs):
            tokens.append(item)
    return tokens


async def test_branch_text_retries_empty_completion_by_default(monkeypatch) -> None:
    from basemode import continue_

    strat = _AlwaysEmptyStrategy()
    monkeypatch.setattr(continue_, "detect_strategy", lambda model, strategy: strat)

    await _drain_branch_text(
        prefix="hello",
        model="openrouter/some-model",
        n=1,
        record_health=False,
    )

    # One initial attempt, one retry with reasoning forced off.
    assert len(strat.calls) == 2
    assert strat.calls[0]["reasoning"] is None
    assert strat.calls[1]["reasoning"] == {"enabled": False}


async def test_branch_text_does_not_retry_when_disabled(monkeypatch) -> None:
    from basemode import continue_

    strat = _AlwaysEmptyStrategy()
    monkeypatch.setattr(continue_, "detect_strategy", lambda model, strategy: strat)

    await _drain_branch_text(
        prefix="hello",
        model="openrouter/some-model",
        n=1,
        record_health=False,
        retry_empty_completion=False,
    )

    assert len(strat.calls) == 1

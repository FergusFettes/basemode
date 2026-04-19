"""Unit tests for prefix normalization edge cases."""

import pytest

from basemode.strategies.utils import (
    needs_leading_space,
    normalize_completion_segment,
    normalize_prefix,
    normalize_stream_newlines,
)


@pytest.mark.parametrize("inp,expected", [
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
    ("the rain falls like static\nbetween stations, the city\nblurs into signal and",
     "the rain falls like static\nbetween stations, the city\nblurs into signal and "),
])
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


# ── needs_leading_space ───────────────────────────────────────────────────────

@pytest.mark.parametrize("prefix,token,expected", [
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
    ("She said:", "hello", True),   # "She said:hello" is wrong
    ("She said:", " hello", False), # space in token is fine
    # No space needed: empty inputs
    ("", "word", False),
    ("prefix", "", False),
    ("", "", False),
    # No space needed: prefix ends with newline
    ("line one\n", "line two", False),
    ("line one\n", " line two", False),
])
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


async def test_normalize_stream_repairs_split_compound_with_space_in_first_chunk() -> None:
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

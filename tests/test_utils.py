"""Unit tests for prefix normalization edge cases."""
import pytest

from basemode.strategies.utils import needs_leading_space, normalize_prefix


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

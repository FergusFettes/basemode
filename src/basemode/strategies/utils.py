import re
from collections.abc import AsyncGenerator, AsyncIterable
from pathlib import Path

_LOOKBEHIND_CHARS = 80
_COMMIT_LAG_CHARS = 32

_DICT_PATH = Path("/usr/share/dict/words")
_SYSTEM_WORDS: frozenset[str] | None = None


def _system_words() -> frozenset[str]:
    global _SYSTEM_WORDS
    if _SYSTEM_WORDS is None:
        if _DICT_PATH.exists():
            _SYSTEM_WORDS = frozenset(_DICT_PATH.read_text().lower().split())
        else:
            _SYSTEM_WORDS = frozenset()
    return _SYSTEM_WORDS


def _is_word(s: str) -> bool:
    return s.lower() in _system_words()


_COMPOUND_RE = re.compile(r"\b([A-Za-z]{2,}) ([A-Za-z]{2,})\b")
_PREFIX_WORD_RE = re.compile(r"([A-Za-z]{2,})$")
_LEADING_WORD_RE = re.compile(r"^ ([A-Za-z]{2,})(\b|(?=[^A-Za-z]))")


def normalize_prefix(prefix: str) -> str:
    """Ensure prefix ends with exactly one space for the model input.

    Chat models respond without a leading space, so we strip trailing whitespace
    and add exactly one space. This makes the model output tokens that join
    correctly when we prepend a space to the first token if needed.

    Not applied to completion/prefill strategies — they handle boundaries natively.
    """
    return prefix.rstrip() + " "


def needs_leading_space(prefix: str, first_token: str) -> bool:
    """Return True if a space must be injected between prefix and first_token.

    After sending normalize_prefix(prefix) to the model, the model outputs
    first_token without a leading space. If the original prefix didn't end
    with whitespace, the space was consumed in the model input and must be
    restored so that prefix + tokens is correct text.
    """
    return (
        bool(prefix)
        and not prefix[-1].isspace()
        and bool(first_token)
        and not first_token[0].isspace()
    )


def _looks_line_oriented(text: str) -> bool:
    lines = [line for line in text.rstrip().splitlines() if line.strip()]
    if len(lines) < 3:
        return False

    recent = lines[-4:]
    avg_len = sum(len(line.strip()) for line in recent) / len(recent)
    punctuation_endings = sum(line.rstrip().endswith((".", "!", "?", ":", ";")) for line in recent)
    markdown_starts = sum(line.lstrip().startswith(("#", ">", "-", "*", "+", "```")) for line in recent)

    return markdown_starts > 0 or (avg_len < 48 and punctuation_endings <= 1)


def _should_collapse_single_newline(prefix: str, prev_char: str, next_char: str) -> bool:
    if _looks_line_oriented(prefix):
        return False
    if not prev_char or not next_char:
        return False
    if prev_char.isspace() or next_char.isspace():
        return False
    if next_char in "#>-*+`|":
        return False
    return True


_JOINABLE_COMPOUNDS = {
    "anybody", "anyone", "anything", "anywhere",
    "cowardice",
    "everybody", "everyone", "everything", "everywhere",
    "herself", "himself", "inside", "itself", "myself",
    "nobody", "nothing", "nowhere",
    "ourselves", "outside",
    "somebody", "someone", "something", "somewhere",
    "themselves", "yourself", "yourselves",
}


def _join_split_compounds(text: str) -> str:
    """Join high-confidence compounds in the mutable stream tail."""

    def replace(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        joined = left + right
        if joined.lower() not in _JOINABLE_COMPOUNDS:
            return match.group(0)
        if left.isupper():
            return joined.upper()
        if left[0].isupper():
            return joined.capitalize()
        return joined

    return _COMPOUND_RE.sub(replace, text)


_SPACE_PUNCT_RE = re.compile(r"(?<=\w) ([,\.;:!?])")
_SPACE_CONTRACTION_RE = re.compile(r"(?<=\w) ('(?:s|t|re|ve|ll|d|m))\b", re.IGNORECASE)
_LEADING_PUNCT_RE = re.compile(r"^ ([,\.;:!?])")
_LEADING_CONTRACTION_RE = re.compile(r"^ ('(?:s|t|re|ve|ll|d|m))\b", re.IGNORECASE)


def _fix_space_before_punctuation(text: str) -> str:
    text = _SPACE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_CONTRACTION_RE.sub(r"\1", text)
    return text


def _repair_prefix_boundary(prefix: str, text: str) -> str:
    prefix_match = _PREFIX_WORD_RE.search(prefix)

    # Space injected before punctuation or contraction at the boundary
    if prefix_match:
        if _LEADING_PUNCT_RE.match(text) or _LEADING_CONTRACTION_RE.match(text):
            return text[1:]

    text_match = _LEADING_WORD_RE.match(text)
    if not prefix_match or not text_match:
        return text

    left = prefix_match.group(1)
    right = text_match.group(1)
    joined = left + right

    # Whitelist: high-confidence compounds where both halves may be real words
    if joined.lower() in _JOINABLE_COMPOUNDS:
        return text[1:]

    # Dictionary: join if combined form is a word and the fragment alone is not
    if _is_word(joined) and not _is_word(right):
        return text[1:]

    return text


async def normalize_stream_newlines(
    prefix: str,
    tokens: AsyncIterable[str],
) -> AsyncGenerator[str, None]:
    """Collapse likely hard-wrapped prose newlines and repair split compounds.

    The final few characters are held back briefly so the next token can repair
    boundaries such as ``coward ice`` -> ``cowardice`` before text is committed
    to the caller.
    """
    prev_char = prefix[-1] if prefix else ""
    pending_newlines = 0
    pending_text = ""

    async for token in tokens:
        out: list[str] = []
        for char in token:
            if char == "\n":
                pending_newlines += 1
                continue

            if pending_newlines:
                if pending_newlines == 1 and _should_collapse_single_newline(prefix, prev_char, char):
                    if prev_char != " ":
                        out.append(" ")
                        prev_char = " "
                else:
                    out.append("\n" * pending_newlines)
                    prev_char = "\n"
                pending_newlines = 0

            out.append(char)
            prev_char = char

        if out:
            pending_text += "".join(out)
            pending_text = _repair_prefix_boundary(prefix, pending_text)
            pending_text = _join_split_compounds(pending_text)
            pending_text = _fix_space_before_punctuation(pending_text)
            if len(pending_text) > _LOOKBEHIND_CHARS:
                emit_len = len(pending_text) - _COMMIT_LAG_CHARS
                yield pending_text[:emit_len]
                pending_text = pending_text[emit_len:]

    if pending_newlines:
        pending_text += "\n" * pending_newlines
    pending_text = _repair_prefix_boundary(prefix, pending_text)
    pending_text = _join_split_compounds(pending_text)
    pending_text = _fix_space_before_punctuation(pending_text)
    if pending_text:
        yield pending_text

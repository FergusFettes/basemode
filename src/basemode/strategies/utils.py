import re
from collections.abc import AsyncGenerator, AsyncIterable

_LOOKBEHIND_CHARS = 80
_COMMIT_LAG_CHARS = 32

# High-confidence compounds where the split form is rarely intended in prose.
_JOINABLE_COMPOUNDS = {
    "anybody",
    "anyone",
    "anything",
    "anywhere",
    "cowardice",
    "everybody",
    "everyone",
    "everything",
    "everywhere",
    "herself",
    "himself",
    "inside",
    "itself",
    "myself",
    "nobody",
    "nothing",
    "nowhere",
    "ourselves",
    "outside",
    "somebody",
    "someone",
    "something",
    "somewhere",
    "themselves",
    "yourself",
    "yourselves",
}

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


def _repair_prefix_boundary(prefix: str, text: str) -> str:
    prefix_match = _PREFIX_WORD_RE.search(prefix)
    text_match = _LEADING_WORD_RE.match(text)
    if not prefix_match or not text_match:
        return text

    joined = prefix_match.group(1) + text_match.group(1)
    if joined.lower() not in _JOINABLE_COMPOUNDS:
        return text

    # The prefix already contains the first half, so the continuation should
    # start with the second half and no inserted boundary space.
    return text[1:]


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
            if len(pending_text) > _LOOKBEHIND_CHARS:
                emit_len = len(pending_text) - _COMMIT_LAG_CHARS
                yield pending_text[:emit_len]
                pending_text = pending_text[emit_len:]

    if pending_newlines:
        pending_text += "\n" * pending_newlines
    pending_text = _repair_prefix_boundary(prefix, pending_text)
    pending_text = _join_split_compounds(pending_text)
    if pending_text:
        yield pending_text

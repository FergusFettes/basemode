from collections.abc import AsyncGenerator, AsyncIterable


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


async def normalize_stream_newlines(
    prefix: str,
    tokens: AsyncIterable[str],
) -> AsyncGenerator[str, None]:
    """Collapse likely hard-wrapped prose newlines while preserving structure."""
    prev_char = prefix[-1] if prefix else ""
    pending_newlines = 0

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
            yield "".join(out)

    if pending_newlines:
        yield "\n" * pending_newlines

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

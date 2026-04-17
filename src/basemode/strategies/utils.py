def normalize_prefix(prefix: str) -> str:
    """Ensure prefix ends with exactly one space for clean chat continuation.

    Chat models respond to the user message without a leading space, so if the
    prefix ends mid-word we get "andglided". Stripping trailing whitespace and
    adding one space makes the model output "glided" which joins correctly.

    Not applied to completion/prefill strategies — they handle boundaries natively.
    """
    return prefix.rstrip() + " "

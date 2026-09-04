import logging

from basemode.logging_setup import RedactingFormatter
from basemode.redaction import REDACTED, redact


def test_openrouter_user_id_is_removed() -> None:
    line = (
        'OpenrouterException - {"error":{"message":"Provider returned error",'
        '"code":400,"metadata":{"provider_name":"OpenAI"}},'
        '"user_id":"user_2fQx1JmKp0Za9LbN"}'
    )

    cleaned = redact(line)

    assert "user_2fQx1JmKp0Za9LbN" not in cleaned
    # The parts that make the log worth keeping survive.
    assert "Provider returned error" in cleaned
    assert '"code":400' in cleaned


def test_xai_team_uuid_is_removed() -> None:
    line = (
        'XaiException - {"code":"permission-denied","error":"Your team '
        '51edf858-83c5-45da-98fd-e6e27aae476e does not have access"}'
    )

    cleaned = redact(line)

    assert "51edf858" not in cleaned
    assert "permission-denied" in cleaned


def test_keys_and_addresses_are_removed() -> None:
    cleaned = redact("Authorization: Bearer sk-or-v1-abcdefghij0123456789 for a@b.com")

    assert "abcdefghij0123456789" not in cleaned
    assert "a@b.com" not in cleaned
    assert cleaned.count(REDACTED) == 2


def test_model_ids_and_parameters_are_left_alone() -> None:
    line = (
        "Unsupported parameter: 'max_tokens' is not supported with "
        "openai/gpt-5.6-luna. Use 'max_completion_tokens' instead."
    )

    assert redact(line) == line


def test_formatter_redacts_the_traceback_not_just_the_message() -> None:
    formatter = RedactingFormatter("%(message)s")
    try:
        raise RuntimeError('{"user_id":"user_2fQx1JmKp0Za9LbN"}')
    except RuntimeError:
        import sys

        record = logging.LogRecord(
            "basemode.continue_",
            logging.ERROR,
            __file__,
            1,
            "stream error after 0 tokens",
            (),
            sys.exc_info(),
        )

    formatted = formatter.format(record)

    assert "user_2fQx1JmKp0Za9LbN" not in formatted
    assert "stream error after 0 tokens" in formatted

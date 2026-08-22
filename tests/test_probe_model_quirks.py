import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import probe_model_quirks as pmq  # noqa: E402


def test_temperature_error_regex_matches_known_provider_wordings() -> None:
    assert pmq._TEMPERATURE_ERROR_RE.search(
        "`temperature` is deprecated for this model."
    )
    assert pmq._TEMPERATURE_ERROR_RE.search(
        "Unsupported value: 'temperature' does not support 0.3 with this model."
    )
    assert not pmq._TEMPERATURE_ERROR_RE.search("rate limit exceeded")


def test_prefill_error_regex_matches_known_provider_wordings() -> None:
    assert pmq._PREFILL_ERROR_RE.search(
        "This model does not support assistant message prefill."
    )
    assert not pmq._PREFILL_ERROR_RE.search("rate limit exceeded")


def test_compact_scalar_arrays_collapses_multiline_string_arrays() -> None:
    rendered = '{\n  "quirks": [\n    "no_temperature",\n    "no_prefill"\n  ]\n}'
    assert pmq._compact_scalar_arrays(rendered) == (
        '{\n  "quirks": ["no_temperature", "no_prefill"]\n}'
    )


def test_provider_splits_on_first_slash() -> None:
    assert pmq._provider("anthropic/claude-sonnet-5") == "anthropic"
    assert pmq._provider("openrouter/moonshotai/kimi-k2.6") == "openrouter"
    assert pmq._provider("standalone") == "unknown"

"""Persistent API key + default-model storage in ~/.config/basemode/auth.json.

File schema (current):
    {"keys": {"openai": "sk-...", ...}, "default_model": "...",
     "strategy_overrides": {"anthropic/claude-opus-5": "few_shot", ...},
     "model_ratings": {"anthropic/claude-opus-5": 1, "openai/gpt-4o": -1}}

Legacy flat schema (auto-migrated on next write):
    {"openai": "sk-...", ...}
"""

import json
import os
from pathlib import Path

_CONFIG_DIR = Path.home() / ".config" / "basemode"
_AUTH_FILE = _CONFIG_DIR / "auth.json"

# Maps short key names → env var names that litellm reads
KEY_ALIASES: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "together": "TOGETHER_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "xai": "XAI_API_KEY",
    "zai": "ZAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

#: The only values a model rating may take: thumbs up, thumbs down.
RATING_UP = 1
RATING_DOWN = -1
_VALID_RATINGS = (RATING_UP, RATING_DOWN)


def _load_raw() -> dict:
    if not _AUTH_FILE.exists():
        return {}
    return json.loads(_AUTH_FILE.read_text())


def _normalize_ratings(raw: object) -> dict[str, int]:
    """Keep only well-formed thumbs from the file; drop anything else.

    A hand-edited or future-schema value should not crash a read that only
    wanted the keys, so unknown values are discarded rather than raised on.
    """
    if not isinstance(raw, dict):
        return {}
    return {
        str(model).lower(): int(value)
        for model, value in raw.items()
        if isinstance(value, int)
        and not isinstance(value, bool)
        and value in _VALID_RATINGS
    }


def _normalize(raw: dict) -> dict:
    """Coerce raw file contents into the current schema."""
    overrides = raw.get("strategy_overrides")
    ratings = _normalize_ratings(raw.get("model_ratings"))
    if isinstance(raw.get("keys"), dict):
        return {
            "keys": raw["keys"],
            "default_model": raw.get("default_model"),
            "strategy_overrides": overrides if isinstance(overrides, dict) else {},
            "model_ratings": ratings,
        }
    # Legacy flat format: every top-level string value is a key.
    keys = {k: v for k, v in raw.items() if isinstance(v, str)}
    return {
        "keys": keys,
        "default_model": None,
        "strategy_overrides": {},
        "model_ratings": ratings,
    }


def _write(data: dict) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Strip None fields so the file stays tidy.
    out = {"keys": data.get("keys", {})}
    if data.get("default_model"):
        out["default_model"] = data["default_model"]
    if data.get("strategy_overrides"):
        out["strategy_overrides"] = data["strategy_overrides"]
    if data.get("model_ratings"):
        out["model_ratings"] = data["model_ratings"]
    _AUTH_FILE.write_text(json.dumps(out, indent=2) + "\n")
    _AUTH_FILE.chmod(0o600)


def _load() -> dict:
    return _normalize(_load_raw())


def load_into_environ() -> None:
    """Inject stored keys into os.environ (override=False — env vars win)."""
    for name, value in _load()["keys"].items():
        env_var = KEY_ALIASES.get(name, name.upper() + "_API_KEY")
        if env_var not in os.environ:
            os.environ[env_var] = value


def set_key(name: str, value: str) -> None:
    data = _load()
    data["keys"][name.lower()] = value
    _write(data)


def get_key(name: str) -> str | None:
    return _load()["keys"].get(name.lower())


def list_keys() -> dict[str, str]:
    """Return all stored keys with values masked."""
    return {name: _mask(value) for name, value in _load()["keys"].items()}


def get_default_model() -> str | None:
    return _load().get("default_model")


def set_default_model(model: str | None) -> None:
    data = _load()
    data["default_model"] = model
    _write(data)


def get_strategy_override(model: str) -> str | None:
    """Strategy this user pinned for `model`, if any (see `basemode bench --save`)."""
    return _load()["strategy_overrides"].get(model.lower())


def set_strategy_override(model: str, strategy: str | None) -> None:
    """Pin (or, with `strategy=None`, unpin) the strategy used for `model`."""
    data = _load()
    if strategy is None:
        data["strategy_overrides"].pop(model.lower(), None)
    else:
        data["strategy_overrides"][model.lower()] = strategy
    _write(data)


def list_strategy_overrides() -> dict[str, str]:
    return dict(_load()["strategy_overrides"])


def get_model_rating(model: str) -> int | None:
    """This user's thumb for `model`: `RATING_UP`, `RATING_DOWN`, or None."""
    return _load()["model_ratings"].get(model.lower())


def set_model_rating(model: str, rating: int | None) -> None:
    """Rate `model` up or down; `rating=None` clears it.

    Ratings are keyed like strategy pins — by the model ID as given, lowered —
    so callers should normalize first if they want `kimi-k3` and
    `moonshot/kimi-k3` to be the same thumb.
    """
    # `True == 1`, so a bool would pass the membership test and then be
    # written as JSON `true` — which the reader drops as malformed.
    if rating is not None and (
        isinstance(rating, bool) or rating not in _VALID_RATINGS
    ):
        raise ValueError(f"rating must be {RATING_UP}, {RATING_DOWN}, or None")
    data = _load()
    if rating is None:
        data["model_ratings"].pop(model.lower(), None)
    else:
        data["model_ratings"][model.lower()] = rating
    _write(data)


def list_model_ratings() -> dict[str, int]:
    return dict(_load()["model_ratings"])


def _mask(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return value[:4] + "..." + value[-4:]

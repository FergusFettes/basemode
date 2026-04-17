"""Persistent API key + default-model storage in ~/.config/basemode/auth.json.

File schema (current):
    {"keys": {"openai": "sk-...", ...}, "default_model": "..."}

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
}


def _load_raw() -> dict:
    if not _AUTH_FILE.exists():
        return {}
    return json.loads(_AUTH_FILE.read_text())


def _normalize(raw: dict) -> dict:
    """Coerce raw file contents into the current schema."""
    if isinstance(raw.get("keys"), dict):
        return {
            "keys": raw["keys"],
            "default_model": raw.get("default_model"),
        }
    # Legacy flat format: every top-level string value is a key.
    keys = {k: v for k, v in raw.items() if isinstance(v, str)}
    return {"keys": keys, "default_model": None}


def _write(data: dict) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Strip None fields so the file stays tidy.
    out = {"keys": data.get("keys", {})}
    if data.get("default_model"):
        out["default_model"] = data["default_model"]
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


def _mask(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return value[:4] + "..." + value[-4:]

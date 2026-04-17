"""Persistent API key storage in ~/.config/basemode/auth.json."""
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
}


def _load() -> dict[str, str]:
    if not _AUTH_FILE.exists():
        return {}
    return json.loads(_AUTH_FILE.read_text())


def _save(data: dict[str, str]) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _AUTH_FILE.write_text(json.dumps(data, indent=2) + "\n")
    _AUTH_FILE.chmod(0o600)


def load_into_environ() -> None:
    """Inject stored keys into os.environ (override=False — env vars win)."""
    for name, value in _load().items():
        env_var = KEY_ALIASES.get(name, name.upper() + "_API_KEY")
        if env_var not in os.environ:
            os.environ[env_var] = value


def set_key(name: str, value: str) -> None:
    name = name.lower()
    data = _load()
    data[name] = value
    _save(data)


def get_key(name: str) -> str | None:
    return _load().get(name.lower())


def list_keys() -> dict[str, str]:
    """Return all stored keys with values masked."""
    return {
        name: _mask(value)
        for name, value in _load().items()
    }


def _mask(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return value[:4] + "..." + value[-4:]

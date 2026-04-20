from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from .keys import load_into_environ


def _find_env() -> Path | None:
    for p in [Path(".env"), Path(__file__).parent.parent.parent / ".env"]:
        if p.exists():
            return p
    return None


# Load order (later sources win):
#   1. ~/.config/basemode/auth.json  — persistent key store
#   2. .env file                     — project/dev override
#   3. existing os.environ           — runtime override (never overwritten)
load_into_environ()
_env_path = _find_env()
if _env_path:
    load_dotenv(_env_path, override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_env_path, extra="allow")

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""
    together_api_key: str = ""
    moonshot_api_key: str = ""
    xai_api_key: str = ""
    zai_api_key: str = ""

    @property
    def available_providers(self) -> list[str]:
        return [
            provider
            for provider, key in [
                ("openai", self.openai_api_key),
                ("anthropic", self.anthropic_api_key),
                ("openrouter", self.openrouter_api_key),
                ("groq", self.groq_api_key),
                ("gemini", self.gemini_api_key),
                ("together_ai", self.together_api_key),
                ("moonshot", self.moonshot_api_key),
                ("xai", self.xai_api_key),
                ("zai", self.zai_api_key),
            ]
            if key
        ]


settings = Settings()

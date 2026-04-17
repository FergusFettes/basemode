from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env() -> Path | None:
    for p in [Path(".env"), Path(__file__).parent.parent.parent / ".env"]:
        if p.exists():
            return p
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_find_env(), extra="allow")

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""
    together_api_key: str = ""

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
            ]
            if key
        ]


settings = Settings()

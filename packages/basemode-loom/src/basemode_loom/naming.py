"""Best-effort readable names for persisted generation trees."""

from __future__ import annotations

import os
import re

import litellm
from basemode.keys import get_key
from basemode.usage import _count_tokens

TITLE_TOKEN_THRESHOLD = 500
_MAX_TITLE_CONTEXT_CHARS = 6000
_OPENAI_TITLE_MODEL = "gpt-4o-mini"
_ANTHROPIC_TITLE_MODEL = "anthropic/claude-3-haiku-20240307"


def choose_title_model() -> str | None:
    """Pick a cheap title model when the user has a supported key configured."""
    if _has_key("openai", "OPENAI_API_KEY"):
        return _OPENAI_TITLE_MODEL
    if _has_key("anthropic", "ANTHROPIC_API_KEY"):
        return _ANTHROPIC_TITLE_MODEL
    return None


def should_name(text: str, *, threshold: int = TITLE_TOKEN_THRESHOLD) -> bool:
    return _count_tokens(_OPENAI_TITLE_MODEL, text) >= threshold


def generate_name(text: str, *, model: str | None = None) -> str | None:
    """Generate a lowercase hyphenated name such as ``this-is-the-topic``."""
    model = model or choose_title_model()
    if model is None:
        return None

    excerpt = _title_context(text)
    try:
        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Name the text with a short readable slug. "
                        "Return only lowercase words joined by hyphens. "
                        "Use 3 to 7 words. Do not add quotes or punctuation."
                    ),
                },
                {"role": "user", "content": excerpt},
            ],
            max_tokens=24,
            temperature=0.2,
        )
    except Exception:
        return None

    content = response.choices[0].message.content if response.choices else ""
    return slugify(str(content))


def slugify(value: str) -> str | None:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    parts = [part for part in slug.split("-") if part]
    if not parts:
        return None
    return "-".join(parts[:7])


def _has_key(provider: str, env_var: str) -> bool:
    return bool(os.environ.get(env_var) or get_key(provider))


def _title_context(text: str) -> str:
    text = text.strip()
    if len(text) <= _MAX_TITLE_CONTEXT_CHARS:
        return text
    half = _MAX_TITLE_CONTEXT_CHARS // 2
    return text[:half] + "\n\n[...]\n\n" + text[-half:]

"""System prompt coercion — works on any chat model."""
from collections.abc import AsyncGenerator

import litellm

from ..params import GenerationParams
from .base import ContinuationStrategy
from .utils import needs_leading_space, normalize_prefix

SYSTEM_PROMPT = """\
You are a text continuation engine. Your only function is to extend the provided \
text naturally, as if you wrote it yourself. Rules:
- Output ONLY the continuation — no acknowledgment, preamble, or commentary
- Continue in the exact same voice, style, and register
- Begin immediately with the next character that naturally follows
- Never start with "Sure", "Of course", "Certainly", or any other acknowledgment"""

# Gemini 2.5+ models are "thinking" models that consume tokens on internal reasoning
# before producing output. Without a thinking budget, max_tokens is exhausted by
# thoughts and the visible output is empty or truncated.
_THINKING_MODELS = {"gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"}
_THINKING_BUDGET = 1024
_THINKING_MIN_OUTPUT = 512


def _is_thinking_model(model: str) -> bool:
    m = model.lower().split("/")[-1]
    return any(t in m for t in _THINKING_MODELS)


def _thinking_kwargs(params: GenerationParams) -> dict:
    if not _is_thinking_model(params.model):
        return {}
    return {
        "thinking": {"type": "enabled", "budget_tokens": _THINKING_BUDGET},
        "max_tokens": max(params.max_tokens, _THINKING_BUDGET + _THINKING_MIN_OUTPUT),
    }


class SystemPromptStrategy(ContinuationStrategy):
    """Generic coercion via system prompt. Fallback for any chat model."""

    name = "system"

    async def stream(self, prefix: str, params: GenerationParams) -> AsyncGenerator[str, None]:
        kwargs = {
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            **_thinking_kwargs(params),  # may override max_tokens for thinking models
            **params.extra,
        }
        response = await litellm.acompletion(
            model=params.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": normalize_prefix(prefix)},
            ],
            stream=True,
            **kwargs,
        )
        first = True
        async for chunk in response:
            token = chunk.choices[0].delta.content or ""
            if not token:
                continue
            if first and needs_leading_space(prefix, token):
                token = " " + token
            first = False
            yield token

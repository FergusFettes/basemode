"""System prompt coercion — works on any chat model."""
from collections.abc import AsyncGenerator

import litellm

from ..params import GenerationParams
from .base import ContinuationStrategy

SYSTEM_PROMPT = """\
You are a text continuation engine. Your only function is to extend the provided \
text naturally, as if you wrote it yourself. Rules:
- Output ONLY the continuation — no acknowledgment, preamble, or commentary
- Continue in the exact same voice, style, and register
- Begin immediately with the next character that naturally follows
- Never start with "Sure", "Of course", "Certainly", or any other acknowledgment"""


class SystemPromptStrategy(ContinuationStrategy):
    """Generic coercion via system prompt. Fallback for any chat model."""

    name = "system"

    async def stream(self, prefix: str, params: GenerationParams) -> AsyncGenerator[str, None]:
        response = await litellm.acompletion(
            model=params.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prefix},
            ],
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            stream=True,
            **params.extra,
        )
        async for chunk in response:
            token = chunk.choices[0].delta.content or ""
            if token:
                yield token

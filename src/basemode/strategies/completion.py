"""OpenAI-compatible /completions endpoint — works natively with base models."""
from collections.abc import AsyncGenerator

import litellm

from ..params import GenerationParams
from .base import ContinuationStrategy


class CompletionStrategy(ContinuationStrategy):
    """Uses the text completions API. Best for true base models (davinci, etc.)."""

    name = "completion"

    async def stream(self, prefix: str, params: GenerationParams) -> AsyncGenerator[str, None]:
        response = await litellm.atext_completion(
            model=params.model,
            prompt=prefix,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            stream=True,
            **params.extra,
        )
        async for chunk in response:
            token = chunk.choices[0].text or ""
            if token:
                yield token

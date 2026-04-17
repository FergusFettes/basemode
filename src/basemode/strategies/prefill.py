"""Anthropic prefill trick — force continuation by seeding the assistant turn."""
from collections.abc import AsyncGenerator

import litellm

from ..params import GenerationParams
from .base import ContinuationStrategy

# How many trailing chars of the prefix to use as the assistant seed.
# Long enough that the model clearly understands it's mid-sentence.
SEED_LEN = 50


class PrefillStrategy(ContinuationStrategy):
    """
    Works by splitting the prefix: the first part goes in the user turn,
    the last SEED_LEN characters become the start of the assistant turn.
    The model is forced to continue from exactly where the prefix ends.
    """

    name = "prefill"

    async def stream(self, prefix: str, params: GenerationParams) -> AsyncGenerator[str, None]:
        if len(prefix) <= SEED_LEN:
            user_content = "Continue:"
            assistant_seed = prefix
        else:
            user_content = prefix[:-SEED_LEN]
            assistant_seed = prefix[-SEED_LEN:]

        response = await litellm.acompletion(
            model=params.model,
            messages=[
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_seed},
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

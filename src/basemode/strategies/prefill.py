"""Anthropic prefill trick — force continuation by seeding the assistant turn."""
import logging
from collections.abc import AsyncGenerator

import litellm

log = logging.getLogger(__name__)

from ..params import GenerationParams
from .base import ContinuationStrategy
from .compat import build_kwargs

# Chars of the prefix to use as the assistant seed.
# This forces Claude to continue from exactly that point.
SEED_LEN = 20


class PrefillStrategy(ContinuationStrategy):
    """
    Places the full prefix in the system prompt for context, then seeds the
    assistant turn with the last SEED_LEN characters. Claude is forced to
    continue from exactly where the prefix ends, with complete context.
    """

    name = "prefill"

    async def stream(self, prefix: str, params: GenerationParams) -> AsyncGenerator[str, None]:
        seed = prefix[-SEED_LEN:] if len(prefix) > SEED_LEN else prefix

        system = (
            "You are continuing the following text. "
            "Output only the continuation — no preamble, no commentary.\n\n"
            f"Text to continue:\n{prefix}"
        )
        if params.context:
            system += f"\n\n<CONTEXT>\n{params.context}\n</CONTEXT>"

        log.debug("PrefillStrategy.stream: model=%s seed_len=%d", params.model, len(seed))
        response = await litellm.acompletion(
            model=params.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "[continue]"},
                {"role": "assistant", "content": seed},
            ],
            stream=True,
            **build_kwargs(params),
        )
        async for chunk in response:
            token = chunk.choices[0].delta.content or ""
            if token:
                yield token

"""System prompt coercion — works on any chat model."""

import logging
from collections.abc import AsyncGenerator

import litellm

from ..healing import needs_leading_space, normalize_prefix
from ..params import GenerationParams
from .base import ContinuationStrategy
from .compat import build_kwargs

log = logging.getLogger(__name__)

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

    async def stream(
        self, prefix: str, params: GenerationParams
    ) -> AsyncGenerator[str, None]:
        system = SYSTEM_PROMPT
        if params.context:
            system += f"\n\n<CONTEXT>\n{params.context}\n</CONTEXT>"
        log.debug("SystemPromptStrategy.stream: model=%s", params.model)
        response = await litellm.acompletion(
            model=params.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": normalize_prefix(prefix)},
            ],
            stream=True,
            **build_kwargs(params),
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

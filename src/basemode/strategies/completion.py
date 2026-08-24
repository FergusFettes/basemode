"""OpenAI-compatible /completions endpoint — works natively with base models."""

from collections.abc import AsyncGenerator

from .. import usage_capture
from ..exceptions import EmptyCompletionError
from ..params import GenerationParams
from ..transport import get_transport
from .base import ContinuationStrategy
from .compat import build_kwargs


class CompletionStrategy(ContinuationStrategy):
    """Uses the text completions API. Best for true base models (davinci, etc.)."""

    name = "completion"

    async def stream(
        self, prefix: str, params: GenerationParams
    ) -> AsyncGenerator[str, None]:
        response = await get_transport().text_completion(
            model=params.model,
            prompt=prefix,
            stream=True,
            **build_kwargs(params),
        )
        yielded = False
        finish_reason = None
        async for chunk in response:
            usage_capture.record(getattr(chunk, "usage", None))
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            token = choice.text or ""
            if token:
                yielded = True
                yield token
        if not yielded:
            raise EmptyCompletionError(
                model=params.model, strategy=self.name, finish_reason=finish_reason
            )

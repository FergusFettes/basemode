"""Fill-in-the-middle — for models that support FIM tokens natively."""

from collections.abc import AsyncGenerator

import litellm

from ..params import GenerationParams
from .base import ContinuationStrategy

# Token formats by model family
_FIM_FORMATS = {
    "deepseek": ("<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>"),
    "starcoder": ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),
    "codellama": ("▁<PRE>", "▁<SUF>", "▁<MID>"),
}


def _fim_prompt(prefix: str, model: str) -> str:
    for key, (pre, suf, mid) in _FIM_FORMATS.items():
        if key in model.lower():
            return f"{pre}{prefix}{suf}{mid}"
    # Generic fallback
    pre, suf, mid = _FIM_FORMATS["starcoder"]
    return f"{pre}{prefix}{suf}{mid}"


class FIMStrategy(ContinuationStrategy):
    """Fill-in-the-middle via text completion. DeepSeek, StarCoder, CodeLlama."""

    name = "fim"

    async def stream(
        self, prefix: str, params: GenerationParams
    ) -> AsyncGenerator[str, None]:
        prompt = _fim_prompt(prefix, params.model)
        response = await litellm.atext_completion(
            model=params.model,
            prompt=prompt,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            stream=True,
            **params.extra,
        )
        async for chunk in response:
            token = chunk.choices[0].text or ""
            if token:
                yield token

"""Few-shot coercion — for stubborn models that ignore system prompts."""

import logging
from collections.abc import AsyncGenerator

import litellm

from ..healing import needs_leading_space, normalize_prefix
from ..params import GenerationParams
from .base import ContinuationStrategy
from .compat import build_kwargs

log = logging.getLogger(__name__)

# Varied examples: fiction, technical, poetry, dialogue
_EXAMPLES = [
    (
        "The experiment had been running for three days when Dr. Chen noticed the anomaly.",
        " At first she thought it was a measurement error—the kind that comes from",
    ),
    (
        "To install the package, first ensure you have Python 3.11 or higher.",
        " Then run the following command in your terminal:\n\n```\npip install",
    ),
    (
        "the rain comes down like static",
        "\nbetween stations, the city\nblurs into signal",
    ),
    (
        "Look, I'm just saying—if you'd been there, you'd understand why I had to",
        " make that call. There was no good option. Either way, someone was going to",
    ),
]


def _build_system_prompt() -> str:
    examples = "\n\n".join(
        f'Input: """{inp}"""\nOutput: """{out}"""' for inp, out in _EXAMPLES
    )
    return (
        "You continue text. Given input text, output only the natural continuation "
        "with no preamble or acknowledgment. Examples:\n\n" + examples
    )


_SYSTEM_PROMPT = _build_system_prompt()


class FewShotStrategy(ContinuationStrategy):
    """Few-shot examples in system prompt. For models that ignore plain instructions."""

    name = "few_shot"

    async def stream(
        self, prefix: str, params: GenerationParams
    ) -> AsyncGenerator[str, None]:
        system = _SYSTEM_PROMPT
        if params.context:
            system += f"\n\n<CONTEXT>\n{params.context}\n</CONTEXT>"
        log.debug("FewShotStrategy.stream: model=%s", params.model)
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

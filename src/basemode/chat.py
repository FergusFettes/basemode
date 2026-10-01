"""Plain chat: ask a model a question and stream its answer.

Everything else in basemode coerces a model into *continuing* text. This is
the other direction -- the model answers as an assistant -- but it rides the
same provider plumbing: model normalization, key loading, the compat quirks in
`strategies.compat.build_kwargs`, usage capture and the observation ledger.

A model with no chat endpoint -- litellm's model map lists it as
`mode: completion` (davinci-002, gpt-3.5-turbo-instruct) -- gets a `User:` /
`Assistant:` transcript sent to the text-completion endpoint instead, cut off where the model starts writing the next user
turn, so `chat_text` works on every model basemode can reach.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from dataclasses import replace

import litellm

from . import usage_capture
from .continue_ import _observe_attempt, _report_usage
from .detect import normalize_model
from .exceptions import EmptyCompletionError
from .keys import load_into_environ
from .observations import ObservationContext, observe_operation
from .params import GenerationParams
from .strategies.compat import build_kwargs
from .strategies.completion import CompletionStrategy
from .transport import get_transport

log = logging.getLogger(__name__)

#: Strategy name recorded in the observation ledger for a direct chat call, so
#: health can tell answers apart from continuations on the same endpoint.
CHAT_STRATEGY = "chat"

#: Strategy recorded for a base model answered through a transcript.
TRANSCRIPT_STRATEGY = "chat_transcript"

#: Where a base model's transcript answer ends: the model starting the next
#: user turn.
TRANSCRIPT_STOP = "\nUser:"

_ROLE_LABELS = {"user": "User", "assistant": "Assistant"}


def build_messages(prompt: str | list[dict], system: str | None = None) -> list[dict]:
    """Normalize a prompt string or message list into chat messages.

    A `system` argument is prepended unless the messages already open with a
    system message, which then wins.
    """
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [dict(m) for m in prompt]
    if not messages:
        raise ValueError("chat_text needs at least one message")
    if system and messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": system})
    return messages


def transcript_prompt(messages: list[dict]) -> str:
    """Render chat messages as a plain-text transcript a base model can extend."""
    parts: list[str] = []
    turns = messages
    if turns and turns[0].get("role") == "system":
        parts.append(f"{turns[0]['content'].strip()}\n")
        turns = turns[1:]
    for message in turns:
        role = message.get("role", "user")
        label = _ROLE_LABELS.get(role, role.capitalize())
        parts.append(f"{label}: {message['content'].strip()}")
    parts.append("Assistant:")
    return "\n".join(parts)


def has_chat_endpoint(model: str) -> bool:
    """Whether `model` takes chat messages, per litellm's model map.

    Only a model the map positively lists as `mode: completion` is treated as
    chat-less. Unknown models are assumed to chat: almost everything served
    today does, and `continue_`'s `completion` heuristic is no guide here --
    it also covers gpt-oss, a chat model that merely continues well.
    """
    stem = model.split("/", 1)[-1]
    info = litellm.model_cost.get(model) or litellm.model_cost.get(stem) or {}
    return info.get("mode") != "completion"


async def chat_text(
    prompt: str | list[dict],
    model: str = "gpt-4o-mini",
    *,
    system: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    transcript: bool | None = None,
    observation: ObservationContext | None = None,
    on_usage: Callable[[list[dict]], None] | None = None,
    **extra,
) -> AsyncGenerator[str, None]:
    """Stream a model's answer to `prompt`.

    `prompt` is either a single user message or a full list of
    `{"role", "content"}` messages (a running conversation). `system` is
    prepended as a system message. `extra` is forwarded to the provider
    request, after basemode's compat quirks.

    Models with no chat endpoint are answered through a transcript sent to
    the text-completion endpoint; everything else goes straight to chat.
    `transcript=True` forces the transcript (a base model a provider serves
    without a chat template), `False` forces chat. `on_usage` behaves as in
    `continue_text`.
    """
    litellm.suppress_debug_info = True
    load_into_environ()
    model = normalize_model(model)
    messages = build_messages(prompt, system)

    params = GenerationParams(
        model=model, max_tokens=max_tokens, temperature=temperature, extra=extra
    )
    if transcript is None:
        transcript = not has_chat_endpoint(model)
    if transcript:
        strategy = TRANSCRIPT_STRATEGY
        raw = _transcript_answer(messages, params)
    else:
        strategy = CHAT_STRATEGY
        raw = _stream_chat(messages, params)
    operation = observe_operation(model, strategy, CHAT_STRATEGY, observation)
    log.debug(
        "chat_text: model=%s strategy=%s messages=%d", model, strategy, len(messages)
    )
    usage_capture.begin_capture()
    emitted = 0
    try:
        async for token in _observe_attempt(raw, operation, "initial"):
            emitted += 1
            yield token
    except (GeneratorExit, asyncio.CancelledError):
        operation.finish(
            "cancelled" if emitted else "inconclusive", returned_content=bool(emitted)
        )
        _report_usage(on_usage)
        raise
    except Exception:
        operation.finish("failure", returned_content=bool(emitted))
        _report_usage(on_usage)
        raise
    operation.finish(
        "success" if emitted else "failure", returned_content=bool(emitted)
    )
    _report_usage(on_usage)


async def _stream_chat(
    messages: list[dict], params: GenerationParams
) -> AsyncGenerator[str, None]:
    response = await get_transport().chat_completion(
        model=params.model,
        messages=messages,
        stream=True,
        **build_kwargs(params),
    )
    yielded = False
    finish_reason = None
    async for chunk in response:
        usage_capture.record_chunk(chunk)
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        finish_reason = getattr(choice, "finish_reason", None) or finish_reason
        token = choice.delta.content or ""
        if token:
            yielded = True
            yield token
    if not yielded:
        raise EmptyCompletionError(
            model=params.model, strategy=CHAT_STRATEGY, finish_reason=finish_reason
        )


async def _transcript_answer(
    messages: list[dict], params: GenerationParams
) -> AsyncGenerator[str, None]:
    """Answer as a base model would: by extending a chat transcript.

    This calls the completion strategy directly rather than `continue_text`,
    because continuation healing collapses the very newline the stop marker
    starts with, and an answer has no prefix seam to heal anyway.
    """
    params = replace(params, extra={"stop": [TRANSCRIPT_STOP], **params.extra})
    raw = CompletionStrategy().stream(transcript_prompt(messages), params)
    first = True
    async for token in _cut_at(raw, TRANSCRIPT_STOP):
        if first:
            token = token.lstrip()
            if not token:
                continue
            first = False
        yield token


async def _cut_at(
    tokens: AsyncGenerator[str, None], stop: str
) -> AsyncGenerator[str, None]:
    """Pass tokens through until `stop` appears, even split across chunks.

    Providers are asked to stop there too, but not every one honours `stop`
    on a stream, so this is the guarantee rather than the optimization.
    """
    try:
        async for token in _cut_before(tokens, stop):
            yield token
    finally:
        # Stopping early leaves the provider stream suspended; close it now
        # rather than whenever it is garbage collected.
        await tokens.aclose()


async def _cut_before(
    tokens: AsyncGenerator[str, None], stop: str
) -> AsyncGenerator[str, None]:
    pending = ""
    async for token in tokens:
        pending += token
        cut = pending.find(stop)
        if cut != -1:
            if pending[:cut]:
                yield pending[:cut]
            return
        # Hold back only what could still be the start of `stop`.
        keep = next(
            (
                n
                for n in range(min(len(stop) - 1, len(pending)), 0, -1)
                if stop.startswith(pending[-n:])
            ),
            0,
        )
        ready, pending = pending[: len(pending) - keep], pending[len(pending) - keep :]
        if ready:
            yield ready
    if pending:
        yield pending

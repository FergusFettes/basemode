import asyncio
import os
import stat
import sys
from typing import Annotated

import typer

from ..keys import get_default_model
from . import app
from .render import console


@app.command()
def ask(
    ctx: typer.Context,
    prompt: Annotated[
        str | None,
        typer.Argument(help="Question to ask; piped stdin is prepended to it"),
    ] = None,
    model: Annotated[str | None, typer.Option("-m", "--model")] = None,
    system: Annotated[
        str | None, typer.Option("-s", "--system", help="System prompt")
    ] = None,
    max_tokens: Annotated[int, typer.Option("-M", "--max-tokens")] = 4096,
    temperature: Annotated[float, typer.Option("-t", "--temperature")] = 0.7,
    transcript: Annotated[
        bool | None,
        typer.Option(
            "--transcript/--no-transcript",
            help="Force (or skip) the User:/Assistant: transcript used for base "
            "models. Default: only for models with no chat endpoint.",
        ),
    ] = None,
    show_usage: Annotated[
        bool,
        typer.Option("--show-usage", help="Show token usage after the answer"),
    ] = False,
    show_cost: Annotated[
        bool, typer.Option("--show-cost", help="Show estimated cost after the answer")
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "-v", "--verbose", help="Show content-free call and health events."
        ),
    ] = False,
) -> None:
    """Ask a model a question and stream a normal chat answer."""
    piped = _read_piped_stdin()
    text = "\n\n".join(part for part in (piped.strip(), prompt) if part)
    if not text:
        console.print(ctx.get_help())
        return
    if verbose:
        from ..logging_setup import setup_verbose_logging

        setup_verbose_logging()
    if model is None:
        model = get_default_model() or "gpt-4o-mini"

    messages = [{"role": "user", "content": text}]
    try:
        answer, usage_events = asyncio.run(
            _stream_answer(messages, model, system, max_tokens, temperature, transcript)
        )
    except KeyboardInterrupt:
        raise typer.Exit(130) from None
    except Exception as exc:
        # One line on stderr, not a provider traceback: callers pipe `ask`.
        detail = str(exc).strip().splitlines()[-1] if str(exc).strip() else ""
        typer.echo(f"error: {type(exc).__name__}: {detail}", err=True)
        raise typer.Exit(1) from None
    if show_usage or show_cost:
        _print_chat_usage(model, messages, system, answer, show_cost, usage_events)


def _read_piped_stdin() -> str:
    """Read stdin only when something was actually piped or redirected in.

    `isatty()` alone is not enough: agents, cron and editors often hand a
    command a non-tty character device that never reaches EOF, and reading
    it would hang `ask` forever.
    """
    try:
        mode = os.fstat(sys.stdin.fileno()).st_mode
    except (AttributeError, OSError, ValueError):
        # No real file descriptor (an in-memory stream, as under test).
        return "" if sys.stdin is None or sys.stdin.isatty() else sys.stdin.read()
    if stat.S_ISFIFO(mode) or stat.S_ISREG(mode):
        return sys.stdin.read()
    return ""


async def _stream_answer(
    messages: list[dict],
    model: str,
    system: str | None,
    max_tokens: int,
    temperature: float,
    transcript: bool | None,
) -> tuple[str, list[dict]]:
    from ..chat import chat_text
    from ..observations import ObservationContext

    chunks: list[str] = []
    usage_events: list[dict] = []
    # Plain stdout rather than the rich console: answers are meant to be
    # piped, and model output must never be read as rich markup.
    async for token in chat_text(
        messages,
        model,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        transcript=transcript,
        observation=ObservationContext(source="cli"),
        on_usage=usage_events.extend,
    ):
        chunks.append(token)
        sys.stdout.write(token)
        sys.stdout.flush()
    answer = "".join(chunks)
    if not answer.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()
    return answer, usage_events


def _print_chat_usage(
    model: str,
    messages: list[dict],
    system: str | None,
    answer: str,
    show_cost: bool,
    usage_events: list[dict],
) -> None:
    from ..chat import build_messages
    from ..detect import normalize_model
    from ..usage import estimate_usage, usage_from_events
    from .run import _usage_table

    resolved = normalize_model(model)
    usage = usage_from_events(resolved, usage_events) if usage_events else None
    if usage is None:
        usage = estimate_usage(
            resolved, "", answer, prompt_messages=build_messages(messages, system)
        )
    # Usage goes to stderr so `basemode ask ... > answer.txt` stays clean.
    from rich.console import Console

    Console(stderr=True).print(_usage_table(usage, show_cost))

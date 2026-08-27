import asyncio
import sys
from typing import Annotated

import typer
from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ..keys import get_default_model
from . import app
from .render import _BRANCH_COLORS, console


@app.command()
def run(
    ctx: typer.Context,
    prefix: Annotated[
        str | None, typer.Argument(help="Text to continue (or pipe via stdin)")
    ] = None,
    model: Annotated[str | None, typer.Option("-m", "--model")] = None,
    n: Annotated[
        int, typer.Option("-n", "--branches", help="Number of parallel continuations")
    ] = 1,
    max_tokens: Annotated[int, typer.Option("-M", "--max-tokens")] = 200,
    temperature: Annotated[float, typer.Option("-t", "--temperature")] = 0.9,
    strategy: Annotated[str | None, typer.Option("-s", "--strategy")] = None,
    rewind: Annotated[
        bool,
        typer.Option(
            "--rewind",
            help="Rewind short trailing word fragments before generation.",
        ),
    ] = False,
    strict_max_tokens: Annotated[
        bool,
        typer.Option(
            "--strict-max-tokens",
            help="Hard-stop streamed output at max_tokens using client-side token counting.",
        ),
    ] = False,
    show_strategy: Annotated[bool, typer.Option("--show-strategy")] = False,
    show_usage: Annotated[
        bool,
        typer.Option(
            "--show-usage", help="Show estimated token usage after generation"
        ),
    ] = False,
    show_cost: Annotated[
        bool, typer.Option("--show-cost", help="Show estimated cost after generation")
    ] = False,
) -> None:
    """Continue text with an LLM (default command)."""
    if prefix is None and not sys.stdin.isatty():
        prefix = sys.stdin.read()
    if prefix is None:
        console.print(ctx.get_help())
        return
    _run_text(
        prefix,
        model,
        n,
        max_tokens,
        temperature,
        strategy,
        rewind,
        strict_max_tokens,
        show_strategy,
        show_usage,
        show_cost,
    )


async def _stream_one(
    prefix: str,
    model: str,
    max_tokens: int,
    temperature: float,
    strategy: str | None,
    rewind: bool = False,
    strict_max_tokens: bool = False,
) -> tuple[str, list[dict]]:
    from ..continue_ import continue_text

    console.print(f"[dim]{prefix}[/dim]", end="")
    chunks: list[str] = []
    usage_events: list[dict] = []
    async for token in continue_text(
        prefix,
        model,
        max_tokens=max_tokens,
        temperature=temperature,
        strategy=strategy,
        rewind=rewind,
        strict_max_tokens=strict_max_tokens,
        on_usage=usage_events.extend,
    ):
        chunks.append(token)
        console.print(token, end="")
    console.print()
    return "".join(chunks), usage_events


async def _stream_branches(
    prefix: str,
    model: str,
    n: int,
    max_tokens: int,
    temperature: float,
    strategy: str | None,
    rewind: bool = False,
    strict_max_tokens: bool = False,
) -> tuple[list[str], list[dict]]:
    from ..continue_ import branch_text

    buffers: list[list[str]] = [[] for _ in range(n)]
    usage_events: list[dict] = []

    with Live(
        _branches_panel(prefix, buffers),
        console=console,
        refresh_per_second=12,
    ) as live:
        async for idx, token in branch_text(
            prefix,
            model,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            strategy=strategy,
            rewind=rewind,
            strict_max_tokens=strict_max_tokens,
            on_usage=lambda _idx, events: usage_events.extend(events),
        ):
            buffers[idx].append(token)
            live.update(_branches_panel(prefix, buffers))

    return ["".join(buf) for buf in buffers], usage_events


def _branches_panel(prefix: str, buffers: list[list[str]]) -> Panel:
    columns = []
    for i, buf in enumerate(buffers):
        color = _BRANCH_COLORS[i % len(_BRANCH_COLORS)]
        text = Text(f"Branch {i + 1}\n", style=f"bold {color}")
        text.append("".join(buf), style=color)
        columns.append(text)
    prompt = Text("Prompt\n", style="bold")
    prompt.append(prefix, style="dim")
    return Panel(
        Group(
            prompt,
            Rule(style="dim"),
            Columns(columns, equal=True, expand=True),
        ),
        title="Branches",
        border_style="dim",
    )


def _run_text(
    prefix: str,
    model: str | None,
    n: int,
    max_tokens: int,
    temperature: float,
    strategy: str | None,
    rewind: bool,
    strict_max_tokens: bool,
    show_strategy: bool,
    show_usage: bool,
    show_cost: bool,
) -> None:
    if model is None:
        model = get_default_model() or "gpt-4o-mini"

    prefix = prefix.rstrip("\n")

    if show_strategy:
        from ..detect import detect_strategy, normalize_model

        strat = detect_strategy(normalize_model(model), strategy)
        console.print(f"[dim]strategy: {strat.name}[/dim]")

    if n == 1:
        completion, usage_events = asyncio.run(
            _stream_one(
                prefix,
                model,
                max_tokens,
                temperature,
                strategy,
                rewind,
                strict_max_tokens,
            )
        )
        if show_usage or show_cost:
            _print_usage_estimate(
                model,
                prefix,
                completion,
                strategy,
                show_cost,
                prompt_requests=1,
                usage_events=usage_events,
            )
    else:
        completions, usage_events = asyncio.run(
            _stream_branches(
                prefix,
                model,
                n,
                max_tokens,
                temperature,
                strategy,
                rewind,
                strict_max_tokens,
            )
        )
        if show_usage or show_cost:
            _print_usage_estimate(
                model,
                prefix,
                "".join(completions),
                strategy,
                show_cost,
                prompt_requests=n,
                usage_events=usage_events,
            )


def _print_usage_estimate(
    model: str,
    prefix: str,
    completion: str,
    strategy: str | None,
    show_cost: bool,
    prompt_requests: int,
    usage_events: list[dict] | None = None,
) -> None:
    from ..detect import normalize_model
    from ..usage import estimate_usage, format_usd, usage_from_events

    resolved = normalize_model(model)
    usage = usage_from_events(resolved, usage_events) if usage_events else None
    if usage is None:
        prompt, messages = _usage_prompt(resolved, prefix, strategy)
        usage = estimate_usage(
            resolved,
            prompt,
            completion,
            prompt_messages=messages,
            prompt_requests=prompt_requests,
        )
    table = Table("Metric", "Value", show_header=False)
    table.add_row("Model", usage.model)
    table.add_row("Source", "provider" if not usage.is_estimate else "estimate")
    table.add_row("Prompt tokens", str(usage.prompt_tokens))
    table.add_row("Completion tokens", str(usage.completion_tokens))
    table.add_row("Total tokens", str(usage.total_tokens))
    if show_cost:
        table.add_row("Estimated cost", format_usd(usage.cost_usd))
        if not usage.pricing_available:
            table.add_row("Cost note", "pricing unavailable in LiteLLM model map")
    console.print(table)


def _usage_prompt(
    model: str, prefix: str, strategy: str | None
) -> tuple[str, list[dict] | None]:
    from ..detect import detect_strategy
    from ..healing import normalize_prefix
    from ..strategies.few_shot import _SYSTEM_PROMPT as FEW_SHOT_SYSTEM_PROMPT
    from ..strategies.fim import _fim_prompt
    from ..strategies.prefill import SEED_LEN
    from ..strategies.system import SYSTEM_PROMPT

    strat = detect_strategy(model, strategy)
    if strat.name == "system":
        return "", [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": normalize_prefix(prefix)},
        ]
    if strat.name == "few_shot":
        return "", [
            {"role": "system", "content": FEW_SHOT_SYSTEM_PROMPT},
            {"role": "user", "content": normalize_prefix(prefix)},
        ]
    if strat.name == "prefill":
        seed = prefix[-SEED_LEN:] if len(prefix) > SEED_LEN else prefix
        return "", [
            {
                "role": "system",
                "content": (
                    "You are continuing the following text. "
                    "Output only the continuation — no preamble, no commentary.\n\n"
                    f"Text to continue:\n{prefix}"
                ),
            },
            {"role": "user", "content": "[continue]"},
            {"role": "assistant", "content": seed},
        ]
    if strat.name == "fim":
        return _fim_prompt(prefix, model), None
    return prefix, None

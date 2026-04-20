import asyncio
import logging
import sys
from typing import Annotated

import click
import typer
import typer.core
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .keys import (
    KEY_ALIASES,
    get_default_model,
    get_key,
    list_keys,
    set_default_model,
    set_key,
)

log = logging.getLogger(__name__)
console = Console()
_BRANCH_COLORS = ["green", "blue", "yellow", "magenta", "cyan"]


_GROUP_FLAGS = {"--help", "-h", "--install-completion", "--show-completion"}


def _default_to(command: str) -> type:
    class _Group(typer.core.TyperGroup):
        def parse_args(self, ctx: click.Context, args: list) -> list:
            if not args or (args[0].startswith("-") and args[0] not in _GROUP_FLAGS):
                args = [command, *args]
            return super().parse_args(ctx, args)

        def resolve_command(self, ctx: click.Context, args: list) -> tuple:
            try:
                return super().resolve_command(ctx, args)
            except click.UsageError:
                args.insert(0, command)
                return super().resolve_command(ctx, args)

    return _Group


app = typer.Typer(
    help="Make any LLM do raw text continuation.",
    cls=_default_to("run"),
)


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
) -> str:
    from .continue_ import continue_text

    console.print(f"[dim]{prefix}[/dim]", end="")
    chunks: list[str] = []
    async for token in continue_text(
        prefix,
        model,
        max_tokens=max_tokens,
        temperature=temperature,
        strategy=strategy,
        rewind=rewind,
    ):
        chunks.append(token)
        console.print(token, end="")
    console.print()
    return "".join(chunks)


async def _stream_branches(
    prefix: str,
    model: str,
    n: int,
    max_tokens: int,
    temperature: float,
    strategy: str | None,
    rewind: bool = False,
) -> list[str]:
    from .continue_ import branch_text

    buffers: list[list[str]] = [[] for _ in range(n)]

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
        ):
            buffers[idx].append(token)
            live.update(_branches_panel(prefix, buffers))

    return ["".join(buf) for buf in buffers]


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
    show_strategy: bool,
    show_usage: bool,
    show_cost: bool,
) -> None:
    if model is None:
        model = get_default_model() or "gpt-4o-mini"

    prefix = prefix.rstrip("\n")

    if show_strategy:
        from .detect import detect_strategy, normalize_model

        strat = detect_strategy(normalize_model(model), strategy)
        console.print(f"[dim]strategy: {strat.name}[/dim]")

    if n == 1:
        completion = asyncio.run(
            _stream_one(prefix, model, max_tokens, temperature, strategy, rewind)
        )
        if show_usage or show_cost:
            _print_usage_estimate(
                model, prefix, completion, strategy, show_cost, prompt_requests=1
            )
    else:
        completions = asyncio.run(
            _stream_branches(
                prefix, model, n, max_tokens, temperature, strategy, rewind
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
            )


def _print_usage_estimate(
    model: str,
    prefix: str,
    completion: str,
    strategy: str | None,
    show_cost: bool,
    prompt_requests: int,
) -> None:
    from .detect import normalize_model
    from .usage import estimate_usage, format_usd

    resolved = normalize_model(model)
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
    from .detect import detect_strategy
    from .healing import normalize_prefix
    from .strategies.few_shot import _SYSTEM_PROMPT as FEW_SHOT_SYSTEM_PROMPT
    from .strategies.fim import _fim_prompt
    from .strategies.prefill import SEED_LEN
    from .strategies.system import SYSTEM_PROMPT

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


@app.command()
def models(
    provider: Annotated[str | None, typer.Option("-p", "--provider")] = None,
    search: Annotated[str | None, typer.Option("-s", "--search")] = None,
    available: Annotated[
        bool, typer.Option("-a", "--available", help="Only show models with keys set")
    ] = False,
) -> None:
    """List available models."""
    from .models import list_models

    results = list_models(provider=provider, search=search, available_only=available)
    if not results:
        console.print("[yellow]No models found.[/yellow]")
        return

    table = Table("Model", show_header=True, header_style="bold")
    for m in results:
        table.add_row(m)
    console.print(table)


@app.command()
def providers() -> None:
    """List all known providers."""
    from .models import list_providers

    for p in list_providers():
        console.print(p)


@app.command()
def strategies() -> None:
    """List available continuation strategies."""
    table = Table("Name", "Description", show_header=True, header_style="bold")
    descriptions = {
        "completion": "OpenAI /completions endpoint — for true base models",
        "prefill": "Anthropic assistant prefill trick",
        "system": "System prompt coercion — generic fallback for any chat model",
        "few_shot": "Few-shot examples in system prompt — for stubborn models",
        "fim": "Fill-in-the-middle tokens — DeepSeek, StarCoder, CodeLlama",
    }
    for name in descriptions:
        table.add_row(name, descriptions.get(name, ""))
    console.print(table)


def _preview(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _format_float(value: float) -> str:
    return f"{value:.2f}"


@app.command()
def info(model: Annotated[str, typer.Argument(help="Model name to inspect")]) -> None:
    """Show strategy, provider, limits, and known pricing for a model."""
    from .detect import detect_strategy, normalize_model
    from .usage import format_per_million, get_price_info

    resolved = normalize_model(model)
    strat = detect_strategy(resolved)
    price = get_price_info(resolved)

    table = Table("Field", "Value", show_header=True, header_style="bold")
    table.add_row("Model", model)
    table.add_row("Resolved", resolved)
    table.add_row("Strategy", strat.name)
    table.add_row("Provider", price.provider or "unknown")
    table.add_row("Input price", format_per_million(price.input_cost_per_token))
    table.add_row("Output price", format_per_million(price.output_cost_per_token))
    table.add_row(
        "Cache read price", format_per_million(price.cache_read_input_token_cost)
    )
    table.add_row(
        "Reasoning output price",
        format_per_million(price.output_cost_per_reasoning_token),
    )
    table.add_row(
        "Max input tokens",
        str(price.max_input_tokens) if price.max_input_tokens else "unknown",
    )
    table.add_row(
        "Max output tokens",
        str(price.max_output_tokens) if price.max_output_tokens else "unknown",
    )
    table.add_row(
        "Supports reasoning",
        str(price.supports_reasoning)
        if price.supports_reasoning is not None
        else "unknown",
    )
    if not price.pricing_available:
        table.add_row("Cost note", "pricing unavailable in LiteLLM model map")
    console.print(table)


@app.command()
def keys(
    action: Annotated[str, typer.Argument(help="Action: set | list | get")],
    provider: Annotated[
        str | None, typer.Argument(help="Provider name (e.g. openai, anthropic)")
    ] = None,
    value: Annotated[
        str | None, typer.Argument(help="API key value (set only; prompted if omitted)")
    ] = None,
) -> None:
    """Manage API keys stored in ~/.config/basemode/auth.json.

    Examples:

      basemode keys set openai

      basemode keys list

      basemode keys get anthropic
    """
    if action == "set":
        if not provider:
            console.print(
                "[red]Provider required. E.g.: basemode keys set openai[/red]"
            )
            raise typer.Exit(1)
        if value is None:
            value = typer.prompt(f"{provider} API key", hide_input=True)
        set_key(provider, value)
        console.print(
            f"[green]✓[/green] Saved [bold]{provider}[/bold] key to ~/.config/basemode/auth.json"
        )

    elif action == "list":
        stored = list_keys()
        if not stored:
            console.print(
                "[yellow]No keys stored. Use: basemode keys set <provider>[/yellow]"
            )
            return
        table = Table(
            "Provider", "Key", "Env var", show_header=True, header_style="bold"
        )
        for name, masked in stored.items():
            env_var = KEY_ALIASES.get(name, name.upper() + "_API_KEY")
            table.add_row(name, masked, env_var)
        console.print(table)

    elif action == "get":
        if not provider:
            console.print(
                "[red]Provider required. E.g.: basemode keys get openai[/red]"
            )
            raise typer.Exit(1)
        val = get_key(provider)
        if val is None:
            console.print(f"[yellow]No key stored for {provider!r}.[/yellow]")
            raise typer.Exit(1)
        console.print(val)

    else:
        console.print(f"[red]Unknown action {action!r}. Use: set | list | get[/red]")
        raise typer.Exit(1)


@app.command()
def default(
    model: Annotated[
        str | None,
        typer.Argument(help="Model to set as default (omit to show current)"),
    ] = None,
    unset: Annotated[
        bool, typer.Option("--unset", help="Clear the stored default")
    ] = False,
) -> None:
    """Show or set the default model (stored in ~/.config/basemode/auth.json).

    Provider prefixes are inferred — `claude-sonnet-4-6` resolves to
    `anthropic/claude-sonnet-4-6`, `gemini-2.5-flash` to `gemini/...`, etc.
    """
    if unset:
        set_default_model(None)
        console.print("[green]✓[/green] Default model cleared.")
        return

    if model is None:
        current = get_default_model()
        if current is None:
            console.print(
                "[yellow]No default model set. E.g.: basemode default claude-sonnet-4-6[/yellow]"
            )
            return
        from .detect import normalize_model

        resolved = normalize_model(current)
        suffix = f" → [dim]{resolved}[/dim]" if resolved != current else ""
        console.print(f"[bold]{current}[/bold]{suffix}")
        return

    set_default_model(model)
    from .detect import normalize_model

    resolved = normalize_model(model)
    suffix = f" → [dim]{resolved}[/dim]" if resolved != model else ""
    console.print(f"[green]✓[/green] Default model set to [bold]{model}[/bold]{suffix}")

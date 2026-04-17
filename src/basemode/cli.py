import asyncio
import sys
from typing import Annotated

import click
import typer
import typer.core
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .continue_ import branch_text, continue_text
from .detect import detect_strategy, normalize_model
from .keys import (
    KEY_ALIASES,
    get_default_model,
    get_key,
    list_keys,
    set_default_model,
    set_key,
)
from .models import list_models, list_providers
from .strategies import REGISTRY

console = Console()
_BRANCH_COLORS = ["green", "blue", "yellow", "magenta", "cyan"]


_GROUP_FLAGS = {"--help", "-h", "--install-completion", "--show-completion"}


class _DefaultToRun(typer.core.TyperGroup):
    """Route unrecognised first args (and empty/flag-led invocations) to 'run'."""

    def parse_args(self, ctx: click.Context, args: list) -> list:
        if not args or (args[0].startswith("-") and args[0] not in _GROUP_FLAGS):
            args = ["run"] + args
        return super().parse_args(ctx, args)

    def resolve_command(self, ctx: click.Context, args: list) -> tuple:
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            args.insert(0, "run")
            return super().resolve_command(ctx, args)


app = typer.Typer(
    help="Make any LLM do raw text continuation.",
    cls=_DefaultToRun,
)


@app.command()
def run(
    ctx: typer.Context,
    prefix: Annotated[str | None, typer.Argument(help="Text to continue (or pipe via stdin)")] = None,
    model: Annotated[str | None, typer.Option("-m", "--model")] = None,
    n: Annotated[int, typer.Option("-n", "--branches", help="Number of parallel continuations")] = 1,
    max_tokens: Annotated[int, typer.Option("-M", "--max-tokens")] = 200,
    temperature: Annotated[float, typer.Option("-t", "--temperature")] = 0.9,
    strategy: Annotated[str | None, typer.Option("-s", "--strategy")] = None,
    show_strategy: Annotated[bool, typer.Option("--show-strategy")] = False,
) -> None:
    """Continue text with an LLM (default command)."""
    if prefix is None and not sys.stdin.isatty():
        prefix = sys.stdin.read()
    if prefix is None:
        console.print(ctx.get_help())
        return

    if model is None:
        model = get_default_model() or "gpt-4o-mini"

    prefix = prefix.rstrip("\n")

    if show_strategy:
        strat = detect_strategy(normalize_model(model), strategy)
        console.print(f"[dim]strategy: {strat.name}[/dim]")

    if n == 1:
        asyncio.run(_stream_one(prefix, model, max_tokens, temperature, strategy))
    else:
        asyncio.run(_stream_branches(prefix, model, n, max_tokens, temperature, strategy))  # noqa: E501


async def _stream_one(prefix: str, model: str, max_tokens: int, temperature: float, strategy: str | None) -> None:
    console.print(f"[dim]{prefix}[/dim]", end="")
    async for token in continue_text(prefix, model, max_tokens=max_tokens, temperature=temperature, strategy=strategy):
        console.print(token, end="")
    console.print()


async def _stream_branches(
    prefix: str, model: str, n: int, max_tokens: int, temperature: float, strategy: str | None
) -> None:
    buffers: list[list[str]] = [[] for _ in range(n)]
    console.print(f"[dim]{prefix}[/dim]\n")

    async for idx, token in branch_text(
        prefix, model, n=n, max_tokens=max_tokens, temperature=temperature, strategy=strategy
    ):
        buffers[idx].append(token)

    panels = []
    for i, buf in enumerate(buffers):
        color = _BRANCH_COLORS[i % len(_BRANCH_COLORS)]
        text = Text(prefix, style="dim")
        text.append("".join(buf), style=color)
        panels.append(Panel(text, title=f"[{color}]Branch {i + 1}[/{color}]"))

    console.print(Columns(panels, equal=True))


@app.command()
def models(
    provider: Annotated[str | None, typer.Option("-p", "--provider")] = None,
    search: Annotated[str | None, typer.Option("-s", "--search")] = None,
    available: Annotated[bool, typer.Option("-a", "--available", help="Only show models with keys set")] = False,
) -> None:
    """List available models."""
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
    for name in REGISTRY:
        table.add_row(name, descriptions.get(name, ""))
    console.print(table)


@app.command()
def info(model: Annotated[str, typer.Argument(help="Model name to inspect")]) -> None:
    """Show which strategy would be used for a given model."""
    strat = detect_strategy(model)
    console.print(f"[bold]{model}[/bold] → [green]{strat.name}[/green]")


@app.command()
def keys(
    action: Annotated[str, typer.Argument(help="Action: set | list | get")],
    provider: Annotated[str | None, typer.Argument(help="Provider name (e.g. openai, anthropic)")] = None,
    value: Annotated[str | None, typer.Argument(help="API key value (set only; prompted if omitted)")] = None,
) -> None:
    """Manage API keys stored in ~/.config/basemode/auth.json.

    Examples:

      basemode keys set openai

      basemode keys list

      basemode keys get anthropic
    """
    if action == "set":
        if not provider:
            console.print("[red]Provider required. E.g.: basemode keys set openai[/red]")
            raise typer.Exit(1)
        if value is None:
            value = typer.prompt(f"{provider} API key", hide_input=True)
        set_key(provider, value)
        console.print(f"[green]✓[/green] Saved [bold]{provider}[/bold] key to ~/.config/basemode/auth.json")

    elif action == "list":
        stored = list_keys()
        if not stored:
            console.print("[yellow]No keys stored. Use: basemode keys set <provider>[/yellow]")
            return
        table = Table("Provider", "Key", "Env var", show_header=True, header_style="bold")
        for name, masked in stored.items():
            env_var = KEY_ALIASES.get(name, name.upper() + "_API_KEY")
            table.add_row(name, masked, env_var)
        console.print(table)

    elif action == "get":
        if not provider:
            console.print("[red]Provider required. E.g.: basemode keys get openai[/red]")
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
    model: Annotated[str | None, typer.Argument(help="Model to set as default (omit to show current)")] = None,
    unset: Annotated[bool, typer.Option("--unset", help="Clear the stored default")] = False,
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
            console.print("[yellow]No default model set. E.g.: basemode default claude-sonnet-4-6[/yellow]")
            return
        resolved = normalize_model(current)
        suffix = f" → [dim]{resolved}[/dim]" if resolved != current else ""
        console.print(f"[bold]{current}[/bold]{suffix}")
        return

    set_default_model(model)
    resolved = normalize_model(model)
    suffix = f" → [dim]{resolved}[/dim]" if resolved != model else ""
    console.print(f"[green]✓[/green] Default model set to [bold]{model}[/bold]{suffix}")

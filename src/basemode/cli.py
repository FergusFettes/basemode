import asyncio
import sys
from typing import Annotated

import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .continue_ import branch_text, continue_text
from .detect import detect_strategy, normalize_model
from .models import list_models, list_providers
from .strategies import REGISTRY

app = typer.Typer(help="Make any LLM do raw text continuation.")
console = Console()

_BRANCH_COLORS = ["green", "blue", "yellow", "magenta", "cyan"]


def _read_prefix(prefix: str | None) -> str | None:
    """Return prefix from arg, stdin pipe, or None."""
    if prefix is not None:
        return prefix
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return None


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prefix: Annotated[str | None, typer.Argument(help="Text to continue (or pipe via stdin)")] = None,
    model: Annotated[str, typer.Option("-m", "--model")] = "gpt-4o-mini",
    n: Annotated[int, typer.Option("-n", "--branches", help="Number of parallel continuations")] = 1,
    max_tokens: Annotated[int, typer.Option("--max-tokens")] = 200,
    temperature: Annotated[float, typer.Option("-t", "--temperature")] = 0.9,
    strategy: Annotated[str | None, typer.Option("-s", "--strategy")] = None,
    show_strategy: Annotated[bool, typer.Option("--show-strategy")] = False,
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    text = _read_prefix(prefix)
    if text is None:
        console.print(ctx.get_help())
        return

    prefix = text.rstrip("\n")

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

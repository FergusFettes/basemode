import asyncio
import json
from typing import Annotated

import typer
from rich.panel import Panel
from rich.table import Table

from ..keys import set_strategy_override
from . import app
from .render import _preview, _score_color, console


@app.command()
def bench(
    model: Annotated[str, typer.Argument(help="Model to benchmark")],
    strategy_list: Annotated[
        str | None,
        typer.Option(
            "-s",
            "--strategies",
            help="Comma-separated strategies to compare (default: system,prefill,few_shot)",
        ),
    ] = None,
    max_tokens: Annotated[int, typer.Option("-M", "--max-tokens")] = 60,
    temperature: Annotated[float, typer.Option("-t", "--temperature")] = 1.0,
    save: Annotated[
        bool,
        typer.Option(
            "--save",
            help="Pin the winning strategy for this model in ~/.config/basemode/auth.json",
        ),
    ] = False,
    show_samples: Annotated[
        bool, typer.Option("--samples", help="Print a sample continuation per strategy")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the ranking as JSON")
    ] = False,
) -> None:
    """Run each strategy against this model and rank them by continuation quality.

    Sends a handful of short continuations per strategy (real API calls, so
    this costs a fraction of a cent) and scores each one for assistant
    behavior: preamble, refusal, echoed prefix, chat turns, stray formatting.

    Examples:

      basemode bench claude-opus-5

      basemode bench kimi-k3 --samples --save
    """
    from ..bench import DEFAULT_STRATEGIES, bench_model, winner
    from ..detect import normalize_model, select_strategy
    from ..strategies import REGISTRY

    resolved = normalize_model(model)
    names = (
        [s.strip() for s in strategy_list.split(",") if s.strip()]
        if strategy_list
        else list(DEFAULT_STRATEGIES)
    )
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        console.print(
            f"[red]Unknown strategy {unknown[0]!r}. Valid: {', '.join(REGISTRY)}[/red]"
        )
        raise typer.Exit(1)

    current = select_strategy(resolved)
    if not as_json:
        console.print(
            f"[dim]Benchmarking {resolved} — currently {current.name} "
            f"(from {current.source})[/dim]"
        )

    results = asyncio.run(
        bench_model(
            resolved,
            strategies=names,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    )

    if as_json:
        console.print(
            json.dumps(
                {
                    "model": resolved,
                    "current_strategy": current.name,
                    "current_source": current.source,
                    "results": [r.as_dict() for r in results],
                },
                indent=2,
            )
        )
        return

    table = Table(
        "Strategy",
        "Score",
        "Flags / error",
        "Mean s",
        show_header=True,
        header_style="bold",
    )
    for result in results:
        # An error explains a row better than the "empty" flag it produces.
        detail = (
            f"[red]{_preview(result.errors[0], limit=64)}[/red]"
            if result.errors
            else ", ".join(result.flags)
        )
        table.add_row(
            result.strategy,
            f"[{_score_color(result.score)}]{result.score:.2f}[/]",
            detail,
            f"{result.mean_elapsed_s:.2f}",
        )
    console.print(table)

    if show_samples:
        for result in results:
            console.print(
                Panel(
                    _preview(
                        result.sample or (result.errors[0] if result.errors else ""),
                        limit=300,
                    ),
                    title=f"{result.strategy} ({result.score:.2f})",
                    border_style="dim",
                )
            )

    best = winner(results)
    if best is None:
        console.print(
            "[red]No strategy produced a usable continuation.[/red] "
            "Check the key for this provider, or try --samples to see the errors."
        )
        raise typer.Exit(1)

    if save:
        set_strategy_override(resolved, best.strategy)
        console.print(
            f"[green]✓[/green] Pinned [bold]{best.strategy}[/bold] for {resolved}"
        )
        return

    # A tie is not a reason to switch: ranking breaks ties on latency, so
    # without this every model where all strategies come back clean would be
    # told to re-pin itself to whichever one happened to answer fastest.
    current_score = next((r.score for r in results if r.strategy == current.name), None)
    if current_score is not None and current_score >= best.score:
        console.print(f"[dim]{current.name} is already the best of these.[/dim]")
    else:
        console.print(
            f"[yellow]{best.strategy}[/yellow] ({best.score:.2f}) beats the current "
            f"[dim]{current.name}[/dim]"
            + (f" ({current_score:.2f})" if current_score is not None else "")
            + f" — pin it with: [bold]basemode bench {model} --save[/bold]"
        )

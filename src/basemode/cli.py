import asyncio
import json
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

from .health import (
    EVENT_RETENTION_DAYS,
    clear_model_health,
    list_model_health,
    model_health,
    verification_history,
)
from .keys import (
    KEY_ALIASES,
    RATING_DOWN,
    RATING_UP,
    get_default_model,
    get_key,
    get_model_rating,
    list_keys,
    list_model_ratings,
    list_strategy_overrides,
    set_default_model,
    set_key,
    set_model_rating,
    set_strategy_override,
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
    from .continue_ import continue_text

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
    from .continue_ import branch_text

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
        from .detect import detect_strategy, normalize_model

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
    from .detect import normalize_model
    from .usage import estimate_usage, format_usd, usage_from_events

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


def _print_live_models(provider: str | None, search: str | None) -> None:
    from .live_models import (
        PROVIDER_ENDPOINTS,
        LiveModelsError,
        dates_look_trustworthy,
        fetch_live_models,
    )
    from .models import list_models
    from .settings import settings

    if not provider:
        console.print(
            "[red]--live requires --provider (one of: "
            f"{', '.join(sorted(PROVIDER_ENDPOINTS))})[/red]"
        )
        raise typer.Exit(1)

    api_key = settings.api_key_for(provider)
    if not api_key:
        console.print(f"[red]no API key configured for provider {provider!r}[/red]")
        raise typer.Exit(1)

    try:
        live = fetch_live_models(provider, api_key)
    except LiveModelsError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if search:
        needle = search.lower()
        live = [m for m in live if needle in m.id.lower()]

    known = set(list_models(provider=provider))
    known_display = {_display_id(provider, m) for m in known}
    reliable_dates = dates_look_trustworthy(live)

    table = Table(
        "Model", "Release Date", "In litellm?", show_header=True, header_style="bold"
    )
    for m in live:
        in_litellm = m.id in known_display or m.id in known
        release_date = m.release_date if reliable_dates else None
        table.add_row(
            m.id,
            release_date
            or ("unknown" if m.release_date_confidence != "unknown" else ""),
            "" if in_litellm else "[bold yellow]NEW[/bold yellow]",
        )
    console.print(table)
    new_count = sum(1 for m in live if m.id not in known_display and m.id not in known)
    console.print(
        f"[dim]{len(live)} models from {provider}'s live API"
        f"{f', {new_count} not in litellm yet' if new_count else ''}[/dim]"
    )
    if not reliable_dates:
        console.print(
            f"[yellow]{provider}'s release dates look bogus (most models share "
            "the same timestamp — likely a list-refresh time, not a per-model "
            "release date) — hidden above.[/yellow]"
        )


def _display_id(provider: str, model: str) -> str:
    prefix = f"{provider}/"
    return model[len(prefix) :] if model.startswith(prefix) else model


@app.command()
def models(
    provider: Annotated[str | None, typer.Option("-p", "--provider")] = None,
    search: Annotated[str | None, typer.Option("-s", "--search")] = None,
    available: Annotated[
        bool, typer.Option("-a", "--available", help="Only show models with keys set")
    ] = False,
    verified: Annotated[
        bool,
        typer.Option(
            "--verified",
            help="Only show models in the verified registry table.",
        ),
    ] = False,
    full: Annotated[
        bool,
        typer.Option(
            "--full",
            help="Show every dated snapshot instead of collapsing them into "
            "their undated alias.",
        ),
    ] = False,
    all_modes: Annotated[
        bool,
        typer.Option(
            "--all-modes",
            help="Include non-text models too (image, audio, embedding, ...).",
        ),
    ] = False,
    since: Annotated[
        str | None,
        typer.Option(
            "--since",
            help="Only show models released within this long, e.g. 10d, 4w, 6m, 1y.",
        ),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit structured JSON for frontend model pickers.",
        ),
    ] = False,
    live: Annotated[
        bool,
        typer.Option(
            "--live",
            help="Bypass litellm and query the provider's own /v1/models "
            "endpoint directly (requires --provider and a configured key). "
            "Flags models litellm doesn't know about yet as NEW.",
        ),
    ] = False,
) -> None:
    """List available models, grouped by provider.

    Compact by default: dated snapshots (`gpt-5.4-2026-03-05`) collapse into
    their undated alias (`gpt-5.4`); non-text models (image/audio/embedding/
    ...) are hidden. Use --full and --all-modes to see everything.
    """
    from .models import list_model_picker_entries, parse_since

    if since:
        try:
            parse_since(since)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc

    if live:
        _print_live_models(provider, search)
        return

    entries = list_model_picker_entries(
        provider=provider,
        search=search,
        available_only=available,
        verified_only=verified,
        text_only=not all_modes,
        compact=not full,
        since=since,
    )

    if as_json:
        console.print(json.dumps(entries, indent=2))
        return

    if not entries:
        console.print("[yellow]No models found.[/yellow]")
        return

    columns = ["Provider", "Model", "Verified", "Rating", "Release Date"]
    if not full:
        columns.append("Snapshots")
    table = Table(*columns, show_header=True, header_style="bold")
    inferred_count = 0
    for e in entries:
        release_date = e.get("release_date") or ""
        if release_date and e.get("release_date_inferred"):
            release_date = f"~{release_date}"
            inferred_count += 1
        verified_mark = "[green]✓[/green]" if e.get("verified") else ""
        row = [
            e["provider"],
            e["display"],
            verified_mark,
            _RATING_MARKS.get(e.get("rating"), ""),
            release_date,
        ]
        if not full:
            snapshots = e.get("snapshots") or []
            row.append(str(len(snapshots)) if snapshots else "")
        table.add_row(*row)
    console.print(table)
    summary = f"[dim]{len(entries)} models"
    if inferred_count:
        summary += f" ([yellow]~{inferred_count}[/yellow] dates guessed from another provider's listing of the same model)"
    console.print(summary + "[/dim]")


_RATING_MARKS = {RATING_UP: "[green]+[/green]", RATING_DOWN: "[red]-[/red]"}
_RATING_WORDS_BY_VALUE = {RATING_UP: "thumbs up", RATING_DOWN: "thumbs down"}
_RATING_WORDS = {
    "up": RATING_UP,
    "+": RATING_UP,
    "down": RATING_DOWN,
    "-": RATING_DOWN,
    "clear": None,
    "none": None,
}


@app.command()
def rate(
    model: Annotated[
        str | None,
        typer.Argument(help="Model to rate. Omit to list every rated model."),
    ] = None,
    rating: Annotated[
        str | None,
        typer.Argument(help="up, down, or clear."),
    ] = None,
) -> None:
    """Rate a model up or down. Rated models sort first (or last) everywhere.

    Ratings are yours alone — stored in ~/.config/basemode/auth.json — and
    outrank the shipped reliability ordering in `basemode models` and in any
    frontend built on the picker list.
    """
    if model is None:
        rated = list_model_ratings()
        if not rated:
            console.print("[yellow]No models rated yet.[/yellow]")
            return
        table = Table("Model", "Rating", show_header=True, header_style="bold")
        for model_id, value in sorted(rated.items(), key=lambda kv: (-kv[1], kv[0])):
            table.add_row(model_id, _RATING_MARKS.get(value, ""))
        console.print(table)
        console.print("[dim]Clear one with: basemode rate MODEL clear[/dim]")
        return

    if rating is None:
        console.print("[red]Give a rating: up, down, or clear.[/red]")
        raise typer.Exit(1)
    word = rating.strip().lower()
    if word not in _RATING_WORDS:
        console.print(f"[red]Unknown rating {rating!r}. Use up, down, or clear.[/red]")
        raise typer.Exit(1)

    from .detect import normalize_model

    resolved = normalize_model(model)
    value = _RATING_WORDS[word]
    set_model_rating(resolved, value)
    if value is None:
        console.print(f"[green]✓[/green] Cleared rating for [bold]{resolved}[/bold]")
    else:
        console.print(
            f"[green]✓[/green] Rated [bold]{resolved}[/bold] "
            f"{'up' if value == RATING_UP else 'down'}"
        )


@app.command()
def providers() -> None:
    """List all known providers."""
    from .models import list_providers

    for p in list_providers():
        console.print(p)


@app.command()
def strategies(
    unpin: Annotated[
        str | None,
        typer.Option("--unpin", help="Remove the pinned strategy for a model"),
    ] = None,
) -> None:
    """List continuation strategies, and any strategies pinned per model."""
    if unpin is not None:
        from .detect import normalize_model

        resolved = normalize_model(unpin)
        if resolved.lower() not in {k.lower() for k in list_strategy_overrides()}:
            console.print(f"[yellow]No pinned strategy for {resolved}.[/yellow]")
            raise typer.Exit(1)
        set_strategy_override(resolved, None)
        console.print(f"[green]✓[/green] Unpinned strategy for [bold]{resolved}[/bold]")
        return

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

    pinned = list_strategy_overrides()
    if pinned:
        console.print()
        pin_table = Table(
            "Model", "Pinned strategy", show_header=True, header_style="bold"
        )
        for model_id, strategy in sorted(pinned.items()):
            pin_table.add_row(model_id, strategy)
        console.print(pin_table)
        console.print("[dim]Clear one with: basemode strategies --unpin MODEL[/dim]")


_SCORE_COLORS = ((0.75, "green"), (0.4, "yellow"))


def _score_color(score: float) -> str:
    for threshold, color in _SCORE_COLORS:
        if score >= threshold:
            return color
    return "red"


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
    from .bench import DEFAULT_STRATEGIES, bench_model, winner
    from .detect import normalize_model, select_strategy
    from .strategies import REGISTRY

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


_STRATEGY_SOURCES = {
    "user": "pinned locally (basemode bench --save)",
    "registry": "verified models registry",
    "heuristic": "model-name heuristic",
    "explicit": "passed explicitly",
}


def _preview(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _format_float(value: float) -> str:
    return f"{value:.2f}"


@app.command()
def info(model: Annotated[str, typer.Argument(help="Model name to inspect")]) -> None:
    """Show strategy, provider, limits, and known pricing for a model."""
    from .detect import normalize_model, select_strategy
    from .strategies.compat import model_quirks
    from .usage import format_per_million, get_price_info

    resolved = normalize_model(model)
    strat = select_strategy(resolved)
    price = get_price_info(resolved)
    quirks = model_quirks(resolved)

    table = Table("Field", "Value", show_header=True, header_style="bold")
    table.add_row("Model", model)
    table.add_row("Resolved", resolved)
    table.add_row("Strategy", strat.name)
    table.add_row("Strategy source", _STRATEGY_SOURCES.get(strat.source, strat.source))
    table.add_row("Quirks", ", ".join(sorted(quirks)) if quirks else "none known")
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

    rating = get_model_rating(resolved)
    table.add_row("Your rating", _RATING_WORDS_BY_VALUE.get(rating, "unrated"))
    observed = model_health(resolved)
    if observed is None:
        table.add_row("Observed health", "never generated with")
    else:
        table.add_row("Observed health", _health_summary(observed))
        if observed["categories"]:
            table.add_row(
                "Recent failures",
                ", ".join(
                    f"{name} x{count}" for name, count in observed["categories"].items()
                ),
            )
    console.print(table)


def _health_summary(observed: dict) -> str:
    rate = observed["failure_rate"]
    attempts = observed["attempts"]
    if not rate:
        return f"{attempts} attempts, no failures"
    return (
        f"{attempts} attempts, {observed['failures']} failed ({rate:.0%}); "
        f"last {observed['last_category']} at {observed['last_failure_at']}"
    )


@app.command()
def health(
    model: Annotated[
        str | None,
        typer.Argument(help="Model to inspect. Omit to show every model seen."),
    ] = None,
    days: Annotated[
        int | None,
        typer.Option("--days", help="Window for the failure breakdown, in days."),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the raw records as JSON.")
    ] = False,
    clear: Annotated[
        bool, typer.Option("--clear", help="Forget the history instead of showing it.")
    ] = False,
    verification: Annotated[
        bool,
        typer.Option(
            "--verification",
            help="Show verification-probe results instead of real generations "
            "(see `basemode bench` / a verification sweep) -- recorded "
            "separately so probing a model doesn't skew its usage stats.",
        ),
    ] = False,
) -> None:
    """Show what models actually did here: attempts, failures, and why.

    Recorded from real generations rather than from the shipped registry, so
    it reflects this machine's keys, rate limits, and usage. Turn recording
    off with BASEMODE_NO_HEALTH=1.
    """
    from .detect import normalize_model

    resolved = normalize_model(model) if model else None

    if clear:
        clear_model_health(resolved)
        target = f"[bold]{resolved}[/bold]" if resolved else "every model"
        console.print(f"[green]✓[/green] Cleared health history for {target}")
        return

    if verification:
        from .usage import format_usd

        records = verification_history(model=resolved, days=days)
        if not records:
            console.print("[yellow]No verification probes recorded.[/yellow]")
            return
        if as_json:
            console.print(json.dumps(records, indent=2))
            return
        table = Table(
            "Model",
            "Attempts",
            "Failed",
            "Transient?",
            "Failures seen",
            "Cost",
            "Last probe",
            show_header=True,
            header_style="bold",
        )
        total_cost = 0.0
        any_cost_known = False
        for model_id, observed in sorted(
            records.items(),
            key=lambda kv: (-kv[1]["failures"], kv[0]),
        ):
            transient = ""
            if observed["failures"]:
                transient = (
                    "[yellow]maybe[/yellow]"
                    if observed["looks_transient"]
                    else "[red]no[/red]"
                )
            cost = observed["cost_usd"]
            if cost is not None:
                total_cost += cost
                any_cost_known = True
            table.add_row(
                model_id,
                str(observed["attempts"]),
                str(observed["failures"]),
                transient,
                ", ".join(
                    f"{name} x{count}" for name, count in observed["categories"].items()
                ),
                format_usd(cost) if cost is not None else "",
                observed["last_at"],
            )
        console.print(table)
        window = f" over the last {days} days" if days else ""
        cost_line = (
            f" total cost {format_usd(total_cost)}"
            if any_cost_known
            else " cost unavailable for these probes"
        )
        console.print(f"[dim]{len(records)} models probed{window};{cost_line}[/dim]")
        console.print(
            "[dim]'Transient?' is 'maybe' only when every failure seen was "
            "rate_limit/timeout/provider_unavailable/network[/dim]"
        )
        return

    if resolved:
        observed = model_health(resolved, days=days)
        if observed is None:
            console.print(f"[yellow]No generations recorded for {resolved}.[/yellow]")
            raise typer.Exit(1)
        records = {resolved: observed}
    else:
        records = list_model_health(days=days)
        if not records:
            console.print("[yellow]No generations recorded yet.[/yellow]")
            return

    if as_json:
        console.print(json.dumps(records, indent=2))
        return

    table = Table(
        "Model",
        "Attempts",
        "Failed",
        "Rate",
        "Failures seen",
        "Last failure",
        show_header=True,
        header_style="bold",
    )
    for model_id, observed in sorted(
        records.items(), key=lambda kv: (-(kv[1]["failure_rate"] or 0), kv[0])
    ):
        rate = observed["failure_rate"]
        rate_text = "" if rate is None else f"{rate:.0%}"
        if rate:
            rate_text = f"[red]{rate_text}[/red]" if rate >= 0.2 else rate_text
        table.add_row(
            model_id,
            str(observed["attempts"]),
            str(observed["failures"]),
            rate_text,
            ", ".join(
                f"{name} x{count}" for name, count in observed["categories"].items()
            ),
            observed["last_failure_at"] or "",
        )
    console.print(table)
    window = f" over the last {days} days" if days else ""
    console.print(
        f"[dim]{len(records)} models with recorded generations; "
        f"failure breakdown{window} from the last "
        f"{EVENT_RETENTION_DAYS} days of events[/dim]"
    )


@app.command("verify")
def verify_command(
    models: Annotated[
        list[str] | None,
        typer.Argument(help="Provider-qualified model IDs to verify."),
    ] = None,
    suite: Annotated[
        str,
        typer.Option("--suite", help="quick, thorough, or transient-recheck"),
    ] = "quick",
    attempts: Annotated[
        int, typer.Option("--attempts", min=1, help="Attempts per probe.")
    ] = 1,
    max_tokens: Annotated[int | None, typer.Option("--max-tokens", min=1)] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Probe models and retain every result in the shared evidence database."""
    from dataclasses import asdict

    from .verify import verify_models

    if suite not in {"quick", "thorough", "transient-recheck"}:
        console.print(
            "[red]--suite must be quick, thorough, or transient-recheck[/red]"
        )
        raise typer.Exit(2)
    if not models and suite != "transient-recheck":
        console.print("[red]Supply at least one model to verify.[/red]")
        raise typer.Exit(2)
    summary = asyncio.run(
        verify_models(models, suite=suite, attempts=attempts, max_tokens=max_tokens)
    )
    payload = asdict(summary)
    if as_json:
        console.print(json.dumps(payload, indent=2))
    else:
        console.print(
            f"[green]✓[/green] {summary.successes}/{summary.attempts} probes "
            f"passed; run {summary.run_id}"
        )


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


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Bind address")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Bind port")] = 8080,
) -> None:
    """Run an OpenAI-completions-compatible server (POST /v1/completions).

    Point tools that expect a llama.cpp-style /v1/completions endpoint
    (e.g. Tapestry Loom) at http://<host>:<port>/v1/completions.
    """
    try:
        from .server import serve as run_server
    except ImportError as exc:
        console.print(
            "[red]Missing server dependencies.[/red] Install with: "
            "pip install 'basemode[server]'"
        )
        raise typer.Exit(1) from exc

    console.print(
        f"[green]Serving[/green] http://{host}:{port}/v1/completions (Ctrl+C to stop)"
    )
    run_server(host=host, port=port)

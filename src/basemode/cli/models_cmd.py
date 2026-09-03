import json
from typing import Annotated

import typer
from rich.table import Table

from ..keys import (
    RATING_UP,
    get_model_rating,
    list_model_ratings,
    list_strategy_overrides,
    set_model_rating,
    set_strategy_override,
)
from ..observation_queries import endpoint_health
from . import app
from .render import (
    _RATING_MARKS,
    _RATING_WORDS,
    _RATING_WORDS_BY_VALUE,
    _STRATEGY_SOURCES,
    _health_summary,
    console,
)


def _print_live_models(provider: str | None, search: str | None) -> None:
    from ..live_models import (
        PROVIDER_ENDPOINTS,
        LiveModelsError,
        dates_look_trustworthy,
        fetch_live_models,
    )
    from ..models import list_models
    from ..settings import settings

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
        "Model",
        "Release Date",
        "$/M in",
        "$/M out",
        "In litellm?",
        show_header=True,
        header_style="bold",
    )
    for m in live:
        in_litellm = m.id in known_display or m.id in known
        release_date = m.release_date if reliable_dates else None
        table.add_row(
            m.id,
            release_date
            or ("unknown" if m.release_date_confidence != "unknown" else ""),
            f"{m.input_price_per_m:.2f}" if m.input_price_per_m is not None else "",
            f"{m.output_price_per_m:.2f}" if m.output_price_per_m is not None else "",
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
    from ..models import list_model_picker_entries, parse_since

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

    from ..detect import normalize_model

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
    from ..models import list_providers

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
        from ..detect import normalize_model

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


@app.command()
def info(model: Annotated[str, typer.Argument(help="Model name to inspect")]) -> None:
    """Show strategy, provider, limits, and known pricing for a model."""
    from ..detect import normalize_model, select_strategy
    from ..strategies.compat import model_quirks
    from ..usage import format_per_million, get_price_info

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
    raw_health = endpoint_health(resolved)
    observed = None
    if raw_health is not None:
        operations = int(raw_health["operations"])
        successes = int(raw_health["successful_operations"])
        observed = {
            "attempts": operations,
            "successes": successes,
            "failures": operations - successes,
            "failure_rate": (operations - successes) / operations if operations else 0,
            "categories": raw_health["failures"],
        }
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

import json
from typing import Annotated

import typer
from rich.table import Table

from ..health import (
    EVENT_RETENTION_DAYS,
    verification_history,
)
from ..observation_queries import (
    clear_endpoint_health,
    endpoint_health,
    list_endpoint_health,
)
from . import app
from .render import console


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
    from ..detect import normalize_model

    resolved = normalize_model(model) if model else None

    if clear:
        clear_endpoint_health(resolved)
        target = f"[bold]{resolved}[/bold]" if resolved else "every model"
        console.print(f"[green]✓[/green] Cleared health history for {target}")
        return

    if verification:
        from ..usage import format_usd

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
        observed = endpoint_health(resolved, days=days)
        if observed is None:
            console.print(f"[yellow]No generations recorded for {resolved}.[/yellow]")
            raise typer.Exit(1)
        records = {resolved: observed}
    else:
        records = list_endpoint_health(days=days)
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
        records.items(),
        key=lambda kv: (-(1 - (kv[1]["logical_success_rate"] or 0)), kv[0]),
    ):
        success_rate = observed["logical_success_rate"]
        rate = None if success_rate is None else 1 - success_rate
        rate_text = "" if rate is None else f"{rate:.0%}"
        if rate:
            rate_text = f"[red]{rate_text}[/red]" if rate >= 0.2 else rate_text
        table.add_row(
            model_id,
            str(observed["operations"]),
            str(observed["operations"] - observed["successful_operations"]),
            rate_text,
            ", ".join(
                f"{name} x{count}" for name, count in observed["failures"].items()
            ),
            observed["last_failed_at"] or "",
        )
    console.print(table)
    window = f" over the last {days} days" if days else ""
    console.print(
        f"[dim]{len(records)} models with recorded generations; "
        f"failure breakdown{window} from the last "
        f"{EVENT_RETENTION_DAYS} days of events[/dim]"
    )

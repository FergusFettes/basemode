import asyncio
import json
from typing import Annotated

import typer
from rich.table import Table

from . import app
from .render import console


def _verify_plan_table(plan) -> Table:
    """Render a dry-run verification plan as a Table of targets."""
    from ..usage import format_usd

    table = Table(
        "Stage",
        "Provider",
        "Model",
        "Prior",
        "Catalog",
        "Release",
        "Probes",
        "Max reqs",
        "Max cost",
    )
    for target in plan.targets:
        table.add_row(
            target.stage,
            target.provider,
            target.model,
            target.prior_status,
            "yes"
            if target.catalog_available
            else "no"
            if target.catalog_available is False
            else "?",
            target.release_date or "?",
            str(target.logical_probes),
            str(target.maximum_requests),
            format_usd(target.estimated_max_cost_usd),
        )
    return table


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
    providers: Annotated[
        list[str] | None,
        typer.Option("--provider", help="Limit to a provider; repeatable."),
    ] = None,
    statuses: Annotated[
        list[str] | None,
        typer.Option(
            "--status",
            help="never-tested, reachable, broken, transient, verified, or stale; repeatable.",
        ),
    ] = None,
    from_catalog: Annotated[
        bool,
        typer.Option("--from-catalog", help="Require current catalog availability."),
    ] = False,
    released_since: Annotated[
        str | None, typer.Option("--released-since", help="Minimum ISO release date.")
    ] = None,
    max_release_age_days: Annotated[
        int | None, typer.Option("--max-release-age-days", min=0)
    ] = None,
    stale_after_days: Annotated[int, typer.Option("--stale-after-days", min=1)] = 30,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Plan only; never contact providers.")
    ] = False,
    run_id: Annotated[
        str | None, typer.Option("--resume", help="Resume this run ID.")
    ] = None,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1)] = 4,
    per_provider_concurrency: Annotated[
        int, typer.Option("--per-provider-concurrency", min=1)
    ] = 2,
    max_probes: Annotated[int | None, typer.Option("--max-probes", min=1)] = None,
    max_requests: Annotated[int | None, typer.Option("--max-requests", min=1)] = None,
    max_elapsed_seconds: Annotated[
        float | None, typer.Option("--max-elapsed", min=0.01)
    ] = None,
    max_cost_usd: Annotated[
        float | None, typer.Option("--max-cost-usd", min=0.000001)
    ] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "-v", "--verbose", help="Show content-free probe and health events."
        ),
    ] = False,
) -> None:
    """Probe models and retain every result in the shared evidence database."""
    from dataclasses import asdict

    from ..usage import format_usd
    from ..verification_plan import plan_verification
    from ..verify import verify_models

    if verbose:
        from ..logging_setup import setup_verbose_logging

        setup_verbose_logging()

    if suite not in {"quick", "thorough", "transient-recheck"}:
        console.print(
            "[red]--suite must be quick, thorough, or transient-recheck[/red]"
        )
        raise typer.Exit(2)
    has_selector = bool(
        providers
        or statuses
        or from_catalog
        or released_since
        or max_release_age_days is not None
    )
    if dry_run and run_id:
        console.print("[red]--dry-run cannot be combined with --resume.[/red]")
        raise typer.Exit(2)
    if not models and suite != "transient-recheck" and not has_selector and not run_id:
        console.print("[red]Supply models or at least one target selector.[/red]")
        raise typer.Exit(2)
    plan = None
    if not run_id:
        try:
            plan = plan_verification(
                models,
                suite=suite,
                attempts=attempts,
                max_tokens=max_tokens,
                providers=providers,
                statuses=statuses,
                catalog_available=from_catalog,
                released_since=released_since,
                max_release_age_days=max_release_age_days,
                stale_after_days=stale_after_days,
            )
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc
    if dry_run:
        assert plan is not None
        if as_json:
            console.print(json.dumps(plan.to_dict(), indent=2))
            return
        console.print(_verify_plan_table(plan))
        console.print(
            f"[dim]{len(plan.targets)} targets; providers {plan.provider_counts}; "
            f"{plan.logical_probes} logical probes; at most {plan.maximum_requests} requests; "
            f"known-price ceiling {format_usd(plan.estimated_known_max_cost_usd)} "
            f"({plan.priced_targets} priced, {plan.unpriced_targets} unknown)[/dim]"
        )
        return
    selected_models = [target.model for target in plan.targets] if plan else None
    if not run_id and not selected_models:
        console.print("[yellow]No eligible verification targets.[/yellow]")
        return
    if verbose:
        import logging

        logging.getLogger("basemode.verify").info(
            "verification starting: suite=%s targets=%s max_probes=%s "
            "max_requests=%s max_cost_usd=%s concurrency=%s",
            suite,
            len(selected_models or []),
            max_probes or "unlimited",
            max_requests or "unlimited",
            max_cost_usd if max_cost_usd is not None else "unlimited",
            concurrency,
        )
    summary = asyncio.run(
        verify_models(
            selected_models,
            suite=suite,
            attempts=attempts,
            max_tokens=max_tokens,
            run_id=run_id,
            concurrency=concurrency,
            per_provider_concurrency=per_provider_concurrency,
            max_probes=max_probes,
            max_requests=max_requests,
            max_elapsed_seconds=max_elapsed_seconds,
            max_cost_usd=max_cost_usd,
        )
    )
    payload = asdict(summary)
    if as_json:
        console.print(json.dumps(payload, indent=2))
    else:
        console.print(
            f"[green]✓[/green] {summary.successes}/{summary.attempts} probes "
            f"passed ({summary.requests} requests, {summary.status}); run {summary.run_id}"
        )

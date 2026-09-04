"""Explicit aggregate-only public contribution commands."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

from .. import observations
from ..contributions import (
    build_contribution,
    export_bundle,
    open_contribution_pr,
    release_bundle,
)
from . import app
from .render import console

contribute_app = typer.Typer(
    help="Preview and export content-free evidence aggregates."
)
app.add_typer(contribute_app, name="contribute")


#: Timestamps are taken as strings rather than Typer's `datetime`, whose
#: fixed format list rejects the trailing `Z` that every documented example
#: and every timestamp in the ledger itself carries.
_TIMESTAMP = "A date, or an ISO-8601 timestamp (a trailing Z is accepted)."


def _parse_timestamp(option: str, value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as error:
        console.print(
            f"[red]{option} must be a date or ISO-8601 timestamp, not {value!r}[/red]"
        )
        raise typer.Exit(2) from error
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _window(since: str | None, until: str | None) -> tuple[datetime, datetime]:
    end = _parse_timestamp("--until", until) or datetime.now(UTC)
    start = _parse_timestamp("--since", since) or end - timedelta(days=7)
    return start, end


def _build(since: str | None, until: str | None):
    start, end = _window(since, until)
    try:
        return build_contribution(since=start, until=end)
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error


@contribute_app.command("preview")
def preview(
    since: Annotated[str | None, typer.Option("--since", help=_TIMESTAMP)] = None,
    until: Annotated[str | None, typer.Option("--until", help=_TIMESTAMP)] = None,
) -> None:
    """Print the exact validated JSON shape an export would write."""
    console.print_json(json.dumps(_build(since, until).bundle))


@contribute_app.command("export")
def export(
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    since: Annotated[str | None, typer.Option("--since", help=_TIMESTAMP)] = None,
    until: Annotated[str | None, typer.Option("--until", help=_TIMESTAMP)] = None,
) -> None:
    """Write a validated bundle and mark the observations it counted."""
    contribution = _build(since, until)
    bundle = contribution.bundle
    target = output or Path(f"basemode-contribution-{bundle['bundle_id']}.json")
    export_bundle(contribution, target)
    console.print(str(target))
    console.print(
        f"[dim]{len(contribution.operation_ids)} operations marked submitted; "
        f"basemode contribute release {bundle['bundle_id']} undoes this[/dim]"
    )


@contribute_app.command("pr")
def pr(
    repo: Annotated[str, typer.Option("--repo")] = "FergusFettes/basemode-evidence",
    since: Annotated[str | None, typer.Option("--since", help=_TIMESTAMP)] = None,
    until: Annotated[str | None, typer.Option("--until", help=_TIMESTAMP)] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Preview, confirm, and submit one aggregate bundle using authenticated gh."""
    contribution = _build(since, until)
    bundle = contribution.bundle
    console.print_json(json.dumps(bundle))
    if not yes and not typer.confirm("Submit exactly this aggregate bundle?"):
        raise typer.Abort()
    exported = Path(f"basemode-contribution-{bundle['bundle_id']}.json")
    try:
        url = open_contribution_pr(contribution, repo=repo, exported_path=exported)
    except RuntimeError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    console.print(url)


@contribute_app.command("release")
def release(
    bundle_id: Annotated[str, typer.Argument(help="Bundle ID from a local export.")],
) -> None:
    """Undo an export that was never submitted, freeing its observations."""
    try:
        released = release_bundle(bundle_id)
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    console.print(f"[green]✓[/green] {released} operations can be contributed again.")


@contribute_app.command("clear-pending")
def clear_pending() -> None:
    """Forget failed or pending submission records without deleting exports."""
    if not observations._DB_FILE.exists():
        console.print("No pending contribution batches.")
        return
    with observations._db() as conn:
        removed = conn.execute(
            "DELETE FROM contribution_batches WHERE status IN ('pending','failed')"
        ).rowcount
    console.print(f"Cleared {removed} pending contribution batch(es).")

"""Explicit aggregate-only public contribution commands."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

from .. import observations
from ..contributions import build_bundle, export_bundle
from ..keys import contribution_enabled, set_contribution_enabled
from . import app
from .render import console

contribute_app = typer.Typer(
    help="Preview and export content-free evidence aggregates."
)
app.add_typer(contribute_app, name="contribute")


def _window(
    since: datetime | None, until: datetime | None
) -> tuple[datetime, datetime]:
    end = until or datetime.now(UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    start = since or end - timedelta(days=7)
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    return start, end


@contribute_app.command("status")
def status() -> None:
    """Show whether future operations may enter contribution aggregates."""
    state = "enabled" if contribution_enabled() else "disabled"
    console.print(f"Public contribution is {state}.")


@contribute_app.command("enable")
def enable() -> None:
    """Opt future operations into aggregate-only public contribution."""
    set_contribution_enabled(True)
    console.print("[green]✓[/green] Future operations are contribution-eligible.")


@contribute_app.command("disable")
def disable() -> None:
    """Keep recording locally but exclude future operations from contribution."""
    set_contribution_enabled(False)
    console.print("[green]✓[/green] Public contribution disabled.")


@contribute_app.command("preview")
def preview(
    since: Annotated[datetime | None, typer.Option("--since")] = None,
    until: Annotated[datetime | None, typer.Option("--until")] = None,
) -> None:
    """Print the exact validated JSON shape an export would write."""
    start, end = _window(since, until)
    console.print_json(json.dumps(build_bundle(since=start, until=end)))


@contribute_app.command("export")
def export(
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    since: Annotated[datetime | None, typer.Option("--since")] = None,
    until: Annotated[datetime | None, typer.Option("--until")] = None,
) -> None:
    """Write a validated contribution bundle and record the exported window."""
    start, end = _window(since, until)
    bundle = build_bundle(since=start, until=end)
    target = output or Path(f"basemode-contribution-{bundle['bundle_id']}.json")
    export_bundle(bundle, target)
    console.print(str(target))


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

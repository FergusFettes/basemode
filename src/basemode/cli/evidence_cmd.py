from typing import Annotated

import typer
from rich.table import Table

from . import app
from .render import console


@app.command("evidence")
def evidence_command(
    view: Annotated[
        str,
        typer.Argument(
            help="overview, providers, statuses, failures, transient, rechecks, runs, corpus, endpoint, or export"
        ),
    ] = "overview",
    model: Annotated[
        str | None, typer.Argument(help="Provider-qualified model for endpoint view.")
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON.")
    ] = False,
) -> None:
    """Inspect durable verification, catalog, and Loom evidence (text models only)."""
    from .. import evidence, evidence_report

    views = {
        "overview": evidence_report.overview,
        "providers": evidence_report.providers,
        "statuses": evidence_report.statuses,
        "failures": evidence_report.failures,
        "transient": evidence_report.transient,
        "rechecks": evidence_report.rechecks,
        "runs": evidence_report.runs,
        "corpus": evidence_report.corpus,
    }
    with evidence.connect() as db:
        if view == "endpoint":
            if not model:
                console.print("[red]The endpoint view requires a model.[/red]")
                raise typer.Exit(2)
            rows = evidence_report.endpoint(db, model)
        elif view == "export":
            records = list(evidence_report.export_records(db))
            if as_json:
                console.print_json(data=records)
            else:
                typer.echo(evidence_report.json_lines(records))
            return
        elif view in views:
            rows = views[view](db)
        else:
            console.print(f"[red]Unknown evidence view: {view}[/red]")
            raise typer.Exit(2)

    if as_json or view == "endpoint":
        console.print_json(data=rows[0] if view == "endpoint" and rows else rows)
        return
    if not rows:
        console.print("[yellow]No matching text-model evidence.[/yellow]")
        return
    columns = list(rows[0])
    table = Table(*(column.replace("_", " ").title() for column in columns))
    for row in rows:
        table.add_row(
            *(
                str(row.get(column, "") if row.get(column) is not None else "")
                for column in columns
            )
        )
    console.print(table)

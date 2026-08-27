from typing import Annotated

import typer
from rich.table import Table

from ..keys import (
    KEY_ALIASES,
    get_default_model,
    get_key,
    list_keys,
    set_default_model,
    set_key,
)
from . import app
from .render import console


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
        from ..detect import normalize_model

        resolved = normalize_model(current)
        suffix = f" → [dim]{resolved}[/dim]" if resolved != current else ""
        console.print(f"[bold]{current}[/bold]{suffix}")
        return

    set_default_model(model)
    from ..detect import normalize_model

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
        from ..server import serve as run_server
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

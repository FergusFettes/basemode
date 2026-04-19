import asyncio
import json as _json
import logging
import sys
from pathlib import Path
from typing import Annotated

log = logging.getLogger(__name__)

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

from .continue_ import branch_text, continue_text
from .detect import detect_strategy, normalize_model
from .keys import (
    KEY_ALIASES,
    get_default_model,
    get_key,
    list_keys,
    set_default_model,
    set_key,
)
from .models import list_models, list_providers
from .naming import generate_name, should_name
from .store import AmbiguousNodeReference, GenerationStore, Node
from .strategies import REGISTRY
from .strategies.few_shot import _SYSTEM_PROMPT as FEW_SHOT_SYSTEM_PROMPT
from .strategies.fim import _fim_prompt
from .strategies.prefill import SEED_LEN
from .strategies.system import SYSTEM_PROMPT
from .strategies.utils import normalize_prefix
from .usage import estimate_usage, format_per_million, format_usd, get_price_info

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

loom_app = typer.Typer(
    help="Persistent branching exploration and SQLite-backed sessions.",
    cls=_default_to("view"),
)
app.add_typer(loom_app, name="loom")


@app.command()
def run(
    ctx: typer.Context,
    prefix: Annotated[str | None, typer.Argument(help="Text to continue (or pipe via stdin)")] = None,
    model: Annotated[str | None, typer.Option("-m", "--model")] = None,
    n: Annotated[int, typer.Option("-n", "--branches", help="Number of parallel continuations")] = 1,
    max_tokens: Annotated[int, typer.Option("-M", "--max-tokens")] = 200,
    temperature: Annotated[float, typer.Option("-t", "--temperature")] = 0.9,
    strategy: Annotated[str | None, typer.Option("-s", "--strategy")] = None,
    show_strategy: Annotated[bool, typer.Option("--show-strategy")] = False,
    show_usage: Annotated[bool, typer.Option("--show-usage", help="Show estimated token usage after generation")] = False,
    show_cost: Annotated[bool, typer.Option("--show-cost", help="Show estimated cost after generation")] = False,
) -> None:
    """Continue text with an LLM (default command)."""
    if prefix is None and not sys.stdin.isatty():
        prefix = sys.stdin.read()
    if prefix is None:
        console.print(ctx.get_help())
        return
    _run_text(prefix, model, n, max_tokens, temperature, strategy, show_strategy, show_usage, show_cost)


async def _stream_one(prefix: str, model: str, max_tokens: int, temperature: float, strategy: str | None) -> str:
    console.print(f"[dim]{prefix}[/dim]", end="")
    chunks: list[str] = []
    async for token in continue_text(prefix, model, max_tokens=max_tokens, temperature=temperature, strategy=strategy):
        chunks.append(token)
        console.print(token, end="")
    console.print()
    return "".join(chunks)


async def _stream_branches(
    prefix: str, model: str, n: int, max_tokens: int, temperature: float, strategy: str | None
) -> list[str]:
    buffers: list[list[str]] = [[] for _ in range(n)]

    with Live(
        _branches_panel(prefix, buffers),
        console=console,
        refresh_per_second=12,
    ) as live:
        async for idx, token in branch_text(
            prefix, model, n=n, max_tokens=max_tokens, temperature=temperature, strategy=strategy
        ):
            buffers[idx].append(token)
            live.update(_branches_panel(prefix, buffers))

    return ["".join(buf) for buf in buffers]


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
    show_strategy: bool,
    show_usage: bool,
    show_cost: bool,
) -> None:
    if model is None:
        model = get_default_model() or "gpt-4o-mini"

    prefix = prefix.rstrip("\n")

    if show_strategy:
        strat = detect_strategy(normalize_model(model), strategy)
        console.print(f"[dim]strategy: {strat.name}[/dim]")

    if n == 1:
        completion = asyncio.run(_stream_one(prefix, model, max_tokens, temperature, strategy))
        if show_usage or show_cost:
            _print_usage_estimate(model, prefix, completion, strategy, show_cost, prompt_requests=1)
    else:
        completions = asyncio.run(
            _stream_branches(prefix, model, n, max_tokens, temperature, strategy)
        )
        if show_usage or show_cost:
            _print_usage_estimate(
                model,
                prefix,
                "".join(completions),
                strategy,
                show_cost,
                prompt_requests=n,
            )


@loom_app.command("run")
def loom_run(
    ctx: typer.Context,
    prefix: Annotated[str | None, typer.Argument(help="Text to continue (or pipe via stdin)")] = None,
    model: Annotated[str | None, typer.Option("-m", "--model")] = None,
    n: Annotated[int, typer.Option("-n", "--branches", help="Number of parallel continuations")] = 1,
    max_tokens: Annotated[int, typer.Option("-M", "--max-tokens")] = 200,
    temperature: Annotated[float, typer.Option("-t", "--temperature")] = 0.9,
    strategy: Annotated[str | None, typer.Option("-s", "--strategy")] = None,
    show_strategy: Annotated[bool, typer.Option("--show-strategy")] = False,
    show_usage: Annotated[bool, typer.Option("--show-usage", help="Show estimated token usage after generation")] = False,
    show_cost: Annotated[bool, typer.Option("--show-cost", help="Show estimated cost after generation")] = False,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite generation database path")] = None,
) -> None:
    """Persist a generation tree in SQLite."""
    if prefix is None and not sys.stdin.isatty():
        prefix = sys.stdin.read()
    if prefix is None:
        console.print(ctx.get_help())
        return
    store = GenerationStore(db)
    _run_loom_generation(
        store,
        prefix,
        None,
        model,
        n,
        max_tokens,
        temperature,
        strategy,
        show_strategy,
        show_usage,
        show_cost,
    )


@loom_app.command("continue")
def loom_continue(
    ctx: typer.Context,
    branch: Annotated[int | None, typer.Option("-b", "--branch", min=1, help="Select a child branch of the active node")] = None,
    model: Annotated[str | None, typer.Option("-m", "--model")] = None,
    n: Annotated[int, typer.Option("-n", "--branches", help="Number of parallel continuations")] = 1,
    max_tokens: Annotated[int, typer.Option("-M", "--max-tokens")] = 200,
    temperature: Annotated[float, typer.Option("-t", "--temperature")] = 0.9,
    strategy: Annotated[str | None, typer.Option("-s", "--strategy")] = None,
    show_strategy: Annotated[bool, typer.Option("--show-strategy")] = False,
    show_usage: Annotated[bool, typer.Option("--show-usage", help="Show estimated token usage after generation")] = False,
    show_cost: Annotated[bool, typer.Option("--show-cost", help="Show estimated cost after generation")] = False,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite generation database path")] = None,
) -> None:
    """Continue from the stored active node."""
    store = GenerationStore(db)
    active = store.get_active_node()
    if active is None:
        console.print("[red]No active node stored yet.[/red]")
        raise typer.Exit(1)
    base_node = _resolve_loom_base(store, active, branch)
    prefix = store.full_text(base_node.id)
    _run_loom_generation(
        store,
        base_node,
        prefix,
        model,
        n,
        max_tokens,
        temperature,
        strategy,
        show_strategy,
        show_usage,
        show_cost,
    )


@loom_app.command("select")
def loom_select(
    node_id: Annotated[str, typer.Argument(help="Node id to mark active")],
    db: Annotated[Path | None, typer.Option("--db", help="SQLite generation database path")] = None,
) -> None:
    """Mark a node as the active cursor."""
    store = GenerationStore(db)
    try:
        node = store.get(node_id)
    except AmbiguousNodeReference as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    if node is None:
        console.print(f"[red]Unknown node: {node_id}[/red]")
        raise typer.Exit(1)
    store.set_active_node(node.id)
    console.print(f"[green]✓[/green] Active node set to {node.id}")


@loom_app.command("nodes")
def loom_nodes(
    limit: Annotated[int, typer.Option("-n", "--limit", help="Number of recent nodes to show")] = 20,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite generation database path")] = None,
) -> None:
    """List recently persisted generation nodes."""
    store = GenerationStore(db)
    rows = store.recent(limit)
    active_id = store.get_active_node_id()
    if not rows:
        console.print(f"[yellow]No nodes found in {store.db_path}.[/yellow]")
        return

    table = Table("Active", "ID", "Name", "Parent", "Model", "Branch", "Created", "Text", show_header=True, header_style="bold")
    for node in rows:
        table.add_row(
            "*" if node.id == active_id else "",
            node.id,
            str(node.metadata.get("name", "")),
            node.parent_id or "",
            node.model or "",
            "" if node.branch_index is None else str(node.branch_index + 1),
            node.created_at,
            _preview(node.text),
        )
    console.print(table)


@loom_app.command("active")
def loom_active(
    db: Annotated[Path | None, typer.Option("--db", help="SQLite generation database path")] = None,
) -> None:
    """Show the currently active node."""
    store = GenerationStore(db)
    node = store.get_active_node()
    if node is None:
        console.print("[yellow]No active node stored yet.[/yellow]")
        return
    table = Table("Field", "Value", show_header=True, header_style="bold")
    table.add_row("ID", node.id)
    table.add_row("Name", str(node.metadata.get("name", "")))
    table.add_row("Parent", node.parent_id or "")
    table.add_row("Branch", "" if node.branch_index is None else str(node.branch_index + 1))
    table.add_row("Text", _preview(node.text, limit=120))
    console.print(table)


@loom_app.command("show")
def loom_show(
    node_id: Annotated[str, typer.Argument(help="Node id to print")],
    segment: Annotated[bool, typer.Option("--segment", help="Print only this node's segment")] = False,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite generation database path")] = None,
) -> None:
    """Print a persisted node's full text."""
    store = GenerationStore(db)
    try:
        node = store.get(node_id)
    except AmbiguousNodeReference as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    if node is None:
        console.print(f"[red]Unknown node: {node_id}[/red]")
        raise typer.Exit(1)
    console.print(node.text if segment else store.full_text(node_id))


@loom_app.command("children")
def loom_children(
    node_id: Annotated[str, typer.Argument(help="Parent node id")],
    db: Annotated[Path | None, typer.Option("--db", help="SQLite generation database path")] = None,
) -> None:
    """List children of a persisted node."""
    store = GenerationStore(db)
    try:
        node = store.get(node_id)
    except AmbiguousNodeReference as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    if node is None:
        console.print(f"[red]Unknown node: {node_id}[/red]")
        raise typer.Exit(1)
    rows = store.children(node.id)
    if not rows:
        console.print("[yellow]No children.[/yellow]")
        return
    table = Table("ID", "Branch", "Model", "Created", "Text", show_header=True, header_style="bold")
    for node in rows:
        table.add_row(
            node.id,
            "" if node.branch_index is None else str(node.branch_index + 1),
            node.model or "",
            node.created_at,
            _preview(node.text),
        )
    console.print(table)


@loom_app.command("roots")
def loom_roots(
    db: Annotated[Path | None, typer.Option("--db", help="SQLite generation database path")] = None,
) -> None:
    """List all root nodes (top-level generation trees)."""
    store = GenerationStore(db)
    rows = store.roots()
    active_id = store.get_active_node_id()
    if not rows:
        console.print(f"[yellow]No roots found in {store.db_path}.[/yellow]")
        return
    table = Table("Active", "ID", "Name", "Children", "Created", "Text", header_style="bold")
    for root in rows:
        child_count = len(store.children(root.id))
        table.add_row(
            "*" if root.id == active_id else "",
            root.id[:8],
            str(root.metadata.get("name", "")),
            str(child_count),
            root.created_at,
            _preview(root.text),
        )
    console.print(table)


@loom_app.command("view")
def loom_view(
    source: Annotated[str | None, typer.Argument(help="Node id, .txt file (use as root), or .json export to import")] = None,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite generation database path")] = None,
) -> None:
    """Interactive loom viewer. hjkl: nav. space: generate. q: quit."""
    from .session import LoomSession
    from .tui.app import BasemodeApp

    store = GenerationStore(db)
    start = _resolve_loom_source(store, source)
    if start is None:
        return
    session = LoomSession(store, start.id)
    BasemodeApp(session).run()


def _resolve_loom_source(store: "GenerationStore", source: "str | None") -> "Node | None":
    """Resolve a source argument to a Node: None→active, file→import/create, str→node id."""
    if source is None:
        node = store.get_active_node()
        if node is None:
            console.print("[yellow]No active node.[/yellow]")
        return node

    p = Path(source)
    if p.suffix == ".json" and p.exists():
        return _import_loom_json(store, p)

    if p.exists() and p.is_file():
        text = p.read_text().rstrip("\n")
        existing = store.find_root_by_text(text)
        if existing:
            console.print(f"[dim]Found existing root {existing.id[:8]}[/dim]")
            return existing
        root = store.create_root(text, metadata={"source_file": str(p)})
        store.set_active_node(root.id)
        console.print(f"[dim]Created root {root.id[:8]} from {p.name}[/dim]")
        return root

    try:
        node = store.get(source)
    except AmbiguousNodeReference as exc:
        console.print(f"[red]{exc}[/red]")
        return None
    if node is None:
        console.print(f"[red]Unknown node or file not found: {source}[/red]")
    return node


def _import_loom_json(store: "GenerationStore", path: Path) -> "Node | None":
    try:
        data = _json.loads(path.read_text())
    except Exception as exc:
        console.print(f"[red]Failed to read {path}: {exc}[/red]")
        return None
    raw_nodes = data.get("nodes", [])
    if not raw_nodes:
        console.print("[red]No nodes found in export.[/red]")
        return None
    from .store import Node as _Node
    nodes = [
        _Node(
            id=n["id"],
            parent_id=n.get("parent_id"),
            root_id=n["root_id"],
            text=n["text"],
            model=n.get("model"),
            strategy=n.get("strategy"),
            max_tokens=n.get("max_tokens"),
            temperature=n.get("temperature"),
            branch_index=n.get("branch_index"),
            created_at=n["created_at"],
            metadata=n.get("metadata", {}),
        )
        for n in raw_nodes
    ]
    inserted = store.import_nodes(nodes)
    skipped = len(nodes) - inserted
    console.print(f"[dim]Imported {inserted} nodes, skipped {skipped} duplicates[/dim]")
    root_id = next((n.root_id for n in nodes if n.parent_id is None), nodes[0].root_id)
    root = store.get(root_id)
    if root:
        store.set_active_node(root.id)
    return root


@loom_app.command("export")
def loom_export(
    to: Annotated[str, typer.Option("--to", help="Output file path or 'json' for stdout")] = "json",
    node_id: Annotated[str | None, typer.Option("--node", help="Any node in the tree (defaults to active)")] = None,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite generation database path")] = None,
) -> None:
    """Export the current generation tree to JSON."""
    store = GenerationStore(db)
    if node_id is not None:
        try:
            node = store.get(node_id)
        except AmbiguousNodeReference as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from None
        if node is None:
            console.print(f"[red]Unknown node: {node_id}[/red]")
            raise typer.Exit(1)
    else:
        node = store.get_active_node()
        if node is None:
            console.print("[red]No active node. Use --node to specify one.[/red]")
            raise typer.Exit(1)

    root = store.root(node.id)
    tree_nodes = store.tree(root.id)
    data = {
        "version": 1,
        "nodes": [
            {
                "id": n.id,
                "parent_id": n.parent_id,
                "root_id": n.root_id,
                "text": n.text,
                "model": n.model,
                "strategy": n.strategy,
                "max_tokens": n.max_tokens,
                "temperature": n.temperature,
                "branch_index": n.branch_index,
                "created_at": n.created_at,
                "metadata": n.metadata,
            }
            for n in tree_nodes
        ],
    }
    serialized = _json.dumps(data, indent=2, ensure_ascii=False)

    if to == "json":
        print(serialized)
    else:
        out = Path(to)
        out.write_text(serialized, encoding="utf-8")
        console.print(f"[dim]Exported {len(tree_nodes)} nodes → {out}[/dim]")


def _resolve_loom_base(store: GenerationStore, active: Node, branch: int | None) -> Node:
    children = store.children(active.id)
    if branch is not None:
        try:
            return store.select_branch(active.id, branch)
        except (IndexError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from None

    if len(children) == 1:
        return children[0]
    if len(children) > 1:
        console.print(
            f"[red]Active node {active.id} has {len(children)} branches. Use -b N to select one.[/red]"
        )
        raise typer.Exit(1)
    return active


def _run_loom_generation(
    store: GenerationStore,
    base_node: Node | None,
    prefix: str,
    model: str | None,
    n: int,
    max_tokens: int,
    temperature: float,
    strategy: str | None,
    show_strategy: bool,
    show_usage: bool,
    show_cost: bool,
) -> None:
    if model is None:
        model = get_default_model() or "gpt-4o-mini"
    prefix = prefix.rstrip("\n")
    if show_strategy:
        strat = detect_strategy(normalize_model(model), strategy)
        console.print(f"[dim]strategy: {strat.name}[/dim]")
    if n == 1:
        completion = asyncio.run(_stream_one(prefix, model, max_tokens, temperature, strategy))
        _save_loom_run(
            store,
            prefix,
            [completion],
            model,
            strategy,
            max_tokens,
            temperature,
            base_node.id if base_node is not None else None,
        )
        if show_usage or show_cost:
            _print_usage_estimate(model, prefix, completion, strategy, show_cost, prompt_requests=1)
    else:
        completions = asyncio.run(
            _stream_branches(prefix, model, n, max_tokens, temperature, strategy)
        )
        _save_loom_run(
            store,
            prefix,
            completions,
            model,
            strategy,
            max_tokens,
            temperature,
            base_node.id if base_node is not None else None,
        )
        if show_usage or show_cost:
            _print_usage_estimate(
                model,
                prefix,
                "".join(completions),
                strategy,
                show_cost,
                prompt_requests=n,
            )


def _save_loom_run(
    store: GenerationStore,
    prefix: str,
    completions: list[str],
    model: str | None,
    strategy: str | None,
    max_tokens: int,
    temperature: float,
    active_node_id: str | None,
) -> None:
    resolved = normalize_model(model or get_default_model() or "gpt-4o-mini")
    strategy_name = detect_strategy(resolved, strategy).name
    parent, children = store.save_continuations(
        prefix,
        completions,
        model=resolved,
        strategy=strategy_name,
        max_tokens=max_tokens,
        temperature=temperature,
        parent_id=active_node_id,
    )
    console.print(f"[dim]saved parent: {parent.id}[/dim]")
    for child in children:
        label = f"branch {child.branch_index + 1}" if child.branch_index is not None else "child"
        console.print(f"[dim]saved {label}: {child.id}[/dim]")
    base_id = active_node_id or parent.id
    store.set_active_node(base_id if len(children) > 1 else children[0].id)
    _maybe_name_tree(store, children)


def _maybe_name_tree(store: GenerationStore, children: list[Node]) -> None:
    if not children:
        return
    root = store.root(children[0].id)
    if root.metadata.get("name"):
        return

    candidates = [(child, store.full_text(child.id)) for child in children]
    child, text = max(candidates, key=lambda item: len(item[1]))
    if not should_name(text):
        return

    name = generate_name(text)
    if name is None:
        return
    store.update_metadata(root.id, {"name": name, "named_from": child.id})
    console.print(f"[dim]named tree: {name}[/dim]")


def _print_usage_estimate(
    model: str,
    prefix: str,
    completion: str,
    strategy: str | None,
    show_cost: bool,
    prompt_requests: int,
) -> None:
    resolved = normalize_model(model)
    prompt, messages = _usage_prompt(resolved, prefix, strategy)
    usage = estimate_usage(resolved, prompt, completion, prompt_messages=messages, prompt_requests=prompt_requests)
    table = Table("Metric", "Value", show_header=False)
    table.add_row("Model", usage.model)
    table.add_row("Prompt tokens", str(usage.prompt_tokens))
    table.add_row("Completion tokens", str(usage.completion_tokens))
    table.add_row("Total tokens", str(usage.total_tokens))
    if show_cost:
        table.add_row("Estimated cost", format_usd(usage.cost_usd))
        if not usage.pricing_available:
            table.add_row("Cost note", "pricing unavailable in LiteLLM model map")
    console.print(table)


def _usage_prompt(model: str, prefix: str, strategy: str | None) -> tuple[str, list[dict] | None]:
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


def _preview(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


@app.command()
def info(model: Annotated[str, typer.Argument(help="Model name to inspect")]) -> None:
    """Show strategy, provider, limits, and known pricing for a model."""
    resolved = normalize_model(model)
    strat = detect_strategy(resolved)
    price = get_price_info(resolved)

    table = Table("Field", "Value", show_header=True, header_style="bold")
    table.add_row("Model", model)
    table.add_row("Resolved", resolved)
    table.add_row("Strategy", strat.name)
    table.add_row("Provider", price.provider or "unknown")
    table.add_row("Input price", format_per_million(price.input_cost_per_token))
    table.add_row("Output price", format_per_million(price.output_cost_per_token))
    table.add_row("Cache read price", format_per_million(price.cache_read_input_token_cost))
    table.add_row("Reasoning output price", format_per_million(price.output_cost_per_reasoning_token))
    table.add_row("Max input tokens", str(price.max_input_tokens) if price.max_input_tokens else "unknown")
    table.add_row("Max output tokens", str(price.max_output_tokens) if price.max_output_tokens else "unknown")
    table.add_row("Supports reasoning", str(price.supports_reasoning) if price.supports_reasoning is not None else "unknown")
    if not price.pricing_available:
        table.add_row("Cost note", "pricing unavailable in LiteLLM model map")
    console.print(table)


@app.command()
def keys(
    action: Annotated[str, typer.Argument(help="Action: set | list | get")],
    provider: Annotated[str | None, typer.Argument(help="Provider name (e.g. openai, anthropic)")] = None,
    value: Annotated[str | None, typer.Argument(help="API key value (set only; prompted if omitted)")] = None,
) -> None:
    """Manage API keys stored in ~/.config/basemode/auth.json.

    Examples:

      basemode keys set openai

      basemode keys list

      basemode keys get anthropic
    """
    if action == "set":
        if not provider:
            console.print("[red]Provider required. E.g.: basemode keys set openai[/red]")
            raise typer.Exit(1)
        if value is None:
            value = typer.prompt(f"{provider} API key", hide_input=True)
        set_key(provider, value)
        console.print(f"[green]✓[/green] Saved [bold]{provider}[/bold] key to ~/.config/basemode/auth.json")

    elif action == "list":
        stored = list_keys()
        if not stored:
            console.print("[yellow]No keys stored. Use: basemode keys set <provider>[/yellow]")
            return
        table = Table("Provider", "Key", "Env var", show_header=True, header_style="bold")
        for name, masked in stored.items():
            env_var = KEY_ALIASES.get(name, name.upper() + "_API_KEY")
            table.add_row(name, masked, env_var)
        console.print(table)

    elif action == "get":
        if not provider:
            console.print("[red]Provider required. E.g.: basemode keys get openai[/red]")
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
    model: Annotated[str | None, typer.Argument(help="Model to set as default (omit to show current)")] = None,
    unset: Annotated[bool, typer.Option("--unset", help="Clear the stored default")] = False,
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
            console.print("[yellow]No default model set. E.g.: basemode default claude-sonnet-4-6[/yellow]")
            return
        resolved = normalize_model(current)
        suffix = f" → [dim]{resolved}[/dim]" if resolved != current else ""
        console.print(f"[bold]{current}[/bold]{suffix}")
        return

    set_default_model(model)
    resolved = normalize_model(model)
    suffix = f" → [dim]{resolved}[/dim]" if resolved != model else ""
    console.print(f"[green]✓[/green] Default model set to [bold]{model}[/bold]{suffix}")

import asyncio
import curses
import sys
import textwrap
from pathlib import Path
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
    node_id: Annotated[str | None, typer.Argument(help="Node id to start at (defaults to active)")] = None,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite generation database path")] = None,
) -> None:
    """Interactive loom viewer. hjkl: parent/child/prev-sibling/next-sibling. q: quit."""
    store = GenerationStore(db)
    if node_id is None:
        start = store.get_active_node()
        if start is None:
            console.print("[yellow]No active node.[/yellow]")
            return
    else:
        try:
            start = store.get(node_id)
        except AmbiguousNodeReference as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from None
        if start is None:
            console.print(f"[red]Unknown node: {node_id}[/red]")
            raise typer.Exit(1)
    curses.wrapper(_loom_tui, store, start.id)


def _loom_tui(stdscr: "curses._CursesWindow", store: GenerationStore, start_id: str) -> None:
    import os
    import subprocess
    import tempfile

    curses.curs_set(0)
    stdscr.nodelay(False)

    current_id = start_id
    selected_idx = 0
    child_path: dict[str, int] = {}
    max_tokens = 200
    temperature = 0.9

    while True:
        height, width = stdscr.getmaxyx()
        children = store.children(current_id)
        if children:
            selected_idx = min(selected_idx, len(children) - 1)
        else:
            selected_idx = 0

        stdscr.clear()
        display_lines = _build_loom_display(store, current_id, selected_idx, child_path, width)

        start_line = max(0, len(display_lines) - (height - 1))
        for row, (text, attr) in enumerate(display_lines[start_line:]):
            if row >= height - 1:
                break
            try:
                stdscr.addstr(row, 0, text.ljust(width)[:width], attr)
            except curses.error:
                pass

        node = store.get(current_id)
        status = (
            f" {current_id[:8]}  max_tokens:{max_tokens}"
            "  hjkl=nav  space=continue  e=editor  w/s=tokens  d=set-tokens  q=quit"
        )
        try:
            stdscr.addstr(height - 1, 0, status[:width].ljust(width)[:width], curses.A_REVERSE)
        except curses.error:
            pass

        stdscr.refresh()
        key = stdscr.getch()

        if key in (ord("q"), 27):
            store.set_active_node(current_id)
            break
        elif key == ord("j") and children and selected_idx < len(children) - 1:
            selected_idx += 1
        elif key == ord("k") and children and selected_idx > 0:
            selected_idx -= 1
        elif key == ord("l") and children:
            child_path[current_id] = selected_idx
            current_id = children[selected_idx].id
            selected_idx = child_path.get(current_id, 0)
        elif key == ord("h") and node and node.parent_id:
            parent_id = node.parent_id
            parent_children = store.children(parent_id)
            for i, c in enumerate(parent_children):
                if c.id == current_id:
                    selected_idx = i
                    break
            else:
                selected_idx = child_path.get(parent_id, 0)
            current_id = parent_id
        elif key == ord("w"):
            max_tokens = min(max_tokens + 200, 8000)
        elif key == ord("s"):
            max_tokens = max(max_tokens - 200, 100)
        elif key == ord("d"):
            result = _loom_ask_int(stdscr, "Max tokens", max_tokens)
            if result is not None and result > 0:
                max_tokens = result
        elif key == ord("e"):
            text = store.full_text(current_id)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(text)
                tmpfile = f.name
            curses.endwin()
            subprocess.run([os.environ.get("EDITOR", "vim"), tmpfile])
            os.unlink(tmpfile)
            stdscr.refresh()
        elif key == ord(" "):
            model = get_default_model() or "gpt-4o-mini"
            prefix = store.full_text(current_id)
            curses.endwin()
            completion = asyncio.run(_stream_one(prefix, model, max_tokens, temperature, None))
            _save_loom_run(store, prefix, [completion], model, None, max_tokens, temperature, current_id)
            new_children = store.children(current_id)
            if new_children:
                child_path[current_id] = len(new_children) - 1
                current_id = new_children[-1].id
                selected_idx = 0
            input("\nPress Enter to return to loom...")
            stdscr.refresh()


def _loom_ask_int(stdscr: "curses._CursesWindow", prompt: str, current: int) -> int | None:
    height, width = stdscr.getmaxyx()
    popup_w = min(50, width - 4)
    popup_h = 5
    py = (height - popup_h) // 2
    px = (width - popup_w) // 2

    win = curses.newwin(popup_h, popup_w, py, px)
    win.box()
    win.addstr(1, 2, prompt, curses.A_BOLD)
    label = f"Current: {current}  New: "
    win.addstr(2, 2, label)
    win.addstr(3, 2, "Enter to confirm, Esc to cancel", curses.A_DIM)
    win.refresh()

    curses.echo()
    curses.curs_set(1)
    try:
        raw = win.getstr(2, 2 + len(label), popup_w - 4 - len(label)).decode().strip()
    except Exception:
        raw = ""
    curses.noecho()
    curses.curs_set(0)
    del win
    stdscr.touchwin()
    stdscr.refresh()

    try:
        return int(raw)
    except ValueError:
        return None


def _loom_word_wrap(text: str, first_width: int, full_width: int) -> list[str]:
    """Word-wrap text: first line to first_width, subsequent lines to full_width."""
    text = text.rstrip("\n")
    if not text:
        return [""]
    words = text.split()
    lines: list[str] = []
    current_line = ""
    current_width = first_width

    for word in words:
        if not current_line:
            current_line = word[:current_width]
            if len(word) > current_width:
                lines.append(current_line)
                current_line = ""
                current_width = full_width
        elif len(current_line) + 1 + len(word) <= current_width:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word[:full_width]
            current_width = full_width

    if current_line:
        lines.append(current_line)
    return lines or [""]


def _build_loom_display(
    store: GenerationStore,
    current_id: str,
    selected_idx: int,
    child_path: dict[str, int],
    width: int,
) -> list[tuple[str, int]]:
    """Build display lines for the loom TUI as (text, curses_attr) pairs."""
    parent_text = store.full_text(current_id)
    children = store.children(current_id)

    # Word-wrap the parent text to terminal width
    parent_lines: list[str] = []
    for segment in parent_text.split("\n"):
        if segment:
            parent_lines.extend(textwrap.wrap(segment, width) or [""])
        else:
            parent_lines.append("")
    if not parent_lines:
        parent_lines = [""]

    if not children:
        return [(line, 0) for line in parent_lines]

    ARROW = " -> "
    last_line = parent_lines[-1]
    indent = len(last_line)
    available = width - indent - len(ARROW)

    if available < 10:
        # Parent's last line is too long; put children on a fresh line
        lines: list[tuple[str, int]] = [(line, 0) for line in parent_lines]
        indent = 0
        available = width - len(ARROW)
        last_line = ""
    else:
        lines = [(line, 0) for line in parent_lines[:-1]]

    selected_child = children[selected_idx]
    sel_grandchildren = store.children(selected_child.id)
    sel_marker = "**" if sel_grandchildren else ""
    child_segment = selected_child.text + sel_marker

    def sibling_marker(child: Node) -> str:
        return "*" if store.children(child.id) else ""

    child_fits = len(child_segment) <= available

    if child_fits and sel_grandchildren:
        # Grandchild joins the stream on the same first line
        gc_idx = min(child_path.get(selected_child.id, 0), len(sel_grandchildren) - 1)
        grandchild = sel_grandchildren[gc_idx]
        gc_marker = "**" if store.children(grandchild.id) else ""
        stream = child_segment + " " + grandchild.text + gc_marker
    else:
        stream = child_segment

    stream_lines = _loom_word_wrap(stream, available, width)
    first_stream = stream_lines[0]
    rest_stream = stream_lines[1:]

    prefix = (last_line + ARROW) if last_line else ARROW
    lines.append((prefix + first_stream, curses.A_BOLD))

    inline = child_fits and sel_grandchildren

    if inline:
        # Siblings appear after the first line; grandchild continuation follows siblings
        for i, child in enumerate(children):
            if i == selected_idx:
                continue
            sib_segment = child.text + sibling_marker(child)
            sib_lines = _loom_word_wrap(sib_segment, available, width)
            for j, sl in enumerate(sib_lines):
                if j == 0:
                    lines.append((" " * indent + ARROW + sl, curses.A_DIM))
                else:
                    lines.append((sl, curses.A_DIM))
        for sl in rest_stream:
            lines.append((sl, curses.A_BOLD))
    else:
        # Child wraps: show all child lines first, then siblings, then grandchild
        for sl in rest_stream:
            lines.append((sl, curses.A_BOLD))
        for i, child in enumerate(children):
            if i == selected_idx:
                continue
            sib_segment = child.text + sibling_marker(child)
            sib_lines = _loom_word_wrap(sib_segment, available, width)
            for j, sl in enumerate(sib_lines):
                if j == 0:
                    lines.append((" " * indent + ARROW + sl, curses.A_DIM))
                else:
                    lines.append((sl, curses.A_DIM))
        if sel_grandchildren:
            gc_idx = min(child_path.get(selected_child.id, 0), len(sel_grandchildren) - 1)
            grandchild = sel_grandchildren[gc_idx]
            gc_marker = "**" if store.children(grandchild.id) else ""
            gc_segment = grandchild.text + gc_marker
            gc_lines = _loom_word_wrap(gc_segment, available, width)
            for j, gl in enumerate(gc_lines):
                if j == 0:
                    lines.append((" " * indent + gl, 0))
                else:
                    lines.append((gl, 0))

    return lines


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

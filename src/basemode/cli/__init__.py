import click
import typer
import typer.core

from ..logging_setup import setup_file_logging
from .render import console

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


@app.callback()
def _main() -> None:
    """Set up CLI-only side effects before any command runs.

    File logging lives here rather than at package import because importing
    `basemode` as a library must not create a state directory or attach a
    handler to someone else's logging config. The CLI is the one context
    that owns the user's terminal, so it is the one that opts in.
    """
    setup_file_logging()


# Importing these modules registers their commands on `app` via decorators.
# Re-exported here so `basemode.cli.<name>` keeps working for anything that
# referenced the pre-split flat module (tests, scripts, docs examples).
from .bench_cmd import bench as bench  # noqa: E402
from .config_cmd import default as default  # noqa: E402
from .config_cmd import keys as keys  # noqa: E402
from .config_cmd import serve as serve  # noqa: E402
from .evidence_cmd import evidence_command as evidence_command  # noqa: E402
from .health_cmd import health as health  # noqa: E402
from .models_cmd import info as info  # noqa: E402
from .models_cmd import models as models  # noqa: E402
from .models_cmd import providers as providers  # noqa: E402
from .models_cmd import rate as rate  # noqa: E402
from .models_cmd import strategies as strategies  # noqa: E402
from .run import _usage_prompt as _usage_prompt  # noqa: E402
from .run import run as run  # noqa: E402
from .verify_cmd import _verify_plan_table as _verify_plan_table  # noqa: E402
from .verify_cmd import verify_command as verify_command  # noqa: E402

__all__ = [
    "app",
    "bench",
    "console",
    "default",
    "evidence_command",
    "health",
    "info",
    "keys",
    "models",
    "providers",
    "rate",
    "run",
    "serve",
    "strategies",
    "verify_command",
]

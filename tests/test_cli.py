from typer.testing import CliRunner

from basemode.cli import app

runner = CliRunner()


def test_top_level_help_is_stateless() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "models" in result.output
    assert "loom" not in result.output


def test_strategies_command_lists_core_strategies() -> None:
    result = runner.invoke(app, ["strategies"])

    assert result.exit_code == 0
    assert "completion" in result.output
    assert "prefill" in result.output

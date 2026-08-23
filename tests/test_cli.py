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


def _fake_bench(script: dict[str, str]):
    """Return a bench_model stub whose output is scripted per strategy."""
    from basemode.bench import ProbeResult, StrategyResult, rank
    from basemode.scoring import score_continuation

    async def bench_model(model, *, strategies, max_tokens=60, temperature=1.0, **kw):
        results = []
        for strategy in strategies:
            text = script.get(strategy, "")
            results.append(
                StrategyResult(
                    strategy=strategy,
                    probes=[
                        ProbeResult(
                            prefix="prefix",
                            text=text,
                            score=score_continuation("prefix", text),
                            elapsed_s=0.5,
                        )
                    ],
                )
            )
        return rank(results)

    return bench_model


def test_bench_ranks_strategies_and_suggests_the_winner(monkeypatch) -> None:
    from basemode import bench as bench_module

    monkeypatch.setattr(
        bench_module,
        "bench_model",
        _fake_bench(
            {"system": "Sure! Here you go:", "few_shot": " and then it rained."}
        ),
    )

    result = runner.invoke(app, ["bench", "gpt-4o-mini", "-s", "system,few_shot"])

    assert result.exit_code == 0
    assert "few_shot" in result.output
    assert "--save" in result.output


def test_bench_save_pins_the_winning_strategy(monkeypatch) -> None:
    from basemode import bench as bench_module
    from basemode.detect import select_strategy
    from basemode.keys import list_strategy_overrides

    monkeypatch.setattr(
        bench_module,
        "bench_model",
        _fake_bench(
            {"system": "Sure! Here you go:", "few_shot": " and then it rained."}
        ),
    )

    result = runner.invoke(
        app, ["bench", "gpt-4o-mini", "-s", "system,few_shot", "--save"]
    )

    assert result.exit_code == 0
    assert list_strategy_overrides() == {"openai/gpt-4o-mini": "few_shot"}
    assert select_strategy("openai/gpt-4o-mini").name == "few_shot"


def test_bench_rejects_an_unknown_strategy() -> None:
    result = runner.invoke(app, ["bench", "gpt-4o-mini", "-s", "telepathy"])

    assert result.exit_code == 1
    assert "Unknown strategy" in result.output


def test_bench_fails_when_no_strategy_produces_output(monkeypatch) -> None:
    from basemode import bench as bench_module

    monkeypatch.setattr(bench_module, "bench_model", _fake_bench({"system": ""}))

    result = runner.invoke(app, ["bench", "gpt-4o-mini", "-s", "system"])

    assert result.exit_code == 1
    assert "No strategy produced a usable continuation" in result.output


def test_bench_json_output_is_machine_readable(monkeypatch) -> None:
    import json

    from basemode import bench as bench_module

    monkeypatch.setattr(
        bench_module,
        "bench_model",
        _fake_bench({"system": " and then it rained."}),
    )

    result = runner.invoke(app, ["bench", "gpt-4o-mini", "-s", "system", "--json"])

    payload = json.loads(result.output)
    assert payload["model"] == "openai/gpt-4o-mini"
    assert payload["results"][0]["strategy"] == "system"
    assert payload["results"][0]["score"] == 1.0


def test_strategies_lists_and_unpins_overrides() -> None:
    from basemode.keys import list_strategy_overrides, set_strategy_override

    set_strategy_override("openai/gpt-4o-mini", "few_shot")

    listed = runner.invoke(app, ["strategies"])
    assert "openai/gpt-4o-mini" in listed.output

    unpinned = runner.invoke(app, ["strategies", "--unpin", "gpt-4o-mini"])
    assert unpinned.exit_code == 0
    assert list_strategy_overrides() == {}


def test_strategies_unpin_reports_when_nothing_is_pinned() -> None:
    result = runner.invoke(app, ["strategies", "--unpin", "gpt-4o-mini"])

    assert result.exit_code == 1
    assert "No pinned strategy" in result.output


def test_info_reports_where_the_strategy_came_from() -> None:
    result = runner.invoke(app, ["info", "kimi-k3"])

    assert result.exit_code == 0
    assert "prefill" in result.output
    assert "registry" in result.output


def test_rate_pins_lists_and_clears_a_thumb() -> None:
    from basemode.keys import list_model_ratings

    up = runner.invoke(app, ["rate", "gpt-4o-mini", "up"])
    assert up.exit_code == 0
    assert list_model_ratings() == {"openai/gpt-4o-mini": 1}

    listed = runner.invoke(app, ["rate"])
    assert "openai/gpt-4o-mini" in listed.output

    cleared = runner.invoke(app, ["rate", "gpt-4o-mini", "clear"])
    assert cleared.exit_code == 0
    assert list_model_ratings() == {}


def test_rate_rejects_an_unknown_rating() -> None:
    result = runner.invoke(app, ["rate", "gpt-4o-mini", "sideways"])

    assert result.exit_code == 1
    assert "Unknown rating" in result.output


def test_health_reports_recorded_outcomes(monkeypatch) -> None:
    from basemode import health

    # The table is rendered by rich, which elides columns at the default test
    # terminal width; give it room so the assertions see real values.
    monkeypatch.setenv("COLUMNS", "200")
    health.record_outcome("openai/gpt-4o-mini", ok=True)
    health.record_outcome("openai/gpt-4o-mini", ok=False, category="rate_limit")

    listed = runner.invoke(app, ["health"])

    assert listed.exit_code == 0
    assert "openai/gpt-4o-mini" in listed.output
    assert "rate_limit" in listed.output


def test_health_for_an_unseen_model_exits_nonzero() -> None:
    result = runner.invoke(app, ["health", "gpt-4o-mini"])

    assert result.exit_code == 1
    assert "No generations recorded" in result.output


def test_health_clear_forgets_the_history() -> None:
    from basemode import health

    health.record_outcome("openai/gpt-4o-mini", ok=True)

    result = runner.invoke(app, ["health", "--clear"])

    assert result.exit_code == 0
    assert health.list_model_health() == {}


def test_health_json_is_machine_readable() -> None:
    import json

    from basemode import health

    health.record_outcome("openai/gpt-4o-mini", ok=False, category="timeout")

    result = runner.invoke(app, ["health", "--json"])

    payload = json.loads(result.output)
    assert payload["openai/gpt-4o-mini"]["failures"] == 1


def test_info_shows_the_rating_and_observed_health() -> None:
    from basemode import health
    from basemode.keys import set_model_rating

    set_model_rating("openai/gpt-4o-mini", 1)
    health.record_outcome("openai/gpt-4o-mini", ok=True)

    result = runner.invoke(app, ["info", "gpt-4o-mini"])

    assert "thumbs up" in result.output
    assert "1 attempts, no failures" in result.output

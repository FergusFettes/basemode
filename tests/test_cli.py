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


def _record_observation(observations, *, ok: bool) -> None:
    operation = observations.observe_operation(
        "openai/gpt-4o-mini", "system", "heuristic", None
    )
    attempt = operation.begin_attempt("initial")
    if ok:
        attempt.saw_content("safe count only")
        attempt.finish("success")
        operation.finish("success", returned_content=True)
        return

    class RateLimitError(RuntimeError):
        status_code = 429

    attempt.finish("failure", RateLimitError("not persisted"))
    operation.finish("failure", returned_content=False)


def test_health_reports_recorded_outcomes(monkeypatch) -> None:
    from basemode import observations

    # The table is rendered by rich, which elides columns at the default test
    # terminal width; give it room so the assertions see real values.
    monkeypatch.setenv("COLUMNS", "200")
    _record_observation(observations, ok=True)
    _record_observation(observations, ok=False)

    listed = runner.invoke(app, ["health"])

    assert listed.exit_code == 0
    assert "openai/gpt-4o-mini" in listed.output
    assert "rate_limit" in listed.output


def test_health_for_an_unseen_model_exits_nonzero() -> None:
    result = runner.invoke(app, ["health", "gpt-4o-mini"])

    assert result.exit_code == 1
    assert "No generations recorded" in result.output


def test_health_clear_forgets_the_history() -> None:
    from basemode import observations
    from basemode.observation_queries import list_endpoint_health

    _record_observation(observations, ok=True)

    result = runner.invoke(app, ["health", "--clear"])

    assert result.exit_code == 0
    assert list_endpoint_health() == {}


def test_health_json_is_machine_readable() -> None:
    import json

    from basemode import observations

    _record_observation(observations, ok=False)

    result = runner.invoke(app, ["health", "--json"])

    payload = json.loads(result.output)
    assert payload["openai/gpt-4o-mini"]["failures"] == {"rate_limit": 1}


def test_info_shows_the_rating_and_observed_health() -> None:
    from basemode import health
    from basemode.keys import set_model_rating

    set_model_rating("openai/gpt-4o-mini", 1)
    health.record_outcome("openai/gpt-4o-mini", ok=True)

    result = runner.invoke(app, ["info", "gpt-4o-mini"])

    assert "thumbs up" in result.output
    assert "1 attempts, no failures" in result.output


# -- extracted render helpers -----------------------------------------------


def test_score_color_thresholds() -> None:
    from basemode.cli.render import _score_color

    assert _score_color(0.9) == "green"
    assert _score_color(0.75) == "green"
    assert _score_color(0.5) == "yellow"
    assert _score_color(0.4) == "yellow"
    assert _score_color(0.1) == "red"


def test_preview_truncates_long_text_and_collapses_whitespace() -> None:
    from basemode.cli.render import _preview

    assert _preview("hello   world\n\nagain") == "hello world again"
    long_text = "x" * 100
    truncated = _preview(long_text, limit=10)
    assert truncated == "xxxxxxx..."
    assert len(truncated) == 10


def test_format_float() -> None:
    from basemode.cli.render import _format_float

    assert _format_float(1.0) == "1.00"
    assert _format_float(3.14159) == "3.14"


def test_health_summary_reports_no_failures() -> None:
    from basemode.cli.render import _health_summary

    assert _health_summary({"failure_rate": 0, "attempts": 5}) == (
        "5 attempts, no failures"
    )


def test_health_summary_reports_failure_details() -> None:
    from basemode.cli.render import _health_summary

    summary = _health_summary(
        {
            "failure_rate": 0.5,
            "attempts": 4,
            "failures": 2,
            "last_category": "timeout",
            "last_failure_at": "2026-01-01T00:00:00+00:00",
        }
    )
    assert summary == (
        "4 attempts, 2 failed (50%); last timeout at 2026-01-01T00:00:00+00:00"
    )


def test_display_id_strips_matching_provider_prefix() -> None:
    from basemode.cli.models_cmd import _display_id

    assert _display_id("openai", "openai/gpt-4o-mini") == "gpt-4o-mini"
    assert _display_id("openai", "anthropic/claude-opus-5") == "anthropic/claude-opus-5"


def test_run_streams_a_single_completion(monkeypatch) -> None:
    from basemode import continue_ as continue_module

    async def fake_continue_text(prefix, model, **kwargs):
        on_usage = kwargs.get("on_usage")
        if on_usage:
            on_usage([{"prompt_tokens": 3, "completion_tokens": 2}])
        for tok in [" and", " more"]:
            yield tok

    monkeypatch.setattr(continue_module, "continue_text", fake_continue_text)

    result = runner.invoke(
        app, ["run", "hello", "-M", "5", "--show-strategy", "--show-usage"]
    )

    assert result.exit_code == 0
    assert "and more" in result.output
    assert "strategy:" in result.output
    assert "Prompt tokens" in result.output


def test_run_reads_stdin_when_no_prefix_given(monkeypatch) -> None:
    from basemode import continue_ as continue_module

    async def fake_continue_text(prefix, model, **kwargs):
        yield "!"

    monkeypatch.setattr(continue_module, "continue_text", fake_continue_text)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdin.read", lambda: "piped text")

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 0
    assert "!" in result.output


def test_models_command_lists_entries() -> None:
    result = runner.invoke(app, ["models", "-p", "openai", "-s", "gpt-4o-mini"])

    assert result.exit_code == 0
    assert "gpt-4o-mini" in result.output


def test_models_command_json_output() -> None:
    import json

    result = runner.invoke(
        app, ["models", "-p", "openai", "-s", "gpt-4o-mini", "--json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert isinstance(payload, list)


def test_models_command_rejects_bad_since() -> None:
    result = runner.invoke(app, ["models", "--since", "not-a-duration"])

    assert result.exit_code == 1


def test_providers_command_lists_providers() -> None:
    result = runner.invoke(app, ["providers"])

    assert result.exit_code == 0
    assert "openai" in result.output


def test_rate_lists_none_rated_yet() -> None:
    result = runner.invoke(app, ["rate"])

    assert result.exit_code == 0
    assert "No models rated yet" in result.output


def test_health_verification_view(monkeypatch) -> None:
    monkeypatch.setenv("COLUMNS", "200")

    def fake_status():
        return {
            "openai/gpt-4o-mini": {
                "controlled_status": "reachable",
                "suite": "quick",
                "required_probes": 1,
                "successful_probes": 1,
                "attempts": 2,
                "failures": {"timeout": 1},
                "last_run_at": "2026-01-01T00:00:00+00:00",
            }
        }

    monkeypatch.setattr("basemode.cli.health_cmd.list_controlled_status", fake_status)

    result = runner.invoke(app, ["health", "--verification"])

    assert result.exit_code == 0
    assert "openai/gpt-4o-mini" in result.output
    assert "timeout" in result.output


def test_keys_set_list_and_get() -> None:
    result = runner.invoke(app, ["keys", "set", "openai", "sk-test-value"])
    assert result.exit_code == 0

    listed = runner.invoke(app, ["keys", "list"])
    assert listed.exit_code == 0
    assert "openai" in listed.output

    got = runner.invoke(app, ["keys", "get", "openai"])
    assert got.exit_code == 0
    assert "sk-test-value" in got.output

    bad = runner.invoke(app, ["keys", "bogus"])
    assert bad.exit_code == 1


def test_default_model_show_set_and_unset() -> None:
    none_set = runner.invoke(app, ["default"])
    assert none_set.exit_code == 0
    assert "No default model set" in none_set.output

    set_result = runner.invoke(app, ["default", "gpt-4o-mini"])
    assert set_result.exit_code == 0
    assert "Default model set" in set_result.output

    shown = runner.invoke(app, ["default"])
    assert "gpt-4o-mini" in shown.output

    unset = runner.invoke(app, ["default", "--unset"])
    assert unset.exit_code == 0
    assert "cleared" in unset.output.lower()


def test_verify_dry_run_prints_plan_table(monkeypatch) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    from basemode.verification_plan import PlannedTarget, VerificationPlan

    plan = VerificationPlan(
        suite="quick",
        targets=(
            PlannedTarget(
                model="openai/gpt-4o-mini",
                provider="openai",
                stage="never-tested",
                prior_status="never-tested",
                catalog_available=True,
                release_date="2026-01-01",
                last_checked_at=None,
                logical_probes=2,
                maximum_requests=2,
                estimated_max_cost_usd=0.001,
            ),
        ),
        logical_probes=2,
        maximum_requests=2,
        provider_counts={"openai": 1},
        estimated_known_max_cost_usd=0.001,
        priced_targets=1,
        unpriced_targets=0,
    )
    monkeypatch.setattr(
        "basemode.verification_plan.plan_verification", lambda *a, **kw: plan
    )

    result = runner.invoke(app, ["verify", "openai/gpt-4o-mini", "--dry-run"])

    assert result.exit_code == 0
    assert "openai/gpt-4o-mini" in result.output
    assert "targets" in result.output


def test_verify_plan_table_renders_a_row_per_target() -> None:
    from basemode.cli.verify_cmd import _verify_plan_table
    from basemode.verification_plan import PlannedTarget, VerificationPlan

    plan = VerificationPlan(
        suite="quick",
        targets=(
            PlannedTarget(
                model="openai/gpt-4o-mini",
                provider="openai",
                stage="never-tested",
                prior_status="never-tested",
                catalog_available=True,
                release_date="2026-01-01",
                last_checked_at=None,
                logical_probes=2,
                maximum_requests=2,
                estimated_max_cost_usd=0.001,
            ),
            PlannedTarget(
                model="anthropic/claude-opus-5",
                provider="anthropic",
                stage="stale",
                prior_status="verified",
                catalog_available=None,
                release_date=None,
                last_checked_at="2026-01-01T00:00:00+00:00",
                logical_probes=1,
                maximum_requests=1,
                estimated_max_cost_usd=None,
            ),
        ),
        logical_probes=3,
        maximum_requests=3,
        provider_counts={"openai": 1, "anthropic": 1},
        estimated_known_max_cost_usd=0.001,
        priced_targets=1,
        unpriced_targets=1,
    )

    table = _verify_plan_table(plan)

    assert table.row_count == 2
    from rich.console import Console

    console = Console(width=200, record=True)
    console.print(table)
    rendered = console.export_text()
    assert "openai/gpt-4o-mini" in rendered
    assert "anthropic/claude-opus-5" in rendered
    assert "yes" in rendered
    assert "2026-01-01" in rendered
    assert "?" in rendered

import pytest

from basemode import bench
from basemode.bench import ProbeResult, StrategyResult, bench_model, rank, winner
from basemode.scoring import score_continuation

PREFIX = "The ship rounded the headland and"


def _probe(text: str, *, elapsed: float = 1.0, error: str | None = None) -> ProbeResult:
    return ProbeResult(
        prefix=PREFIX,
        text=text,
        score=score_continuation(PREFIX, text),
        elapsed_s=elapsed,
        error=error,
    )


def test_strategy_result_aggregates_score_and_flags() -> None:
    result = StrategyResult(
        strategy="system",
        probes=[
            _probe(" the harbour opened out.", elapsed=1.0),
            _probe("Sure! Here you go:", elapsed=3.0),
        ],
    )

    assert result.score == 0.7  # mean of a clean 1.0 and a chatty 0.4
    assert result.flags == ("preamble",)
    assert result.mean_elapsed_s == 2.0
    assert result.sample == "the harbour opened out."


def test_flags_are_ordered_by_frequency() -> None:
    result = StrategyResult(
        strategy="system",
        probes=[
            _probe("Sure! Here you go:"),
            _probe("Certainly, I will do that."),
            _probe('"quoted continuation"'),
        ],
    )

    assert result.flags[0] == "preamble"
    assert "quoted" in result.flags


def test_empty_strategy_result_is_zero_not_an_error() -> None:
    result = StrategyResult(strategy="system")

    assert result.score == 0.0
    assert result.mean_elapsed_s == 0.0
    assert result.sample == ""
    assert result.errors == ()


def test_errors_are_collected() -> None:
    result = StrategyResult(
        strategy="prefill",
        probes=[_probe("", error="BadRequestError: prefill not allowed")],
    )

    assert result.score == 0.0
    assert result.errors == ("BadRequestError: prefill not allowed",)
    assert result.as_dict()["errors"] == ["BadRequestError: prefill not allowed"]


def test_rank_orders_by_score_then_speed() -> None:
    slow_clean = StrategyResult(
        "few_shot", [_probe(" the fog closed in.", elapsed=9.0)]
    )
    fast_clean = StrategyResult("system", [_probe(" the fog closed in.", elapsed=1.0)])
    chatty = StrategyResult("prefill", [_probe("Sure! Here you go:", elapsed=0.1)])

    assert [r.strategy for r in rank([chatty, slow_clean, fast_clean])] == [
        "system",
        "few_shot",
        "prefill",
    ]


def test_winner_is_none_when_nothing_worked() -> None:
    dead = StrategyResult("system", [_probe("", error="boom")])

    assert winner([dead]) is None
    assert winner([]) is None


def test_winner_picks_the_top_ranked() -> None:
    good = StrategyResult("few_shot", [_probe(" the fog closed in.")])
    bad = StrategyResult("system", [_probe("Sure! Here you go:")])

    assert winner([bad, good]).strategy == "few_shot"


async def test_bench_model_runs_every_strategy_and_ranks(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    async def fake_continue_text(prefix, model, *, max_tokens, temperature, strategy):
        calls.append((strategy, prefix))
        text = " the fog closed in." if strategy == "few_shot" else "Sure! Here you go:"
        for token in text.split(" "):
            yield token + " "

    monkeypatch.setattr(
        bench,
        "_run_probe",
        _probe_via(fake_continue_text),
    )

    results = await bench_model(
        "openai/gpt-4o-mini",
        strategies=["system", "few_shot"],
        probes=[PREFIX],
        max_tokens=10,
    )

    assert [r.strategy for r in results] == ["few_shot", "system"]
    assert results[0].score > results[1].score
    assert {strategy for strategy, _ in calls} == {"system", "few_shot"}


def _probe_via(fake_continue_text):
    async def _run(model, strategy, prefix, *, max_tokens, temperature):
        text = "".join(
            [
                token
                async for token in fake_continue_text(
                    prefix,
                    model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    strategy=strategy,
                )
            ]
        )
        return ProbeResult(
            prefix=prefix,
            text=text,
            score=score_continuation(prefix, text),
            elapsed_s=0.1,
        )

    return _run


async def test_probe_records_provider_errors_without_raising(monkeypatch) -> None:
    async def exploding(*args, **kwargs):
        raise RuntimeError("provider rejected prefill")
        yield  # pragma: no cover

    monkeypatch.setattr("basemode.continue_.continue_text", exploding)

    result = await bench._run_probe(
        "anthropic/claude-opus-5",
        "prefill",
        PREFIX,
        max_tokens=10,
        temperature=1.0,
    )

    assert result.error == "RuntimeError: provider rejected prefill"
    assert result.score.score == 0.0


async def test_probe_times_out_cleanly(monkeypatch) -> None:
    import asyncio

    async def stalling(*args, **kwargs):
        await asyncio.sleep(10)
        yield ""  # pragma: no cover

    monkeypatch.setattr("basemode.continue_.continue_text", stalling)
    monkeypatch.setattr(bench, "PROBE_TIMEOUT", 0.01)

    result = await bench._run_probe(
        "openai/gpt-4o-mini", "system", PREFIX, max_tokens=10, temperature=1.0
    )

    assert result.error is not None
    assert "timed out" in result.error


@pytest.mark.parametrize("strategy", bench.DEFAULT_STRATEGIES)
def test_default_strategies_are_real_strategies(strategy: str) -> None:
    from basemode.strategies import REGISTRY

    assert strategy in REGISTRY

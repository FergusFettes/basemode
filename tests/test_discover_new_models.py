import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import discover_new_models as dnm  # noqa: E402

from basemode.live_models import LiveModel  # noqa: E402


def test_openai_blacklist_filters_non_text_models() -> None:
    assert dnm._is_openai_blacklisted("text-embedding-3-large")
    assert dnm._is_openai_blacklisted("whisper-1")
    assert dnm._is_openai_blacklisted("gpt-4o-realtime-preview")
    assert not dnm._is_openai_blacklisted("gpt-6-mini")


def test_probe_judgement_uses_the_shared_continuation_scorer() -> None:
    prefix = dnm.PROBE_PREFIX

    ok, reason = dnm.looks_clean(prefix, "")
    assert not ok
    assert "empty" in reason

    ok, reason = dnm.looks_clean(prefix, "Sure, here's a continuation:")
    assert not ok
    assert "preamble" in reason

    assert dnm.looks_clean(prefix, " dog. It was a sunny day.") == (True, None)


def _live(id_: str, release_date: str | None) -> LiveModel:
    return LiveModel(
        id=id_, release_date=release_date, release_date_confidence="release"
    )


def test_select_candidates_dedupes_sorts_and_caps() -> None:
    raw = [
        _live("gpt-5.9-mini", "2026-01-01"),
        _live("gpt-5.8", "2026-03-01"),
        _live("gpt-5.7", "2026-02-01"),
        _live("text-embedding-3-large", "2026-04-01"),
        _live("gpt-4o-mini", "2026-05-01"),  # already known
    ]
    candidates = dnm.select_candidates(
        provider="openai",
        raw_models=raw,
        known={"openai/gpt-4o-mini"},
        rejected=set(),
        limit=2,
    )

    assert [c.normalized_id for c in candidates] == ["openai/gpt-5.8", "openai/gpt-5.7"]


def test_select_candidates_skips_previously_rejected() -> None:
    raw = [_live("gpt-5.7", "2026-01-01")]
    candidates = dnm.select_candidates(
        provider="openai",
        raw_models=raw,
        known=set(),
        rejected={"openai/gpt-5.7"},
        limit=10,
    )
    assert candidates == []


def test_release_date_ts_parses_iso_date() -> None:
    assert dnm._release_date_ts("2026-04-20") > 0
    assert dnm._release_date_ts(None) == 0.0
    assert dnm._release_date_ts("not-a-date") == 0.0


def test_guess_openrouter_id_matches_stem() -> None:
    index = {"anthropic/claude-opus-4.9": {}, "openai/gpt-4o-mini": {}}
    assert (
        dnm._guess_openrouter_id("anthropic/claude-opus-4-9", index)
        == "anthropic/claude-opus-4.9"
    )
    assert dnm._guess_openrouter_id("openai/gpt-9-nonexistent", index) is None


async def test_run_adds_working_and_rejects_broken_models(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(dnm, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(dnm, "REJECTED_PATH", tmp_path / "rejected.json")
    monkeypatch.setattr(dnm, "SUMMARY_PATH", tmp_path / "summary.md")
    # Don't let a real ~/.config/basemode/auth.json leak keys into this test.
    monkeypatch.setattr(dnm, "load_into_environ", lambda: None)

    monkeypatch.setattr(dnm.settings, "api_key_for", lambda provider: "test-key")

    def fake_fetch_live_models(provider, api_key):
        if provider != "openai":
            raise dnm.LiveModelsError(f"skip {provider}")
        return [
            _live("gpt-9-good", "2026-02-01"),
            _live("gpt-9-bad", "2026-01-01"),
        ]

    monkeypatch.setattr(dnm, "fetch_live_models", fake_fetch_live_models)
    monkeypatch.setattr(dnm, "_fetch_openrouter_index", lambda: {})

    async def fake_probe(candidate):
        if candidate.normalized_id == "openai/gpt-9-good":
            return True, "system", "ok", {"prefill": "empty continuation"}, False
        return (
            False,
            None,
            "empty continuation",
            {"system": "empty continuation"},
            False,
        )

    monkeypatch.setattr(dnm, "_probe", fake_probe)

    result = await dnm._run(limit=10, providers={"openai"})
    assert result == 0

    registry = json.loads((tmp_path / "registry.json").read_text())
    assert [m["model"] for m in registry["models"]] == ["openai/gpt-9-good"]
    assert registry["models"][0]["prompt_method"] == "system"

    rejected = json.loads((tmp_path / "rejected.json").read_text())
    assert [m["model"] for m in rejected["models"]] == ["openai/gpt-9-bad"]
    assert rejected["models"][0]["attempted_strategies"] == {
        "system": "empty continuation"
    }

    summary = (tmp_path / "summary.md").read_text()
    assert "openai/gpt-9-good" in summary
    assert "openai/gpt-9-bad" in summary


async def test_probe_falls_back_through_strategies(monkeypatch) -> None:
    candidate = dnm.Candidate("anthropic", "claude-x", "anthropic/claude-x", 0)
    monkeypatch.setattr(
        dnm, "detect_strategy", lambda model: SimpleNamespace(name="prefill")
    )

    attempted: list[str | None] = []

    async def fake_probe_strategy(candidate, strategy):
        attempted.append(strategy)
        if strategy == "system":
            return True, "ok", " it worked", False
        return False, "BadRequestError: no prefill support", "", False

    monkeypatch.setattr(dnm, "_probe_strategy", fake_probe_strategy)

    worked, strategy_used, detail, attempts, needs_budget = await dnm._probe(candidate)

    assert worked is True
    assert strategy_used == "system"
    assert detail == "ok"
    assert needs_budget is False
    # Auto attempt (None) plus "system" fallback; "prefill" fallback skipped
    # because detect_strategy already resolved to "prefill".
    assert attempted == [None, "system"]
    assert attempts == {"prefill": "BadRequestError: no prefill support"}


async def test_probe_reports_all_failures_when_every_strategy_fails(
    monkeypatch,
) -> None:
    candidate = dnm.Candidate("anthropic", "claude-x", "anthropic/claude-x", 0)
    monkeypatch.setattr(
        dnm, "detect_strategy", lambda model: SimpleNamespace(name="prefill")
    )

    async def fake_probe_strategy(candidate, strategy):
        return False, f"failed for {strategy}", "", False

    monkeypatch.setattr(dnm, "_probe_strategy", fake_probe_strategy)

    worked, strategy_used, detail, attempts, needs_budget = await dnm._probe(candidate)

    assert worked is False
    assert strategy_used is None
    assert detail == "failed for None"
    assert needs_budget is False
    assert set(attempts) == {"prefill", "system", "few_shot"}


async def test_probe_strategy_retries_empty_output_with_wider_reasoning_budget(
    monkeypatch,
) -> None:
    """A reasoning model that only produces text once given a bigger budget
    should be accepted and flagged, not rejected as broken."""
    candidate = dnm.Candidate("anthropic", "claude-x", "anthropic/claude-x", 0)

    calls: list[int] = []

    async def fake_collect_chunks(
        candidate, strategy, *, max_tokens=dnm.PROBE_MAX_TOKENS
    ):
        calls.append(max_tokens)
        if max_tokens == dnm.PROBE_MAX_TOKENS:
            return []  # starved by hidden reasoning tokens
        return [" a clean continuation."]

    monkeypatch.setattr(dnm, "_collect_chunks", fake_collect_chunks)

    worked, _detail, text, needs_budget = await dnm._probe_strategy(candidate, None)

    assert worked is True
    assert needs_budget is True
    assert text.strip() == "a clean continuation."
    assert calls == [dnm.PROBE_MAX_TOKENS, dnm.REASONING_RETRY_MAX_TOKENS]


async def test_probe_strategy_stays_rejected_when_wider_budget_still_empty(
    monkeypatch,
) -> None:
    candidate = dnm.Candidate("anthropic", "claude-x", "anthropic/claude-x", 0)

    async def fake_collect_chunks(
        candidate, strategy, *, max_tokens=dnm.PROBE_MAX_TOKENS
    ):
        return []

    monkeypatch.setattr(dnm, "_collect_chunks", fake_collect_chunks)

    worked, detail, _text, needs_budget = await dnm._probe_strategy(candidate, None)

    assert worked is False
    assert needs_budget is False
    assert "empty" in detail


async def test_run_tags_reasoning_budget_quirk(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dnm, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(dnm, "REJECTED_PATH", tmp_path / "rejected.json")
    monkeypatch.setattr(dnm, "SUMMARY_PATH", tmp_path / "summary.md")
    monkeypatch.setattr(dnm, "load_into_environ", lambda: None)
    monkeypatch.setattr(dnm.settings, "api_key_for", lambda provider: "test-key")

    def fake_fetch_live_models(provider, api_key):
        if provider != "openai":
            raise dnm.LiveModelsError(f"skip {provider}")
        return [_live("gpt-9-thinker", "2026-02-01")]

    monkeypatch.setattr(dnm, "fetch_live_models", fake_fetch_live_models)
    monkeypatch.setattr(dnm, "_fetch_openrouter_index", lambda: {})

    async def fake_probe(candidate):
        return True, None, "ok (needed a reasoning budget)", {}, True

    monkeypatch.setattr(dnm, "_probe", fake_probe)

    result = await dnm._run(limit=10, providers={"openai"})
    assert result == 0

    registry = json.loads((tmp_path / "registry.json").read_text())
    assert registry["models"][0]["quirks"] == ["reasoning_budget"]

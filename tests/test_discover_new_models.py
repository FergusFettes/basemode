import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import discover_new_models as dnm  # noqa: E402


def test_openai_blacklist_filters_non_text_models() -> None:
    assert dnm._is_openai_blacklisted("text-embedding-3-large")
    assert dnm._is_openai_blacklisted("whisper-1")
    assert dnm._is_openai_blacklisted("gpt-4o-realtime-preview")
    assert not dnm._is_openai_blacklisted("gpt-6-mini")


def test_looks_clean_rejects_empty_and_chatty() -> None:
    assert dnm._looks_clean("") == (False, "empty continuation")
    ok, reason = dnm._looks_clean("Sure, here's a continuation:")
    assert not ok
    assert "preamble" in reason
    assert dnm._looks_clean(" dog. It was a sunny day.") == (True, None)


def test_select_candidates_dedupes_sorts_and_caps() -> None:
    raw = [
        {"id": "gpt-5.9-mini", "created": 100},
        {"id": "gpt-5.8", "created": 300},
        {"id": "gpt-5.7", "created": 200},
        {"id": "text-embedding-3-large", "created": 999},
        {"id": "gpt-4o-mini", "created": 500},  # already known
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
    raw = [{"id": "gpt-5.7", "created": 1}]
    candidates = dnm.select_candidates(
        provider="openai",
        raw_models=raw,
        known=set(),
        rejected={"openai/gpt-5.7"},
        limit=10,
    )
    assert candidates == []


def test_select_candidates_skips_ids_normalize_resolves_to_other_provider() -> None:
    # A bare Anthropic id fed into the openai selection pass shouldn't
    # get pulled in just because normalize_model recognized it.
    raw = [{"id": "claude-opus-4-9", "created": 1}]
    candidates = dnm.select_candidates(
        provider="openai",
        raw_models=raw,
        known=set(),
        rejected=set(),
        limit=10,
    )
    assert candidates == []


def test_anthropic_created_parses_iso_timestamp() -> None:
    assert dnm._anthropic_created({"created_at": "2026-04-20T00:00:00Z"}) > 0
    assert dnm._anthropic_created({}) == 0.0
    assert dnm._anthropic_created({"created_at": "not-a-date"}) == 0.0


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

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    monkeypatch.setattr(
        dnm,
        "_fetch_openai_models",
        lambda api_key: [
            {"id": "gpt-9-good", "created": 200},
            {"id": "gpt-9-bad", "created": 100},
        ],
    )
    monkeypatch.setattr(dnm, "_fetch_openrouter_index", lambda: {})

    async def fake_probe(candidate):
        if candidate.normalized_id == "openai/gpt-9-good":
            return True, "ok", " a fine continuation"
        return False, "empty continuation", ""

    monkeypatch.setattr(dnm, "_probe", fake_probe)

    result = await dnm._run(limit=10)
    assert result == 0

    registry = json.loads((tmp_path / "registry.json").read_text())
    assert [m["model"] for m in registry["models"]] == ["openai/gpt-9-good"]

    rejected = json.loads((tmp_path / "rejected.json").read_text())
    assert [m["model"] for m in rejected["models"]] == ["openai/gpt-9-bad"]

    summary = (tmp_path / "summary.md").read_text()
    assert "openai/gpt-9-good" in summary
    assert "openai/gpt-9-bad" in summary

import json
import sqlite3
from datetime import UTC, datetime

import pytest

from basemode import evidence


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("openrouter/google/lyria-3-clip-preview", "lyria"),
        ("deepinfra/baai/bge-m3", "bge"),
        ("deepinfra/google/embeddinggemma-300m", "embeddinggemma"),
        ("deepinfra/hexgrad/kokoro-82m", "kokoro"),
        ("openai/gpt-audio-mini", "audio"),
        ("deepinfra/bria/remove_background", "bria"),
        ("deepinfra/google/nano-banana-pro", "nano-banana"),
        ("together_ai/bytedance-seed/seedream-4.0", "seedream"),
        ("deepinfra/sentence-transformers/all-minilm-l6-v2", "sentence-transformers"),
        ("groq/canopylabs/orpheus-v1-english", "orpheus"),
        ("openai/gpt-realtime-2.1", "realtime"),
        ("openai/gpt-4o-transcribe", "transcribe"),
        ("openrouter/nvidia/nemotron-content-safety:free", "content-safety"),
        ("together_ai/deepseek-ai/deepseek-ocr-2", "ocr"),
        ("deepinfra/stabilityai/sdxl-turbo", "sdxl"),
    ],
)
def test_text_classification_excludes_non_text_families(
    model: str, family: str
) -> None:
    assert evidence.classify_text_endpoint(model) == (
        False,
        f"non-text model family: {family}",
    )


def test_text_classification_keeps_text_reasoning_models() -> None:
    assert evidence.classify_text_endpoint("openrouter/meta/muse-spark-1.2") == (
        True,
        None,
    )
    assert evidence.classify_text_endpoint("openrouter/sakana/fugu-ultra") == (
        True,
        None,
    )


def test_schema_and_thorough_status_are_durable() -> None:
    with evidence.connect() as db:
        run = evidence.start_run("thorough", conn=db)
        evidence.record_attempt(
            run,
            "openrouter/example/model",
            probe_kind="continuation",
            attempt_number=1,
            outcome="success",
            conn=db,
            request_params={"max_tokens": 160},
        )
        evidence.finish_run(run, conn=db)
        assert db.execute("PRAGMA user_version").fetchone()[0] == 4
        assert evidence.current_status(conn=db)["openrouter/example/model"] == {
            "reachable": True,
            "last_checked_at": db.execute(
                "SELECT finished_at FROM verification_attempts"
            ).fetchone()[0],
            "currently_broken": False,
            "transient_failure": False,
            "verified": True,
        }


def test_transient_failure_is_selected_for_recheck() -> None:
    with evidence.connect() as db:
        run = evidence.start_run("quick", conn=db)
        evidence.record_attempt(
            run,
            "groq/flaky",
            probe_kind="continuation",
            attempt_number=1,
            outcome="failure",
            failure_class="rate_limit",
            conn=db,
        )
        evidence.finish_run(run, conn=db)
        assert evidence.transient_recheck_models(conn=db) == []
        assert evidence.transient_recheck_models(
            conn=db, now="9999-01-01T00:00:00+00:00"
        ) == ["groq/flaky"]
        status = evidence.current_status(conn=db)["groq/flaky"]
        assert status["transient_failure"] is True
        assert status["currently_broken"] is False
        assert status["operational_status"] == "suspected_transient"
        assert status["next_check_at"] is not None


def test_v3_migration_backfills_existing_transient_queue(tmp_path) -> None:
    path = tmp_path / "legacy-v3.sqlite"
    with evidence.connect(path) as db:
        run = evidence.start_run("quick", conn=db)
        evidence.record_attempt(
            run,
            "groq/flaky",
            probe_kind="continuation",
            attempt_number=1,
            outcome="failure",
            failure_class="rate_limit",
            finished_at="2026-08-01T00:00:00+00:00",
            conn=db,
        )
        evidence.finish_run(run, conn=db)
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE recheck_schedules")
        db.execute("PRAGMA user_version=3")
    with evidence.connect(path) as db:
        state = evidence.recheck_statuses(conn=db)["groq/flaky"]
        assert state["next_check_at"] == "2026-08-01T00:15:00+00:00"


def test_aborted_run_does_not_change_recheck_schedule() -> None:
    with evidence.connect() as db:
        run = evidence.start_run("quick", conn=db)
        evidence.record_attempt(
            run,
            "groq/interrupted",
            probe_kind="continuation",
            attempt_number=1,
            outcome="failure",
            failure_class="timeout",
            conn=db,
        )
        evidence.finish_run(run, "aborted", conn=db)
        assert "groq/interrupted" not in evidence.recheck_statuses(conn=db)


def _failure_run(db, model: str, at: str, failure_class: str) -> None:
    run = evidence.start_run("transient-recheck", conn=db)
    evidence.record_attempt(
        run,
        model,
        probe_kind="continuation",
        attempt_number=1,
        outcome="failure",
        failure_class=failure_class,
        finished_at=at,
        conn=db,
    )
    evidence.finish_run(run, conn=db)


def test_recheck_backoff_becomes_persistent_only_across_separate_runs() -> None:
    with evidence.connect() as db:
        first = evidence.start_run("quick", conn=db)
        for number in (1, 2):
            evidence.record_attempt(
                first,
                "groq/flaky",
                probe_kind="continuation",
                attempt_number=number,
                outcome="failure",
                failure_class="timeout",
                finished_at="2026-08-01T00:00:00+00:00",
                conn=db,
            )
        evidence.finish_run(first, conn=db)
        state = evidence.recheck_statuses(conn=db)["groq/flaky"]
        assert state["consecutive_operational_failures"] == 1
        assert state["next_check_at"] == "2026-08-01T00:15:00+00:00"

        _failure_run(db, "groq/flaky", "2026-08-01T02:00:00+00:00", "timeout")
        state = evidence.recheck_statuses(conn=db)["groq/flaky"]
        assert state["consecutive_operational_failures"] == 2
        assert state["next_check_at"] == "2026-08-01T04:00:00+00:00"

        _failure_run(db, "groq/flaky", "2026-08-02T04:00:00+00:00", "timeout")
        state = evidence.recheck_statuses(conn=db)["groq/flaky"]
        assert state["operational_status"] == "persistent_operational"
        assert state["next_check_at"] == "2026-08-09T04:00:00+00:00"


def test_success_marks_operational_failure_recovered() -> None:
    with evidence.connect() as db:
        _failure_run(db, "groq/flaky", "2026-08-01T00:00:00+00:00", "rate_limit")
        run = evidence.start_run("transient-recheck", conn=db)
        evidence.record_attempt(
            run,
            "groq/flaky",
            probe_kind="continuation",
            attempt_number=1,
            outcome="success",
            finished_at="2026-08-01T00:16:00+00:00",
            conn=db,
        )
        evidence.finish_run(run, conn=db)
        state = evidence.recheck_statuses(conn=db)["groq/flaky"]
        assert state["operational_status"] == "recovered"
        assert state["next_check_at"] is None


def test_authentication_failure_is_account_limited_and_not_scheduled() -> None:
    with evidence.connect() as db:
        _failure_run(
            db, "anthropic/private", "2026-08-01T00:00:00+00:00", "authentication"
        )
        state = evidence.recheck_statuses(conn=db)["anthropic/private"]
        assert state["operational_status"] == "account_limited"
        assert state["next_check_at"] is None


def test_quota_shaped_rate_limit_is_account_limited() -> None:
    with evidence.connect() as db:
        run = evidence.start_run("quick", conn=db)
        evidence.record_attempt(
            run,
            "openai/no-credit",
            probe_kind="continuation",
            attempt_number=1,
            outcome="failure",
            failure_class="rate_limit",
            safe_error_code="insufficient_quota",
            conn=db,
        )
        evidence.finish_run(run, conn=db)
        state = evidence.recheck_statuses(conn=db)["openai/no-credit"]
        assert state["operational_status"] == "account_limited"


def test_persistent_failed_route_is_distinguished_from_working_route() -> None:
    with evidence.connect() as db:
        evidence.ensure_endpoint(
            "openrouter/vendor/model", model_family_id="vendor-model", conn=db
        )
        evidence.ensure_endpoint(
            "vendor/model", model_family_id="vendor-model", conn=db
        )
        good = evidence.start_run("quick", conn=db)
        evidence.record_attempt(
            good,
            "vendor/model",
            probe_kind="continuation",
            attempt_number=1,
            outcome="success",
            conn=db,
        )
        evidence.finish_run(good, conn=db)
        for day in (1, 2, 3):
            _failure_run(
                db,
                "openrouter/vendor/model",
                f"2026-08-0{day}T00:00:00+00:00",
                "provider_unavailable",
            )
        state = evidence.recheck_statuses(conn=db)["openrouter/vendor/model"]
        assert state["operational_status"] == "provider_route_unavailable"


def test_catalog_availability_is_an_independent_status() -> None:
    evidence.record_catalog_observation(
        "deepinfra/new", source="provider_api", available=True
    )
    assert evidence.current_status()["deepinfra/new"]["available"] is True


def test_import_sweep_is_idempotent(tmp_path) -> None:
    source = tmp_path / "sweep.jsonl"
    source.write_text(json.dumps({"model": "deepinfra/new", "ok": True}) + "\n")
    first = evidence.import_sweep_jsonl(source)
    second = evidence.import_sweep_jsonl(source)
    assert first == second
    with evidence.connect() as db:
        assert (
            db.execute("SELECT count(*) FROM verification_attempts").fetchone()[0] == 1
        )


def test_import_sweep_preserves_claude_timing_and_safe_error_fields(tmp_path) -> None:
    source = tmp_path / "sweep.jsonl"
    source.write_text(
        json.dumps(
            {
                "model": "deepinfra/new",
                "ok": False,
                "elapsed_s": 1.25,
                "category": "invalid_request",
                "status": 422,
                "code": "unsupported_parameter",
                "param": "thinking",
            }
        )
        + "\n"
    )
    evidence.import_sweep_jsonl(source)
    with evidence.connect() as db:
        row = db.execute("SELECT * FROM verification_attempts").fetchone()
    assert row["latency_ms"] == 1250
    assert row["http_status"] == 422
    assert row["safe_error_code"] == "unsupported_parameter"
    assert row["safe_error_parameter"] == "thinking"
    assert (
        row["finished_at"]
        == datetime.fromtimestamp(source.stat().st_mtime, UTC).isoformat()
    )


def test_import_provider_status_is_not_stored_as_an_http_status(tmp_path) -> None:
    source = tmp_path / "provider.jsonl"
    source.write_text(
        json.dumps({"model": "openai/old", "status": "xfail_retired_model"}) + "\n"
    )
    evidence.import_provider_health_jsonl(source)
    with evidence.connect() as db:
        row = db.execute("SELECT * FROM verification_attempts").fetchone()
    assert row["failure_class"] == "xfail_retired_model"
    assert row["http_status"] is None


def test_import_provider_history_preserves_run_timestamp(tmp_path) -> None:
    source = tmp_path / "provider.jsonl"
    source.write_text(
        json.dumps(
            {
                "model": "openai/model",
                "status": "ok",
                "run_at": "2026-08-19T12:00:00+00:00",
            }
        )
        + "\n"
    )
    evidence.import_provider_health_jsonl(source)
    with evidence.connect() as db:
        row = db.execute("SELECT finished_at FROM verification_attempts").fetchone()
    assert row[0] == "2026-08-19T12:00:00+00:00"


def test_import_live_catalog_cache_is_idempotent(tmp_path) -> None:
    source = tmp_path / "live.json"
    source.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-08-24T00:00:00Z",
                "providers": {
                    "deepinfra": {
                        "models": {"org/new-model": "2026-08-01"},
                        "reliable_dates": True,
                    }
                },
            }
        )
    )
    assert evidence.import_live_catalog_cache(source) == 1
    assert evidence.import_live_catalog_cache(source) == 0
    status = evidence.current_status()["deepinfra/org/new-model"]
    assert status["available"] is True


def test_structured_catalog_cache_preserves_capabilities_and_uses_text_output(
    tmp_path,
) -> None:
    source = tmp_path / "live-structured.json"
    source.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-08-24T00:00:00Z",
                "providers": {
                    "openrouter": {
                        "models": {
                            "vendor/vision-chat": {
                                "release_date": "2026-08-01",
                                "input_modalities": ["text", "image"],
                                "output_modalities": ["text"],
                                "supported_parameters": ["temperature"],
                            },
                            "vendor/image-only": {
                                "input_modalities": ["text"],
                                "output_modalities": ["image"],
                            },
                        },
                        "reliable_dates": True,
                    }
                },
            }
        )
    )
    assert evidence.import_live_catalog_cache(source) == 2
    with evidence.connect() as db:
        rows = db.execute(
            "SELECT normalized_model_id,text_eligible FROM model_endpoints ORDER BY normalized_model_id"
        ).fetchall()
        metadata = json.loads(
            db.execute(
                "SELECT metadata_json FROM catalog_observations c "
                "JOIN model_endpoints e ON e.id=c.endpoint_id "
                "WHERE e.normalized_model_id='openrouter/vendor/vision-chat'"
            ).fetchone()[0]
        )
    assert [tuple(row) for row in rows] == [
        ("openrouter/vendor/image-only", 0),
        ("openrouter/vendor/vision-chat", 1),
    ]
    assert metadata["input_modalities"] == ["text", "image"]
    assert metadata["supported_parameters"] == ["temperature"]


def test_explicit_text_output_can_correct_older_name_heuristic() -> None:
    evidence.ensure_endpoint("openrouter/vendor/image-specialist")
    evidence.record_catalog_observation(
        "openrouter/vendor/image-specialist",
        source="provider",
        available=True,
        metadata={"input_modalities": ["image"], "output_modalities": ["text"]},
    )
    with evidence.connect() as db:
        row = db.execute(
            "SELECT modality,text_eligible,exclusion_reason FROM model_endpoints"
        ).fetchone()
    assert tuple(row) == ("text", 1, None)


def test_import_verified_registry_retains_intent_without_granting_success(
    tmp_path,
) -> None:
    source = tmp_path / "registry.json"
    source.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": "anthropic/claude-test",
                        "prompt_method": "prefill",
                        "quirks": ["no_temperature"],
                    }
                ]
            }
        )
    )
    assert evidence.import_verified_registry(source) == 1
    assert evidence.import_verified_registry(source) == 0
    with evidence.connect() as db:
        annotation = db.execute(
            "SELECT value_json FROM model_annotations WHERE kind='registry_intent'"
        ).fetchone()
    assert json.loads(annotation[0]) == {
        "prompt_method": "prefill",
        "quirks": ["no_temperature"],
    }
    assert "anthropic/claude-test" not in evidence.current_status()


def test_import_legacy_health_verification_only(tmp_path) -> None:
    source = tmp_path / "health.sqlite"
    db = sqlite3.connect(source)
    db.execute("""CREATE TABLE model_events(model TEXT,timestamp TEXT,ok INTEGER,
        category TEXT,status INTEGER,error_code TEXT,error_param TEXT,source TEXT,cost_usd REAL)""")
    db.execute(
        "INSERT INTO model_events VALUES(?,?,?,?,?,?,?,?,?)",
        (
            "openai/test",
            "2026-01-01T00:00:00+00:00",
            1,
            None,
            None,
            None,
            None,
            "verification",
            0.01,
        ),
    )
    db.execute(
        "INSERT INTO model_events VALUES(?,?,?,?,?,?,?,?,?)",
        (
            "openai/test",
            "2026-01-01T00:00:00+00:00",
            0,
            "timeout",
            None,
            None,
            None,
            "generation",
            None,
        ),
    )
    db.commit()
    db.close()
    evidence.import_health_sqlite(source)
    with evidence.connect() as target:
        assert (
            target.execute("SELECT count(*) FROM verification_attempts").fetchone()[0]
            == 1
        )


def test_import_real_legacy_health_at_column(tmp_path) -> None:
    source = tmp_path / "health.sqlite"
    db = sqlite3.connect(source)
    db.execute("""CREATE TABLE model_events(model TEXT,at TEXT,ok INTEGER,
        category TEXT,status INTEGER,error_code TEXT,error_param TEXT,source TEXT,cost_usd REAL)""")
    db.execute(
        "INSERT INTO model_events VALUES(?,?,?,?,?,?,?,?,?)",
        (
            "openai/test",
            "2026-08-24T18:00:00+00:00",
            1,
            None,
            None,
            None,
            None,
            "verification",
            0.01,
        ),
    )
    db.commit()
    db.close()
    evidence.import_health_sqlite(source)
    with evidence.connect() as target:
        row = target.execute("SELECT started_at FROM verification_attempts").fetchone()
    assert row[0] == "2026-08-24T18:00:00+00:00"


def test_corpus_adapter_stores_only_aggregates() -> None:
    assert (
        evidence.publish_corpus_observations(
            [
                {
                    "model": "anthropic/claude",
                    "depth_bucket": "15-29",
                    "issue_kind": "boundary_space",
                    "generated_count": 10,
                    "successful_count": 9,
                    "flagged_count": 1,
                    "timing_summary": {"p50_ms": 300},
                }
            ],
            source_instance="loom-test",
        )
        == 1
    )
    with evidence.connect() as db:
        row = db.execute("SELECT * FROM corpus_observations").fetchone()
        assert row["generated_count"] == 10
        assert json.loads(row["timing_summary_json"]) == {"p50_ms": 300}


def test_corpus_publication_replaces_same_source_window() -> None:
    kwargs = {
        "source_instance": "loom-test",
        "window_start": "2026-08-01",
        "window_end": "2026-08-31",
    }
    evidence.publish_corpus_observations(
        [{"model": "openai/test", "generated_count": 10}], **kwargs
    )
    evidence.publish_corpus_observations(
        [{"model": "openai/test", "generated_count": 12}], **kwargs
    )
    with evidence.connect() as db:
        rows = db.execute("SELECT generated_count FROM corpus_observations").fetchall()
    assert [row[0] for row in rows] == [12]


def test_rating_clear_supersedes_without_erasing_history() -> None:
    evidence.set_model_rating("openai/example", 1)
    assert evidence.get_model_rating("openai/example") == 1
    evidence.set_model_rating("openai/example", None)
    assert evidence.get_model_rating("openai/example") is None
    with evidence.connect() as db:
        assert db.execute("SELECT count(*) FROM model_annotations").fetchone()[0] == 2


def test_text_enforcement_excludes_non_text_from_all_derived_status() -> None:
    with evidence.connect() as db:
        run = evidence.start_run("quick", conn=db)
        evidence.record_attempt(
            run,
            "together/black-forest-labs/flux.1",
            probe_kind="continuation",
            attempt_number=1,
            outcome="failure",
            failure_class="rate_limit",
            conn=db,
        )
        evidence.finish_run(run, conn=db)
        first = evidence.enforce_text_only_and_supersede_obsolete_failures(conn=db)
        second = evidence.enforce_text_only_and_supersede_obsolete_failures(conn=db)
        assert first["non_text_endpoints_excluded"] == 0  # excluded on insertion
        assert second["non_text_endpoints_excluded"] == 0
        assert evidence.excluded_non_text_models(conn=db) == {
            "together/black-forest-labs/flux.1"
        }
        assert evidence.current_status(conn=db) == {}
        assert evidence.transient_recheck_models(conn=db) == []


def test_recovered_invalid_request_is_superseded_but_retained() -> None:
    with evidence.connect() as db:
        run = evidence.start_run("quick", conn=db)
        evidence.record_attempt(
            run,
            "openai/fixed",
            probe_kind="continuation",
            attempt_number=1,
            outcome="failure",
            failure_class="invalid_request",
            finished_at="2026-01-01T00:00:00+00:00",
            conn=db,
        )
        evidence.record_attempt(
            run,
            "openai/fixed",
            probe_kind="continuation",
            attempt_number=2,
            outcome="success",
            finished_at="2026-02-01T00:00:00+00:00",
            conn=db,
        )
        evidence.finish_run(run, conn=db)
        result = evidence.enforce_text_only_and_supersede_obsolete_failures(conn=db)
        assert result["obsolete_invalid_requests_superseded"] == 1
        rows = db.execute(
            "SELECT outcome,status_eligible,status_exclusion_reason FROM verification_attempts ORDER BY id"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["status_eligible"] == 0
        assert (
            rows[0]["status_exclusion_reason"]
            == "superseded by later successful request"
        )
        assert evidence.current_status(conn=db)["openai/fixed"]["reachable"] is True


def test_unrecovered_invalid_request_remains_status_bearing() -> None:
    with evidence.connect() as db:
        run = evidence.start_run("quick", conn=db)
        evidence.record_attempt(
            run,
            "openai/still-broken",
            probe_kind="continuation",
            attempt_number=1,
            outcome="failure",
            failure_class="invalid_request",
            conn=db,
        )
        evidence.finish_run(run, conn=db)
        result = evidence.enforce_text_only_and_supersede_obsolete_failures(conn=db)
        assert result["obsolete_invalid_requests_superseded"] == 0
        assert (
            evidence.current_status(conn=db)["openai/still-broken"]["currently_broken"]
            is True
        )


def test_unsupported_parameter_is_compatibility_not_endpoint_health() -> None:
    with evidence.connect() as db:
        run = evidence.start_run("quick", conn=db)
        evidence.record_attempt(
            run,
            "anthropic/new-model",
            probe_kind="continuation",
            attempt_number=1,
            outcome="failure",
            failure_class="invalid_request",
            safe_error_code="unsupported_parameter",
            safe_error_parameter="thinking",
            conn=db,
        )
        evidence.finish_run(run, conn=db)
        result = evidence.enforce_text_only_and_supersede_obsolete_failures(conn=db)
        row = db.execute(
            "SELECT status_eligible,status_exclusion_reason FROM verification_attempts"
        ).fetchone()
        assert result["request_shaping_failures_superseded"] == 1
        assert row["status_eligible"] == 0
        assert row["status_exclusion_reason"] == (
            "basemode request-shaping incompatibility"
        )
        assert evidence.current_status(conn=db) == {}

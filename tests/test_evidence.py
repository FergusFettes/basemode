import json
import sqlite3

from basemode import evidence


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
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2
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
        assert evidence.transient_recheck_models(conn=db) == ["groq/flaky"]
        status = evidence.current_status(conn=db)["groq/flaky"]
        assert status["transient_failure"] is True
        assert status["currently_broken"] is False


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

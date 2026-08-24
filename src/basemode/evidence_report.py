"""Read-only summaries and sanitized exports for model evidence."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from typing import Any

from . import evidence

_NON_TEXT = ("image", "video", "audio", "embedding", "rerank", "moderation")


def _text_clause(alias: str = "e") -> str:
    terms = " AND ".join(
        f"lower(coalesce({alias}.modality,'')) NOT LIKE '%{kind}%'"
        for kind in _NON_TEXT
    )
    return f"({terms})"


def _rows(
    db: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()
) -> list[dict[str, Any]]:
    return [dict(row) for row in db.execute(sql, params).fetchall()]


def providers(db: sqlite3.Connection) -> list[dict[str, Any]]:
    text = _text_clause()
    return _rows(
        db,
        f"""SELECT e.provider,
      count(*) endpoints,
      sum(CASE WHEN lc.available=1 THEN 1 ELSE 0 END) available,
      sum(CASE WHEN la.endpoint_id IS NOT NULL THEN 1 ELSE 0 END) probed,
      sum(CASE WHEN la.outcome='success' THEN 1 ELSE 0 END) reachable,
      sum(CASE WHEN la.outcome!='success' AND la.failure_transience='durable' THEN 1 ELSE 0 END) broken,
      sum(CASE WHEN la.outcome!='success' AND la.failure_transience='suspected' THEN 1 ELSE 0 END) transient
      FROM model_endpoints e
      LEFT JOIN (SELECT endpoint_id,available,row_number() OVER
        (PARTITION BY endpoint_id ORDER BY observed_at DESC,id DESC) rn FROM catalog_observations) lc
        ON lc.endpoint_id=e.id AND lc.rn=1
      LEFT JOIN (SELECT a.endpoint_id,a.outcome,a.failure_transience,row_number() OVER
        (PARTITION BY a.endpoint_id ORDER BY a.finished_at DESC,a.id DESC) rn
        FROM verification_attempts a JOIN verification_runs r ON r.id=a.run_id
        WHERE r.status='completed') la ON la.endpoint_id=e.id AND la.rn=1
      WHERE {text} GROUP BY e.provider ORDER BY endpoints DESC,e.provider""",
    )


def overview(db: sqlite3.Connection) -> list[dict[str, Any]]:
    ps = providers(db)
    totals = {
        key: sum(int(row[key] or 0) for row in ps)
        for key in (
            "endpoints",
            "available",
            "probed",
            "reachable",
            "broken",
            "transient",
        )
    }
    totals["providers"] = len(ps)
    totals["attempts"] = db.execute(f"""SELECT count(*) FROM verification_attempts a
      JOIN model_endpoints e ON e.id=a.endpoint_id WHERE {_text_clause()}""").fetchone()[0]
    totals["runs"] = db.execute("SELECT count(*) FROM verification_runs").fetchone()[0]
    totals["corpus_generations"] = (
        db.execute(f"""SELECT coalesce(sum(c.generated_count),0)
      FROM corpus_observations c JOIN model_endpoints e ON e.id=c.endpoint_id
      WHERE {_text_clause()}""").fetchone()[0]
    )
    return [{"metric": key, "value": value} for key, value in totals.items()]


def statuses(db: sqlite3.Connection) -> list[dict[str, Any]]:
    current = evidence.current_status(conn=db)
    modalities = {
        r[0]: r[1]
        for r in db.execute("SELECT normalized_model_id,modality FROM model_endpoints")
    }
    output = []
    for model, state in current.items():
        modality = (modalities.get(model) or "").lower()
        if any(kind in modality for kind in _NON_TEXT):
            continue
        output.append({"model": model, **state})
    return sorted(output, key=lambda row: row["model"])


def failures(db: sqlite3.Connection) -> list[dict[str, Any]]:
    return _rows(
        db,
        f"""SELECT a.failure_class,coalesce(a.failure_transience,'unknown') transience,
      count(*) attempts,count(DISTINCT a.endpoint_id) endpoints,
      min(a.finished_at) first_seen,max(a.finished_at) last_seen
      FROM verification_attempts a JOIN model_endpoints e ON e.id=a.endpoint_id
      WHERE a.outcome!='success' AND {_text_clause()} GROUP BY 1,2
      ORDER BY attempts DESC,failure_class""",
    )


def transient(db: sqlite3.Connection) -> list[dict[str, Any]]:
    return [row for row in statuses(db) if row.get("transient_failure")]


def runs(db: sqlite3.Connection) -> list[dict[str, Any]]:
    return _rows(
        db,
        f"""SELECT r.id,r.suite,r.status,r.started_at,r.completed_at,
      count(a.id) attempts,sum(CASE WHEN a.outcome='success' THEN 1 ELSE 0 END) successes,
      count(DISTINCT a.endpoint_id) endpoints,r.basemode_version,r.git_commit,r.litellm_version
      FROM verification_runs r LEFT JOIN verification_attempts a ON a.run_id=r.id
      LEFT JOIN model_endpoints e ON e.id=a.endpoint_id
      WHERE a.id IS NULL OR {_text_clause()} GROUP BY r.id ORDER BY r.started_at DESC""",
    )


def corpus(db: sqlite3.Connection) -> list[dict[str, Any]]:
    return _rows(
        db,
        f"""SELECT e.normalized_model_id model,c.depth_bucket,c.issue_kind,
      sum(c.generated_count) generated,sum(coalesce(c.flagged_count,0)) flagged,
      sum(coalesce(c.corrected_count,0)) corrected,sum(coalesce(c.open_issue_count,0)) open_issues
      FROM corpus_observations c JOIN model_endpoints e ON e.id=c.endpoint_id
      WHERE {_text_clause()} GROUP BY e.id,c.depth_bucket,c.issue_kind
      ORDER BY model,c.depth_bucket,c.issue_kind""",
    )


def endpoint(db: sqlite3.Connection, model: str) -> list[dict[str, Any]]:
    row = db.execute(
        "SELECT * FROM model_endpoints WHERE normalized_model_id=?", (model.lower(),)
    ).fetchone()
    if row is None:
        return []
    endpoint_id = row["id"]
    state = evidence.current_status(conn=db).get(row["normalized_model_id"], {})
    attempts = _rows(
        db,
        """SELECT r.suite,a.probe_kind,a.attempt_number,a.finished_at,a.outcome,
      a.failure_class,a.failure_transience,a.http_status,a.latency_ms,a.ttft_ms,a.completion_tokens,
      a.output_tokens_per_second,a.cost_usd FROM verification_attempts a
      JOIN verification_runs r ON r.id=a.run_id WHERE a.endpoint_id=? ORDER BY a.finished_at DESC""",
        (endpoint_id,),
    )
    quality = _rows(
        db,
        """SELECT depth_bucket,issue_kind,sum(generated_count) generated,
      sum(coalesce(flagged_count,0)) flagged,sum(coalesce(corrected_count,0)) corrected,
      sum(coalesce(open_issue_count,0)) open_issues FROM corpus_observations
      WHERE endpoint_id=? GROUP BY depth_bucket,issue_kind""",
        (endpoint_id,),
    )
    return [
        {"endpoint": {**dict(row), **state}, "attempts": attempts, "corpus": quality}
    ]


def export_records(db: sqlite3.Connection) -> Iterable[dict[str, Any]]:
    """Yield a stable, sanitized export. No prompts, responses, config, or account data."""
    for row in _rows(
        db,
        f"SELECT normalized_model_id model,provider,provider_model_id,model_family_id,upstream_provider,display_name,modality,release_date,first_seen_at,last_seen_at FROM model_endpoints e WHERE {_text_clause()}",
    ):
        yield {"type": "endpoint", **row}
    for row in runs(db):
        yield {"type": "run", **row}
    for row in _rows(
        db,
        f"""SELECT r.id run_id,e.normalized_model_id model,r.suite,a.probe_kind,a.attempt_number,
      a.started_at,a.finished_at,a.prompt_method,a.outcome,a.failure_class,a.failure_transience,
      a.http_status,a.safe_error_code,a.safe_error_parameter,a.latency_ms,a.ttft_ms,a.generation_ms,
      a.prompt_tokens,a.completion_tokens,a.reasoning_tokens,a.output_characters,
      a.output_tokens_per_second,a.cost_usd,a.cost_source FROM verification_attempts a
      JOIN verification_runs r ON r.id=a.run_id JOIN model_endpoints e ON e.id=a.endpoint_id
      WHERE {_text_clause()} ORDER BY a.id""",
    ):
        yield {"type": "attempt", **row}
    for row in corpus(db):
        yield {"type": "corpus", **row}


def json_lines(records: Iterable[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(row, sort_keys=True) for row in records)

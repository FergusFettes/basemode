"""Run due transient rechecks only when explicitly enabled and credentialed."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from basemode import evidence
from basemode.keys import get_key
from basemode.verify import verify_models


def _configured(models: list[str]) -> list[str]:
    return [model for model in models if get_key(model.partition("/")[0])]


def _bootstrap_packaged_evidence() -> None:
    """Seed an empty scheduled store from sanitized repository history."""
    with evidence.connect() as db:
        empty = (
            db.execute("SELECT 1 FROM verification_attempts LIMIT 1").fetchone() is None
        )
    if not empty:
        return
    provider_history = Path("data/provider_health_history.jsonl")
    rejected = Path("data/verified_models_rejected.json")
    registry = Path("data/verified_models_registry.json")
    if provider_history.exists():
        evidence.import_provider_health_jsonl(provider_history)
    if rejected.exists():
        evidence.import_rejected_registry(rejected)
    if registry.exists():
        evidence.import_verified_registry(registry)


def _sanitized_result(run_id: str, selected: list[str]) -> dict:
    with evidence.connect() as db:
        attempts = [
            dict(row)
            for row in db.execute(
                """SELECT e.normalized_model_id model,a.probe_kind,a.attempt_number,
                a.started_at,a.finished_at,a.outcome,a.failure_class,a.failure_transience,
                a.http_status,a.safe_error_code,a.safe_error_parameter,a.latency_ms,a.ttft_ms
                FROM verification_attempts a JOIN model_endpoints e ON e.id=a.endpoint_id
                WHERE a.run_id=? ORDER BY a.id""",
                (run_id,),
            )
        ]
        states = evidence.recheck_statuses(conn=db)
    return {
        "run_id": run_id,
        "selected_models": selected,
        "attempts": attempts,
        "recheck_statuses": {model: states.get(model, {}) for model in selected},
    }


def main() -> int:
    if os.environ.get("BASEMODE_SCHEDULED_RECHECKS") != "1":
        print("scheduled rechecks are disabled; set BASEMODE_SCHEDULED_RECHECKS=1")
        return 0
    _bootstrap_packaged_evidence()
    due = evidence.transient_recheck_models()
    selected = _configured(due)
    output = Path(
        os.environ.get("BASEMODE_RECHECK_ARTIFACT", "dist/transient-recheck.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not selected:
        output.write_text(
            json.dumps(
                {"run_id": None, "selected_models": [], "attempts": []}, indent=2
            )
            + "\n"
        )
        print("no due endpoints have configured provider keys")
        return 0
    summary = asyncio.run(
        verify_models(selected, suite="transient-recheck", attempts=1)
    )
    output.write_text(
        json.dumps(_sanitized_result(summary.run_id, selected), indent=2) + "\n"
    )
    print(f"wrote {output} ({summary.successes}/{summary.attempts} successful)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

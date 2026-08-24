"""Controlled model verification suites and durable evidence recording."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import litellm

from .continue_ import continue_text
from .evidence import (
    connect,
    finish_run,
    record_attempt,
    record_probe_result,
    start_run,
    transient_recheck_models,
)
from .health import classify_error, error_details
from .usage import estimate_usage, usage_from_events

QUICK_PREFIXES = ("The path curved through the trees and",)
THOROUGH_PREFIXES = (
    "The path curved through the trees and",
    "After checking every measurement twice, the researcher concluded",
    "Rain pressed against the windows. On the desk, the unopened letter",
)


@dataclass(frozen=True)
class VerificationSummary:
    run_id: str
    models: int
    attempts: int
    successes: int


async def verify_models(
    models: list[str] | None,
    *,
    suite: str = "quick",
    attempts: int = 1,
    max_tokens: int | None = None,
) -> VerificationSummary:
    if suite not in {"quick", "thorough", "transient-recheck"}:
        raise ValueError("suite must be quick, thorough, or transient-recheck")
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if suite == "transient-recheck" and not models:
        models = transient_recheck_models()
    models = list(dict.fromkeys(models or []))
    token_budget = max_tokens or (160 if suite == "thorough" else 64)
    prefixes = THOROUGH_PREFIXES if suite == "thorough" else QUICK_PREFIXES
    try:
        from importlib.metadata import version

        basemode_version = version("basemode")
    except Exception:  # pragma: no cover - editable source without metadata
        basemode_version = None
    run_id = start_run(
        suite,
        configuration={
            "attempts": attempts,
            "max_tokens": token_budget,
            "prefix_count": len(prefixes),
            "self_healing": ["reasoning_off", "larger_budget"],
        },
        target_policy={"models": models},
        basemode_version=basemode_version,
        litellm_version=getattr(litellm, "__version__", None),
    )
    total = successes = 0
    try:
        with connect() as db:
            for model in models:
                number = 0
                for prefix in prefixes:
                    for _ in range(attempts):
                        number += 1
                        total += 1
                        ok = await _probe(
                            db, run_id, model, prefix, number, token_budget
                        )
                        successes += int(ok)
        finish_run(run_id)
    except BaseException:
        finish_run(run_id, "aborted")
        raise
    return VerificationSummary(run_id, len(models), total, successes)


async def _probe(
    db, run_id: str, model: str, prefix: str, number: int, max_tokens: int
) -> bool:
    configurations: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = [
        (max_tokens, {}, []),
        (
            max_tokens,
            {"reasoning": {"enabled": False}},
            [{"action": "disable_reasoning"}],
        ),
        (
            max(max_tokens * 4, 256),
            {},
            [
                {
                    "action": "increase_token_budget",
                    "from": max_tokens,
                    "to": max(max_tokens * 4, 256),
                }
            ],
        ),
    ]
    actions: list[dict[str, Any]] = []
    for heal_index, (budget, extra, proposed) in enumerate(configurations):
        if heal_index:
            actions.extend(proposed)
        started = time.perf_counter()
        first_token: float | None = None
        pieces: list[str] = []
        usage_events: list[dict] = []
        try:
            async for token in continue_text(
                prefix,
                model,
                max_tokens=budget,
                temperature=0.7,
                record_health=False,
                on_usage=usage_events.extend,
                **extra,
            ):
                if first_token is None:
                    first_token = time.perf_counter()
                pieces.append(token)
            finished = time.perf_counter()
            output = "".join(pieces)
            ok = bool(output.strip())
            usage = usage_from_events(model, usage_events) or estimate_usage(
                model, prefix, output
            )
            attempt_id = record_attempt(
                run_id,
                model,
                probe_kind="continuation",
                attempt_number=number * 10 + heal_index,
                outcome="success" if ok else "failure",
                failure_class=None if ok else "empty_response",
                conn=db,
                request_params={
                    "max_tokens": budget,
                    "temperature": 0.7,
                    "reasoning": extra.get("reasoning"),
                },
                compatibility_actions=[
                    *actions,
                    {"result": "success" if ok else "failed"},
                ]
                if actions
                else [],
                latency_ms=(finished - started) * 1000,
                ttft_ms=(first_token - started) * 1000 if first_token else None,
                generation_ms=(finished - (first_token or started)) * 1000,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                output_characters=len(output),
                cost_usd=usage.cost_usd,
                cost_source="estimated" if usage.is_estimate else "provider",
                output_fingerprint=hashlib.sha256(output.encode()).hexdigest()
                if output
                else None,
            )
            record_probe_result(attempt_id, "non_empty", ok, ok, conn=db)
            if ok:
                return True
        except Exception as exc:
            finished = time.perf_counter()
            category, status = classify_error(exc)
            code, param = error_details(exc)
            record_attempt(
                run_id,
                model,
                probe_kind="continuation",
                attempt_number=number * 10 + heal_index,
                outcome="failure",
                failure_class=category,
                http_status=status,
                safe_error_code=code,
                safe_error_parameter=param,
                conn=db,
                request_params={
                    "max_tokens": budget,
                    "temperature": 0.7,
                    "reasoning": extra.get("reasoning"),
                },
                compatibility_actions=[*actions, {"result": "failed"}]
                if actions
                else [],
                latency_ms=(finished - started) * 1000,
            )
            # Authentication and rate limiting are not request-shape defects.
            if category in {
                "authentication",
                "rate_limit",
                "network",
                "provider_unavailable",
                "timeout",
            }:
                return False
    return False

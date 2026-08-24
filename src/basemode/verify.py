"""Controlled model verification suites and durable evidence recording."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from .continue_ import continue_text
from .evidence import (
    classify_text_endpoint,
    connect,
    finish_run,
    record_attempt,
    record_probe_result,
    resume_run,
    start_run,
    transient_recheck_models,
)
from .health import classify_error, error_details
from .transport import litellm_version
from .usage import estimate_usage, get_price_info, usage_from_events

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
    requests: int = 0
    skipped: int = 0
    cost_usd: float = 0.0
    status: str = "completed"


@dataclass(frozen=True)
class _LogicalProbe:
    model: str
    prefix: str
    number: int


class _Limits:
    def __init__(
        self,
        *,
        max_requests: int | None,
        max_cost_usd: float | None,
        deadline: float | None,
    ):
        self.max_requests = max_requests
        self.max_cost_usd = max_cost_usd
        self.deadline = deadline
        self.requests = 0
        self.cost = 0.0
        self.reserved_cost = 0.0
        self.limited = False
        self.lock = asyncio.Lock()

    def remaining(self) -> float | None:
        return (
            None
            if self.deadline is None
            else max(0.0, self.deadline - time.monotonic())
        )

    async def reserve_request(self, known_cost_bound: float | None) -> bool:
        async with self.lock:
            if self.deadline is not None and time.monotonic() >= self.deadline:
                self.limited = True
                return False
            if self.max_requests is not None and self.requests >= self.max_requests:
                self.limited = True
                return False
            if (
                self.max_cost_usd is not None
                and known_cost_bound is not None
                and self.cost + self.reserved_cost + known_cost_bound
                > self.max_cost_usd
            ):
                self.limited = True
                return False
            self.requests += 1
            self.reserved_cost += known_cost_bound or 0.0
            return True

    async def add_cost(self, value: float | None, reserved: float | None) -> None:
        async with self.lock:
            self.reserved_cost -= reserved or 0.0
            self.cost += value or 0.0
            if self.max_cost_usd is not None and self.cost >= self.max_cost_usd:
                self.limited = True


async def verify_models(
    models: list[str] | None,
    *,
    suite: str = "quick",
    attempts: int = 1,
    max_tokens: int | None = None,
    run_id: str | None = None,
    concurrency: int = 4,
    per_provider_concurrency: int = 2,
    max_probes: int | None = None,
    max_requests: int | None = None,
    max_elapsed_seconds: float | None = None,
    max_cost_usd: float | None = None,
) -> VerificationSummary:
    if suite not in {"quick", "thorough", "transient-recheck"}:
        raise ValueError("suite must be quick, thorough, or transient-recheck")
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if concurrency < 1 or per_provider_concurrency < 1:
        raise ValueError("concurrency limits must be at least 1")
    for name, value in (("max_probes", max_probes), ("max_requests", max_requests)):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be at least 1")
    if max_elapsed_seconds is not None and max_elapsed_seconds <= 0:
        raise ValueError("max_elapsed_seconds must be positive")
    if max_cost_usd is not None and max_cost_usd <= 0:
        raise ValueError("max_cost_usd must be positive")
    resumed = None
    if run_id:
        resumed = resume_run(run_id)
        saved = json.loads(resumed["configuration_json"])
        saved_targets = json.loads(resumed["target_policy_json"])
        suite = resumed["suite"]
        attempts = int(saved["attempts"])
        max_tokens = int(saved["max_tokens"])
        models = list(saved_targets["models"])
    if suite == "transient-recheck" and not models:
        models = transient_recheck_models()
    models = list(dict.fromkeys(models or []))
    db = connect()
    try:
        known_eligibility = (
            {
                row["normalized_model_id"]: bool(row["text_eligible"])
                for row in db.execute(
                    f"SELECT normalized_model_id,text_eligible FROM model_endpoints "
                    f"WHERE normalized_model_id IN ({','.join('?' for _ in models)})",
                    models,
                )
            }
            if models
            else {}
        )
    finally:
        db.close()
    non_text = [
        model
        for model in models
        if not known_eligibility.get(model, classify_text_endpoint(model)[0])
    ]
    if non_text:
        raise ValueError(
            "verification accepts text-generation endpoints only: "
            + ", ".join(non_text)
        )
    token_budget = max_tokens or (160 if suite == "thorough" else 64)
    prefixes = THOROUGH_PREFIXES if suite == "thorough" else QUICK_PREFIXES
    try:
        from importlib.metadata import version

        basemode_version = version("basemode")
    except Exception:  # pragma: no cover - editable source without metadata
        basemode_version = None
    configuration = {
        "attempts": attempts,
        "max_tokens": token_budget,
        "prefix_count": len(prefixes),
        "self_healing": ["reasoning_off", "larger_budget"],
        "runner": {
            "concurrency": concurrency,
            "per_provider_concurrency": per_provider_concurrency,
            "max_probes": max_probes,
            "max_requests": max_requests,
            "max_elapsed_seconds": max_elapsed_seconds,
            "max_cost_usd": max_cost_usd,
        },
    }
    if run_id is None:
        run_id = start_run(
            suite,
            configuration=configuration,
            target_policy={"models": models},
            basemode_version=basemode_version,
            litellm_version=litellm_version(),
        )
    probes = _fair_probes(models, prefixes, attempts)
    completed = _completed_probes(run_id)
    skipped = len(
        [probe for probe in probes if (probe.model, probe.number) in completed]
    )
    probes = [probe for probe in probes if (probe.model, probe.number) not in completed]
    limited_by_probe_count = max_probes is not None and len(probes) > max_probes
    if max_probes is not None:
        probes = probes[:max_probes]
    deadline = time.monotonic() + max_elapsed_seconds if max_elapsed_seconds else None
    limits = _Limits(
        max_requests=max_requests, max_cost_usd=max_cost_usd, deadline=deadline
    )
    provider_semaphores: dict[str, asyncio.Semaphore] = defaultdict(
        lambda: asyncio.Semaphore(per_provider_concurrency)
    )
    global_semaphore = asyncio.Semaphore(concurrency)
    total = successes = 0

    async def execute(probe: _LogicalProbe) -> bool | None:
        provider = probe.model.partition("/")[0]
        # Take the provider slot first: tasks queued behind a saturated
        # provider must not occupy scarce global capacity.
        async with provider_semaphores[provider], global_semaphore:
            return await _probe(
                run_id, probe.model, probe.prefix, probe.number, token_budget, limits
            )

    try:
        results = await asyncio.gather(*(execute(probe) for probe in probes))
        total = sum(result is not None for result in results)
        successes = sum(result is True for result in results)
        status = "limited" if limits.limited or limited_by_probe_count else "completed"
        finish_run(run_id, status)
    except BaseException:
        finish_run(run_id, "aborted")
        raise
    return VerificationSummary(
        run_id,
        len(models),
        total,
        successes,
        limits.requests,
        skipped,
        limits.cost,
        status,
    )


def _fair_probes(
    models: list[str], prefixes: tuple[str, ...], attempts: int
) -> list[_LogicalProbe]:
    """Round-robin providers so a large reseller cannot starve smaller queues."""
    queues: dict[str, deque[_LogicalProbe]] = defaultdict(deque)
    for model in models:
        number = 0
        for prefix in prefixes:
            for _ in range(attempts):
                number += 1
                queues[model.partition("/")[0]].append(
                    _LogicalProbe(model, prefix, number)
                )
    ordered: list[_LogicalProbe] = []
    while queues:
        for provider in list(queues):
            ordered.append(queues[provider].popleft())
            if not queues[provider]:
                del queues[provider]
    return ordered


def _completed_probes(run_id: str) -> set[tuple[str, int]]:
    # A logical probe is complete only after success or exhausting all three heals.
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    with connect() as db:
        outcomes = db.execute(
            """SELECT e.normalized_model_id,a.attempt_number,a.outcome FROM verification_attempts a
            JOIN model_endpoints e ON e.id=a.endpoint_id WHERE a.run_id=?""",
            (run_id,),
        ).fetchall()
    done: set[tuple[str, int]] = set()
    for row in outcomes:
        logical = row["attempt_number"] // 10
        key = (row["normalized_model_id"], logical)
        grouped[key].append(row["attempt_number"] % 10)
        if row["outcome"] == "success":
            done.add(key)
    done.update(key for key, heals in grouped.items() if max(heals) >= 2)
    return done


async def _probe(
    run_id: str, model: str, prefix: str, number: int, max_tokens: int, limits: _Limits
) -> bool | None:
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
    existing: set[int] = set()
    with connect() as db:
        endpoint = db.execute(
            "SELECT id FROM model_endpoints WHERE normalized_model_id=?",
            (model.lower(),),
        ).fetchone()
        if endpoint:
            existing = {
                row[0] % 10
                for row in db.execute(
                    "SELECT attempt_number FROM verification_attempts WHERE run_id=? AND endpoint_id=? AND attempt_number BETWEEN ? AND ?",
                    (run_id, endpoint[0], number * 10, number * 10 + 2),
                )
            }
    for heal_index, (budget, extra, proposed) in enumerate(configurations):
        if heal_index:
            actions.extend(proposed)
        if heal_index in existing:
            continue
        price = get_price_info(model)
        known_cost_bound = None
        if price.pricing_available:
            prompt_estimate = estimate_usage(model, prefix, "")
            known_cost_bound = (prompt_estimate.cost_usd or 0.0) + budget * (
                price.output_cost_per_token or 0.0
            )
        if not await limits.reserve_request(known_cost_bound):
            return None
        started = time.perf_counter()
        first_token: float | None = None
        pieces: list[str] = []
        usage_events: list[dict] = []
        try:
            async with asyncio.timeout(limits.remaining()):
                async for token in continue_text(
                    prefix,
                    model,
                    max_tokens=budget,
                    temperature=0.7,
                    record_health=False,
                    retry_empty_completion=False,
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
            record_probe_result(attempt_id, "non_empty", ok, ok)
            await limits.add_cost(usage.cost_usd, known_cost_bound)
            if ok:
                return True
        except asyncio.CancelledError:
            await limits.add_cost(None, known_cost_bound)
            record_attempt(
                run_id,
                model,
                probe_kind="continuation",
                attempt_number=number * 10 + heal_index,
                outcome="aborted",
                failure_class="cancelled",
                status_eligible=False,
                status_exclusion_reason="verification run interrupted",
                request_params={
                    "max_tokens": budget,
                    "temperature": 0.7,
                    "reasoning": extra.get("reasoning"),
                },
                compatibility_actions=[*actions, {"result": "aborted"}]
                if actions
                else [],
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            raise
        except Exception as exc:
            await limits.add_cost(None, known_cost_bound)
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

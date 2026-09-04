"""Deterministic, read-only planning for controlled verification runs."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from .model_modality import classify_text_endpoint
from .models import list_available_endpoint_metadata, list_catalog_endpoint_metadata
from .observation_queries import list_controlled_status, list_endpoint_health
from .observations import list_endpoint_metadata
from .usage import get_price_info
from .verify import QUICK_PREFIXES, THOROUGH_PREFIXES

STATUSES = frozenset(
    {"never-tested", "reachable", "broken", "transient", "verified", "stale"}
)
_STAGE_ORDER = {
    "transient": 0,
    "broken": 1,
    "never-tested": 2,
    "stale": 3,
    "reachable": 4,
    "verified": 5,
}


@dataclass(frozen=True)
class PlannedTarget:
    model: str
    provider: str
    stage: str
    prior_status: str
    catalog_available: bool | None
    release_date: str | None
    last_checked_at: str | None
    logical_probes: int
    maximum_requests: int
    estimated_max_cost_usd: float | None


@dataclass(frozen=True)
class VerificationPlan:
    suite: str
    targets: tuple[PlannedTarget, ...]
    logical_probes: int
    maximum_requests: int
    provider_counts: dict[str, int]
    estimated_known_max_cost_usd: float
    priced_targets: int
    unpriced_targets: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["targets"] = [asdict(target) for target in self.targets]
        return payload


def plan_verification(
    models: list[str] | None = None,
    *,
    suite: str = "quick",
    attempts: int = 1,
    max_tokens: int | None = None,
    providers: list[str] | None = None,
    statuses: list[str] | None = None,
    catalog_available: bool = False,
    available_only: bool = False,
    released_since: str | None = None,
    max_release_age_days: int | None = None,
    stale_after_days: int = 30,
) -> VerificationPlan:
    """Select and describe targets without contacting any provider."""
    if suite not in {"quick", "thorough", "transient-recheck"}:
        raise ValueError("suite must be quick, thorough, or transient-recheck")
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    requested_statuses = set(statuses or [])
    unknown = requested_statuses - STATUSES
    if unknown:
        raise ValueError("unknown status: " + ", ".join(sorted(unknown)))
    provider_filter = {provider.strip().lower() for provider in providers or []}
    explicit = {model.strip().lower() for model in models or []}
    since = _parse_date(released_since) if released_since else None
    if max_release_age_days is not None:
        age_since = datetime.now(UTC).date() - timedelta(days=max_release_age_days)
        since = max(filter(None, (since, age_since)), default=None)

    operational = list_endpoint_health()
    controlled = list_controlled_status(stale_after_days=stale_after_days)
    derived: dict[str, dict[str, Any]] = {}
    metadata_by_model = {
        str(row["model"]): dict(row) for row in list_endpoint_metadata()
    }
    if catalog_available:
        for row in list_catalog_endpoint_metadata():
            existing = metadata_by_model.setdefault(row["model"], row)
            existing["catalog_available"] = True
            existing["release_date"] = existing.get("release_date") or row.get(
                "release_date"
            )
    available_provider_names: set[str] = set()
    if available_only:
        available_rows = list_available_endpoint_metadata()
        available_provider_names = {row["provider"] for row in available_rows}
        for row in available_rows:
            metadata_by_model.setdefault(row["model"], row)
    metadata_rows = list(metadata_by_model.values())
    rows = []
    for metadata in metadata_rows:
        model = str(metadata["model"])
        local = operational.get(model, {})
        checked = controlled.get(model, {})
        controlled_status = checked.get("controlled_status")
        operational_status = local.get("operational_status")
        derived[model] = {
            "available": metadata["catalog_available"],
            "last_checked_at": checked.get("last_run_at") or local.get("window_end"),
            "transient_failure": operational_status == "suspected_transient",
            "currently_broken": operational_status == "failing"
            or controlled_status == "failed",
            "verified": controlled_status == "verified",
            "reachable": controlled_status in {"reachable", "verified"}
            or operational_status in {"healthy", "recovered"},
        }
        rows.append(
            {
                "normalized_model_id": model,
                "provider": metadata["provider"],
                "release_date": metadata["release_date"],
                "text_eligible": metadata["text_eligible"],
            }
        )
    known_models = {row["normalized_model_id"] for row in rows}
    for model in sorted(explicit - known_models):
        eligible, _ = classify_text_endpoint(model)
        if eligible:
            provider, separator, _ = model.partition("/")
            rows.append(
                {
                    "normalized_model_id": model,
                    "provider": provider if separator else "unknown",
                    "release_date": None,
                    "text_eligible": 1,
                }
            )

    logical_per_model = attempts * (
        len(THOROUGH_PREFIXES) if suite == "thorough" else len(QUICK_PREFIXES)
    )
    targets: list[PlannedTarget] = []
    for row in rows:
        model = row["normalized_model_id"]
        if not row["text_eligible"]:
            continue
        state = derived.get(model, {})
        prior = _prior_status(state, stale_after_days)
        is_explicit = model in explicit
        if explicit and not is_explicit:
            continue
        if provider_filter and row["provider"] not in provider_filter:
            continue
        if available_only and row["provider"] not in available_provider_names:
            continue
        if requested_statuses and prior not in requested_statuses:
            continue
        if suite == "transient-recheck" and not explicit and prior != "transient":
            continue
        if catalog_available and state.get("available") is not True:
            continue
        release = _parse_date(row["release_date"]) if row["release_date"] else None
        if since and (release is None or release < since):
            continue
        cost = _estimate_max_cost(model, logical_per_model, max_tokens, suite)
        targets.append(
            PlannedTarget(
                model=model,
                provider=row["provider"],
                stage=prior,
                prior_status=prior,
                catalog_available=state.get("available"),
                release_date=row["release_date"],
                last_checked_at=state.get("last_checked_at"),
                logical_probes=logical_per_model,
                maximum_requests=logical_per_model * 3,
                estimated_max_cost_usd=cost,
            )
        )
    missing = explicit - {target.model for target in targets}
    if missing:
        raise ValueError(
            "explicit models not eligible under selectors: "
            + ", ".join(sorted(missing))
        )
    targets.sort(key=lambda t: (_STAGE_ORDER[t.stage], t.provider, t.model))
    counts = Counter(target.provider for target in targets)
    priced = [
        target.estimated_max_cost_usd
        for target in targets
        if target.estimated_max_cost_usd is not None
    ]
    return VerificationPlan(
        suite=suite,
        targets=tuple(targets),
        logical_probes=sum(target.logical_probes for target in targets),
        maximum_requests=sum(target.maximum_requests for target in targets),
        provider_counts=dict(sorted(counts.items())),
        estimated_known_max_cost_usd=sum(priced),
        priced_targets=len(priced),
        unpriced_targets=len(targets) - len(priced),
    )


def _prior_status(state: dict[str, Any], stale_after_days: int) -> str:
    checked = state.get("last_checked_at")
    if not checked:
        return "never-tested"
    try:
        observed = datetime.fromisoformat(checked.replace("Z", "+00:00"))
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=UTC)
        if observed < datetime.now(UTC) - timedelta(days=stale_after_days):
            return "stale"
    except ValueError:
        pass
    if state.get("transient_failure"):
        return "transient"
    if state.get("currently_broken"):
        return "broken"
    if state.get("verified"):
        return "verified"
    return "reachable" if state.get("reachable") else "broken"


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value[:10])
    except ValueError as exc:
        raise ValueError(f"invalid ISO date: {value}") from exc


def _estimate_max_cost(
    model: str, logical_probes: int, max_tokens: int | None, suite: str
) -> float | None:
    price = get_price_info(model)
    if not price.pricing_available:
        return None
    budget = max_tokens or (160 if suite == "thorough" else 64)
    # Worst case includes both ordinary healing retries plus the enlarged-budget retry.
    completion_tokens = logical_probes * (budget * 2 + max(budget * 4, 256))
    prefixes = THOROUGH_PREFIXES if suite == "thorough" else QUICK_PREFIXES
    prompt_tokens = (
        logical_probes * max(1, sum(len(p) // 4 for p in prefixes) // len(prefixes)) * 3
    )
    return prompt_tokens * (price.input_cost_per_token or 0) + completion_tokens * (
        price.output_cost_per_token or 0
    )

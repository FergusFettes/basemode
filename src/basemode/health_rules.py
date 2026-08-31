"""Versioned conservative rules for endpoint operational status."""

from __future__ import annotations

RULES_VERSION = 1
MIN_CLASSIFIED_SAMPLE = 3
HEALTHY_SUCCESS_RATE = 0.95
DEGRADED_SUCCESS_RATE = 0.70


def operational_status(
    *,
    operations: int,
    successful_operations: int,
    transient_failures: int,
    persistent_failures: int,
    account_failures: int,
    last_outcome: str | None,
) -> str:
    """Project observations into one explicit, testable status."""
    if operations == 0:
        return "account_limited" if account_failures else "unknown"
    failures = operations - successful_operations
    if failures == 0:
        return "healthy"
    if last_outcome in {"success", "cancelled"} and operations >= 2:
        return "recovered"
    if persistent_failures == 0 and transient_failures:
        return "suspected_transient"
    success_rate = successful_operations / operations
    if operations < MIN_CLASSIFIED_SAMPLE:
        return "degraded"
    if success_rate >= HEALTHY_SUCCESS_RATE:
        return "healthy"
    if success_rate >= DEGRADED_SUCCESS_RATE:
        return "degraded"
    return "failing"

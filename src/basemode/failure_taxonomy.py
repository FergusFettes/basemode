"""Provider-neutral, content-free failure classification."""

from __future__ import annotations

import re
from typing import Any, Final

from .exceptions import EmptyCompletionError

EMPTY_RESPONSE = "empty_response"

#: Failure classes worth retrying or rechecking: the endpoint may well work
#: on the next request.
TRANSIENT_FAILURES: Final = frozenset(
    {
        "rate_limit",
        "timeout",
        "network",
        "provider_unavailable",
        "empty_response",
        "provider_error",
    }
)


def classify_error(error: BaseException) -> tuple[str, int | None]:
    if isinstance(error, EmptyCompletionError):
        return EMPTY_RESPONSE, None
    status = error_status(error)
    name = type(error).__name__.lower()
    if status in {401, 403} or "auth" in name or "permission" in name:
        return "authentication", status
    if status == 429 or "ratelimit" in name or "rate_limit" in name:
        return "rate_limit", status
    if isinstance(error, TimeoutError) or "timeout" in name:
        return "timeout", status
    if status is not None and status >= 500:
        return "provider_unavailable", status
    if status in {400, 404, 409, 422} or "unsupportedparam" in name:
        return "invalid_request", status
    if "connection" in name or "network" in name:
        return "network", status
    return "provider_error", status


def error_details(error: BaseException) -> tuple[str | None, str | None]:
    """Extract only bounded error codes and parameter names."""
    code = _safe_error_field(getattr(error, "code", None))
    param = _safe_error_field(getattr(error, "param", None))
    for payload in (getattr(error, "body", None), getattr(error, "error", None)):
        payload_code, payload_param = _payload_error_details(payload)
        code = code or payload_code
        param = param or payload_param
    message = str(error)
    if code is None:
        if "UnsupportedParamsError" in type(error).__name__ or re.search(
            r"unsupported parameter", message, re.IGNORECASE
        ):
            code = "unsupported_parameter"
        elif re.search(
            r"unsupported value|invalid temperature", message, re.IGNORECASE
        ):
            code = "unsupported_value"
    if param is None:
        match = re.search(
            r"(?:parameter|value):\s*['`]([A-Za-z_][A-Za-z0-9_]*)['`]",
            message,
            re.IGNORECASE,
        ) or re.search(r"parameters:\s*\[['\"]([^'\"]+)", message)
        if match:
            param = match.group(1)
        elif "invalid temperature" in message.lower():
            param = "temperature"
    return code, param


def _safe_error_field(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value[:120] if value else None


def _payload_error_details(payload: Any) -> tuple[str | None, str | None]:
    if not isinstance(payload, dict):
        return None, None
    nested = payload.get("error")
    if isinstance(nested, dict):
        payload = nested
    return _safe_error_field(payload.get("code")), _safe_error_field(
        payload.get("param")
    )


def error_status(error: BaseException) -> int | None:
    candidates = (getattr(error, "status_code", None), getattr(error, "status", None))
    response = getattr(error, "response", None)
    if response is not None:
        candidates += (getattr(response, "status_code", None),)
    return next(
        (
            value
            for value in candidates
            if isinstance(value, int)
            and not isinstance(value, bool)
            and 100 <= value <= 599
        ),
        None,
    )


def failure_attribution(failure_class: str | None) -> str | None:
    """Say whose problem a failure is, so it lands on the right ledger.

    An account without access to a model says nothing about the model, and a
    request basemode shaped wrongly says nothing about the provider. Only
    provider- and endpoint-attributed failures belong in public endpoint
    health.
    """
    if failure_class is None:
        return None
    if failure_class in {"authentication", "quota"}:
        return "account"
    if failure_class == "invalid_request":
        return "basemode"
    if failure_class == "cancelled":
        return "client"
    if failure_class in {
        "provider_unavailable",
        "provider_error",
        "content_filter",
        "rate_limit",
    }:
        return "provider"
    if failure_class == "empty_response":
        return "endpoint"
    return "unknown"


def failure_transience(failure_class: str | None) -> str | None:
    """Say whether retrying the same request could plausibly succeed."""
    if failure_class is None:
        return None
    if failure_class in TRANSIENT_FAILURES:
        return "transient"
    if failure_class in {"authentication", "quota", "invalid_request"}:
        return "persistent"
    return "unknown"

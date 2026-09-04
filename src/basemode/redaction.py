"""Strip account identifiers out of text on its way to a local log file.

The observation ledger never stores a raw provider error, but the rotating
diagnostic log does — that is what it is for, and a traceback with the
provider's own message in it is often the only way to work out why a request
was rejected. Those messages carry identifiers that belong to the account
rather than the request: an OpenRouter `user_id`, an xAI team UUID, an
occasional key or address. Nothing reads the log except the person who ran
the command, but a log is the file people paste into an issue, so the
identifiers come out before they are written.

This is deliberately about identity, not secrecy: it does not try to hide
model IDs, parameters, or error codes, which are exactly what makes the log
worth keeping.
"""

from __future__ import annotations

import re
from typing import Final

REDACTED: Final = "[redacted]"

#: JSON-ish `"user_id": "…"` / `user_id=…` fields naming a person or account.
_FIELD_RE: Final = re.compile(
    r"""(?P<key>["']?\b(?:user|user_id|userid|account|account_id|org|org_id|
        organization|organization_id|team|team_id|customer|customer_id|
        api_key|apikey|key|token|access_token|email)\b["']?\s*[:=]\s*)
        (?P<quote>["']?)(?P<value>[A-Za-z0-9][A-Za-z0-9._@+\-]{3,})(?P=quote)""",
    re.IGNORECASE | re.VERBOSE,
)
_UUID_RE: Final = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
#: Provider key shapes: `sk-…`, `sk-or-v1-…`, `xai-…`, `gsk_…`, `AIza…`.
_KEY_RE: Final = re.compile(
    r"\b(?:sk|pk|rk|xai|sk-or-v1|gsk|AIza)[_-][A-Za-z0-9_\-]{12,}\b"
)
_EMAIL_RE: Final = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def redact(text: str) -> str:
    """Replace account-identifying substrings with a fixed marker."""
    text = _KEY_RE.sub(REDACTED, text)
    text = _EMAIL_RE.sub(REDACTED, text)
    text = _FIELD_RE.sub(rf"\g<key>\g<quote>{REDACTED}\g<quote>", text)
    return _UUID_RE.sub(REDACTED, text)

"""Completion transport boundary.

Basemode owns endpoint identity, catalog metadata, and compatibility policy.
Transports only send an already-normalized request and return its native stream.
"""

from __future__ import annotations

from typing import Any, Protocol

import litellm


class CompletionTransport(Protocol):
    """Narrow interface used by continuation strategies.

    Return values deliberately remain transport-native async iterables.  This
    keeps the boundary small while allowing strategies to consume the choice
    and usage fields they already understand.
    """

    async def chat_completion(self, **request: Any) -> Any: ...

    async def text_completion(self, **request: Any) -> Any: ...


class LiteLLMTransport:
    """Default multi-provider transport backed by LiteLLM.

    A provider-qualified model is passed through unchanged.  It does not need
    to occur in LiteLLM's bundled model catalog; basemode's endpoint and live
    catalog data are authoritative for model discovery.
    """

    async def chat_completion(self, **request: Any) -> Any:
        return await litellm.acompletion(**request)

    async def text_completion(self, **request: Any) -> Any:
        return await litellm.atext_completion(**request)


_transport: CompletionTransport = LiteLLMTransport()


def get_transport() -> CompletionTransport:
    """Return the process-wide completion transport."""
    return _transport


def set_transport(transport: CompletionTransport) -> CompletionTransport:
    """Install a transport and return the previous one.

    This is primarily an integration escape hatch.  Callers should restore the
    returned transport after a scoped override.
    """
    global _transport
    previous = _transport
    _transport = transport
    return previous


def litellm_version() -> str:
    """Return the LiteLLM version for verification-run provenance."""
    version = getattr(litellm, "__version__", None)
    if isinstance(version, str):
        return version
    try:
        from importlib.metadata import version as package_version

        return package_version("litellm")
    except Exception:  # pragma: no cover - only unusual vendored installs
        return "unknown"

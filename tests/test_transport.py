from __future__ import annotations

from typing import Any

import pytest

from basemode.transport import LiteLLMTransport, get_transport, set_transport


@pytest.mark.asyncio
async def test_litellm_transport_passes_unknown_qualified_model_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}
    sentinel = object()

    async def fake_acompletion(**request: Any) -> object:
        seen.update(request)
        return sentinel

    monkeypatch.setattr("basemode.transport.litellm.acompletion", fake_acompletion)

    result = await LiteLLMTransport().chat_completion(
        model="deepinfra/vendor/model-not-in-litellm", messages=[], stream=True
    )

    assert result is sentinel
    assert seen == {
        "model": "deepinfra/vendor/model-not-in-litellm",
        "messages": [],
        "stream": True,
    }


def test_transport_can_be_overridden_and_restored() -> None:
    class AlternateTransport:
        async def chat_completion(self, **request: Any) -> Any:
            return request

        async def text_completion(self, **request: Any) -> Any:
            return request

    original = get_transport()
    alternate = AlternateTransport()
    previous = set_transport(alternate)
    try:
        assert previous is original
        assert get_transport() is alternate
    finally:
        set_transport(original)

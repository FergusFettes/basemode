from basemode.params import GenerationParams
from basemode.strategies.compat import build_kwargs


def test_generation_prefix_does_not_rewind_by_default() -> None:
    from basemode.continue_ import _generation_prefix

    assert _generation_prefix("twas brilig and the sli", "system", False) == (
        "twas brilig and the sli",
        "",
    )


def test_generation_prefix_rewinds_when_requested() -> None:
    from basemode.continue_ import _generation_prefix

    assert _generation_prefix("twas brilig and the sli", "system", True) == (
        "twas brilig and the ",
        "sli",
    )


def test_moonshot_kimi_does_not_send_thinking_param() -> None:
    kwargs = build_kwargs(GenerationParams(model="moonshot/kimi-k2.5", max_tokens=200))

    assert kwargs["max_tokens"] == 4608
    assert "temperature" not in kwargs
    assert "thinking" not in kwargs
    assert "extra_body" not in kwargs


def test_moonshot_kimi_thinking_keeps_budget_without_control_param() -> None:
    kwargs = build_kwargs(
        GenerationParams(model="moonshot/kimi-k2-thinking", max_tokens=200)
    )

    assert kwargs["max_tokens"] == 4608
    assert "thinking" not in kwargs
    assert "extra_body" not in kwargs


def test_openrouter_kimi_uses_extra_body_thinking() -> None:
    kwargs = build_kwargs(
        GenerationParams(model="openrouter/moonshotai/kimi-k2.5", max_tokens=200)
    )

    assert kwargs["max_tokens"] == 200
    assert kwargs["extra_body"] == {"thinking": {"budget_tokens": 4096}}


def test_openrouter_kimi_k26_uses_extra_body_thinking() -> None:
    kwargs = build_kwargs(
        GenerationParams(model="openrouter/moonshotai/kimi-k2.6", max_tokens=200)
    )

    assert kwargs["max_tokens"] == 200
    assert kwargs["extra_body"] == {"thinking": {"budget_tokens": 4096}}


def test_gemini_gemma_4_uses_thinking_level_payload() -> None:
    kwargs = build_kwargs(
        GenerationParams(model="gemini/gemma-4-26b-a4b-it", max_tokens=200)
    )

    assert kwargs["max_tokens"] == 4608
    assert kwargs["extra_body"] == {
        "generationConfig": {"thinkingConfig": {"thinkingLevel": "high"}}
    }
    assert "thinking" not in kwargs


def test_zai_glm_disables_thinking_via_extra_body() -> None:
    kwargs = build_kwargs(GenerationParams(model="zai/glm-4.7", max_tokens=200))

    assert kwargs["max_tokens"] == 200
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "thinking" not in kwargs


async def test_continue_text_loads_persisted_keys(monkeypatch) -> None:
    from basemode import continue_ as cont
    from basemode.params import GenerationParams

    calls = {"count": 0}

    def fake_load() -> None:
        calls["count"] += 1

    class DummyStrategy:
        name = "system"

        async def stream(self, prefix: str, params: GenerationParams):
            yield "x"

    monkeypatch.setattr(cont, "load_into_environ", fake_load)
    monkeypatch.setattr(cont, "normalize_model", lambda model: model)
    monkeypatch.setattr(cont, "detect_strategy", lambda model, override=None: DummyStrategy())
    monkeypatch.setattr(cont.litellm, "suppress_debug_info", False)

    out = []
    async for token in cont.continue_text("abc", model="openrouter/moonshotai/kimi-k2.6"):
        out.append(token)

    assert calls["count"] == 1
    assert cont.litellm.suppress_debug_info is True
    assert "".join(out)

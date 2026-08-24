import pytest

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


def test_moonshot_kimi_k26_does_not_send_temperature_or_thinking_param() -> None:
    kwargs = build_kwargs(GenerationParams(model="moonshot/kimi-k2.6", max_tokens=200))

    assert kwargs["max_tokens"] == 4608
    assert "temperature" not in kwargs
    assert "thinking" not in kwargs
    assert "extra_body" not in kwargs


def test_moonshot_kimi_k3_does_not_send_temperature() -> None:
    kwargs = build_kwargs(GenerationParams(model="moonshot/kimi-k3", max_tokens=200))

    assert "temperature" not in kwargs


def test_moonshot_kimi_k3_bumps_max_tokens_for_reasoning_budget() -> None:
    kwargs = build_kwargs(GenerationParams(model="moonshot/kimi-k3", max_tokens=200))

    assert kwargs["max_tokens"] == 4608
    assert "thinking" not in kwargs
    assert "extra_body" not in kwargs


def test_claude_opus_5_omits_thinking_kwarg_entirely() -> None:
    """Claude 5.x rejects `thinking.type: "enabled"` outright and, as of
    2026-08-24, also rejects `thinking.type: "disabled"` ("Thinking defaults
    to adaptive mode when not specified") — so the only accepted shape is no
    `thinking` kwarg at all."""
    kwargs = build_kwargs(
        GenerationParams(model="anthropic/claude-opus-5", max_tokens=200)
    )

    assert kwargs["max_tokens"] == 200
    assert "thinking" not in kwargs


def test_claude_sonnet_5_omits_thinking_kwarg_entirely() -> None:
    kwargs = build_kwargs(
        GenerationParams(model="anthropic/claude-sonnet-5", max_tokens=200)
    )

    assert kwargs["max_tokens"] == 200
    assert "thinking" not in kwargs


def test_claude_opus_4_family_keeps_enabled_thinking_shape_if_ever_tagged(
    monkeypatch,
) -> None:
    """Only the 5.x family needs the disable workaround — older Claude
    models that pick up the registry's generic `reasoning_budget` quirk
    should still get the (working) `"enabled"` shape."""
    import basemode.strategies.compat as compat

    monkeypatch.setattr(
        compat, "model_quirks", lambda model: frozenset({"reasoning_budget"})
    )

    kwargs = build_kwargs(
        GenerationParams(model="anthropic/claude-opus-4-7", max_tokens=200)
    )

    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}


def test_openai_gpt_5_6_sol_bumps_max_tokens_without_thinking_param() -> None:
    kwargs = build_kwargs(GenerationParams(model="openai/gpt-5.6-sol", max_tokens=200))

    assert kwargs["max_tokens"] == 2560
    assert "thinking" not in kwargs
    assert "extra_body" not in kwargs


def test_together_reasoning_model_uses_budget_without_thinking_param() -> None:
    """Together rejects LiteLLM's Anthropic-shaped ``thinking`` argument."""
    kwargs = build_kwargs(
        GenerationParams(
            model="together_ai/deepseek-ai/DeepSeek-V4-Pro-0813", max_tokens=20
        )
    )

    assert kwargs["max_tokens"] == 5120
    assert "thinking" not in kwargs
    assert "extra_body" not in kwargs


def test_moonshot_kimi_k27_highspeed_does_not_send_temperature() -> None:
    kwargs = build_kwargs(
        GenerationParams(model="moonshot/kimi-k2.7-code-highspeed", max_tokens=20)
    )

    assert "temperature" not in kwargs


def test_registry_reasoning_budget_quirk_applies_generic_bump(monkeypatch) -> None:
    """A model with no tuned _THINKING_MODELS entry, but tagged with the
    registry's generic `reasoning_budget` quirk (auto-added by
    scripts/discover_new_models.py or scripts/probe_model_quirks.py), still
    gets a budget bump instead of silently going empty."""
    import basemode.strategies.compat as compat

    monkeypatch.setattr(
        compat, "model_quirks", lambda model: frozenset({"reasoning_budget"})
    )

    kwargs = build_kwargs(
        GenerationParams(model="anthropic/claude-some-new-reasoner", max_tokens=200)
    )

    assert kwargs["max_tokens"] == 5120
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}


def test_registry_reasoning_budget_quirk_ignored_without_quirk(monkeypatch) -> None:
    import basemode.strategies.compat as compat

    monkeypatch.setattr(compat, "model_quirks", lambda model: frozenset())

    kwargs = build_kwargs(
        GenerationParams(model="anthropic/claude-some-new-model", max_tokens=200)
    )

    assert kwargs["max_tokens"] == 200
    assert "thinking" not in kwargs


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-sonnet-5",
        "anthropic/claude-opus-5",
        "anthropic/claude-fable-5",
    ],
)
def test_claude_5_family_does_not_send_temperature(model: str) -> None:
    kwargs = build_kwargs(GenerationParams(model=model, max_tokens=200))

    assert "temperature" not in kwargs


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-sonnet-5",
        "anthropic/claude-opus-5",
        "anthropic/claude-fable-5",
        "anthropic/claude-opus-4-7",
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-opus-4-6",
    ],
)
def test_claude_5_family_never_uses_prefill(model: str) -> None:
    """These models reject an assistant prefill, whatever the registry says.

    Which non-prefill strategy they land on is the registry's call
    (`claude-opus-5` is registered as `few_shot`); the invariant here is only
    that prefill is never chosen for a `no_prefill` model.
    """
    from basemode.detect import detect_strategy

    assert detect_strategy(model).name != "prefill"


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


def test_gemini_gemma_4_preserves_requested_max_tokens() -> None:
    kwargs = build_kwargs(
        GenerationParams(model="gemini/gemma-4-26b-a4b-it", max_tokens=200)
    )

    assert kwargs["max_tokens"] == 200
    assert "extra_body" not in kwargs
    assert "thinking" not in kwargs


async def test_continue_text_strict_max_tokens_clips_visible_output(
    monkeypatch,
) -> None:
    from basemode import continue_ as cont
    from basemode.params import GenerationParams

    class DummyStrategy:
        name = "system"

        async def stream(self, prefix: str, params: GenerationParams):
            yield "alpha "
            yield "beta "
            yield "gamma"

    # 2 tokens for "alpha beta", 3 for "alpha beta gamma"
    def fake_token_counter(*, model: str, text: str):
        return 0 if not text.strip() else len(text.strip().split())

    monkeypatch.setattr(cont, "load_into_environ", lambda: None)
    monkeypatch.setattr(cont, "normalize_model", lambda model: model)
    monkeypatch.setattr(
        cont, "detect_strategy", lambda model, override=None: DummyStrategy()
    )
    monkeypatch.setattr(cont.litellm, "token_counter", fake_token_counter)

    out = []
    async for token in cont.continue_text(
        "prefix",
        model="openai/gpt-4o-mini",
        max_tokens=2,
        strict_max_tokens=True,
    ):
        out.append(token)
    assert "".join(out).strip() == "alpha beta"


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
    monkeypatch.setattr(
        cont, "detect_strategy", lambda model, override=None: DummyStrategy()
    )
    monkeypatch.setattr(cont.litellm, "suppress_debug_info", False)

    out = []
    async for token in cont.continue_text(
        "abc", model="openrouter/moonshotai/kimi-k2.6"
    ):
        out.append(token)

    assert calls["count"] == 1
    assert cont.litellm.suppress_debug_info is True
    assert "".join(out)


async def test_branch_text_propagates_worker_errors(monkeypatch) -> None:
    from basemode import continue_ as cont
    from basemode.params import GenerationParams

    class FailingStrategy:
        name = "system"

        async def stream(self, prefix: str, params: GenerationParams):
            raise RuntimeError("provider unavailable")
            yield  # pragma: no cover - keeps this an async generator

    monkeypatch.setattr(cont, "load_into_environ", lambda: None)
    monkeypatch.setattr(cont, "normalize_model", lambda model: model)
    monkeypatch.setattr(
        cont, "detect_strategy", lambda model, override=None: FailingStrategy()
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        async for _item in cont.branch_text("prefix", n=2):
            pass


async def test_branch_text_requires_a_positive_branch_count() -> None:
    from basemode import continue_ as cont

    with pytest.raises(ValueError, match="at least 1"):
        async for _item in cont.branch_text("prefix", n=0):
            pass

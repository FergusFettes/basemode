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


def test_reseller_widens_budget_instead_of_thinking_kwarg(monkeypatch) -> None:
    """`model_quirks` is keyed by stem, so a reseller hosting a model under
    the same stem as one tagged `reasoning_budget` inherits that quirk too --
    deepinfra sharing deepseek-v4-flash's stem, probed live 2026-08-24, 400s
    on the Anthropic-shaped `thinking` kwarg ("does not support parameters:
    ['thinking']"). Only anthropic/gemini are trusted to accept that shape
    natively; every other provider widens max_tokens instead, regardless of
    which registry entry the quirk came from."""
    import basemode.strategies.compat as compat

    monkeypatch.setattr(
        compat, "model_quirks", lambda model: frozenset({"reasoning_budget"})
    )

    kwargs = build_kwargs(
        GenerationParams(
            model="deepinfra/deepseek-ai/DeepSeek-V4-Flash", max_tokens=200
        )
    )

    assert kwargs["max_tokens"] > 200
    assert "thinking" not in kwargs
    assert "extra_body" not in kwargs


def test_openrouter_kimi_k26_uses_extra_body_thinking() -> None:
    kwargs = build_kwargs(
        GenerationParams(model="openrouter/moonshotai/kimi-k2.6", max_tokens=200)
    )

    assert kwargs["max_tokens"] == 200
    assert kwargs["extra_body"] == {"thinking": {"budget_tokens": 4096}}


def test_gemini_gemma_4_widens_budget_instead_of_thinking_kwarg() -> None:
    """Gemma rejects any thinking-budget shape outright ("Thinking budget is
    not supported for this model", probed live 2026-08-24) even though it
    silently burns hidden reasoning tokens -- confirmed live: a caller's
    small max_tokens leaves nothing after reasoning eats its share
    (finish_reason="length", zero visible tokens). Widening max_tokens is the
    only lever available, same as the OpenAI-compatible-provider case."""
    kwargs = build_kwargs(
        GenerationParams(model="gemini/gemma-4-26b-a4b-it", max_tokens=200)
    )

    assert kwargs["max_tokens"] > 200
    assert "extra_body" not in kwargs
    assert "thinking" not in kwargs
    assert "reasoning_effort" not in kwargs


def test_gemini_3_flash_disables_reasoning() -> None:
    """gemini-3.x flash and its "latest" alias default reasoning on and can
    silently consume the whole visible budget (finish_reason="length",
    probed live 2026-08-24); unlike gemma or gemini-pro-latest, they accept
    reasoning_effort="none" outright, which is cheaper than a bigger budget."""
    for model in (
        "gemini/gemini-3.5-flash",
        "gemini/gemini-3.6-flash",
        "gemini/gemini-3.7-flash",
        "gemini/gemini-flash-latest",
    ):
        kwargs = build_kwargs(GenerationParams(model=model, max_tokens=200))
        assert kwargs["reasoning_effort"] == "none"
        assert kwargs["max_tokens"] == 200


def test_gemini_pro_latest_needs_a_nonzero_thinking_budget() -> None:
    """Unlike the flash family, gemini-pro-latest rejects reasoning_effort
    "none" outright ("Budget 0 is invalid. This model only works in thinking
    mode.", probed live 2026-08-24) -- same tuning as gemini-2.5-pro."""
    kwargs = build_kwargs(
        GenerationParams(model="gemini/gemini-pro-latest", max_tokens=200)
    )

    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert kwargs["max_tokens"] == 2560


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


def test_zai_glm_5_3_cannot_disable_thinking() -> None:
    """glm-5.3 rejects `thinking.type: "disabled"` outright ("This model
    always engages in thinking and cannot be disabled; please use low, high,
    or max", probed live 2026-08-24) unlike the rest of the glm-5.x family,
    which the "glm-5" prefix match would otherwise catch it under."""
    kwargs = build_kwargs(GenerationParams(model="zai/glm-5.3", max_tokens=200))

    assert kwargs["extra_body"] == {"thinking": {"type": "enabled", "effort": "low"}}
    assert kwargs["max_tokens"] > 200


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


async def test_continue_text_retries_openrouter_empty_with_reasoning_off(
    monkeypatch,
) -> None:
    """OpenRouter often defaults reasoning on for a model, which can eat the
    whole visible budget and yield nothing. A single retry with reasoning
    switched off should recover transparently."""
    from basemode import continue_ as cont
    from basemode.exceptions import EmptyCompletionError
    from basemode.params import GenerationParams

    calls = []

    class FlakyStrategy:
        name = "system"

        async def stream(self, prefix: str, params: GenerationParams):
            calls.append(dict(params.extra))
            if params.extra.get("reasoning") == {"enabled": False}:
                yield "recovered"
                return
            raise EmptyCompletionError(
                model=params.model, strategy=self.name, finish_reason="length"
            )

    monkeypatch.setattr(cont, "load_into_environ", lambda: None)
    monkeypatch.setattr(cont, "normalize_model", lambda model: model)
    monkeypatch.setattr(
        cont, "detect_strategy", lambda model, override=None: FlakyStrategy()
    )

    out = []
    async for token in cont.continue_text(
        "prefix", model="openrouter/some/model", record_health=False
    ):
        out.append(token)

    assert "".join(out) == "recovered"
    assert calls == [{}, {"reasoning": {"enabled": False}}]


async def test_continue_text_does_not_retry_non_openrouter_empty(monkeypatch) -> None:
    from basemode import continue_ as cont
    from basemode.exceptions import EmptyCompletionError
    from basemode.params import GenerationParams

    calls = []

    class AlwaysEmptyStrategy:
        name = "system"

        async def stream(self, prefix: str, params: GenerationParams):
            calls.append(dict(params.extra))
            raise EmptyCompletionError(
                model=params.model, strategy=self.name, finish_reason="length"
            )
            yield  # pragma: no cover - keeps this an async generator

    monkeypatch.setattr(cont, "load_into_environ", lambda: None)
    monkeypatch.setattr(cont, "normalize_model", lambda model: model)
    monkeypatch.setattr(
        cont, "detect_strategy", lambda model, override=None: AlwaysEmptyStrategy()
    )

    with pytest.raises(EmptyCompletionError):
        async for _ in cont.continue_text(
            "prefix", model="anthropic/claude-opus-5", record_health=False
        ):
            pass

    assert calls == [{}]


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

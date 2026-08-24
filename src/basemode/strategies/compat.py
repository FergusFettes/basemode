"""Compatibility helpers for model-specific API quirks.

Per-model quirks (`no_temperature`, `no_prefill`, ...) and the verified
prompt method are sourced from the
verified-models registry (`data/verified_models_registry.json`, packaged as
`verified_models_details.json`) rather than hardcoded here — see the
`"quirks"` field on a registry entry. That registry is kept current by a
weekly probe (`scripts/probe_model_quirks.py`), which re-tests every known
quirk and looks for new ones, opening a PR when it finds drift. Regex
patterns below remain for provider-wide families too broad to enumerate
individually in the registry.

The registry's `"prompt_method"` field records which strategy was observed
to actually work for a model — written by `scripts/discover_new_models.py`
when it probes a newly-listed model, and read back here by
`registry_prompt_method` so `detect.py` selects that strategy at runtime
instead of re-deriving a guess from the model name.
"""

import json
import re
from functools import lru_cache
from importlib import resources

from ..params import GenerationParams

# Model stems that only accept temperature=1 (probed 2026-04-18):
#   - gpt-5 / gpt-5-mini / gpt-5-nano / gpt-5-codex  (but NOT gpt-5.1, 5.4, etc.)
#   - o-series reasoning models: o1, o3, o3-mini, o4-mini, ...
_NO_TEMPERATURE_PATTERNS = [
    re.compile(r"^gpt-5(-[a-z]+)?$"),
    re.compile(r"^o\d+(-[a-z]+)?$"),
]


@lru_cache(maxsize=1)
def _registry_rows() -> list[dict]:
    """Rows of the packaged verified-models data (empty if unreadable)."""
    try:
        text = (
            resources.files("basemode")
            .joinpath("data", "verified_models_details.json")
            .read_text()
        )
        payload = json.loads(text)
    except Exception:
        return []
    rows = payload.get("rows")
    return rows if isinstance(rows, list) else []


@lru_cache(maxsize=1)
def _registry_quirks() -> dict[str, frozenset[str]]:
    """Map model stem -> quirk names, read from the packaged verified-models data."""
    result: dict[str, set[str]] = {}
    for row in _registry_rows():
        model = row.get("model")
        quirks = row.get("quirks") or []
        if isinstance(model, str) and quirks:
            stem = model.lower().split("/")[-1]
            result.setdefault(stem, set()).update(quirks)
    return {stem: frozenset(quirks) for stem, quirks in result.items()}


@lru_cache(maxsize=1)
def _registry_prompt_methods() -> dict[str, str]:
    """Map model stem -> the prompt method verified to work for it."""
    result: dict[str, str] = {}
    for row in _registry_rows():
        model = row.get("model")
        method = row.get("prompt_method")
        if isinstance(model, str) and isinstance(method, str) and method:
            result[model.lower().split("/")[-1]] = method
    return result


def _model_stem(model: str) -> str:
    return model.lower().split("/")[-1]


def model_quirks(model: str) -> frozenset[str]:
    """Registry-declared quirk names for this model (empty if none known)."""
    return _registry_quirks().get(_model_stem(model), frozenset())


def registry_prompt_method(model: str) -> str | None:
    """Strategy name verified to work for this model, if the registry knows one."""
    return _registry_prompt_methods().get(_model_stem(model))


# Known live Anthropic model IDs (from /v1/models, 2026-04-17).
# Used for best-effort alias resolution — typing `sonnet-4-5` expands to
# `claude-sonnet-4-5-20250929` when exactly one known model contains that substring.
KNOWN_ANTHROPIC_MODELS = {
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
}

# Thinking models: consume token budget on internal reasoning before output.
# Without a large token budget the visible output is empty or truncated.
# Key = substring to match in model name (after last /), value = (budget, min_output)
_THINKING_MODELS: dict[str, tuple[int, int]] = {
    "gemini-2.5-flash": (1024, 512),
    "gemini-2.5-flash-lite": (512, 256),
    "gemini-2.5-pro": (2048, 512),
    # gemini-pro-latest currently aliases a pro-tier model that rejects a
    # zero thinking budget outright ("This model only works in thinking
    # mode", probed live 2026-08-24) -- same tuning as gemini-2.5-pro.
    "gemini-pro-latest": (2048, 512),
    "kimi-k2.5": (4096, 512),  # Kimi K2.5 uses a large reasoning budget
    "kimi-k2.6": (4096, 512),  # Kimi K2.6 exhibits similar long-reasoning behavior
    "kimi-k2-thinking": (4096, 512),
    "kimi-k3": (4096, 512),  # same always-on reasoning behavior as k2-thinking
    "gpt-5.6-sol": (2048, 512),  # o-series-style reasoning, always on
    "gpt-5.6-terra": (2048, 512),
}

# gemini-3.x flash and its "latest" alias, probed live 2026-08-24: these
# default reasoning on and can silently burn the whole visible max_tokens
# budget on hidden thinking (finish_reason="length", nothing yielded).
# Unlike gemini-pro-latest, they accept reasoning_effort="none" and disabling
# is both cheaper and simpler than guessing a big-enough budget.
_GEMINI_DISABLE_THINKING_STEMS = frozenset(
    {"gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-flash-latest"}
)

_ZAI_DISABLE_THINKING_PREFIXES = (
    "glm-4.5",
    "glm-4.6",
    "glm-4.7",
    "glm-5",
)

# glm-5.3, probed live 2026-08-24: unlike the rest of the glm-5.x family
# above, it rejects `thinking.type: "disabled"` outright ("This model always
# engages in thinking and cannot be disabled; please use low, high, or max")
# and needs an effort level plus a wider budget instead -- checked before
# the prefix match above, which would otherwise catch it via "glm-5".
_ZAI_MANDATORY_THINKING_STEMS = frozenset({"glm-5.3"})

# Claude 5.x (opus-5, sonnet-5, ...), probed live 2026-08-24: this family
# rejects the older `thinking.type: "enabled"` shape outright ("Use
# thinking.type.adaptive and output_config.effort instead"), and also now
# rejects `thinking.type: "disabled"` ("Thinking defaults to adaptive mode
# when not specified"). So the only shape Anthropic still accepts is no
# `thinking` kwarg at all, which is also the fastest: adaptive-by-default
# produces clean output instantly instead of racing an unbounded reasoning
# budget.
_ANTHROPIC_ADAPTIVE_ONLY_PATTERN = re.compile(
    r"^claude-(opus|sonnet|haiku|fable)-5(\.\d+)?(-\d{8})?$"
)

# Claude 5's adaptive reasoning can consume a very small completion allowance
# before emitting visible text.  Anthropic does not accept an explicit budget
# for these models, so reserve provider-side headroom with max_tokens instead.
# Callers that need an exact visible limit use continue_text's
# strict_max_tokens=True; Loom does this for every branch.
_ANTHROPIC_ADAPTIVE_MIN_TOKENS = 512


def no_temperature(model: str) -> bool:
    if "no_temperature" in model_quirks(model):
        return True
    return any(p.match(_model_stem(model)) for p in _NO_TEMPERATURE_PATTERNS)


def no_prefill(model: str) -> bool:
    return "no_prefill" in model_quirks(model)


# Fallback budget for models tagged with the registry's generic
# `reasoning_budget` quirk (see `scripts/discover_new_models.py` and
# `scripts/probe_model_quirks.py`, which detect and add it automatically).
# Deliberately more generous than any tuned entry in `_THINKING_MODELS`,
# since it's applied without knowing how much a given model actually needs.
_GENERIC_REASONING_BUDGET = (4096, 1024)


#: Providers confirmed live to accept the bare Anthropic-shaped
#: ``thinking: {"type": "enabled", "budget_tokens": ...}`` kwarg as-is
#: (litellm translates it for gemini under the hood). Every other provider
#: defaults to the safe "just widen max_tokens" path: `model_quirks` is
#: keyed by model *stem*, so a reseller hosting the same model under the
#: same stem (e.g. `deepinfra/deepseek-ai/DeepSeek-V4-Flash` sharing
#: `deepseek/deepseek-v4-flash`'s `reasoning_budget` quirk, probed live
#: 2026-08-24: deepinfra 400s on an unrecognized `thinking` param) inherits
#: this quirk without ever being reviewed for whether it accepts the shape.
_NATIVE_THINKING_KWARG_PROVIDERS = frozenset({"anthropic", "gemini"})


def _reasoning_budget_kwargs(
    *,
    budget: int,
    min_out: int,
    max_tokens: int,
    provider: str,
    via_openrouter: bool,
    via_gemma: bool = False,
) -> dict:
    adjusted = max(max_tokens, budget + min_out)
    if via_gemma:
        # Gemma models reject any thinking-budget shape outright ("Thinking
        # budget is not supported for this model", probed live 2026-08-24)
        # even though they silently burn hidden reasoning tokens with no way
        # to see or cap them -- widen-only, same as a non-native provider,
        # despite gemma living under the "gemini" provider.
        return {"max_tokens": adjusted}
    if via_openrouter:
        # OpenRouter separates visible completion cap from thinking budget.
        # Preserve the caller's max_tokens instead of inflating it.
        return {
            "max_tokens": max_tokens,
            "extra_body": {"thinking": {"budget_tokens": budget}},
        }
    if provider not in _NATIVE_THINKING_KWARG_PROVIDERS:
        # Safe default: widen the raw token budget so there is room left
        # after hidden reasoning, without guessing at a provider-specific
        # thinking-control shape it may reject outright (or, for gemma-style
        # models, may not support at all -- see _GEMMA_NO_THINKING_CONTROL).
        return {"max_tokens": adjusted}
    return {
        "thinking": {"type": "enabled", "budget_tokens": budget},
        "max_tokens": adjusted,
    }


def thinking_kwargs(model: str, max_tokens: int) -> dict:
    stem = _model_stem(model)
    lower_model = model.lower()
    provider = lower_model.split("/", 1)[0] if "/" in lower_model else "unknown"
    via_openrouter = provider == "openrouter"
    via_zai = provider == "zai"
    via_anthropic = provider == "anthropic"
    via_gemma = stem.startswith("gemma-")
    if via_zai and stem in _ZAI_MANDATORY_THINKING_STEMS:
        budget, min_out = _GENERIC_REASONING_BUDGET
        return {
            "extra_body": {"thinking": {"type": "enabled", "effort": "low"}},
            "max_tokens": max(max_tokens, budget + min_out),
        }
    if via_zai and stem.startswith(_ZAI_DISABLE_THINKING_PREFIXES):
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    if via_anthropic and _ANTHROPIC_ADAPTIVE_ONLY_PATTERN.match(stem):
        return {"max_tokens": max(max_tokens, _ANTHROPIC_ADAPTIVE_MIN_TOKENS)}
    if stem in _GEMINI_DISABLE_THINKING_STEMS:
        return {"reasoning_effort": "none"}
    for fragment, (budget, min_out) in _THINKING_MODELS.items():
        if fragment in stem:
            return _reasoning_budget_kwargs(
                budget=budget,
                min_out=min_out,
                max_tokens=max_tokens,
                provider=provider,
                via_openrouter=via_openrouter,
                via_gemma=via_gemma,
            )
    if "reasoning_budget" in model_quirks(model):
        budget, min_out = _GENERIC_REASONING_BUDGET
        return _reasoning_budget_kwargs(
            budget=budget,
            min_out=min_out,
            max_tokens=max_tokens,
            provider=provider,
            via_openrouter=via_openrouter,
            via_gemma=via_gemma,
        )
    return {}


def build_kwargs(params: GenerationParams) -> dict:
    """Build litellm kwargs with model-specific compatibility applied."""
    kwargs: dict = {
        "max_tokens": params.max_tokens,
        # Ask for real token/cost usage on the streamed response instead of
        # relying solely on local tokenizer estimates (see `usage_capture.py`).
        # litellm passes this through for OpenAI-compatible providers; for
        # Anthropic it's a no-op since usage is always included on stream.
        "stream_options": {"include_usage": True},
    }
    if not no_temperature(params.model):
        kwargs["temperature"] = params.temperature
    kwargs.update(thinking_kwargs(params.model, params.max_tokens))
    kwargs.update(params.extra)
    return kwargs

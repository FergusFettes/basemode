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
    "kimi-k2.5": (4096, 512),  # Kimi K2.5 uses a large reasoning budget
    "kimi-k2.6": (4096, 512),  # Kimi K2.6 exhibits similar long-reasoning behavior
    "kimi-k2-thinking": (4096, 512),
    "kimi-k3": (4096, 512),  # same always-on reasoning behavior as k2-thinking
    "gpt-5.6-sol": (2048, 512),  # o-series-style reasoning, always on
    "gpt-5.6-terra": (2048, 512),
}

_ZAI_DISABLE_THINKING_PREFIXES = (
    "glm-4.5",
    "glm-4.6",
    "glm-4.7",
    "glm-5",
)

# Claude 5.x (opus-5, sonnet-5, ...), probed live 2026-08-23: this family
# rejects the older `thinking.type: "enabled"` shape outright ("Use
# thinking.type.adaptive and output_config.effort instead"), and the
# replacement `"adaptive"` shape has wildly unpredictable reasoning-token
# consumption — even 5000+ tokens sometimes produces no visible output at
# all. Disabling thinking entirely (same trick as the zai/glm family above)
# is simpler and reliable: basemode wants raw continuation, not chat
# reasoning, and it produces clean output instantly at any budget instead of
# racing an unbounded one.
_ANTHROPIC_ADAPTIVE_ONLY_PATTERN = re.compile(
    r"^claude-(opus|sonnet|haiku|fable)-5(\.\d+)?(-\d{8})?$"
)


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


def _reasoning_budget_kwargs(
    *,
    budget: int,
    min_out: int,
    max_tokens: int,
    via_moonshot: bool,
    via_openai: bool,
    via_openrouter: bool,
) -> dict:
    adjusted = max(max_tokens, budget + min_out)
    if via_moonshot or via_openai:
        # Neither takes an Anthropic-style `thinking` kwarg — just widen the
        # raw token budget so there's room left after hidden reasoning.
        return {"max_tokens": adjusted}
    if via_openrouter:
        # OpenRouter separates visible completion cap from thinking budget.
        # Preserve the caller's max_tokens instead of inflating it.
        return {
            "max_tokens": max_tokens,
            "extra_body": {"thinking": {"budget_tokens": budget}},
        }
    return {
        "thinking": {"type": "enabled", "budget_tokens": budget},
        "max_tokens": adjusted,
    }


def thinking_kwargs(model: str, max_tokens: int) -> dict:
    stem = _model_stem(model)
    lower_model = model.lower()
    via_openrouter = lower_model.startswith("openrouter/")
    via_moonshot = lower_model.startswith("moonshot/")
    via_zai = lower_model.startswith("zai/")
    via_openai = lower_model.startswith("openai/")
    via_anthropic = lower_model.startswith("anthropic/")
    if via_zai and stem.startswith(_ZAI_DISABLE_THINKING_PREFIXES):
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    if via_anthropic and _ANTHROPIC_ADAPTIVE_ONLY_PATTERN.match(stem):
        return {"thinking": {"type": "disabled"}}
    for fragment, (budget, min_out) in _THINKING_MODELS.items():
        if fragment in stem:
            return _reasoning_budget_kwargs(
                budget=budget,
                min_out=min_out,
                max_tokens=max_tokens,
                via_moonshot=via_moonshot,
                via_openai=via_openai,
                via_openrouter=via_openrouter,
            )
    if "reasoning_budget" in model_quirks(model):
        budget, min_out = _GENERIC_REASONING_BUDGET
        return _reasoning_budget_kwargs(
            budget=budget,
            min_out=min_out,
            max_tokens=max_tokens,
            via_moonshot=via_moonshot,
            via_openai=via_openai,
            via_openrouter=via_openrouter,
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

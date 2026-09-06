"""Canonical `provider/creator/model` identity, separate from the wire ID.

Two different questions get asked about a model and they need different
answers. *What do I call this to reach it?* is the wire ID — whatever litellm
and the provider expect, and nothing here may change it. *Which model is
this, and who made it?* is the canonical ID, and that is what records,
grouping and comparison want.

Providers answer the second question inconsistently or not at all. Resellers
publish `creator/model` but disagree on the creator's name: the same
organisation appears as `deepseek-ai` and `deepseek`, `zai-org` and `z-ai`,
`minimax` and `minimaxai`. First-party providers publish no creator at all,
because from where they stand it is obvious. So the canonical form is derived
here rather than trusted from the ID.

`canonical_id` is a pure function of the wire ID and the packaged catalog. It
is never sent to a provider.
"""

from __future__ import annotations

from functools import cache

#: Creator names that mean the same organisation, folded onto one spelling.
#: Keyed by the variants providers actually publish; extend it when a new
#: reseller invents another spelling for someone who is already here.
CREATOR_ALIASES: dict[str, str] = {
    "deepseek-ai": "deepseek",
    "z-ai": "zai",
    "zai-org": "zai",
    "minimaxai": "minimax",
    "meta-llama": "meta",
    "moonshotai": "moonshot",
    "x-ai": "xai",
    "mistralai": "mistral",
    "alibaba": "qwen",
    "qwen-ai": "qwen",
    "google-deepmind": "google",
    "nvidia-nim": "nvidia",
    "nim": "nvidia",
}

#: The creator a provider means when it publishes no creator at all. Only
#: correct for a provider serving its own models, so a reseller must not
#: appear here — see `_CREATOR_BY_STEM_PREFIX` for those.
FIRST_PARTY_CREATOR: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "google",
    "xai": "xai",
    "zai": "zai",
    "moonshot": "moonshot",
    "deepseek": "deepseek",
}

#: Resellers that publish bare IDs anyway. Matched as a prefix of the model
#: stem, longest first, because the names are hyphenated rather than pathed
#: (`cerebras/zai-glm-4.6`, `groq/gemma-7b-it`).
_CREATOR_BY_STEM_PREFIX: tuple[tuple[str, str], ...] = (
    ("zai-glm", "zai"),
    ("glm", "zai"),
    ("gpt-oss", "openai"),
    ("gpt", "openai"),
    ("gemma", "google"),
    ("gemini", "google"),
    ("llama", "meta"),
    ("qwen", "qwen"),
    ("kimi", "moonshot"),
    ("deepseek", "deepseek"),
    ("mistral", "mistral"),
    ("mixtral", "mistral"),
    ("minimax", "minimax"),
    ("grok", "xai"),
    ("phi", "microsoft"),
    ("nemotron", "nvidia"),
)

UNKNOWN_CREATOR = "unknown"


def canonical_creator(name: str) -> str:
    """Fold one published creator name onto its canonical spelling.

    OpenRouter prefixes a creator with `~` on its floating "latest" aliases
    (`~anthropic/claude-opus-latest`); that marks the model as an alias, not
    a different organisation.
    """
    lowered = name.strip().lower().lstrip("~")
    return CREATOR_ALIASES.get(lowered, lowered)


@cache
def _creator_by_stem() -> dict[str, str]:
    """Model stem -> creator, learned from every catalog that publishes one.

    A reseller that names a model bare still usually has a sibling somewhere
    that names it `creator/model`, so the catalog answers for most of them
    without a hand-written entry.
    """
    from .live_models import _cached_catalog

    index: dict[str, str] = {}
    for catalog in _cached_catalog().values():
        models = catalog.get("models")
        if not isinstance(models, dict):
            continue
        for model_id in models:
            creator, separator, stem = str(model_id).lower().partition("/")
            if separator and stem:
                index.setdefault(stem, canonical_creator(creator))
    return index


def split_wire_id(wire_id: str) -> tuple[str, str]:
    """Split a wire ID into its provider route and the rest."""
    route, separator, rest = wire_id.lower().partition("/")
    return (route, rest) if separator else ("unknown", route)


def creator_of(wire_id: str) -> str:
    """Who made the model this wire ID reaches."""
    route, rest = split_wire_id(wire_id)
    published, separator, _ = rest.partition("/")
    if separator and published:
        return canonical_creator(published)
    first_party = FIRST_PARTY_CREATOR.get(route)
    if first_party:
        return first_party
    known = _creator_by_stem().get(rest)
    if known:
        return known
    for prefix, creator in _CREATOR_BY_STEM_PREFIX:
        if rest.startswith(prefix):
            return creator
    return UNKNOWN_CREATOR


def model_stem(wire_id: str) -> str:
    """The model's own name, without provider route or creator."""
    return split_wire_id(wire_id)[1].rpartition("/")[2]


def canonical_id(wire_id: str) -> str:
    """`provider/creator/model` for a wire ID. Never send this to a provider."""
    route, _ = split_wire_id(wire_id)
    return f"{route}/{creator_of(wire_id)}/{model_stem(wire_id)}"

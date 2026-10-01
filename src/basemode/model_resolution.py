"""Resolve user model names to catalog wire IDs without provider requests."""

from difflib import get_close_matches
from functools import cache

from .identity import canonical_id, qualify_model_id
from .models import _all_provider_pairs


@cache
def _catalog_index() -> tuple[set[str], dict[str, list[str]], dict[str, list[str]]]:
    wires = {qualify_model_id(p, m) for p, m in _all_provider_pairs()}
    exact: dict[str, list[str]] = {}
    canonical: dict[str, list[str]] = {}
    for wire in sorted(wires):
        exact.setdefault(wire.lower(), []).append(wire)
        canonical.setdefault(canonical_id(wire), []).append(wire)
    return wires, exact, canonical


def match_catalog_model(normalized: str) -> str | None:
    """Resolve a qualified name, allowing unknown wire IDs to pass through."""
    wire_ids, by_wire, by_canonical = _catalog_index()
    lookup = normalized.lower()
    if normalized in wire_ids:
        return normalized
    exact = by_wire.get(lookup, [])
    canonical = canonical_id(normalized)
    matches = exact or by_canonical.get(canonical, [])
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(
            f"Ambiguous model {normalized!r}. Choose: {', '.join(matches)}"
        )
    return None


def resolve_catalog_model(model: str) -> str:
    """Accept aliases, wire IDs and canonical IDs; reject unknown/ambiguous IDs."""
    from .detect import _normalize_model

    normalized = _normalize_model(model)
    matched = match_catalog_model(normalized)
    if matched:
        return matched
    wire_ids = sorted(_catalog_index()[0])
    lookup = normalized.lower()
    names = {wire.lower(): wire for wire in wire_ids}
    names.update({canonical_id(wire): wire for wire in wire_ids})
    route = lookup.partition("/")[0]
    candidates = [name for name in names if name.startswith(f"{route}/")]
    suggestions = list(
        dict.fromkeys(
            names[name]
            for name in get_close_matches(lookup, candidates, n=3, cutoff=0.6)
        )
    )
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise ValueError(
        f"Unknown model {model!r} in the local catalog.{hint} "
        "Use --force for an unlisted model."
    )

"""Resolve user model names to catalog wire IDs without provider requests."""

from difflib import get_close_matches

from .detect import normalize_model
from .identity import canonical_id
from .models import _all_provider_pairs


def resolve_catalog_model(model: str) -> str:
    """Accept aliases, wire IDs and canonical IDs; reject unknown/ambiguous IDs."""
    normalized = normalize_model(model)
    wire_ids = sorted(
        {
            name if name.startswith(f"{provider}/") else f"{provider}/{name}"
            for provider, name in _all_provider_pairs()
        }
    )
    lookup = normalized.lower()
    exact = [wire for wire in wire_ids if wire.lower() == lookup]
    matches = exact or [wire for wire in wire_ids if canonical_id(wire) == lookup]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(f"Ambiguous model {model!r}. Choose: {', '.join(matches)}")
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

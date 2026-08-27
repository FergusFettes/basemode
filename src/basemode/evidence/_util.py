"""Small shared helpers used across the evidence package."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

# These are deliberately conservative. Unknown endpoints remain eligible: many
# provider chat catalogs have no modality field at all. We only exclude a model
# when provider metadata or its product name makes the non-text purpose clear.
TEXT_MODALITIES = frozenset({"text", "chat", "completion", "responses", "language"})
NON_TEXT_MODALITIES = frozenset(
    {
        "audio",
        "audio_generation",
        "embedding",
        "embeddings",
        "image",
        "image_generation",
        "moderation",
        "rerank",
        "speech",
        "stt",
        "tts",
        "transcription",
        "video",
        "video_generation",
    }
)
_NON_TEXT_NAME_RE = re.compile(
    r"(?:^|[/_.:-])(?:audio|bge|bria|chatterbox|clip-vit|content-safety|csm|dall-e|"
    r"diariz\w*|embedding(?:gemma)?|embed|e5|flux|gte|higgsaudio|ideogram|image|"
    r"imagen|i2v|kokoro|llama-guard|llama-prompt-guard|lyria|moderation|"
    r"nano-banana|ocr|orpheus|pixverse|r2v|realtime|rerank|safeguard|seedance|"
    r"seedream|sentence-transformers|sora|speech|sdxl|stable-diffusion|t2v|text2vec|"
    r"transcrib\w*|tts|veo|video|vidu|wan|whisper)"
    r"(?:$|[/_.:-])",
    re.IGNORECASE,
)
_NON_GENERATION_NAME_RE = re.compile(
    r"(?:^|[/_.:-])(?:content-safety|llama-guard|llama-prompt-guard|moderation|"
    r"rerank|safeguard)(?:$|[/_.:-])",
    re.IGNORECASE,
)


def classify_text_endpoint(
    model: str, modality: str | None = None
) -> tuple[bool, str | None]:
    """Return text eligibility and a durable, human-readable exclusion reason."""
    specialized = _NON_GENERATION_NAME_RE.search(model)
    if specialized:
        family = specialized.group(0).strip("/_.:-").lower()
        return False, f"non-generation model family: {family}"
    normalized_modality = (modality or "").strip().lower()
    if normalized_modality in NON_TEXT_MODALITIES:
        return False, f"provider modality: {normalized_modality}"
    if normalized_modality in TEXT_MODALITIES:
        return True, None
    match = _NON_TEXT_NAME_RE.search(model)
    if match:
        return False, f"non-text model family: {match.group(0).strip('/_.:-').lower()}"
    return True, None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _as_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _catalog_text_modality(metadata: Mapping[str, Any]) -> str | None:
    """Project rich catalog capabilities onto text eligibility.

    Output capability is intentionally decisive. Image input with text output
    is a valid text-generation endpoint; image-only output is not.
    """
    outputs = {
        str(value).strip().lower()
        for value in metadata.get("output_modalities", [])
        if isinstance(value, str)
    }
    if outputs:
        known_non_text = outputs & NON_TEXT_MODALITIES
        if known_non_text:
            return sorted(known_non_text)[0]
        if outputs & TEXT_MODALITIES:
            return "text"
    methods = {
        str(value).strip().lower()
        for value in metadata.get("supported_methods", [])
        if isinstance(value, str)
    }
    if "generatecontent" in methods:
        return "text"
    provider_type = metadata.get("provider_type")
    if isinstance(provider_type, str):
        normalized_type = provider_type.strip().lower()
        if normalized_type in TEXT_MODALITIES | NON_TEXT_MODALITIES:
            return normalized_type
    value = metadata.get("modality") or metadata.get("mode")
    return str(value) if isinstance(value, str) else None

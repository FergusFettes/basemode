"""Conservative text-generation endpoint classification."""

from __future__ import annotations

import re

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
    r"(?:^|[/_.:-])(?:asr|audio|bge|bria|chatterbox|clip-vit|content-safety|csm|"
    r"dall-e|diariz\w*|embedding(?:gemma)?|embed|e5|flux|gte|higgsaudio\w*|"
    r"ideogram|image|imagen|i2v|kokoro|llama-guard|llama-prompt-guard|lyria|moderation|"
    r"nano-banana|ocr|orpheus|pixverse|r2v|rerank|safeguard|seedance|"
    r"seedream|sentence-transformers|sora|speech|sdxl|stable-diffusion|t2v|text2vec|"
    r"transcrib\w*|tts|veo|video|vidu|voxtral|wan|whisper)(?:$|[/_.:-])",
    re.IGNORECASE,
)
# Endpoints that speak a protocol of their own rather than plain chat
# completion. They answer a listing request like any text model and then
# reject an ordinary completion outright, so a sweep that trusts the catalog
# spends three requests each discovering that (observed 2026-09-04 across
# Gemini Live, Omni, robotics and computer-use IDs).
_PROTOCOL_SPECIFIC_NAME_RE = re.compile(
    r"(?:^|[/_.:-])(?:bidi|computer-use|live|omni|realtime|robotics)"
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
    """Exclude only endpoints known by modality or product family to be non-text."""
    specialized = _NON_GENERATION_NAME_RE.search(model)
    if specialized:
        family = specialized.group(0).strip("/_.:-").lower()
        return False, f"non-generation model family: {family}"
    protocol = _PROTOCOL_SPECIFIC_NAME_RE.search(model)
    if protocol:
        family = protocol.group(0).strip("/_.:-").lower()
        return False, f"protocol-specific endpoint family: {family}"
    normalized = (modality or "").strip().lower()
    if normalized in NON_TEXT_MODALITIES:
        return False, f"provider modality: {normalized}"
    match = _NON_TEXT_NAME_RE.search(model)
    if match:
        return False, f"non-text model family: {match.group(0).strip('/_.:-').lower()}"
    if normalized in TEXT_MODALITIES:
        return True, None
    return True, None

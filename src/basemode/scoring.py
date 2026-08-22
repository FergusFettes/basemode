"""Score how well a continuation behaves like a continuation.

Coercing a chat model into base-mode either works or it doesn't, and the
failure modes are recognizable: an acknowledgment ("Sure, here's..."), a
refusal, the prefix parroted back, a code fence, a chat transcript. This
module turns those into a single 0..1 score plus the flags that explain it,
so strategy choice can be measured instead of guessed.

Used by `basemode bench` to rank strategies for a model, and by
`scripts/discover_new_models.py` to decide whether a newly-listed model is
worth registering at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Each penalty is (flag, weight). Weights are subtracted from 1.0 and the
# result is clamped, so a single heavy flag is enough to disqualify a
# strategy while a couple of light ones only nudge the ranking.
_REFUSAL_WEIGHT = 0.9
_PREAMBLE_WEIGHT = 0.6
_ECHO_WEIGHT = 0.5
_CHAT_TURN_WEIGHT = 0.4
_META_WEIGHT = 0.4
_QUOTED_WEIGHT = 0.3
_CODE_FENCE_WEIGHT = 0.3
_LIST_WEIGHT = 0.25
_BOUNDARY_WEIGHT = 0.15

_PREAMBLE_STARTS = (
    "sure",
    "of course",
    "certainly",
    "absolutely",
    "got it",
    "understood",
    "here's",
    "here is",
    "here are",
    "i'll",
    "i will",
    "i'd",
    "i would",
    "i can",
    "as an ai",
    "as a language model",
    "continuing",
    "continuation:",
)

_REFUSAL_RE = re.compile(
    r"^\W*(i'm sorry|i am sorry|i can't|i cannot|i can not|i won't|i will not"
    r"|i'm unable|i am unable|sorry,)",
    re.IGNORECASE,
)

# Talking *about* the text rather than being the text.
_META_RE = re.compile(
    r"\b(the (?:above |provided |given )?text|your text|the passage|the prompt"
    r"|this continuation|the continuation)\b",
    re.IGNORECASE,
)

_CHAT_TURN_RE = re.compile(
    r"(^|\n)\s*(user|assistant|human|system)\s*:|<\|im_(start|end)\|>",
    re.IGNORECASE,
)

_LIST_START_RE = re.compile(r"^\s*(?:[-*•]\s|\d+[.)]\s|#{1,6}\s)")
_CODE_FENCE_START_RE = re.compile(r"^\s*(?:```|~~~)")
_QUOTE_WRAPPED_RE = re.compile(r'^\s*(?:"""|\'\'\'|"|“)')

_WORD_RE = re.compile(r"[A-Za-z0-9']+")

# How much prefix tail to look for at the head of the continuation, and the
# shortest repeat that counts as an echo rather than a coincidence.
_ECHO_LOOKBEHIND = 60
_MIN_ECHO_CHARS = 12


@dataclass(frozen=True)
class ContinuationScore:
    """A 0..1 continuation-purity score and the flags that produced it."""

    score: float
    flags: tuple[str, ...]

    @property
    def clean(self) -> bool:
        """True when nothing disqualifying was found."""
        return self.score >= 0.75

    @property
    def detail(self) -> str:
        return ", ".join(self.flags) if self.flags else "clean"


def _looks_like_prose(text: str) -> bool:
    """Rough check: is this prose, where a list or fence would be out of place?"""
    stripped = text.strip()
    if not stripped:
        return False
    if _CODE_FENCE_START_RE.match(stripped) or _LIST_START_RE.match(stripped):
        return False
    return bool(_WORD_RE.search(stripped))


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def _echo_length(prefix: str, text: str) -> int:
    """Longest tail of `prefix` that the continuation repeats at its head."""
    tail = _normalized(prefix[-_ECHO_LOOKBEHIND:])
    head = _normalized(text)[: _ECHO_LOOKBEHIND * 2]
    if not tail or not head:
        return 0
    for size in range(min(len(tail), len(head)), _MIN_ECHO_CHARS - 1, -1):
        if head.startswith(tail[-size:]):
            return size
    return 0


def score_continuation(prefix: str, text: str) -> ContinuationScore:
    """Score `text` as a continuation of `prefix`.

    1.0 is a clean continuation; 0.0 is unusable (empty, refused, or buried
    under enough assistant behavior to be worthless). Flags name every
    problem found, heaviest first.
    """
    if not text.strip():
        return ContinuationScore(0.0, ("empty",))

    penalties: list[tuple[str, float]] = []
    body = text.lstrip()
    lowered = body.lower()

    if _REFUSAL_RE.match(body):
        penalties.append(("refusal", _REFUSAL_WEIGHT))
    elif lowered.startswith(_PREAMBLE_STARTS):
        penalties.append(("preamble", _PREAMBLE_WEIGHT))

    if _echo_length(prefix, text):
        penalties.append(("echoed_prefix", _ECHO_WEIGHT))
    if _CHAT_TURN_RE.search(text):
        penalties.append(("chat_turn", _CHAT_TURN_WEIGHT))
    if _META_RE.search(text):
        penalties.append(("meta_commentary", _META_WEIGHT))
    if _QUOTE_WRAPPED_RE.match(body):
        penalties.append(("quoted", _QUOTED_WEIGHT))

    prose_prefix = _looks_like_prose(prefix)
    if prose_prefix and _CODE_FENCE_START_RE.match(body):
        penalties.append(("code_fence", _CODE_FENCE_WEIGHT))
    if prose_prefix and _LIST_START_RE.match(body):
        penalties.append(("list_formatting", _LIST_WEIGHT))

    # Tie-breaker only: a welded word boundary matters when the continuation
    # is otherwise clean. On top of a preamble or refusal it is just noise —
    # "Sure!" starts with a letter too — so it would double-count a failure
    # that is already fully accounted for.
    if not penalties and _missing_boundary_space(prefix, text):
        penalties.append(("bad_boundary", _BOUNDARY_WEIGHT))

    score = max(0.0, 1.0 - sum(weight for _, weight in penalties))
    flags = tuple(flag for flag, _ in sorted(penalties, key=lambda p: -p[1]))
    return ContinuationScore(round(score, 3), flags)


def _missing_boundary_space(prefix: str, text: str) -> bool:
    """True when the continuation welds a new word onto the prefix's last one."""
    if not prefix or not text or prefix[-1].isspace() or text[0].isspace():
        return False
    # Punctuation joins are legitimate ("headland" + ", then...").
    return prefix[-1].isalnum() and text[0].isalnum()


def looks_clean(prefix: str, text: str) -> tuple[bool, str | None]:
    """Boolean form of `score_continuation`, for pass/fail probes.

    Returns `(ok, reason)` where `reason` is None on success.
    """
    result = score_continuation(prefix, text)
    if result.clean:
        return True, None
    return False, f"{result.detail}: {text.strip()[:120]!r}"
